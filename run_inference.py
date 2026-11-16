import argparse
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent / "reid"))
from embedder import PersonEmbedder  # noqa: E402
from frame_alignment import ALIGN_INTERVAL_S, AlignmentTrack, FrameAligner  # noqa: E402

# SCOUT Phase 1: persistent tracking + CSV telemetry export.
# Uses an Ultralytics built-in tracker (TrackTrack by default, see
# DEFAULT_TRACKER) instead of a separate tracking library. Replaces the old annotated-.mp4 output with the
# structured [Camera, Timestamp, Track_ID, Class, Center_X, Center_Y] rows
# that Stage 1 of the SCOUT/SPACE workflow needs.
#
# The "Camera" column is a permanent camera_id (see homography_middleware.py),
# NOT the video filename -- a physical camera's calibration is keyed on that
# id and needs to resolve to the same value across every future capture
# session, regardless of what that day's video happens to be named. Pass
# --camera-map to assign real camera_ids; without it, the filename stem is
# used as a same-session convenience default, matching how it behaved before
# camera_id existed.
#
# --extract-embeddings computes appearance fingerprints (several per track,
# see FINGERPRINTS_PER_TRACK) for the person class only, and saves them to a
# companion .npz file -- it does NOT assign a Global_ID here. Fingerprints are
# vectors of numbers, not images: no picture of anyone is stored. That used to happen inline in this script, but
# real cross-camera identity resolution needs real-world coordinates (to
# check spatial-temporal plausibility and reconcile overlapping cameras),
# which don't exist until AFTER homography_middleware.py transform runs.
# Global_ID assignment now happens in mcmt_fusion.py, operating across every
# camera at a site at once, on already-transformed CSVs. This script's job is
# strictly per-camera: detect, track, and extract the embeddings a later
# stage will need -- matching what "all processing happens at the camera"
# means for the edge deployment target.
#
# Camera motion: a camera map entry may name a "reference_frame" image -- the
# image that camera's calibration points were clicked on (frame_alignment.py
# reference). Every frame is then lined up with it and the Center/Foot pixel
# columns are written in the reference image's pixels, so a camera that drifts
# keeps landing on the right ground position. Detection still runs on the
# original frames; the as-seen positions are kept in Raw_Foot_X/Raw_Foot_Y.
# Positions that can't be aligned are left blank rather than guessed.
PERSON_CLASS_NAME = "person"

# Default detector (2026-11-16): the stock COCO-trained YOLOv11m, which Ultralytics
# downloads on first use. On the Bell Tower footage it beat the project's own
# fine-tuned V2 model (weights/best.pt in this
# repository, still usable with --weights) on every continuity measure -- longer tracks, no
# persistent detections on tree canopy, head-on bicycles found -- and the medium
# size gave most of the gain over the small one (see the README's design notes). V2 is still the only model with scooter
# and golf-cart classes.
DEFAULT_WEIGHTS = "yolo11m.pt"

# Classes SCOUT records. A model trained on COCO detects 80 everyday classes
# (chairs, potted plants, handbags ...); only these are kept, and they are
# filtered before tracking so other objects can't join or interrupt tracks.
# Models trained on SCOUT's own classes (V2) keep everything they detect.
SCOUT_CLASSES = {"person", "bicycle", "scooter", "skateboard", "motorcycle", "car", "truck", "bus",
                 "vehicle", "golfcart", "wheelchair"}

# Appearance fingerprints: each person track keeps several, from different
# moments across the whole track, rather than one averaged from its first few
# frames. A person seen from behind, then the side, then the front looks
# different each time; keeping the views separate (instead of blurring them
# into one average) is what lets a later sighting from any side find its match.
CROP_SAMPLE_INTERVAL_S = 0.2    # consecutive frames are near-identical; one candidate per 0.2 s per track is plenty
MAX_CANDIDATES_PER_TRACK = 60   # when exceeded, every other candidate is dropped, keeping even coverage of the whole track
FINGERPRINTS_PER_TRACK = 20     # stored per track; mcmt_fusion.py decides how many to actually use
EDGE_MARGIN_PX = 2              # a box touching the frame edge shows only part of the person
MAX_OVERLAP_FRACTION = 0.2      # a box this covered by another person's box mixes two people's appearance

# A video joined from several recordings (a phone stopped and restarted, then
# the clips merged) has a time jump at each join. A camera map entry can list
# "segments" -- [{"from_video_s": 0, "start_timestamp": ...}, {"from_video_s":
# <join>, "start_timestamp": ...}, ...] -- giving each recording its own start
# time (clock_sync.py finds them). Track IDs in segment k are offset by
# k * SEGMENT_TRACK_STRIDE so that no track continues across a join.
SEGMENT_TRACK_STRIDE = 1_000_000

# Tracking thresholds. A detection must reach --conf (0.35, chosen to keep
# shadows from producing detections) to START a track, but once a track
# exists, the tracker may extend it with a weaker detection down to
# TRACK_KEEP_CONF -- typically the same person partly hidden by a branch or
# another pedestrian, whose score drops for a few frames. That second-stage
# association is ByteTrack's central idea; discarding everything below 0.35
# before the tracker sees it switched it off. A shadow still can't start a
# track. On 3 min of the Bell Tower PBT_02 camera this cut tracks 10% and
# raised the share of real-person detections in tracks of 10 s or more from
# 60% to 66%, with no added position jumps.
TRACK_KEEP_CONF = 0.10
# A track lost for this long is ended. Set in seconds and converted to frames
# per video: trackers count frames, so ByteTrack's 30-frame default meant 1 s
# at 30 fps but only 0.5 s at 60 fps. Raised from 2 s to 5 s on 2026-11-12
# with the switch to TrackTrack (below): people walking behind a tree, post or
# another pedestrian are often hidden for more than 2 s, and every such
# occlusion used to end the track and start a new ID.
TRACK_BUFFER_S = 5.0
# A track unseen for longer than the tracker would hold it is finished: its
# appearance fingerprints are computed and its crops freed.
TRACK_END_SECONDS = TRACK_BUFFER_S + 1.0

# Default tracker, chosen 2026-11-12 from a screening of every Ultralytics
# tracker on 8 min of all four Bell Tower cameras (yolo11m, full frame rate;
# see Claude.md). TrackTrack (CVPR 2025) starts new tracks only where no
# existing track already covers the detection and matches in several rounds
# with progressively looser gates. Against ByteTrack it cut new tracks per
# minute of person presence by 7-45% per camera, found 3% more person-time,
# and had no appearance-flagged ID swaps at two-person crossings (ByteTrack 2).
# Its own defaults are adjusted for occlusion:
#   lost_match_thr 0.9 -- a second, looser pass that re-binds tracks lost for a
#     while (0 = off by default; 0.9 is the value its documentation suggests
#     for long occlusions);
#   min_track_len 1 -- a new track is output from its first frame (scaled to 2
#     frames at 60 fps) instead of after 3; the 3-frame wait dropped brief real
#     glimpses of people through branches and the first frames of every track.
DEFAULT_TRACKER = "tracktrack"
TRACKER_DEFAULT_SETTINGS = {"tracktrack": {"lost_match_thr": 0.9, "min_track_len": 1}}


def get_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else 30.0


def load_camera_map(path: str | None) -> dict:
    """Loads {video_filename_stem: {"camera_id": ..., "start_timestamp": ...,
    "reference_frame": ...}}. Every field is optional per video; missing ones
    fall back to the filename stem / 0.0 / no alignment respectively. A
    relative reference_frame path is taken from the working directory, or
    else from the camera map's own folder."""
    if not path:
        return {}
    with open(path, "r") as f:
        camera_map = json.load(f)
    for entry in camera_map.values():
        if isinstance(entry, dict) and entry.get("reference_frame"):
            ref = Path(entry["reference_frame"])
            if not ref.is_absolute() and not ref.exists():
                ref = Path(path).parent / ref
            if not ref.exists():
                raise SystemExit(f"reference_frame not found: {entry['reference_frame']}")
            entry["reference_frame"] = str(ref)
    return camera_map


def session_clock(entry: dict):
    """video seconds -> (session timestamp, segment index) for one camera map
    entry: a single start_timestamp, or per-recording "segments"."""
    segments = sorted(entry.get("segments") or [], key=lambda seg: seg["from_video_s"])
    if not segments:
        start = entry.get("start_timestamp", 0.0)
        return lambda video_t: (start + video_t, 0)
    starts = np.array([seg["from_video_s"] for seg in segments], dtype=np.float64)

    def clock(video_t):
        k = max(int(np.searchsorted(starts, video_t, side="right")) - 1, 0)
        return segments[k]["start_timestamp"] + video_t - segments[k]["from_video_s"], k
    return clock


def align_rows(rows: list[dict], track: AlignmentTrack, video_times: list[float]) -> list[dict]:
    """Rewrites each row's Center/Foot pixels into the reference image's
    pixels, keeping the as-seen foot point in Raw_Foot_X/Raw_Foot_Y."""
    if not rows:
        return rows
    df = pd.DataFrame(rows)
    t = np.asarray(video_times, dtype=np.float64)
    df["Raw_Foot_X"], df["Raw_Foot_Y"] = df["Foot_X"], df["Foot_Y"]
    for x, y in (("Center_X", "Center_Y"), ("Foot_X", "Foot_Y")):
        mapped = track.map_points(t, df[[x, y]].to_numpy(dtype=np.float64))
        df[x], df[y] = np.round(mapped[:, 0], 1), np.round(mapped[:, 1], 1)
    unaligned = int(df["Foot_X"].isna().sum())
    if unaligned:
        print(f"  {unaligned:,} of {len(df):,} detections fell in stretches that couldn't be aligned -- positions left blank")
    return df.to_dict("records")


# Ultralytics' tracker settings that count frames. Their defaults are tuned
# for ~30 fps benchmark video, so they are scaled by fps / 30 to mean the same
# time at any frame rate (a 60 fps camera gets twice the frames).
FRAME_COUNT_SETTINGS = ("delta_t", "reset_velocity_offset_occ", "reset_pos_offset_occ",
                        "active_occ_to_lost_thresh", "occ_reappear_window", "min_track_len")
REFERENCE_FPS = 30.0


def tracker_config(start_conf: float, keep_conf: float, fps: float, tracker_type: str = DEFAULT_TRACKER,
                   reid: bool = False, buffer_s: float = TRACK_BUFFER_S, overrides: dict | None = None) -> str:
    """A tracker config file for this video's frame rate, built on Ultralytics'
    defaults for tracker_type, with SCOUT's thresholds applied to every type:
    start_conf starts a track, detections down to keep_conf may only extend one
    (the low-score second pass), and a track lost for buffer_s ends.
    overrides set any other key of the tracker's config before frame-count
    settings are scaled, so frame counts there mean frames at 30 fps."""
    import yaml
    from ultralytics.utils import ROOT as ULTRALYTICS_ROOT
    with open(ULTRALYTICS_ROOT / "cfg" / "trackers" / f"{tracker_type}.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg.update(track_high_thresh=start_conf, new_track_thresh=start_conf, track_low_thresh=keep_conf,
               track_buffer=max(1, int(round(fps * buffer_s))))
    if "use_byte" in cfg:      # OC-SORT family: same low-score second pass as ByteTrack
        cfg["use_byte"] = True
    if "gmc_method" in cfg:    # fixed cameras; slow drift is handled by frame_alignment
        cfg["gmc_method"] = "none"
    if "with_reid" in cfg:     # appearance from the detector's own features ("auto")
        cfg["with_reid"] = reid
    elif reid:
        raise ValueError(f"tracker '{tracker_type}' has no appearance (ReID) option")
    for key, value in {**TRACKER_DEFAULT_SETTINGS.get(tracker_type, {}), **(overrides or {})}.items():
        if key not in cfg:
            raise ValueError(f"tracker '{tracker_type}' has no setting '{key}'")
        cfg[key] = value
    for key in FRAME_COUNT_SETTINGS:
        if key in cfg:
            cfg[key] = max(1, int(round(cfg[key] * fps / REFERENCE_FPS)))
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
    return f.name


def process_video(model: YOLO, video_path: Path, camera_id: str, clock, conf: float, iou: float, imgsz: int,
                   max_frames: int | None = None, embedder: PersonEmbedder | None = None,
                   embedding_records: list | None = None, aligner: FrameAligner | None = None,
                   tracker: str | None = None, keep_conf: float = TRACK_KEEP_CONF,
                   process_fps: float | None = None, tracker_type: str = DEFAULT_TRACKER,
                   tracker_reid: bool = False, track_buffer_s: float = TRACK_BUFFER_S,
                   tracker_overrides: dict | None = None, classes: list[int] | None = None) -> list[dict]:
    video_fps = get_fps(video_path)
    # Optional frame skipping (an edge device may not keep up with every frame):
    # process every stride-th frame, chosen so the processed rate is as close as
    # possible to process_fps. Everything downstream that counts frames (the
    # tracker's lost-track timeout, alignment sampling) uses the processed rate.
    stride = max(1, int(round(video_fps / process_fps))) if process_fps else 1
    fps = video_fps / stride
    if stride > 1:
        print(f"  processing every {stride} frames: {fps:.2f} of {video_fps:.2f} frames per second")
    # Without an explicit tracker config, detections down to keep_conf go to
    # the tracker and a frame-rate-scaled config decides what starts a track.
    detection_floor = conf if tracker else keep_conf
    tracker = tracker or tracker_config(conf, keep_conf, fps, tracker_type, tracker_reid,
                                        track_buffer_s, tracker_overrides)
    embeddings_enabled = embedder is not None
    alignment = AlignmentTrack() if aligner is not None else None
    align_every = max(1, int(round(fps * ALIGN_INTERVAL_S)))
    # Alignment runs on the CPU in a background thread (OpenCV releases the
    # GIL), overlapping with detection on the GPU instead of stalling it.
    align_pool = ThreadPoolExecutor(max_workers=1) if aligner is not None else None
    pending_alignments = []

    results = model.track(
        source=str(video_path),
        persist=True,
        tracker=tracker,
        conf=detection_floor,
        iou=iou,
        imgsz=imgsz,
        stream=True,
        vid_stride=stride,
        classes=classes,
        verbose=False,
    )

    rows, video_times = [], []
    # Per person track: candidate crops [(timestamp, height_px, crop)], and in
    # video seconds, when the last candidate was taken and when the track was
    # last seen at all.
    candidates, last_sampled, last_seen = {}, {}, {}

    def finish_track(track_id):
        pool = candidates.pop(track_id, [])
        last_sampled.pop(track_id, None)
        last_seen.pop(track_id, None)
        if not pool:
            return
        # One fingerprint per stretch of the track (so views vary), each the
        # largest crop in its stretch (largest = most pixels to compare).
        bins = np.array_split(np.arange(len(pool)), min(FINGERPRINTS_PER_TRACK, len(pool)))
        chosen = [pool[max(b, key=lambda i: pool[i][1])] for b in bins if len(b)]
        vectors = embedder.embed([crop for _, _, crop in chosen])
        for (ts, height, _), vector in zip(chosen, vectors):
            embedding_records.append({"camera_id": camera_id, "track_id": track_id, "timestamp": ts,
                                      "height_px": height, "vector": vector})

    for frame_idx, result in enumerate(results):
        if max_frames is not None and frame_idx >= max_frames:
            break

        video_t = frame_idx / fps
        session_t, segment = clock(video_t)
        timestamp = round(session_t, 2)
        if alignment is not None and frame_idx % align_every == 0:
            pending_alignments.append((video_t, align_pool.submit(aligner.align, result.orig_img.copy())))
        if embeddings_enabled:
            for track_id in [t for t, seen in last_seen.items() if video_t - seen > TRACK_END_SECONDS]:
                finish_track(track_id)

        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = [t + segment * SEGMENT_TRACK_STRIDE for t in boxes.id.int().cpu().tolist()]
        class_ids = boxes.cls.int().cpu().tolist()
        scores = boxes.conf.cpu().tolist()
        frame = result.orig_img if embeddings_enabled else None
        if embeddings_enabled:
            frame_h, frame_w = frame.shape[:2]
            person_boxes = xyxy[[model.names[c] == PERSON_CLASS_NAME for c in class_ids]]

        for (x1, y1, x2, y2), track_id, cls_id, score in zip(xyxy, track_ids, class_ids, scores):
            class_name = model.names[cls_id]
            center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            rows.append({
                "Camera": camera_id,
                "Timestamp": timestamp,
                "Track_ID": track_id,
                "Class": class_name,
                "Center_X": round(float(center_x), 1),
                "Center_Y": round(float(center_y), 1),
                # Bottom-center (feet), not the box center, is what Phase 2's
                # homography needs to map to the ground plane accurately.
                "Foot_X": round(float(center_x), 1),
                "Foot_Y": round(float(y2), 1),
                "Detection_Conf": round(float(score), 3),
            })
            video_times.append(video_t)

            if embeddings_enabled and class_name == PERSON_CLASS_NAME:
                last_seen[track_id] = video_t
                if video_t - last_sampled.get(track_id, -np.inf) < CROP_SAMPLE_INTERVAL_S:
                    continue
                if score < conf:  # a weak detection (partly hidden person) would blur the fingerprint
                    continue
                if x1 <= EDGE_MARGIN_PX or y1 <= EDGE_MARGIN_PX or x2 >= frame_w - EDGE_MARGIN_PX or y2 >= frame_h - EDGE_MARGIN_PX:
                    continue
                area = max((x2 - x1) * (y2 - y1), 1.0)
                ix = np.clip(np.minimum(x2, person_boxes[:, 2]) - np.maximum(x1, person_boxes[:, 0]), 0, None)
                iy = np.clip(np.minimum(y2, person_boxes[:, 3]) - np.maximum(y1, person_boxes[:, 1]), 0, None)
                overlap = ix * iy / area
                overlap[np.argmax(overlap)] = 0.0  # this box's overlap with itself (1.0)
                if overlap.max(initial=0.0) > MAX_OVERLAP_FRACTION:
                    continue
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0:
                    continue
                pool = candidates.setdefault(track_id, [])
                pool.append((timestamp, float(y2 - y1), crop.copy()))
                last_sampled[track_id] = video_t
                if len(pool) > MAX_CANDIDATES_PER_TRACK:
                    candidates[track_id] = pool[::2]

    # Tracks still open when the video ends are finished with whatever they have.
    for track_id in list(last_seen):
        finish_track(track_id)

    if alignment is not None:
        for t, future in pending_alignments:
            alignment.add(t, *future.result())
        align_pool.shutdown()
        print(f"  {alignment.summary()}")
        rows = align_rows(rows, alignment, video_times)
    return rows


def main():
    parser = argparse.ArgumentParser(description="SCOUT Phase 1: tracking inference -> CSV telemetry")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help=f"Detection model (default {DEFAULT_WEIGHTS}, downloaded automatically on first use; the "
                             "project's V2 model is weights/best.pt)")
    parser.add_argument("--all-classes", action="store_true",
                        help="Record every class the model detects, not just SCOUT's (see SCOUT_CLASSES)")
    parser.add_argument("--source", default="videos/sdd_videos", help="Directory of .mp4/.avi/.mov videos, or a single video file")
    parser.add_argument("--output-dir", default="Site_Analyzer_Batch_Runs",
                         help="One <camera_id>_tracking.csv (and, with --extract-embeddings, one <camera_id>_tracking_embeddings.npz) "
                              "is written per camera here -- matching the real deployment architecture where each camera/edge node "
                              "only ever writes its own file (see Claude.md, CSV Output Architecture). Videos sharing a camera_id "
                              "(e.g. multiple sessions from the same physical camera) are combined into that camera's one file.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence needed to start a track (default 0.35)")
    parser.add_argument("--track-keep-conf", type=float, default=TRACK_KEEP_CONF,
                         help=f"Weaker detections down to this can extend an existing track (default {TRACK_KEEP_CONF})")
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tracker-type", default=DEFAULT_TRACKER,
                         choices=["bytetrack", "botsort", "ocsort", "deepocsort", "fasttrack", "tracktrack"],
                         help=f"Ultralytics tracker, run with SCOUT's thresholds and lost-track time (default {DEFAULT_TRACKER})")
    parser.add_argument("--tracker-reid", action="store_true",
                         help="Let the tracker also match by appearance, using the detector's own features "
                              "(botsort, deepocsort, tracktrack only)")
    parser.add_argument("--track-buffer-s", type=float, default=TRACK_BUFFER_S,
                         help=f"Seconds a lost track is kept before it ends (default {TRACK_BUFFER_S})")
    parser.add_argument("--tracker-set", action="append", default=[], metavar="KEY=VALUE",
                         help="Set any other setting of the chosen tracker's Ultralytics config (repeatable); "
                              "frame counts are frames at 30 fps and are scaled to the video's rate")
    parser.add_argument("--tracker", default=None,
                         help="An Ultralytics tracker config to use as is (detections below --conf are then dropped before "
                              "tracking); default: --tracker-type built from --conf, --track-keep-conf and --track-buffer-s")
    parser.add_argument("--camera-map", default=None,
                         help='Optional JSON file mapping {video_filename_stem: {"camera_id": ..., "start_timestamp": ..., '
                              '"reference_frame": ...}}. Assigns the permanent camera_id used by the calibration registry and '
                              'for cross-camera timestamp sync; reference_frame (the image the camera was calibrated on) turns '
                              "on camera-motion compensation; segments (see SEGMENT_TRACK_STRIDE) gives each recording in a "
                              "joined video its own start time. Defaults to the video's filename stem / start_timestamp=0.0 / "
                              "no alignment for any field or video not listed.")
    parser.add_argument("--max-frames", type=int, default=None, help="Debug: stop each video after N frames (for quick smoke tests)")
    parser.add_argument("--process-fps", type=float, default=None,
                         help="Process only enough frames for about this many per second (every Nth frame, "
                              "N = round(video fps / this)); default: every frame")
    parser.add_argument("--extract-embeddings", action="store_true",
                         help="Extract a per-track appearance embedding for the person class, saved to a per-camera "
                              "<camera_id>_tracking_embeddings.npz in --output-dir. Does NOT assign a Global_ID -- see "
                              "mcmt_fusion.py, which resolves cross-camera identity on already-transformed (real-world coordinate) CSVs.")
    args = parser.parse_args()
    import yaml
    tracker_overrides = {k: yaml.safe_load(v) for k, v in (item.split("=", 1) for item in args.tracker_set)}

    source_path = Path(args.source)
    if source_path.is_file():
        video_files = [source_path]
    else:
        video_files = sorted(p for p in source_path.glob("*") if p.suffix.lower() in (".mp4", ".avi", ".mov"))

    if not video_files:
        print(f"No videos found at {source_path}")
        return

    camera_map = load_camera_map(args.camera_map)

    model = YOLO(args.weights)
    keep = None if args.all_classes else [i for i, n in model.names.items() if n in SCOUT_CLASSES]
    print(f"Loaded {args.weights} | recording: {[model.names[i] for i in keep] if keep is not None else 'all classes'}")

    embedder = None
    embedding_records = []
    if args.extract_embeddings:
        embedder = PersonEmbedder()
        print(f"Embedding extraction enabled: PersonEmbedder on {embedder.device}")

    rows_by_camera = {}
    for video_path in video_files:
        video_entry = camera_map.get(video_path.stem, {})
        camera_id = video_entry.get("camera_id", video_path.stem)
        timing = (f"{len(video_entry['segments'])} recording segments" if video_entry.get("segments")
                  else f"start_timestamp={video_entry.get('start_timestamp', 0.0)}")
        reference = video_entry.get("reference_frame")
        aligner = FrameAligner(cv2.imread(reference)) if reference else None
        print(f"\n--- Tracking {video_path.name} as camera_id='{camera_id}' ({timing}"
              f"{', aligned to ' + reference if reference else ''}) ---")
        rows = process_video(model, video_path, camera_id, session_clock(video_entry), args.conf, args.iou, args.imgsz,
                             args.max_frames, embedder, embedding_records, aligner, args.tracker, args.track_keep_conf,
                             args.process_fps, args.tracker_type, args.tracker_reid, args.track_buffer_s,
                             tracker_overrides, keep)
        print(f"  -> {len(rows)} detections logged")
        rows_by_camera.setdefault(camera_id, []).extend(rows)

    if args.extract_embeddings:
        n_tracks = len({(r["camera_id"], r["track_id"]) for r in embedding_records})
        print(f"\nExtracted {len(embedding_records)} appearance fingerprints for {n_tracks} person tracks across this run")

    if not rows_by_camera:
        print("\nNo tracking data generated. Check your video files and confidence threshold.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for camera_id, rows in rows_by_camera.items():
        if not rows:
            print(f"\nCamera '{camera_id}': no detections -- no file written")
            continue
        df = pd.DataFrame(rows).sort_values(by="Timestamp")
        csv_path = output_dir / f"{camera_id}_tracking.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nCamera '{camera_id}': {len(df)} rows written to {csv_path}")

        if args.extract_embeddings:
            camera_embeddings = [r for r in embedding_records if r["camera_id"] == camera_id]
            if camera_embeddings:
                embeddings_path = output_dir / f"{camera_id}_tracking_embeddings.npz"
                # One row per fingerprint (several per track): which track, when it
                # was taken, how tall the person was in pixels, and the vector.
                np.savez(
                    embeddings_path,
                    camera_ids=np.array([r["camera_id"] for r in camera_embeddings]),
                    track_ids=np.array([r["track_id"] for r in camera_embeddings]),
                    timestamps=np.array([r["timestamp"] for r in camera_embeddings]),
                    heights_px=np.array([r["height_px"] for r in camera_embeddings]),
                    vectors=np.stack([r["vector"] for r in camera_embeddings]),
                )
                print(f"Camera '{camera_id}': embeddings saved to {embeddings_path}")


if __name__ == "__main__":
    main()
