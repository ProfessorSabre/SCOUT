import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent / "reid"))
from embedder import PersonEmbedder  # noqa: E402

# SCOUT Phase 1: persistent tracking + CSV telemetry export.
# Uses Ultralytics' built-in ByteTrack (tracker="bytetrack.yaml") instead of a
# separate tracking library. Replaces the old annotated-.mp4 output with the
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
PERSON_CLASS_NAME = "person"

# Appearance fingerprints: each person track keeps several, from different
# moments across the whole track, rather than one averaged from its first few
# frames. A person seen from behind, then the side, then the front looks
# different each time; keeping the views separate (instead of blurring them
# into one average) is what lets a later sighting from any side find its match.
CROP_SAMPLE_INTERVAL_S = 0.2    # consecutive frames are near-identical; one candidate per 0.2 s per track is plenty
MAX_CANDIDATES_PER_TRACK = 60   # when exceeded, every other candidate is dropped, keeping even coverage of the whole track
FINGERPRINTS_PER_TRACK = 20     # stored per track; mcmt_fusion.py decides how many to actually use
TRACK_END_SECONDS = 3.0         # a track unseen this long is finished: fingerprint it and free its crops (ByteTrack drops lost tracks after ~1 s)
EDGE_MARGIN_PX = 2              # a box touching the frame edge shows only part of the person
MAX_OVERLAP_FRACTION = 0.2      # a box this covered by another person's box mixes two people's appearance


def get_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else 30.0


def load_camera_map(path: str | None) -> dict:
    """Loads {video_filename_stem: {"camera_id": ..., "start_timestamp": ...}}.
    Both fields are optional per video; missing ones fall back to the
    filename stem / 0.0 respectively."""
    if not path:
        return {}
    with open(path, "r") as f:
        return json.load(f)


def process_video(model: YOLO, video_path: Path, camera_id: str, start_timestamp: float, conf: float, iou: float, imgsz: int,
                   max_frames: int | None = None, embedder: PersonEmbedder | None = None,
                   embedding_records: list | None = None) -> list[dict]:
    fps = get_fps(video_path)
    embeddings_enabled = embedder is not None

    results = model.track(
        source=str(video_path),
        persist=True,
        tracker="bytetrack.yaml",
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        stream=True,
        verbose=False,
    )

    rows = []
    # Per person track: candidate crops [(timestamp, height_px, crop)], when the
    # last candidate was taken, and when the track was last seen at all.
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

        timestamp = round(start_timestamp + frame_idx / fps, 2)
        if embeddings_enabled:
            for track_id in [t for t, seen in last_seen.items() if timestamp - seen > TRACK_END_SECONDS]:
                finish_track(track_id)

        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        frame = result.orig_img if embeddings_enabled else None
        if embeddings_enabled:
            frame_h, frame_w = frame.shape[:2]
            person_boxes = xyxy[[model.names[c] == PERSON_CLASS_NAME for c in class_ids]]

        for (x1, y1, x2, y2), track_id, cls_id in zip(xyxy, track_ids, class_ids):
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
            })

            if embeddings_enabled and class_name == PERSON_CLASS_NAME:
                last_seen[track_id] = timestamp
                if timestamp - last_sampled.get(track_id, -np.inf) < CROP_SAMPLE_INTERVAL_S:
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
                last_sampled[track_id] = timestamp
                if len(pool) > MAX_CANDIDATES_PER_TRACK:
                    candidates[track_id] = pool[::2]

    # Tracks still open when the video ends are finished with whatever they have.
    for track_id in list(last_seen):
        finish_track(track_id)

    return rows


def main():
    parser = argparse.ArgumentParser(description="SCOUT Phase 1: ByteTrack inference -> CSV telemetry")
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--source", default="videos/sdd_videos", help="Directory of .mp4/.avi/.mov videos, or a single video file")
    parser.add_argument("--output-dir", default="Site_Analyzer_Batch_Runs",
                         help="One <camera_id>_tracking.csv (and, with --extract-embeddings, one <camera_id>_tracking_embeddings.npz) "
                              "is written per camera here -- matching the real deployment architecture where each camera/edge node "
                              "only ever writes its own file (see Claude.md, CSV Output Architecture). Videos sharing a camera_id "
                              "(e.g. multiple sessions from the same physical camera) are combined into that camera's one file.")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--camera-map", default=None,
                         help='Optional JSON file mapping {video_filename_stem: {"camera_id": ..., "start_timestamp": ...}}. '
                              'Assigns the permanent camera_id used by the calibration registry and for cross-camera timestamp sync. '
                              "Defaults to the video's filename stem / start_timestamp=0.0 for any field or video not listed.")
    parser.add_argument("--max-frames", type=int, default=None, help="Debug: stop each video after N frames (for quick smoke tests)")
    parser.add_argument("--extract-embeddings", action="store_true",
                         help="Extract a per-track appearance embedding for the person class, saved to a per-camera "
                              "<camera_id>_tracking_embeddings.npz in --output-dir. Does NOT assign a Global_ID -- see "
                              "mcmt_fusion.py, which resolves cross-camera identity on already-transformed (real-world coordinate) CSVs.")
    args = parser.parse_args()

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
    print(f"Loaded {args.weights} | classes: {model.names}")

    embedder = None
    embedding_records = []
    if args.extract_embeddings:
        embedder = PersonEmbedder()
        print(f"Embedding extraction enabled: PersonEmbedder on {embedder.device}")

    rows_by_camera = {}
    for video_path in video_files:
        video_entry = camera_map.get(video_path.stem, {})
        camera_id = video_entry.get("camera_id", video_path.stem)
        start_ts = video_entry.get("start_timestamp", 0.0)
        print(f"\n--- Tracking {video_path.name} as camera_id='{camera_id}' (start_timestamp={start_ts}) ---")
        rows = process_video(model, video_path, camera_id, start_ts, args.conf, args.iou, args.imgsz, args.max_frames, embedder, embedding_records)
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
