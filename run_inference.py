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
# --extract-embeddings computes a per-track appearance embedding for the
# person class only, and saves it to a companion .npz file -- it does NOT
# assign a Global_ID here. That used to happen inline in this script, but
# real cross-camera identity resolution needs real-world coordinates (to
# check spatial-temporal plausibility and reconcile overlapping cameras),
# which don't exist until AFTER homography_middleware.py transform runs.
# Global_ID assignment now happens in mcmt_fusion.py, operating across every
# camera at a site at once, on already-transformed CSVs. This script's job is
# strictly per-camera: detect, track, and extract the embeddings a later
# stage will need -- matching what "all processing happens at the camera"
# means for the edge deployment target.
PERSON_CLASS_NAME = "person"
EMBEDDING_WARMUP_FRAMES = 5  # crops averaged into one embedding per track, to smooth over one bad/occluded frame


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
    # Per local Track_ID, crops collected until EMBEDDING_WARMUP_FRAMES is
    # reached, at which point one averaged embedding is computed and stored.
    # Rows are emitted immediately regardless (no Global_ID to wait for
    # anymore), unlike the old buffer-until-resolved design.
    pending_crops = {}
    embedded_track_ids = set()

    def flush_embedding(track_id):
        crops = pending_crops.pop(track_id)
        if not crops:
            return
        vectors = embedder.embed(crops)
        mean_vector = vectors.mean(axis=0)
        mean_vector = mean_vector / np.linalg.norm(mean_vector)
        embedding_records.append({"camera_id": camera_id, "track_id": track_id, "vector": mean_vector})
        embedded_track_ids.add(track_id)

    for frame_idx, result in enumerate(results):
        if max_frames is not None and frame_idx >= max_frames:
            break

        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue

        timestamp = round(start_timestamp + frame_idx / fps, 2)
        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        frame = result.orig_img if embeddings_enabled else None

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

            if embeddings_enabled and class_name == PERSON_CLASS_NAME and track_id not in embedded_track_ids:
                crops = pending_crops.setdefault(track_id, [])
                if len(crops) < EMBEDDING_WARMUP_FRAMES:
                    crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
                    if crop.size > 0:
                        crops.append(crop)
                    if len(crops) >= EMBEDDING_WARMUP_FRAMES:
                        flush_embedding(track_id)

    # Any person tracks still warming up when the video ended: embed with
    # whatever crops they did accumulate (even just one) rather than drop them.
    for track_id in list(pending_crops.keys()):
        flush_embedding(track_id)

    return rows


def main():
    parser = argparse.ArgumentParser(description="SCOUT Phase 1: ByteTrack inference -> CSV telemetry")
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--source", default="videos", help="Directory of .mp4/.avi/.mov videos, or a single video file")
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
        print(f"\nExtracted embeddings for {len(embedding_records)} person tracks across this run")

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
                np.savez(
                    embeddings_path,
                    camera_ids=np.array([r["camera_id"] for r in camera_embeddings]),
                    track_ids=np.array([r["track_id"] for r in camera_embeddings]),
                    vectors=np.stack([r["vector"] for r in camera_embeddings]),
                )
                print(f"Camera '{camera_id}': embeddings saved to {embeddings_path}")


if __name__ == "__main__":
    main()
