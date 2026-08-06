import argparse
import hashlib
from pathlib import Path

import cv2
from ultralytics import YOLO

# SCOUT Phase 3 prep: negative-mining frame extraction.
# Finds frames with zero detections from the current model, so V3 training
# has real examples of "nothing here" from the actual deployment domains
# (including off-campus scenes like the boardwalk, which is exactly where
# the current model's false-positive tendencies showed up). This produces
# CANDIDATE images only -- a human still needs to review them before
# uploading to Roboflow; this script doesn't label or curate, just surfaces
# a diverse, evenly-spread sample of empty frames to look through.


def sharpness_score(frame) -> float:
    """Variance of the Laplacian -- a standard, cheap blur metric. A sharp
    image has lots of high-frequency edge content (high variance); a motion-
    blurred one is smoothed out (low variance). Needed for PTZ cameras
    (pan-tilt-zoom): most individual frames are perfectly good stills, but
    ones caught mid-sweep are genuinely blurred and shouldn't be used as
    training data regardless of what the detector does or doesn't see in them."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def find_empty_frames(model: YOLO, video_path: Path, conf: float, iou: float, imgsz: int, frame_stride: int, soft_conf_threshold: float | None = None, target_class: str | None = None, min_sharpness: float | None = None) -> list[int]:
    """Returns frame indices that are candidates for negative mining, checking
    every frame_stride-th frame. Decodes sequentially (grab() every frame,
    only retrieve()+predict() on the ones we keep) rather than seeking via
    CAP_PROP_POS_FRAMES -- repeated random-access seeks on long compressed
    video are expensive (each one decodes forward from the nearest keyframe),
    the same limitation noted in Claude.md's frame-seek caveat. Sequential
    decoding avoids that entirely.

    Three modes, mutually exclusive:
    - default: a frame with literally zero detections of ANY class qualifies.
      This is the only mode that produces true "nothing here" negatives safe
      to upload with zero boxes.
    - soft_conf_threshold set: a frame where every detection present is below
      that confidence also qualifies -- targets scenes that are never truly
      empty (e.g. a busy public space) but have a persistent low-confidence
      false positive worth teaching the model to suppress, like the
      recurring frame-edge "vehicle" artifact found in the beach footage.
      Only use on footage you're confident has no real objects being missed
      at that confidence level -- a real, uncounted object would poison the
      training data by teaching the model to ignore something it can
      actually see.
    - target_class set: a frame qualifies if it has zero detections of that
      SPECIFIC class, regardless of what else is in frame (e.g. "person"
      absent even though a parked bicycle is visible). These are NOT safe
      zero-box negatives -- if other real objects are present, they need
      proper labels before upload, or the unlabeled object poisons the data
      the same way. main() routes these to a separately-named output so
      they're never confused with the zero-box case.

    min_sharpness, if set, additionally rejects any candidate frame below
    that Laplacian-variance score -- for PTZ camera footage, where a frame
    can be technically empty-of-detections simply because it's too blurred
    mid-sweep for the detector to see anything, real or not."""
    target_class_id = None
    if target_class is not None:
        name_to_id = {name: cid for cid, name in model.names.items()}
        if target_class not in name_to_id:
            raise ValueError(f"'{target_class}' is not a class this model knows about: {list(name_to_id)}")
        target_class_id = name_to_id[target_class]

    cap = cv2.VideoCapture(str(video_path))
    candidate_frames = []
    frame_idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if frame_idx % frame_stride == 0:
            ret, frame = cap.retrieve()
            if ret:
                if min_sharpness is not None and sharpness_score(frame) < min_sharpness:
                    frame_idx += 1
                    continue

                result = model.predict(source=frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
                n = len(result.boxes) if result.boxes is not None else 0

                if target_class_id is not None:
                    if n == 0 or target_class_id not in result.boxes.cls.int().tolist():
                        candidate_frames.append(frame_idx)
                elif n == 0:
                    candidate_frames.append(frame_idx)
                elif soft_conf_threshold is not None and result.boxes.conf.max().item() < soft_conf_threshold:
                    candidate_frames.append(frame_idx)
        frame_idx += 1

    cap.release()
    return candidate_frames


def evenly_sample(candidates: list, n: int) -> list:
    """Spreads the sample across the full candidate range rather than taking
    the first N, so mined negatives cover different times/lighting/content
    instead of clustering in one stretch of video."""
    if len(candidates) <= n:
        return candidates
    step = len(candidates) / n
    return [candidates[int(i * step)] for i in range(n)]


def extract_and_save(video_path: Path, frame_indices: list, output_dir: Path, camera_label: str) -> int:
    cap = cv2.VideoCapture(str(video_path))
    saved = 0
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = output_dir / f"{camera_label}_frame{frame_idx:07d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1
    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="SCOUT Phase 3 prep: extract candidate negative-mining frames (zero detections) for manual review before Roboflow upload")
    parser.add_argument("--weights", default="weights/best.pt", help="Path relative to wherever you run this from -- run from the repo root to use the default")
    parser.add_argument("--source", default="videos", help="Directory of videos, or a single video file")
    parser.add_argument("--output-dir", default="negative_mining_candidates")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--frame-stride", type=int, default=15, help="Check every Nth frame (default 15; checking every single frame is unnecessary and slow)")
    parser.add_argument("--images-per-video", type=int, default=40, help="Target number of candidate images per video, evenly spread across its empty-frame pool")
    parser.add_argument("--soft-confidence-threshold", type=float, default=None,
                         help="Also accept frames where every detection is below this confidence (in addition to truly zero-detection frames). "
                              "Only use on footage you're confident has no real objects at that confidence level -- see the warning in find_empty_frames.")
    parser.add_argument("--target-class", default=None,
                         help="Find frames missing only this class (e.g. 'person'), regardless of other classes present. "
                              "NOT safe zero-box negatives if other real objects are in frame -- routed to a separate '_needs_labeling' output folder, "
                              "not the main negative-mining output, since any other visible object needs a proper label before upload.")
    parser.add_argument("--min-sharpness", type=float, default=None,
                         help="Reject candidate frames below this Laplacian-variance sharpness score. Use for PTZ camera footage, "
                              "where individual frames caught mid-pan/tilt/zoom are genuinely blurred and not usable stills, "
                              "regardless of what the detector does or doesn't see in them.")
    args = parser.parse_args()

    if args.target_class and args.soft_confidence_threshold:
        print("--target-class and --soft-confidence-threshold are mutually exclusive modes; pick one.")
        return

    source_path = Path(args.source)
    video_files = [source_path] if source_path.is_file() else sorted(
        p for p in source_path.glob("*") if p.suffix.lower() in (".mp4", ".avi", ".mov")
    )
    if not video_files:
        print(f"No videos found at {source_path}")
        return

    if args.target_class:
        output_dir = Path(args.output_dir) / f"{args.target_class}_absent_needs_labeling"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    print(f"Loaded {args.weights}")

    total_saved = 0
    for video_path in video_files:
        print(f"\n--- Scanning {video_path.name} for candidate frames (every {args.frame_stride}th frame) ---")
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        candidate_frames = find_empty_frames(model, video_path, args.conf, args.iou, args.imgsz, args.frame_stride,
                                              args.soft_confidence_threshold, args.target_class, args.min_sharpness)
        print(f"  -> {len(candidate_frames)} candidate frames found out of {total_frames // args.frame_stride} checked")

        sampled = evenly_sample(candidate_frames, args.images_per_video)
        # A truncated name plus a short hash of the FULL stem, not just the
        # truncation alone -- two videos that share the same first 40
        # characters (e.g. multiple recordings of the same livestream,
        # differing only in a trailing timestamp) would otherwise produce
        # identical camera_labels and silently overwrite each other's saved
        # frames whenever their frame indices happened to collide. Found
        # this the hard way: the 3 Church Street Marketplace clips share an
        # identical 40-char prefix and were clobbering each other.
        name_hash = hashlib.sha1(video_path.stem.encode()).hexdigest()[:8]
        camera_label = video_path.stem[:40].replace(" ", "_") + "_" + name_hash
        saved = extract_and_save(video_path, sampled, output_dir, camera_label)
        print(f"  -> saved {saved} candidate images")
        total_saved += saved

    if args.target_class:
        print(f"\nDone. {total_saved} '{args.target_class}'-absent candidate images in {output_dir}/ -- "
              f"these may still contain OTHER real objects (e.g. parked bicycles) that need proper labels before upload. Not zero-box negatives.")
    else:
        print(f"\nDone. {total_saved} candidate negative-mining images in {output_dir}/ -- review before uploading to Roboflow.")


if __name__ == "__main__":
    main()
