import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from reid.gallery import MCMTGallery, DEFAULT_MAX_SPEED_MPS, DEFAULT_OVERLAP_RADIUS_METERS, DEFAULT_SIMILARITY_THRESHOLD

# SCOUT cross-camera fusion stage. Runs AFTER homography_middleware.py
# transform: its input is a tracking CSV that already has real-world
# coordinates (Local_X_Meters/Local_Y_Meters), because spatial-temporal
# plausibility gating is meaningless without them (raw pixels aren't
# comparable across cameras, and raw lat/long degrees aren't a constant
# real-world distance). Its companion embeddings .npz files come from
# run_inference.py --extract-embeddings, one per source video/camera.
#
# One Global_ID is assigned per real person across every camera at the site,
# using reid.gallery.MCMTGallery (appearance + spatial-temporal plausibility,
# resolved in time-sorted batches via the Hungarian algorithm rather than
# greedily) -- see reid/gallery.py for why the earlier one-at-a-time,
# appearance-only matcher had to be replaced.
#
# Input is one homography-transformed CSV per camera (the output of
# run_inference.py's per-camera writing, after transform), not one
# pre-combined file -- matching the real deployment architecture where each
# camera/edge node only ever writes its own file (see Claude.md, CSV Output
# Architecture). Two kinds of output are written:
#   1. Per-camera fused CSVs: each input CSV with Global_ID and Confidence
#      columns added for every person-class row (every row of a track gets
#      that track's resolved Global_ID/confidence; non-person rows are left
#      blank -- cross-camera identity is scoped to people only, see Claude.md
#      Phase 2.5). These are the raw/audit layer, one per camera, same as the
#      inputs.
#   2. One composite/fused CSV for the whole site: one row per (Global_ID,
#      time bucket) resolving simultaneous overlapping-camera sightings of
#      the same person into a single location -- the highest-confidence
#      camera's position wins for that instant, per the "generic/automatic"
#      overlap-resolution decision in Claude.md. This is the single canonical
#      file third-party tools (GIS, Grasshopper, Blender) should read.

# Person-track candidates are grouped into windows of this many seconds and
# resolved together, so the Hungarian assignment considers every plausible
# pairing within a real time slice rather than one track at a time. Wide
# enough to catch a person walking camera-to-camera at ordinary speed, narrow
# enough to keep each window's candidate/gallery cost matrix small.
DEFAULT_WINDOW_SECONDS = 5.0

# Rounding granularity for the composite CSV's "simultaneous sighting" bucket
# -- two cameras' frames are essentially never sampled at identical
# timestamps, so sightings of the same Global_ID this close together are
# treated as one real-world instant rather than two separate site-position
# rows.
DEFAULT_COMPOSITE_BUCKET_SECONDS = 0.5

PERSON_CLASS_NAME = "person"


def load_embeddings(npz_paths: list[Path]) -> dict:
    """Returns {(camera_id, track_id): vector}, merged across every embeddings file given."""
    lookup = {}
    for path in npz_paths:
        data = np.load(path, allow_pickle=True)
        for camera_id, track_id, vector in zip(data["camera_ids"], data["track_ids"], data["vectors"]):
            lookup[(str(camera_id), int(track_id))] = vector
    return lookup


def build_candidates(df: pd.DataFrame, embeddings: dict) -> list[dict]:
    """One candidate per (Camera, Track_ID) person track: entry point (first
    sighting) is used to query the gallery -- "could this plausibly be
    someone last seen elsewhere" -- since that's the moment relevant to a
    cross-camera hand-off. The track's exit point (last sighting) is applied
    afterward as that Global_ID's new last-known state (see
    reid/gallery.py's update_last_known docstring)."""
    person_df = df[df["Class"] == PERSON_CLASS_NAME]
    candidates = []
    skipped_no_embedding = 0

    for (camera_id, track_id), group in person_df.groupby(["Camera", "Track_ID"]):
        key = (str(camera_id), int(track_id))
        if key not in embeddings:
            skipped_no_embedding += 1
            continue
        group = group.sort_values("Timestamp")
        entry = group.iloc[0]
        exit_ = group.iloc[-1]
        candidates.append({
            "camera_id": str(camera_id),
            "track_id": int(track_id),
            "timestamp": float(entry["Timestamp"]),
            "x": float(entry["Local_X_Meters"]),
            "y": float(entry["Local_Y_Meters"]),
            "vector": np.asarray(embeddings[key], dtype=np.float32),
            "exit_timestamp": float(exit_["Timestamp"]),
            "exit_x": float(exit_["Local_X_Meters"]),
            "exit_y": float(exit_["Local_Y_Meters"]),
        })

    if skipped_no_embedding:
        print(f"  Warning: {skipped_no_embedding} person track(s) had no matching embedding and were skipped "
              f"(re-run run_inference.py --extract-embeddings for that video if this is unexpected)")

    return candidates


def resolve_global_ids(candidates: list[dict], gallery: MCMTGallery, window_seconds: float) -> dict:
    """Returns {(camera_id, track_id): {"global_id":, "confidence":}}."""
    candidates = sorted(candidates, key=lambda c: c["timestamp"])
    resolved = {}

    i = 0
    while i < len(candidates):
        window_start = candidates[i]["timestamp"]
        window = []
        while i < len(candidates) and candidates[i]["timestamp"] < window_start + window_seconds:
            window.append(candidates[i])
            i += 1

        results = gallery.resolve_window(window)
        for candidate, result in zip(window, results):
            resolved[(candidate["camera_id"], candidate["track_id"])] = result
            # Correct the gallery's last-known state to this track's exit
            # point now that matching (which needed the entry point) is done.
            gallery.update_last_known(
                result["global_id"], candidate["vector"],
                candidate["exit_x"], candidate["exit_y"], candidate["exit_timestamp"], candidate["camera_id"],
            )

    return resolved


def build_composite_csv(df: pd.DataFrame, bucket_seconds: float) -> pd.DataFrame:
    person_df = df[(df["Class"] == PERSON_CLASS_NAME) & df["Global_ID"].notna()].copy()
    if person_df.empty:
        return person_df

    person_df["_bucket"] = (person_df["Timestamp"] / bucket_seconds).round().astype(int)
    person_df["Confidence"] = pd.to_numeric(person_df["Confidence"], errors="coerce")

    # Highest-confidence sighting wins for each (Global_ID, time bucket); a
    # missing/None confidence (a brand-new, unmatched identity -- nothing to
    # compare it against yet) is treated as the lowest priority, not silently
    # dropped, so a genuinely single-camera sighting still appears.
    person_df["_confidence_rank"] = person_df["Confidence"].fillna(-1.0)
    person_df = person_df.sort_values("_confidence_rank", ascending=False)
    composite = person_df.drop_duplicates(subset=["Global_ID", "_bucket"], keep="first")
    composite = composite.drop(columns=["_bucket", "_confidence_rank"]).sort_values(["Global_ID", "Timestamp"])
    return composite


def main():
    parser = argparse.ArgumentParser(description="SCOUT cross-camera fusion: assigns Global_ID across every camera at one site")
    parser.add_argument("--tracking-csvs", nargs="+", required=True,
                         help="One or more per-camera, homography-transformed CSVs (output of homography_middleware.py transform, "
                              "one file per camera_id -- see run_inference.py's --output-dir)")
    parser.add_argument("--embeddings", nargs="+", required=True, help="One or more *_embeddings.npz files from run_inference.py --extract-embeddings")
    parser.add_argument("--output-dir", default="Site_Analyzer_Batch_Runs",
                         help="One <camera_id>_fused.csv (input CSV + Global_ID/Confidence) is written per camera here -- the "
                              "per-camera raw/audit layer, mirroring the input files.")
    parser.add_argument("--composite-output", default="Site_Analyzer_Batch_Runs/full_site_composite.csv",
                         help="One row per (Global_ID, time bucket): the single site-wide unified person-path dataset")
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--composite-bucket-seconds", type=float, default=DEFAULT_COMPOSITE_BUCKET_SECONDS)
    parser.add_argument("--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--max-speed-mps", type=float, default=DEFAULT_MAX_SPEED_MPS)
    parser.add_argument("--overlap-radius-meters", type=float, default=DEFAULT_OVERLAP_RADIUS_METERS)
    args = parser.parse_args()

    per_camera_frames = []
    for tracking_csv in args.tracking_csvs:
        camera_df = pd.read_csv(tracking_csv)
        for col in ("Local_X_Meters", "Local_Y_Meters"):
            if col not in camera_df.columns:
                raise SystemExit(f"'{col}' not found in {tracking_csv} -- run homography_middleware.py transform first")
        per_camera_frames.append(camera_df)
    df = pd.concat(per_camera_frames, ignore_index=True)

    embeddings = load_embeddings([Path(p) for p in args.embeddings])
    print(f"Loaded {len(embeddings)} person-track embeddings from {len(args.embeddings)} file(s)")

    candidates = build_candidates(df, embeddings)
    print(f"Built {len(candidates)} person-track candidates for cross-camera resolution")

    if not candidates:
        print("No person tracks with embeddings found -- nothing to fuse.")
        return

    feature_dim = candidates[0]["vector"].shape[0]
    gallery = MCMTGallery(
        feature_dim=feature_dim,
        similarity_threshold=args.similarity_threshold,
        max_speed_mps=args.max_speed_mps,
        overlap_radius_meters=args.overlap_radius_meters,
    )
    resolved = resolve_global_ids(candidates, gallery, args.window_seconds)
    print(f"Resolved {len(candidates)} tracks into {len(gallery.global_ids)} distinct Global_IDs")

    df["Global_ID"] = None
    df["Confidence"] = None
    for (camera_id, track_id), result in resolved.items():
        mask = (df["Camera"].astype(str) == camera_id) & (df["Track_ID"] == track_id) & (df["Class"] == PERSON_CLASS_NAME)
        df.loc[mask, "Global_ID"] = result["global_id"]
        df.loc[mask, "Confidence"] = result["confidence"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for camera_id, camera_df in df.groupby("Camera"):
        camera_path = output_dir / f"{camera_id}_fused.csv"
        camera_df.to_csv(camera_path, index=False)
        print(f"Camera '{camera_id}': fused CSV written to {camera_path}")

    composite = build_composite_csv(df, args.composite_bucket_seconds)
    composite_path = Path(args.composite_output)
    composite_path.parent.mkdir(parents=True, exist_ok=True)
    composite.to_csv(composite_path, index=False)
    print(f"Composite site-wide person-path CSV written to {composite_path} ({len(composite)} rows)")


if __name__ == "__main__":
    main()
