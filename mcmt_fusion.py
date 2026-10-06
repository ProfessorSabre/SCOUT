import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as Polygon
from scipy.spatial import ConvexHull

from coverage_map import render_coverage_map
from homography_middleware import camera_entries, load_site_registry, site_origin_utm
from reid.gallery import (MCMTGallery, DEFAULT_MAX_SPEED_MPS, DEFAULT_OVERLAP_RADIUS_METERS,
                          DEFAULT_RETENTION_SECONDS, DEFAULT_SIMILARITY_THRESHOLD, set_similarity)

# SCOUT site fusion stage. Runs AFTER homography_middleware.py transform, on
# one transformed CSV per camera (positions in real-world meters), plus the
# appearance fingerprints from run_inference.py --extract-embeddings.
#
# What it produces, for one site and one session:
#   - One Global_ID per real person across every camera (reid/gallery.py),
#     with each person's travel mode on every row: on foot, or riding a
#     bicycle / scooter / skateboard / wheelchair, decided by whether the
#     person moves together with one (see associate_modes).
#   - <camera_id>_fused.csv per camera: the input rows + Global_ID, Confidence
#     and Mode. The per-camera raw/audit layer.
#   - full_site_composite.csv: one row per person per time bucket -- where
#     several cameras saw the same person at once, the most precise view wins
#     (largest in frame, inside its camera's calibrated area). The single
#     canonical person-path file for GIS / Grasshopper / Blender.
#   - full_site_vehicles.csv: every vehicle (bicycles, scooters, cars, buses,
#     ...) position over time, with no identity -- the same vehicle seen by two
#     overlapping cameras at once is kept once.
#   - counts_over_time.csv: how many people (by mode) and vehicles (by class)
#     are present, per interval. Presence counts don't depend on identity
#     linking at all, so they hold even where identities are fragmented.
#   - coverage_map.png: where each camera places people, and camera overlap.
#   - run_manifest.json: every setting, input and calibration value the run
#     used, so the run can be reported and reproduced exactly.
#
# A single camera works too; it is simply the N=1 case. Nothing here stores or
# needs images of people: fingerprints are vectors of numbers.

# Person-track candidates are grouped into windows of this many seconds and
# resolved together, so the Hungarian assignment considers every plausible
# pairing within a real time slice rather than one track at a time.
DEFAULT_WINDOW_SECONDS = 5.0

# Rounding granularity for "the same moment" in the composite, vehicles and
# counts outputs -- two cameras' frames are essentially never sampled at
# identical timestamps.
DEFAULT_BUCKET_SECONDS = 0.5

# Fingerprints smaller than this (person box height, pixels) aren't used for
# appearance matching, and a track with none left keeps its own Global_ID. This
# is about how many pixels there are to compare, not who the person is -- in
# practice it screens out distance; anyone close to a camera passes. A 33 px
# crop gave meaningless similarities in the Auburn validation while ~140 px
# crops separated people cleanly, and OSNet takes 256 px-tall input (a 60 px
# crop is already upscaled 4x). Judgment-based; adjust with --min-reid-height-px.
DEFAULT_MIN_REID_HEIGHT_PX = 60.0

# How many of a track's stored fingerprints are used for matching (spread
# across the track, largest per stretch). Chosen from the Founders Square
# comparison of 1, 3, 5, 10 and 20 (2026-10-06; 456 same-person track halves
# vs 629 certain-stranger pairs): at the bar letting 1% of strangers through,
# 3 or 5 kept 86% of same-person pairs, 1 kept 69%, and 10 or 20 kept 81% --
# every extra view is one more chance for a coincidental stranger look-alike.
# Same-camera pairs, so this ranks the settings; it doesn't predict
# cross-camera accuracy. See Claude.md / paper_notes.md.
DEFAULT_FINGERPRINTS_USED = 5

# Similarity-bar auto-calibration (--similarity-threshold auto): within each
# block of time, the bar is set so that only this share of pairs of tracks
# KNOWN to be different people (same camera, same moment) would pass as a
# match. Blocks with too few such pairs use the whole run's bar; a run with
# too few uses DEFAULT_SIMILARITY_THRESHOLD and says so.
DEFAULT_FALSE_MATCH_RATE = 0.01
DEFAULT_CALIBRATION_BLOCK_SECONDS = 3600.0
MIN_STRANGER_PAIRS = 30
# A same-camera pair only counts as two certain strangers if both tracks are
# in the same frames at least this often, standing at least this many body
# heights apart (median over those frames, box-foot to box-foot in pixels).
# Without these, the "strangers" include one person whose tracker ID
# flickered between two numbers (overlapping time spans, but never in one
# frame together) and doubled boxes on one person (same frames, a fraction
# of a body height apart). On Founders Square those were nearly all of the
# pairs scoring above 0.9, and they lifted the bar from ~0.85 to 0.92.
MIN_STRANGER_SHARED_FRAMES = 5
MIN_STRANGER_SEPARATION_HEIGHTS = 1.0

# Overlap hand-off by position (merge_co_sightings): two cameras' tracks that
# stay within this distance for at least this long are one person. On Founders
# Square, PFS_02/PFS_03 (leave-one-out calibration error ~0.35 m each) put
# 57 track pairs within 1.5 m of each other, then very few between 2 and 4 m,
# then the spread of unrelated people: 1.5 m sits at that gap. 2 s rules out
# strangers crossing paths. A camera whose own calibration error is near this
# radius can't be merged this way reliably -- the run warns about it.
CO_SIGHTING_RADIUS_M = 1.5
CO_SIGHTING_MIN_SECONDS = 2.0

# Travel mode: a person is riding when they and a bicycle/scooter/skateboard/
# wheelchair stay within this distance (meters, ground positions) AND move
# together: both moving at MIN_SPEED_MPS or more, with velocities (speed and
# direction, measured over the last second or the track's life so far) within
# VELOCITY_TOLERANCE_MPS of each other, over a span of at least
# MIN_ASSOCIATION_S. Moving together is what separates a rider from someone
# standing beside a parked bike. No speed categories: a cyclist at walking
# pace and one at 20+ mph are both riders. The radius allows for a rider's
# feet being on pedals rather than the ground, which a shallow camera angle
# stretches. A rider's person box flickers on and off (on Founders Square most
# rider person tracks lasted under a second), hence the short history and span.
RIDEABLE_CLASSES = {"bicycle", "scooter", "skateboard", "wheelchair"}
ASSOCIATION_RADIUS_M = 2.0
CO_MOVEMENT_WINDOW_S = 1.0
MIN_MOVEMENT_HISTORY_S = 0.25
MIN_SPEED_MPS = 0.5
VELOCITY_TOLERANCE_MPS = 1.0
MIN_ASSOCIATION_S = 0.5
ON_FOOT = "on foot"

# Two detections of the same vehicle class from DIFFERENT cameras at the same
# moment, closer than this, are one vehicle seen twice. Bigger for larger
# vehicles, whose ground point shifts more with viewing angle.
VEHICLE_MERGE_RADIUS_M = {"vehicle": 6.0, "bus": 10.0, "golfcart": 4.0}
DEFAULT_VEHICLE_MERGE_RADIUS_M = 3.0

DEFAULT_COUNT_INTERVAL_SECONDS = 60.0

# Position precision for choosing between cameras: a detection outside the
# area its camera's calibration points span is extrapolated, so it's weighted
# down; the hull is enlarged a little since the edge of that area is still fine.
OUTSIDE_CALIBRATION_WEIGHT = 0.5
CALIBRATION_HULL_MARGIN = 1.2

PERSON_CLASS_NAME = "person"


# --- Inputs -----------------------------------------------------------------

def load_fingerprints(npz_paths: list[Path]) -> dict:
    """{(camera_id, track_id): {"vectors": (k, D), "heights": (k,) or None,
    "timestamps": (k,) or None}}. Files from before multi-fingerprint tracks
    (one vector per track, no heights) load as single fingerprints."""
    lookup = {}
    for path in npz_paths:
        data = np.load(path, allow_pickle=True)
        has_meta = "heights_px" in data.files
        cams, tids, vecs = data["camera_ids"], data["track_ids"], data["vectors"]
        for idx in range(len(tids)):
            key = (str(cams[idx]), int(tids[idx]))
            entry = lookup.setdefault(key, {"vectors": [], "heights": [] if has_meta else None,
                                            "timestamps": [] if has_meta else None})
            entry["vectors"].append(vecs[idx])
            if has_meta:
                entry["heights"].append(float(data["heights_px"][idx]))
                entry["timestamps"].append(float(data["timestamps"][idx]))
    for entry in lookup.values():
        entry["vectors"] = np.asarray(entry["vectors"], dtype=np.float32)
        for k in ("heights", "timestamps"):
            if entry[k] is not None:
                entry[k] = np.asarray(entry[k])
    return lookup


def select_fingerprints(entry: dict, min_height_px: float, k: int) -> np.ndarray | None:
    """Up to k usable fingerprints, spread across the track: the largest in
    each of k stretches of time. None if the track has none tall enough."""
    if entry["heights"] is None:
        return entry["vectors"]
    usable = np.flatnonzero(entry["heights"] >= min_height_px)
    if len(usable) == 0:
        return None
    usable = usable[np.argsort(entry["timestamps"][usable])]
    bins = np.array_split(usable, min(k, len(usable)))
    return entry["vectors"][[b[np.argmax(entry["heights"][b])] for b in bins if len(b)]]


def build_candidates(df: pd.DataFrame, fingerprints: dict, min_height_px: float, k: int) -> list[dict]:
    """One candidate per (Camera, Track_ID) person track with a ground
    position. Its entry point is matched against the gallery; its exit point
    is where the identity is left. Tracks with no usable fingerprint are still
    candidates -- they get their own identity rather than disappearing."""
    person = df[df["Class"] == PERSON_CLASS_NAME].sort_values("Timestamp")
    # Old single-fingerprint files carry no heights: judge those tracks on the
    # box height of their first frames (height = 2 * (Foot_Y - Center_Y)).
    first_heights = (2 * (person["Foot_Y"] - person["Center_Y"])).groupby([person["Camera"], person["Track_ID"]]) \
        .apply(lambda s: s.head(5).median())

    positioned = person[person["Local_X_Meters"].notna() & person["Local_Y_Meters"].notna()]
    candidates = []
    for (camera_id, track_id), group in positioned.groupby(["Camera", "Track_ID"]):
        key = (str(camera_id), int(track_id))
        entry = fingerprints.get(key)
        exemplars = None
        if entry is not None:
            if entry["heights"] is None:
                exemplars = entry["vectors"] if first_heights.loc[(camera_id, track_id)] >= min_height_px else None
            else:
                exemplars = select_fingerprints(entry, min_height_px, k)
        entry_row, exit_row = group.iloc[0], group.iloc[-1]
        candidates.append({
            "camera_id": key[0], "track_id": key[1],
            "timestamp": float(entry_row["Timestamp"]),
            "x": float(entry_row["Local_X_Meters"]), "y": float(entry_row["Local_Y_Meters"]),
            "exit_timestamp": float(exit_row["Timestamp"]),
            "exit_x": float(exit_row["Local_X_Meters"]), "exit_y": float(exit_row["Local_Y_Meters"]),
            "exemplars": exemplars,
            "eligible": exemplars is not None,
        })
    return candidates


# --- Similarity bar ------------------------------------------------------------

def _clearly_apart(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """True when two same-camera tracks share enough frames, far enough apart,
    to be two different people (see MIN_STRANGER_* above)."""
    shared = a.index.intersection(b.index)
    if len(shared) < MIN_STRANGER_SHARED_FRAMES:
        return False
    a, b = a.loc[shared], b.loc[shared]
    gap = np.hypot(a["Foot_X"] - b["Foot_X"], a["Foot_Y"] - b["Foot_Y"])
    height = np.maximum(a["Height"], b["Height"])
    return float((gap / height).median()) >= MIN_STRANGER_SEPARATION_HEIGHTS


def stranger_pairs(candidates: list[dict], df: pd.DataFrame) -> np.ndarray:
    """(moment, similarity) for every pair of eligible tracks seen together in
    the same camera's frames, clearly apart -- pairs that are certainly two
    different people. Every recording contains these for free, so every run
    can measure how alike strangers look to the appearance model at THIS site.
    Scored with the same best-matching-view rule matching uses."""
    person = df[df["Class"] == PERSON_CLASS_NAME]
    boxes = person[["Camera", "Track_ID", "Timestamp", "Foot_X", "Foot_Y"]].assign(
        Height=2 * (person["Foot_Y"] - person["Center_Y"]))
    by_camera = {}
    for c in candidates:
        if c["eligible"]:
            by_camera.setdefault(c["camera_id"], []).append(c)
    eligible_keys = {(c["camera_id"], c["track_id"]) for c in candidates if c["eligible"]}
    frames = {key: g.drop(columns=["Camera", "Track_ID"]).drop_duplicates("Timestamp").set_index("Timestamp")
              for key, g in boxes.groupby([boxes["Camera"], boxes["Track_ID"].astype(int)]) if key in eligible_keys}

    out = []
    for tracks in by_camera.values():
        t0 = np.array([c["timestamp"] for c in tracks])
        t1 = np.array([c["exit_timestamp"] for c in tracks])
        for i in range(len(tracks)):
            a = tracks[i]
            for j in np.flatnonzero((t0[i + 1:] <= t1[i]) & (t0[i] <= t1[i + 1:])) + i + 1:
                b = tracks[j]
                if _clearly_apart(frames[(a["camera_id"], a["track_id"])], frames[(b["camera_id"], b["track_id"])]):
                    out.append((max(t0[i], t0[j]), set_similarity(a["exemplars"], b["exemplars"])))
    return np.array(out, dtype=np.float64).reshape(-1, 2)


def calibrate_thresholds(candidates: list[dict], df: pd.DataFrame, setting: str, false_match_rate: float,
                         block_seconds: float) -> dict:
    """Sets candidate["threshold"] for every eligible candidate and returns a
    record of how. A fixed number applies everywhere; "auto" sets one bar per
    block of time from that block's stranger pairs, so changes over a session
    (light, weather, day to night) move the bar with them."""
    pairs = stranger_pairs(candidates, df)
    report = {"setting": setting, "stranger_pairs": int(len(pairs))}
    if len(pairs):
        report["stranger_similarity"] = {q: round(float(np.quantile(pairs[:, 1], p)), 3)
                                         for q, p in (("median", 0.5), ("p95", 0.95), ("p99", 0.99))}

    if setting != "auto":
        bar = float(setting)
        for c in candidates:
            c["threshold"] = bar
        report.update(mode="fixed", run_threshold=bar,
                      share_of_strangers_passing=round(float((pairs[:, 1] >= bar).mean()), 3) if len(pairs) else None)
        return report

    quantile = 1.0 - false_match_rate
    if len(pairs) >= MIN_STRANGER_PAIRS:
        run_bar, source = float(np.quantile(pairs[:, 1], quantile)), "calibrated"
    else:
        run_bar, source = DEFAULT_SIMILARITY_THRESHOLD, f"fallback (fewer than {MIN_STRANGER_PAIRS} stranger pairs)"
    blocks = {}
    if len(pairs):
        for b in np.unique(np.floor(pairs[:, 0] / block_seconds)).astype(int):
            sims = pairs[np.floor(pairs[:, 0] / block_seconds) == b, 1]
            calibrated = len(sims) >= MIN_STRANGER_PAIRS
            blocks[int(b)] = {"start_s": b * block_seconds, "stranger_pairs": int(len(sims)),
                              "threshold": round(float(np.quantile(sims, quantile)) if calibrated else run_bar, 4),
                              "source": "calibrated" if calibrated else "run-wide (too few pairs in this block)"}
    for c in candidates:
        block = blocks.get(int(np.floor(c["timestamp"] / block_seconds)))
        c["threshold"] = block["threshold"] if block else run_bar
    report.update(mode="auto", false_match_rate=false_match_rate, block_seconds=block_seconds,
                  run_threshold=round(run_bar, 4), run_threshold_source=source, blocks=list(blocks.values()))
    return report


# --- Identity ---------------------------------------------------------------

def resolve_global_ids(candidates: list[dict], gallery: MCMTGallery, window_seconds: float) -> dict:
    """{(camera_id, track_id): {"global_id", "confidence"}}. Tracks with no
    usable fingerprint get their own Global_ID and never enter the gallery."""
    resolved = {}
    eligible = []
    for c in sorted(candidates, key=lambda c: c["timestamp"]):
        if c["eligible"]:
            eligible.append(c)
        else:
            resolved[(c["camera_id"], c["track_id"])] = {"global_id": gallery.mint_unlinked_id(), "confidence": None}
    i = 0
    while i < len(eligible):
        window_start = eligible[i]["timestamp"]
        window = []
        while i < len(eligible) and eligible[i]["timestamp"] < window_start + window_seconds:
            window.append(eligible[i])
            i += 1
        for candidate, result in zip(window, gallery.resolve_window(window)):
            resolved[(candidate["camera_id"], candidate["track_id"])] = result
    return resolved


def merge_co_sightings(df: pd.DataFrame, resolved: dict, bucket_seconds: float) -> dict:
    """Where cameras overlap, one person standing on one spot of ground at one
    moment is one person, whatever they look like. Two tracks from different
    cameras whose ground positions stay within CO_SIGHTING_RADIUS_M of each
    other (median over their shared moments) for at least
    CO_SIGHTING_MIN_SECONDS are given one Global_ID. Each track pairs with at
    most one track per other camera (closest first), and a merge that would
    put one identity on two of a camera's tracks at once is skipped.

    Needed because appearance alone can't bridge different viewpoints: on
    Founders Square, cross-camera pairs standing on the same spot scored a
    median similarity of 0.64, no higher than known strangers (0.61).
    Updates `resolved` in place; returns counts by camera pair."""
    person = df[(df["Class"] == PERSON_CLASS_NAME) & df["Local_X_Meters"].notna()]
    bucket = (person["Timestamp"] / bucket_seconds).round().astype(int)
    pos = person.groupby([person["Camera"], person["Track_ID"].astype(int), bucket])[["Local_X_Meters", "Local_Y_Meters"]] \
        .mean().reset_index()
    pos.columns = ["Camera", "Track_ID", "Bucket", "X", "Y"]
    min_buckets = max(1, int(round(CO_SIGHTING_MIN_SECONDS / bucket_seconds)))

    pairs = []
    cameras = sorted(pos["Camera"].unique())
    for i, cam_a in enumerate(cameras):
        for cam_b in cameras[i + 1:]:
            m = pos[pos["Camera"] == cam_a].merge(pos[pos["Camera"] == cam_b], on="Bucket", suffixes=("_a", "_b"))
            m["Distance"] = np.hypot(m["X_a"] - m["X_b"], m["Y_a"] - m["Y_b"])
            together = m.groupby(["Track_ID_a", "Track_ID_b"]).agg(n=("Distance", "size"), d=("Distance", "median")).reset_index()
            together = together[(together["n"] >= min_buckets) & (together["d"] <= CO_SIGHTING_RADIUS_M)]
            for r in together.itertuples():
                pairs.append((r.d, -r.n, (cam_a, int(r.Track_ID_a)), (cam_b, int(r.Track_ID_b))))

    spans = person.groupby([person["Camera"], person["Track_ID"].astype(int)])["Timestamp"].agg(["min", "max"])
    members = {}
    for key, r in resolved.items():
        members.setdefault(r["global_id"], set()).add(key)

    def conflicts(group_a, group_b):
        for ka in group_a:
            for kb in group_b:
                if ka[0] == kb[0] and ka != kb and ka in spans.index and kb in spans.index:
                    (a0, a1), (b0, b1) = spans.loc[ka], spans.loc[kb]
                    if a0 <= b1 and b0 <= a1:
                        return True
        return False

    partnered = set()
    linked = {}
    for _, _, ka, kb in sorted(pairs):
        if (ka, kb[0]) in partnered or (kb, ka[0]) in partnered or ka not in resolved or kb not in resolved:
            continue
        gid_a, gid_b = resolved[ka]["global_id"], resolved[kb]["global_id"]
        partnered.update({(ka, kb[0]), (kb, ka[0])})
        if gid_a == gid_b or conflicts(members[gid_a], members[gid_b]):
            continue
        keep, drop = sorted([gid_a, gid_b], key=lambda g: int(g.rsplit("_", 1)[-1]))
        for key in members.pop(drop):
            resolved[key]["global_id"] = keep
            members[keep].add(key)
        label = f"{ka[0]}+{kb[0]}"
        linked[label] = linked.get(label, 0) + 1
    return linked


def same_camera_duplicates(df: pd.DataFrame) -> int:
    """(Camera, Timestamp, Global_ID) combinations held by more than one track
    -- one identity in two places in one camera at once. Must be 0."""
    person = df[(df["Class"] == PERSON_CLASS_NAME) & df["Global_ID"].notna()]
    counts = person.groupby(["Camera", "Timestamp", "Global_ID"])["Track_ID"].nunique()
    return int((counts > 1).sum())


# --- Travel mode ------------------------------------------------------------

def _with_recent_movement(rows: pd.DataFrame) -> pd.DataFrame:
    """Adds vx/vy (m/s): each row's track velocity over the last
    CO_MOVEMENT_WINDOW_S, or over the track's whole life so far when it is
    younger than that (NaN under MIN_MOVEMENT_HISTORY_S). Riders' person boxes
    come and go, so their tracks are often too short for a full window."""
    rows = rows.sort_values("Timestamp").copy()
    first = rows.groupby(["Camera", "Track_ID"])["Timestamp"].transform("min")
    rows["_t_back"] = np.maximum(rows["Timestamp"] - CO_MOVEMENT_WINDOW_S, first)
    rows = rows.sort_values("_t_back")
    past = rows[["Camera", "Track_ID", "Timestamp", "Local_X_Meters", "Local_Y_Meters"]] \
        .rename(columns={"Timestamp": "_t_past", "Local_X_Meters": "_x_past", "Local_Y_Meters": "_y_past"}) \
        .sort_values("_t_past")
    rows = pd.merge_asof(rows, past, left_on="_t_back", right_on="_t_past", by=["Camera", "Track_ID"],
                         direction="forward", tolerance=CO_MOVEMENT_WINDOW_S / 2)
    elapsed = rows["Timestamp"] - rows["_t_past"]
    elapsed = elapsed.where(elapsed >= MIN_MOVEMENT_HISTORY_S)
    rows["vx"] = (rows["Local_X_Meters"] - rows["_x_past"]) / elapsed
    rows["vy"] = (rows["Local_Y_Meters"] - rows["_y_past"]) / elapsed
    return rows.drop(columns=["_t_back", "_t_past", "_x_past", "_y_past"])


def associate_modes(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Mode for every person row (index-aligned with df): ON_FOOT, or the class
    of the bicycle/scooter/skateboard/wheelchair the person is riding. A person
    and a rideable that stay within ASSOCIATION_RADIUS_M and move together for
    at least MIN_ASSOCIATION_S are an associated pair; the person is then
    riding whenever that pair is within the radius -- including stopped at a
    crossing. The rideable itself stays an anonymous vehicle detection."""
    positioned = df["Local_X_Meters"].notna()
    is_person = df["Class"] == PERSON_CLASS_NAME
    mode = pd.Series(np.where(is_person, ON_FOOT, None), index=df.index, dtype=object)
    cols = ["Camera", "Track_ID", "Timestamp", "Local_X_Meters", "Local_Y_Meters"]
    people = df.loc[is_person & positioned, cols].reset_index()
    rides = df.loc[df["Class"].isin(RIDEABLE_CLASSES) & positioned, cols + ["Class"]]
    stats = {"rider_associations": 0, "moving_rideable_rows_without_rider": 0}
    if people.empty or rides.empty:
        return mode, stats

    people = _with_recent_movement(people)
    rides = _with_recent_movement(rides)
    near = people.merge(rides, on=["Camera", "Timestamp"], suffixes=("", "_v"))
    near = near[np.hypot(near["Local_X_Meters"] - near["Local_X_Meters_v"],
                         near["Local_Y_Meters"] - near["Local_Y_Meters_v"]) <= ASSOCIATION_RADIUS_M].copy()
    moving_together = (np.hypot(near["vx"], near["vy"]) >= MIN_SPEED_MPS) \
        & (np.hypot(near["vx_v"], near["vy_v"]) >= MIN_SPEED_MPS) \
        & (np.hypot(near["vx"] - near["vx_v"], near["vy"] - near["vy_v"]) <= VELOCITY_TOLERANCE_MPS)
    together = near[moving_together].groupby(["Camera", "Track_ID", "Track_ID_v"])["Timestamp"].agg(["min", "max"])
    pairs = together[(together["max"] - together["min"]) >= MIN_ASSOCIATION_S].reset_index()[["Camera", "Track_ID", "Track_ID_v"]]
    stats["rider_associations"] = int(len(pairs))

    riding = near.merge(pairs, on=["Camera", "Track_ID", "Track_ID_v"])
    if not riding.empty:
        riding["_d"] = np.hypot(riding["Local_X_Meters"] - riding["Local_X_Meters_v"], riding["Local_Y_Meters"] - riding["Local_Y_Meters_v"])
        nearest = riding.sort_values("_d").drop_duplicates("index")
        mode.loc[nearest["index"].to_numpy()] = nearest["Class"].to_numpy()

    moving = rides[np.hypot(rides["vx"], rides["vy"]) >= MIN_SPEED_MPS]
    ridden = moving.merge(pairs.rename(columns={"Track_ID": "_p", "Track_ID_v": "Track_ID"}), on=["Camera", "Track_ID"])
    stats["moving_rideable_rows_without_rider"] = int(len(moving) - len(ridden.drop_duplicates(["Camera", "Track_ID", "Timestamp"])))
    return mode, stats


# --- Position precision and site-wide outputs ----------------------------------

def position_precision(df: pd.DataFrame, registry: dict) -> pd.Series:
    """How much to trust each row's position, for choosing between cameras that
    saw the same thing at once: its box height in pixels (larger = closer =
    more precise), halved when it lies outside the area its camera's
    calibration points span (extrapolated)."""
    height = (2 * (df["Foot_Y"] - df["Center_Y"])).clip(lower=0)
    weight = pd.Series(1.0, index=df.index)
    entries = camera_entries(registry)
    origin = np.array(site_origin_utm(registry)) if any(e.get("mode") == "global" for e in entries.values()) else None
    for camera_id, rows in df.groupby("Camera"):
        cal = entries.get(camera_id)
        if cal is None:
            continue
        pts = np.array(cal["utm_coords"]) - origin if cal.get("mode") == "global" else np.array(cal["site_coords"], dtype=float)
        if len(pts) < 3:
            continue
        hull = pts[ConvexHull(pts).vertices]
        centre = hull.mean(axis=0)
        polygon = Polygon(centre + (hull - centre) * CALIBRATION_HULL_MARGIN)
        xy = rows[["Local_X_Meters", "Local_Y_Meters"]].to_numpy(dtype=float)
        inside = polygon.contains_points(np.nan_to_num(xy, nan=1e12))
        weight.loc[rows.index[~inside]] = OUTSIDE_CALIBRATION_WEIGHT
    return height * weight


def build_composite_csv(df: pd.DataFrame, bucket_seconds: float) -> pd.DataFrame:
    """One row per person per time bucket; where several cameras saw that
    person at once, the most precise view's row is kept."""
    person = df[(df["Class"] == PERSON_CLASS_NAME) & df["Global_ID"].notna() & df["Local_X_Meters"].notna()].copy()
    if person.empty:
        return person
    person["_bucket"] = (person["Timestamp"] / bucket_seconds).round().astype(int)
    person = person.sort_values("_precision", ascending=False).drop_duplicates(["Global_ID", "_bucket"], keep="first")
    return person.drop(columns=["_bucket"]).sort_values(["Global_ID", "Timestamp"])


def build_vehicles_csv(df: pd.DataFrame, bucket_seconds: float) -> pd.DataFrame:
    """Every vehicle position over time, no identity. Detections of one class
    from different cameras in the same moment, within that class's merge
    radius, are one vehicle seen twice: the most precise one is kept. Two
    detections from the SAME camera are always two vehicles."""
    rows = df[(df["Class"] != PERSON_CLASS_NAME) & df["Local_X_Meters"].notna()].copy()
    if rows.empty:
        return rows
    rows["_bucket"] = (rows["Timestamp"] / bucket_seconds).round().astype(int)
    rows = rows.sort_values("_precision", ascending=False)
    keep = []
    for (_, cls), group in rows.groupby(["_bucket", "Class"], sort=False):
        radius = VEHICLE_MERGE_RADIUS_M.get(cls, DEFAULT_VEHICLE_MERGE_RADIUS_M)
        kept = []  # [x, y, camera, cameras whose duplicate it has already absorbed]
        for idx, x, y, cam in zip(group.index, group["Local_X_Meters"], group["Local_Y_Meters"], group["Camera"]):
            # A vehicle appears at most once per camera, so a kept detection can
            # stand in for at most ONE detection from each other camera; two
            # bikes side by side in cam1 can't both be cam2's one bike.
            matches = [(np.hypot(x - k[0], y - k[1]), k) for k in kept
                       if k[2] != cam and cam not in k[3] and np.hypot(x - k[0], y - k[1]) <= radius]
            if matches:
                min(matches, key=lambda m: m[0])[1][3].add(cam)
                continue
            keep.append(idx)
            kept.append([x, y, cam, set()])
    out = rows.loc[keep, ["_bucket", "Class", "Local_X_Meters", "Local_Y_Meters", "Longitude", "Latitude", "Camera"]]
    out = out.rename(columns={"Camera": "Source_Camera"})
    out.insert(0, "Timestamp", out.pop("_bucket") * bucket_seconds)
    return out.sort_values(["Timestamp", "Class"]).reset_index(drop=True)


def build_counts_csv(composite: pd.DataFrame, vehicles: pd.DataFrame, t_start: float, t_end: float,
                     bucket_seconds: float, interval_seconds: float) -> pd.DataFrame:
    """Per interval, for people by mode and vehicles by class: the average
    number present at a moment, and the most present at once."""
    buckets = np.arange(round(t_start / bucket_seconds), round(t_end / bucket_seconds) + 1).astype(int)
    series = {}
    if not composite.empty:
        b = (composite["Timestamp"] / bucket_seconds).round().astype(int)
        for mode, grp in composite.groupby(composite["Mode"].fillna(ON_FOOT)):
            series[f"person ({mode})"] = grp.groupby(b.loc[grp.index])["Global_ID"].nunique()
    if not vehicles.empty:
        b = (vehicles["Timestamp"] / bucket_seconds).round().astype(int)
        for cls, grp in vehicles.groupby("Class"):
            series[cls] = grp.groupby(b.loc[grp.index]).size()
    rows = []
    interval_of = np.floor(buckets * bucket_seconds / interval_seconds).astype(int)
    for category, counts in series.items():
        full = counts.reindex(buckets, fill_value=0).to_numpy()
        for interval in np.unique(interval_of):
            sel = full[interval_of == interval]
            rows.append({"Interval_Start_s": interval * interval_seconds, "Category": category,
                         "Mean_Present": round(float(sel.mean()), 2), "Peak_Present": int(sel.max())})
    return pd.DataFrame(rows, columns=["Interval_Start_s", "Category", "Mean_Present", "Peak_Present"])


# --- Run --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SCOUT site fusion: identities, travel modes, vehicle counts and coverage for one site")
    parser.add_argument("--tracking-csvs", nargs="+", required=True,
                         help="Per-camera transformed CSVs (homography_middleware.py transform output), one per camera")
    parser.add_argument("--embeddings", nargs="+", required=True, help="The *_embeddings.npz files from run_inference.py --extract-embeddings")
    parser.add_argument("--site-id", required=True, help="The site's calibration registry (calibrations/<site_id>.json)")
    parser.add_argument("--output-dir", default="Site_Analyzer_Batch_Runs", help="Where per-camera <camera_id>_fused.csv files go")
    parser.add_argument("--composite-output", default=None,
                         help="The site-wide person-path CSV (default <output-dir>/full_site_composite.csv). "
                              "The vehicles, counts, coverage map and manifest files go next to it.")
    parser.add_argument("--basemap", default="esri",
                         help="Coverage map imagery: esri | esri-clarity | indiana | none | an XYZ tile URL | an ArcGIS MapServer/ImageServer URL")
    parser.add_argument("--similarity-threshold", default="auto",
                         help="'auto' (default): calibrate the appearance bar from this run's known-stranger pairs; or a fixed number, e.g. 0.8")
    parser.add_argument("--false-match-rate", type=float, default=DEFAULT_FALSE_MATCH_RATE,
                         help=f"With auto: share of known strangers the bar may let through (default {DEFAULT_FALSE_MATCH_RATE:g})")
    parser.add_argument("--calibration-block-seconds", type=float, default=DEFAULT_CALIBRATION_BLOCK_SECONDS,
                         help=f"With auto: one bar per block of this length (default {DEFAULT_CALIBRATION_BLOCK_SECONDS:g} s)")
    parser.add_argument("--fingerprints-used", type=int, default=DEFAULT_FINGERPRINTS_USED,
                         help=f"Fingerprints per track used for matching (default {DEFAULT_FINGERPRINTS_USED})")
    parser.add_argument("--min-reid-height-px", type=float, default=DEFAULT_MIN_REID_HEIGHT_PX,
                         help=f"Fingerprints of smaller person boxes aren't used for matching (default {DEFAULT_MIN_REID_HEIGHT_PX:g} px)")
    parser.add_argument("--retention-seconds", type=float, default=DEFAULT_RETENTION_SECONDS,
                         help=f"How long an identity stays matchable after it was last seen (default {DEFAULT_RETENTION_SECONDS:g} s)")
    parser.add_argument("--max-speed-mps", type=float, default=DEFAULT_MAX_SPEED_MPS,
                         help=f"Fastest anyone could plausibly move between sightings (default {DEFAULT_MAX_SPEED_MPS:g} m/s)")
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--bucket-seconds", type=float, default=DEFAULT_BUCKET_SECONDS)
    parser.add_argument("--count-interval-seconds", type=float, default=DEFAULT_COUNT_INTERVAL_SECONDS)
    parser.add_argument("--overlap-radius-meters", type=float, default=DEFAULT_OVERLAP_RADIUS_METERS)
    args = parser.parse_args()
    if args.composite_output is None:
        args.composite_output = str(Path(args.output_dir) / "full_site_composite.csv")

    frames = []
    for tracking_csv in args.tracking_csvs:
        camera_df = pd.read_csv(tracking_csv)
        if "Local_X_Meters" not in camera_df.columns:
            raise SystemExit(f"'Local_X_Meters' not found in {tracking_csv} -- run homography_middleware.py transform first")
        frames.append(camera_df)
    df = pd.concat(frames, ignore_index=True)
    df["Camera"] = df["Camera"].astype(str)
    registry = load_site_registry(args.site_id)

    fingerprints = load_fingerprints([Path(p) for p in args.embeddings])
    candidates = build_candidates(df, fingerprints, args.min_reid_height_px, args.fingerprints_used)
    n_eligible = sum(c["eligible"] for c in candidates)
    print(f"{len(candidates)} person tracks with a ground position; {n_eligible} have fingerprints >= {args.min_reid_height_px:g} px "
          f"to match on appearance, {len(candidates) - n_eligible} keep their own Global_ID")
    if not candidates:
        print("No positioned person tracks -- nothing to fuse.")
        return

    calibration = calibrate_thresholds(candidates, df, args.similarity_threshold, args.false_match_rate, args.calibration_block_seconds)
    if calibration["stranger_pairs"]:
        s = calibration["stranger_similarity"]
        print(f"Appearance check: {calibration['stranger_pairs']:,} pairs of tracks are certainly different people (same camera, same moment); "
              f"their similarity: median {s['median']}, 95th pct {s['p95']}, 99th pct {s['p99']}")
    if calibration["mode"] == "auto":
        print(f"Similarity bar (auto, lets through {args.false_match_rate:.0%} of known strangers): run-wide {calibration['run_threshold']} "
              f"[{calibration['run_threshold_source']}]" + "".join(f"; from {b['start_s'] / 60:.0f} min: {b['threshold']} ({b['source']})"
                                                                  for b in calibration["blocks"]))
    else:
        print(f"Similarity bar (fixed): {calibration['run_threshold']}; it would let through "
              f"{calibration['share_of_strangers_passing']:.0%} of known strangers" if calibration["share_of_strangers_passing"] is not None
              else f"Similarity bar (fixed): {calibration['run_threshold']}")

    eligible = [c for c in candidates if c["eligible"]]
    gallery = MCMTGallery(
        feature_dim=eligible[0]["exemplars"].shape[1] if eligible else 512,
        max_speed_mps=args.max_speed_mps,
        overlap_radius_meters=args.overlap_radius_meters,
        retention_seconds=args.retention_seconds,
        # An identity keeps no more fingerprints than a single track uses: the
        # similarity bar was calibrated on sets of that size, and a bigger set
        # gives a stranger more chances to look alike by accident.
        max_exemplars=args.fingerprints_used,
    )
    resolved = resolve_global_ids(candidates, gallery, args.window_seconds)
    print(f"Resolved {len(candidates)} tracks into {len({r['global_id'] for r in resolved.values()})} Global_IDs "
          f"({len(gallery.global_ids)} of them from appearance-matched tracks)")
    co_sightings = merge_co_sightings(df, resolved, args.bucket_seconds)
    print(f"Overlap hand-offs by position (same ground within {CO_SIGHTING_RADIUS_M:g} m for {CO_SIGHTING_MIN_SECONDS:g}+ s): "
          + (", ".join(f"{k}: {v}" for k, v in sorted(co_sightings.items())) or "none")
          + f"; {len({r['global_id'] for r in resolved.values()})} Global_IDs after")
    loose = {cam: e["error_m"]["loo_median"] for cam, e in camera_entries(registry).items()
             if cam in set(df["Camera"]) and isinstance(e.get("error_m"), dict)
             and e["error_m"].get("loo_median", 0) > CO_SIGHTING_RADIUS_M / 2}
    for cam, err in loose.items():
        print(f"  WARNING: {cam}'s calibration is accurate to only ~{err:g} m (leave-one-out median), too loose to "
              f"hand people off by position reliably -- recalibrating it with more, better-spread points would help")

    assignments = pd.DataFrame([(cam, tid, r["global_id"], r["confidence"]) for (cam, tid), r in resolved.items()],
                               columns=["Camera", "Track_ID", "Global_ID", "Confidence"])
    assignments["Class"] = PERSON_CLASS_NAME
    df = df.merge(assignments, on=["Camera", "Track_ID", "Class"], how="left")

    df["Mode"], mode_stats = associate_modes(df)
    riding_rows = int(((df["Class"] == PERSON_CLASS_NAME) & (df["Mode"] != ON_FOOT)).sum())
    print(f"Travel mode: {mode_stats['rider_associations']} person-rideable pairings found; {riding_rows:,} person rows riding; "
          f"{mode_stats['moving_rideable_rows_without_rider']:,} moving bicycle/scooter/skateboard rows had no detected rider")

    duplicates = same_camera_duplicates(df)
    print(f"Same-camera duplicate identities (one Global_ID on two tracks at once): {duplicates}"
          + ("" if duplicates == 0 else "  <-- WARNING: should be 0; the one-identity-per-camera constraint failed"))

    df["_precision"] = position_precision(df, registry)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for camera_id, camera_df in df.drop(columns=["_precision"]).groupby("Camera"):
        camera_df.to_csv(output_dir / f"{camera_id}_fused.csv", index=False)
    print(f"Per-camera fused CSVs written to {output_dir}")

    site_dir = Path(args.composite_output).parent
    site_dir.mkdir(parents=True, exist_ok=True)
    composite = build_composite_csv(df, args.bucket_seconds).drop(columns=["_precision"])
    composite.to_csv(args.composite_output, index=False)
    vehicles = build_vehicles_csv(df, args.bucket_seconds)
    vehicles.to_csv(site_dir / "full_site_vehicles.csv", index=False)
    counts = build_counts_csv(composite, vehicles, df["Timestamp"].min(), df["Timestamp"].max(), args.bucket_seconds, args.count_interval_seconds)
    counts.to_csv(site_dir / "counts_over_time.csv", index=False)
    print(f"Site outputs written to {site_dir}: composite ({len(composite):,} rows), vehicles ({len(vehicles):,} rows), counts_over_time")

    areas = render_coverage_map({c: g for c, g in df.groupby("Camera")}, args.site_id, site_dir / "coverage_map.png", args.basemap)
    print("Coverage map: " + "  ".join(f"{k}: {v:,.0f} sq m" for k, v in areas.items()))

    manifest = {
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "site_id": args.site_id,
        "inputs": {"tracking_csvs": args.tracking_csvs, "embeddings": args.embeddings},
        "settings": {k: v for k, v in vars(args).items() if k not in ("tracking_csvs", "embeddings")},
        "similarity_calibration": calibration,
        "results": {
            "person_tracks": len(candidates), "appearance_matched_tracks": n_eligible,
            "global_ids": len({r["global_id"] for r in resolved.values()}),
            "global_ids_from_appearance_matching": len(gallery.global_ids),
            "overlap_handoffs_by_position": co_sightings, "cameras_too_loosely_calibrated_for_position_handoff": loose,
            "same_camera_duplicates": duplicates, "travel_mode": mode_stats, "person_rows_riding": riding_rows,
            "composite_rows": len(composite), "vehicle_rows": len(vehicles),
            "coverage_sq_m": {k: round(v) for k, v in areas.items()},
        },
        "outputs": [str(p) for p in [args.composite_output, site_dir / "full_site_vehicles.csv", site_dir / "counts_over_time.csv",
                                    site_dir / "coverage_map.png"]],
    }
    (site_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Run manifest written to {site_dir / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
