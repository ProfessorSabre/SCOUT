import argparse
import json
import math
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyproj import Transformer

# SCOUT Phase 2: pixel -> ground-plane middleware.
#
# Identity model: every physical camera gets a permanent (site_id, camera_id)
# pair, assigned once and never changed for the life of that camera mounting.
# This is deliberately decoupled from video filenames, which vary across
# capture sessions (timestamps, batch naming) even though the physical camera
# hasn't moved. Calibration is "memory" keyed on that permanent pair: once a
# camera is calibrated, every future analysis run against that same
# camera_id reuses it with no extra work, for as long as the camera's mount
# doesn't change. See calibrations/<site_id>.json.
#
# Two ways to build a calibration entry:
#   - calibrate            : interactive OpenCV window, exactly 4 corners of
#                             a known rectangular reference object (tape
#                             measure or typed GPS). Quick, but no redundancy
#                             to catch a bad click.
#   - calibrate-from-points : ingests >=6 (pixel, lat/lon) correspondences
#                             from the standalone browser tool
#                             (scout_calibration.html), fits a least-squares
#                             homography over all of them (averaging out click
#                             imprecision), and reports each point's
#                             leave-one-out error so a bad click stands out.
#
# Local (tape-measure) mode has no absolute frame to anchor to, so it stays
# on the simple 4-corner path. Global (GPS) mode converts anchors straight to
# UTM via pyproj and fits the homography directly against those UTM
# coordinates -- true scale AND orientation in one step, no separate
# rotation correction needed. Multi-camera fusion at a single site requires
# global mode: local mode gives each camera its own private, unrelated
# origin, so only global mode puts every camera's output in one shared
# real-world coordinate frame. In global mode, Local_X/Y_Meters are meters
# east/north of one site-wide origin stored in the registry (see
# site_origin_utm), shared by every camera at the site.

REFERENCE_WINDOW = "SCOUT Calibration"
CALIBRATION_DIR = Path("calibrations")
DEFAULT_STALE_DAYS = 180


# --- Registry I/O ---

def registry_path(site_id: str) -> Path:
    return CALIBRATION_DIR / f"{site_id}.json"


def load_site_registry(site_id: str) -> dict:
    path = registry_path(site_id)
    return json.loads(path.read_text()) if path.exists() else {}


def save_site_registry(site_id: str, registry: dict):
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    registry_path(site_id).write_text(json.dumps(registry, indent=2))


# Site-level settings live in the same registry file as the cameras, under a
# key no camera_id can collide with.
SITE_KEY = "_site"


def camera_entries(registry: dict) -> dict:
    return {k: v for k, v in registry.items() if not k.startswith("_")}


def site_origin_utm(registry: dict) -> list:
    """The single UTM point every GPS-calibrated camera at a site measures its
    Local_X/Y_Meters from. It has to be shared: cross-camera matching compares
    positions from different cameras directly, which only works if they're
    offsets from the same origin. Registries saved before the site origin
    existed fall back to their first GPS camera's first point -- exactly what
    that camera used before, so single-camera output doesn't change."""
    if SITE_KEY in registry:
        return registry[SITE_KEY]["origin_utm"]
    for cal in camera_entries(registry).values():
        if cal.get("mode") == "global":
            return cal["utm_coords"][0]
    raise ValueError("No GPS-calibrated camera in this registry to take a site origin from")


# --- Core math ---

def solve_local_quad(L1: float, L2: float, L3: float, L4: float, L5: float) -> list:
    """Back-solve 4 planar (x, y) points, in meters, from the 5 pairwise
    distances of a quadrilateral clicked clockwise on the ground
    (edges L1-L4, diagonal L5 = corner1-corner3), via the law of cosines.
    Used for the local (tape-measure) path, which has no absolute frame to
    anchor to."""
    p1 = (0.0, 0.0)
    p2 = (L1, 0.0)

    cos_t1 = max(-1.0, min(1.0, (L1 ** 2 + L5 ** 2 - L2 ** 2) / (2 * L1 * L5)))
    theta1 = math.acos(cos_t1)
    p3 = (L5 * math.cos(theta1), L5 * math.sin(theta1))

    cos_t2 = max(-1.0, min(1.0, (L5 ** 2 + L4 ** 2 - L3 ** 2) / (2 * L5 * L4)))
    theta2 = math.acos(cos_t2)
    total_theta = theta1 + theta2
    p4 = (L4 * math.cos(total_theta), L4 * math.sin(total_theta))

    return [list(p1), list(p2), list(p3), list(p4)]


def utm_epsg_for(lon: float, lat: float) -> int:
    zone = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def build_local_calibration(cam_pixels: list, lengths: list) -> dict:
    L1, L2, L3, L4, L5 = lengths
    return {
        "mode": "local",
        "cam_pixels": cam_pixels,
        "site_coords": solve_local_quad(L1, L2, L3, L4, L5),
        "last_calibrated": date.today().isoformat(),
    }


def build_global_calibration_4pt(cam_pixels: list, gps_anchors: list) -> dict:
    """Exact 4-point path for the interactive OpenCV tool. gps_anchors: list
    of 4 (lat, lon) tuples, same click order as cam_pixels."""
    lat0, lon0 = gps_anchors[0]
    epsg = utm_epsg_for(lon0, lat0)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    utm_coords = [list(to_utm.transform(lon, lat)) for lat, lon in gps_anchors]
    return {
        "mode": "global",
        "cam_pixels": cam_pixels,
        "utm_coords": utm_coords,
        "utm_epsg": epsg,
        "last_calibrated": date.today().isoformat(),
    }


def fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares homography over ALL correspondences -- used both when a
    calibration is fit and when it's applied in `transform`, so the error
    report describes exactly the transform that gets used.

    Deliberately not RANSAC. With the handful of hand-clicked points a
    calibration has (typically 6-10), RANSAC has too little redundancy to
    separate a bad click from ordinary imprecision: a tight inlier threshold
    (it was 1 m) can leave a fit resting on the bare 4-point minimum, which
    matches those four exactly and extrapolates badly everywhere else, while
    reporting near-zero error on them. A bad click is found instead by
    leave-one-out error (see fit_global_calibration_from_points) and fixed by
    the person who made it."""
    matrix, _ = cv2.findHomography(np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64), 0)
    if matrix is None:
        raise ValueError("Homography fit failed - check for duplicate or collinear points")
    return matrix


def _project(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2), matrix).reshape(-1, 2)


def in_front_of_horizon(matrix: np.ndarray, calibration_pixels: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """True where a pixel lies below the ground plane's horizon, i.e. could be
    a point on the ground in front of the camera.

    A homography's third (scale) coordinate changes sign at the horizon line.
    Every calibration point is on the ground, so its sign marks the valid
    side. A pixel on the other side -- far-field sky/buildings, or a detection
    whose "feet" are on a building facade -- has no ground position at all,
    and projecting it anyway lands it on the opposite side of the camera, often
    hundreds of meters away. Without this check, those positions would reach
    cross-camera matching as people who appear to teleport."""
    scale = lambda px: (np.c_[np.asarray(px, dtype=np.float64), np.ones(len(px))] @ matrix.T)[:, 2]
    ground_sign = np.sign(np.median(scale(calibration_pixels)))
    return np.sign(scale(pixels)) == ground_sign


# A point is flagged when its leave-one-out error is both well above the
# calibration's typical point (2x the median) and large in absolute terms
# (> 2 m) -- the first condition finds the odd one out, the second keeps a
# uniformly good calibration from flagging its least-good point.
SUSPECT_LOO_RATIO = 2.0
SUSPECT_LOO_FLOOR_M = 2.0


def fit_global_calibration_from_points(pixel_points: list, latlon_points: list, epsg: int | None = None) -> dict:
    """Least-squares fit for >=6 (pixel, lat/lon) correspondences from the
    browser tool. Returns the calibration entry plus a per-point error report
    in meters, so a bad click is visible before it's trusted.

    Two errors are reported per point:
      - residual: how far the all-points fit lands from that point. Optimistic,
        since the point helped shape the fit it's being judged against.
      - leave-one-out (LOO): fit on every OTHER point, then measure this one.
        This is the honest per-point check. A high LOO error means either the
        click pair is wrong, or the point sits where no other point can vouch
        for it (e.g. alone in a corner of the frame) -- adding points near it
        tells the two apart."""
    if len(pixel_points) < 6:
        raise ValueError(f"Need at least 6 point pairs, got {len(pixel_points)}")
    if len(pixel_points) != len(latlon_points):
        raise ValueError("pixel_points and latlon_points must be the same length")

    lat0, lon0 = latlon_points[0]
    epsg = epsg or utm_epsg_for(lon0, lat0)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    utm_coords = [list(to_utm.transform(lon, lat)) for lat, lon in latlon_points]

    src = np.array(pixel_points, dtype=np.float64)
    dst = np.array(utm_coords, dtype=np.float64)
    matrix = fit_homography(src, dst)
    residuals = np.linalg.norm(_project(matrix, src) - dst, axis=1)

    loo = np.full(len(src), np.nan)
    for i in range(len(src)):
        keep = np.arange(len(src)) != i
        try:
            loo[i] = np.linalg.norm(_project(fit_homography(src[keep], dst[keep]), src[i:i + 1])[0] - dst[i])
        except ValueError:
            pass  # remaining points degenerate without this one; leave NaN

    median_loo = float(np.nanmedian(loo))
    suspects = [i for i, e in enumerate(loo)
                if np.isfinite(e) and e > SUSPECT_LOO_RATIO * median_loo and e > SUSPECT_LOO_FLOOR_M]

    def rounded(values):
        return [None if not np.isfinite(v) else round(float(v), 3) for v in values]

    return {
        "mode": "global",
        "cam_pixels": pixel_points,
        "utm_coords": utm_coords,
        "utm_epsg": epsg,
        "last_calibrated": date.today().isoformat(),
        "point_count": len(pixel_points),
        "fit_method": "least_squares_all_points",
        "error_m": {
            "residual_per_point": rounded(residuals),
            "residual_mean": round(float(residuals.mean()), 3),
            "loo_per_point": rounded(loo),
            "loo_median": round(median_loo, 3),
            "loo_max": round(float(np.nanmax(loo)), 3),
            "suspect_points": suspects,  # 0-based indices; the browser tool numbers points from 1
        },
    }


# --- Interactive calibration (run on a machine with a real display) ---

class ClickCollector:
    def __init__(self):
        self.points = []

    def callback(self, event, x, y, flags, param):
        frame = param["frame"]
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append([x, y])
            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
            if len(self.points) > 1:
                cv2.line(frame, tuple(self.points[-2]), tuple(self.points[-1]), (0, 255, 0), 2)
            if len(self.points) == 4:
                cv2.line(frame, tuple(self.points[3]), tuple(self.points[0]), (0, 255, 0), 2)
                cv2.line(frame, tuple(self.points[0]), tuple(self.points[2]), (255, 0, 255), 2)
            cv2.imshow(REFERENCE_WINDOW, frame)


def calibrate_camera_interactive(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read a frame from {video_path}")

    collector = ClickCollector()
    cv2.namedWindow(REFERENCE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(REFERENCE_WINDOW, collector.callback, param={"frame": frame})
    print(f"\n--- Calibrating {video_path.stem} ---")
    print("Click the 4 corners of a known rectangular reference object, clockwise. Press ENTER when done.")
    while True:
        cv2.imshow(REFERENCE_WINDOW, frame)
        if cv2.waitKey(1) == 13 and len(collector.points) == 4:
            break
    cv2.destroyAllWindows()

    print("\n[L] Local (tape measure)   [G] Global (GPS)")
    choice = input("Scale mode: ").strip().upper()

    if choice == "G":
        gps_anchors = []
        for i in range(1, 5):
            raw = input(f"Lat, Long for point {i}: ")
            lat, lon = (float(v) for v in raw.replace(" ", "").split(","))
            gps_anchors.append((lat, lon))
        return build_global_calibration_4pt(collector.points, gps_anchors)
    else:
        lengths = [float(input(f"Length {i} (meters): ")) for i in range(1, 6)]
        return build_local_calibration(collector.points, lengths)


def run_calibration(source: str, output_site_id: str):
    source_path = Path(source)
    video_files = [source_path] if source_path.is_file() else sorted(
        p for p in source_path.glob("*") if p.suffix.lower() in (".mp4", ".avi", ".mov")
    )
    if not video_files:
        print(f"No videos found at {source_path}")
        return

    registry = load_site_registry(output_site_id)

    for video_path in video_files:
        camera_id = video_path.stem
        registry[camera_id] = calibrate_camera_interactive(video_path)
        print(f"Calibrated camera_id='{camera_id}' for site_id='{output_site_id}'")

    save_site_registry(output_site_id, registry)
    print(f"\nCalibration saved to {registry_path(output_site_id)}")


def run_calibration_from_points(points_json: str, site_id: str, camera_id: str):
    payload = json.loads(Path(points_json).read_text())

    site_id = site_id or payload.get("site_id")
    camera_id = camera_id or payload.get("camera_id")
    if not site_id or not camera_id:
        raise ValueError("site_id and camera_id must be given either as --site-id/--camera-id or embedded in the points JSON")

    points = payload["points"]
    pixel_points = [p["video_xy"] for p in points]
    latlon_points = [[p["lat"], p["lon"]] for p in points]

    registry = load_site_registry(site_id)
    if SITE_KEY not in registry:
        existing = [c for c in camera_entries(registry).values() if c.get("mode") == "global"]
        if existing:
            registry = {SITE_KEY: {"origin_utm": existing[0]["utm_coords"][0], "utm_epsg": existing[0]["utm_epsg"]}, **registry}
    site = registry.get(SITE_KEY)

    # Every camera at a site is fit in the site's UTM zone, so a site that
    # happens to straddle a zone boundary can't end up in two coordinate frames.
    entry = fit_global_calibration_from_points(pixel_points, latlon_points, epsg=site["utm_epsg"] if site else None)

    if site is None:
        # First GPS calibration at this site: fix the site origin here, once.
        # It never moves afterwards, even if this camera is recalibrated.
        registry = {SITE_KEY: {"origin_utm": entry["utm_coords"][0], "utm_epsg": entry["utm_epsg"]}, **registry}
    registry[camera_id] = entry
    save_site_registry(site_id, registry)

    err = entry["error_m"]
    print(f"Calibrated camera_id='{camera_id}' for site_id='{site_id}' from {entry['point_count']} points")
    print(f"Leave-one-out error (meters): median={err['loo_median']} max={err['loo_max']}   "
          f"(fit residual mean={err['residual_mean']})")
    print("  point  residual_m  leave-one-out_m")
    for i, (res, loo) in enumerate(zip(err["residual_per_point"], err["loo_per_point"])):
        flag = "   <-- check this point" if i in err["suspect_points"] else ""
        print(f"  #{i + 1:<5} {res:>9}  {loo if loo is not None else 'n/a':>15}{flag}")
    if err["suspect_points"]:
        print(f"WARNING: point(s) {[i + 1 for i in err['suspect_points']]} (numbered as in the browser tool) disagree with "
              f"the rest -- either a mismatched click pair, or a point no other point is near enough to vouch for.")
    print(f"Saved to {registry_path(site_id)}")


# --- Transform: apply a saved calibration to a Phase 1 tracking CSV ---

def transform_tracking_csv(tracking_csv: str, site_id: str, output_csv: str, stale_days: int):
    registry = load_site_registry(site_id)
    if not registry:
        print(f"No calibration registry found for site_id='{site_id}' ({registry_path(site_id)})")
        return

    df = pd.read_csv(tracking_csv)
    for col in ("Local_X_Meters", "Local_Y_Meters", "Longitude", "Latitude"):
        df[col] = np.nan
    df["Beyond_Horizon"] = False

    unmatched_cameras = set(df["Camera"].unique()) - set(camera_entries(registry))

    for camera_id, cal in camera_entries(registry).items():
        mask = df["Camera"] == camera_id
        if not mask.any():
            continue

        last_calibrated = cal.get("last_calibrated")
        if last_calibrated:
            age_days = (date.today() - datetime.fromisoformat(last_calibrated).date()).days
            if age_days > stale_days:
                print(f"WARNING: camera_id='{camera_id}' calibration is {age_days} days old "
                      f"(calibrated {last_calibrated}) - verify the camera hasn't moved before trusting this run")

        src = np.array(cal["cam_pixels"], dtype=np.float64)
        dst = np.array(cal["utm_coords"] if cal["mode"] == "global" else cal["site_coords"], dtype=np.float64)
        matrix = fit_homography(src, dst)

        foot = df.loc[mask, ["Foot_X", "Foot_Y"]].to_numpy(dtype=np.float64)
        on_ground = in_front_of_horizon(matrix, src, foot)
        rows = df.index[mask]
        df.loc[rows[~on_ground], "Beyond_Horizon"] = True
        if not on_ground.all():
            print(f"camera_id='{camera_id}': {(~on_ground).sum():,} of {len(foot):,} rows have their foot point above the "
                  f"ground plane's horizon -- not on the ground, coordinates left blank (Beyond_Horizon=True)")
        valid_rows, transformed = rows[on_ground], _project(matrix, foot[on_ground])

        if cal["mode"] == "global":
            origin = site_origin_utm(registry)
            df.loc[valid_rows, "Local_X_Meters"] = np.round(transformed[:, 0] - origin[0], 2)
            df.loc[valid_rows, "Local_Y_Meters"] = np.round(transformed[:, 1] - origin[1], 2)
            to_wgs84 = Transformer.from_crs(f"EPSG:{cal['utm_epsg']}", "EPSG:4326", always_xy=True)
            lon, lat = to_wgs84.transform(transformed[:, 0], transformed[:, 1])
            df.loc[valid_rows, "Longitude"] = np.round(lon, 6)
            df.loc[valid_rows, "Latitude"] = np.round(lat, 6)
        else:
            df.loc[valid_rows, "Local_X_Meters"] = np.round(transformed[:, 0], 2)
            df.loc[valid_rows, "Local_Y_Meters"] = np.round(transformed[:, 1], 2)

    if unmatched_cameras:
        print(f"Warning: no calibration found for cameras: {sorted(unmatched_cameras)} (rows left blank)")

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Done. {len(df)} rows written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="SCOUT Phase 2: homography middleware")
    sub = parser.add_subparsers(dest="command", required=True)

    cal = sub.add_parser("calibrate", help="Interactively define a 4-point ground-plane anchor per camera (needs a display)")
    cal.add_argument("--source", default="videos/sdd_videos")
    cal.add_argument("--site-id", required=True, help="Permanent site identifier, e.g. auburn_toomers_corner")

    calp = sub.add_parser("calibrate-from-points", help="Fit a calibration from >=6 point pairs exported by scout_calibration.html")
    calp.add_argument("--points-json", required=True)
    calp.add_argument("--site-id", default=None, help="Overrides the site_id embedded in the points JSON, if given")
    calp.add_argument("--camera-id", default=None, help="Overrides the camera_id embedded in the points JSON, if given")

    trans = sub.add_parser("transform", help="Apply a site's saved calibrations to a Phase 1 tracking CSV")
    trans.add_argument("--tracking-csv", default="Site_Analyzer_Batch_Runs/full_site_tracking.csv")
    trans.add_argument("--site-id", required=True)
    trans.add_argument("--output", default="Site_Analyzer_Batch_Runs/full_site_movement.csv")
    trans.add_argument("--stale-days", type=int, default=DEFAULT_STALE_DAYS,
                        help=f"Warn if a matched camera's calibration is older than this many days (default {DEFAULT_STALE_DAYS})")

    args = parser.parse_args()
    if args.command == "calibrate":
        run_calibration(args.source, args.site_id)
    elif args.command == "calibrate-from-points":
        run_calibration_from_points(args.points_json, args.site_id, args.camera_id)
    else:
        transform_tracking_csv(args.tracking_csv, args.site_id, args.output, args.stale_days)


if __name__ == "__main__":
    main()
