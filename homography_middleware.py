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
#                             (scout_calibration.html) and fits via RANSAC,
#                             which both averages out click imprecision and
#                             reports a per-point reprojection error.
#
# Local (tape-measure) mode has no absolute frame to anchor to, so it stays
# on the simple 4-corner path. Global (GPS) mode converts anchors straight to
# UTM via pyproj and fits the homography directly against those UTM
# coordinates -- true scale AND orientation in one step, no separate
# rotation correction needed. Multi-camera fusion at a single site requires
# global mode: local mode gives each camera its own private, unrelated
# origin, so only global mode puts every camera's output in one shared
# real-world coordinate frame.

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


def fit_global_calibration_from_points(pixel_points: list, latlon_points: list) -> dict:
    """RANSAC-fit path for >=6 (pixel, lat/lon) correspondences from the
    browser tool. Returns the calibration entry plus a per-point reprojection
    error report (in meters) so a bad click is visible before it's trusted."""
    if len(pixel_points) < 6:
        raise ValueError(f"Need at least 6 point pairs for a RANSAC fit, got {len(pixel_points)}")
    if len(pixel_points) != len(latlon_points):
        raise ValueError("pixel_points and latlon_points must be the same length")

    lat0, lon0 = latlon_points[0]
    epsg = utm_epsg_for(lon0, lat0)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    utm_coords = [list(to_utm.transform(lon, lat)) for lat, lon in latlon_points]

    src = np.array(pixel_points, dtype=np.float32)
    dst = np.array(utm_coords, dtype=np.float32)
    matrix, inlier_mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    if matrix is None:
        raise ValueError("Homography fit failed - check for duplicate or collinear points")

    reprojected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors_m = np.linalg.norm(reprojected - dst, axis=1)
    inliers = inlier_mask.ravel().astype(bool) if inlier_mask is not None else np.ones(len(src), dtype=bool)

    return {
        "mode": "global",
        "cam_pixels": pixel_points,
        "utm_coords": utm_coords,
        "utm_epsg": epsg,
        "last_calibrated": date.today().isoformat(),
        "point_count": len(pixel_points),
        "reprojection_error_m": {
            "per_point": [round(float(e), 3) for e in errors_m],
            "mean": round(float(errors_m.mean()), 3),
            "max": round(float(errors_m.max()), 3),
            "outlier_points": [i for i, is_in in enumerate(inliers) if not is_in],
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

    entry = fit_global_calibration_from_points(pixel_points, latlon_points)

    registry = load_site_registry(site_id)
    registry[camera_id] = entry
    save_site_registry(site_id, registry)

    err = entry["reprojection_error_m"]
    print(f"Calibrated camera_id='{camera_id}' for site_id='{site_id}' from {entry['point_count']} points")
    print(f"Reprojection error (meters): mean={err['mean']} max={err['max']}")
    if err["outlier_points"]:
        print(f"WARNING: point(s) {err['outlier_points']} flagged as RANSAC outliers - consider re-clicking them")
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

    unmatched_cameras = set(df["Camera"].unique()) - set(registry.keys())

    for camera_id, cal in registry.items():
        mask = df["Camera"] == camera_id
        if not mask.any():
            continue

        last_calibrated = cal.get("last_calibrated")
        if last_calibrated:
            age_days = (date.today() - datetime.fromisoformat(last_calibrated).date()).days
            if age_days > stale_days:
                print(f"WARNING: camera_id='{camera_id}' calibration is {age_days} days old "
                      f"(calibrated {last_calibrated}) - verify the camera hasn't moved before trusting this run")

        src = np.array(cal["cam_pixels"], dtype=np.float32)
        dst = np.array(cal["utm_coords"] if cal["mode"] == "global" else cal["site_coords"], dtype=np.float32)
        matrix, _ = cv2.findHomography(src, dst, method=cv2.RANSAC if len(src) > 4 else 0)

        points = df.loc[mask, ["Foot_X", "Foot_Y"]].to_numpy(dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)

        if cal["mode"] == "global":
            origin = dst[0]
            df.loc[mask, "Local_X_Meters"] = np.round(transformed[:, 0] - origin[0], 2)
            df.loc[mask, "Local_Y_Meters"] = np.round(transformed[:, 1] - origin[1], 2)
            to_wgs84 = Transformer.from_crs(f"EPSG:{cal['utm_epsg']}", "EPSG:4326", always_xy=True)
            lon, lat = to_wgs84.transform(transformed[:, 0], transformed[:, 1])
            df.loc[mask, "Longitude"] = np.round(lon, 6)
            df.loc[mask, "Latitude"] = np.round(lat, 6)
        else:
            df.loc[mask, "Local_X_Meters"] = np.round(transformed[:, 0], 2)
            df.loc[mask, "Local_Y_Meters"] = np.round(transformed[:, 1], 2)

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
