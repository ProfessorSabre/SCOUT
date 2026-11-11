"""Synthetic checks for the per-camera pipeline pieces: frame alignment,
clock sync, recording segments, and the ground-plane homography.
Run: python tests/test_pipeline.py"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import frame_alignment as fa  # noqa: E402
from clock_sync import estimate_offset  # noqa: E402
from homography_middleware import _project, fit_homography, in_front_of_horizon  # noqa: E402
from run_inference import SEGMENT_TRACK_STRIDE, session_clock  # noqa: E402

rng = np.random.default_rng(11)
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))


# --- frame alignment ---
# A textured synthetic "scene", and the same scene seen after the camera turned slightly (a known homography).
scene = cv2.GaussianBlur((rng.random((1080, 1920)) * 255).astype(np.uint8), (0, 0), 3)
scene = cv2.cvtColor(cv2.equalizeHist(scene), cv2.COLOR_GRAY2BGR)
H_true = np.array([[1.0, 0.004, 35.0], [-0.003, 1.0, -22.0], [0, 0, 1.0]])     # frame -> reference
frame = cv2.warpPerspective(scene, np.linalg.inv(H_true), (1920, 1080))
H, inliers = fa.FrameAligner(scene).align(frame)
pts = np.array([[400, 300], [1500, 800], [960, 540]], np.float64)
err = np.abs(_project(H, pts) - _project(H_true, pts)).max() if H is not None else np.inf
check("frame aligner recovers a known camera shift to within 1 px", err < 1.0, f"max error {err:.2f} px, {inliers} inliers")
H_none, _ = fa.FrameAligner(scene).align(np.zeros_like(scene))
check("a frame with nothing to match is rejected, not guessed", H_none is None)

T = lambda dx: np.array([[1, 0, dx], [0, 1, 0], [0, 0, 1.0]])
tr = fa.AlignmentTrack(); tr.add(0, T(0), 100); tr.add(1, T(10), 100); tr.add(2, None, 5); tr.add(20, T(50), 100)
m = tr.map_points(np.array([0.5, 3.5, 11, 19, 24, -4]), np.zeros((6, 2)))[:, 0]
check("alignment interpolates between samples and leaves long gaps blank",
      np.allclose(m, [5, 10, np.nan, 50, np.nan, np.nan], equal_nan=True), str(m))

# --- clock sync ---
# People wander across an area both cameras see; the second camera's clock is 37.3 s behind.
rows_a, rows_b = [], []
for pid in range(60):
    t0 = rng.uniform(0, 500); x0, y0 = rng.uniform(0, 20, 2); v = rng.normal(0, 1.2, 2)
    for t in np.arange(t0, t0 + rng.uniform(5, 20), 0.2):
        x, y = x0 + v[0] * (t - t0), y0 + v[1] * (t - t0)
        rows_a.append(("A", round(t, 2), pid, "person", x, y))
        rows_b.append(("B", round(t - 37.3, 2), pid, "person", x + rng.normal(0, 0.3), y + rng.normal(0, 0.3)))
cols = ["Camera", "Timestamp", "Track_ID", "Class", "Local_X_Meters", "Local_Y_Meters"]
res = estimate_offset(pd.DataFrame(rows_a, columns=cols), pd.DataFrame(rows_b, columns=cols))
check("clock sync recovers a known offset", abs(res["offset_seconds"] - 37.3) < 0.3 and res["reliable"], str(res["offset_seconds"]))
far = pd.DataFrame(rows_b, columns=cols).assign(Local_X_Meters=lambda d: d.Local_X_Meters + 500)
try:
    estimate_offset(pd.DataFrame(rows_a, columns=cols), far)
    check("cameras that don't overlap are refused, not synced", False)
except SystemExit:
    check("cameras that don't overlap are refused, not synced", True)

# --- recording segments ---
clock = session_clock({"segments": [{"from_video_s": 0, "start_timestamp": -22.47}, {"from_video_s": 359.92, "start_timestamp": 486.78}]})
check("segment clock: each recording gets its own start time",
      abs(clock(100)[0] - 77.53) < 1e-6 and clock(100)[1] == 0 and abs(clock(400)[0] - 526.86) < 1e-6 and clock(400)[1] == 1)
check("single start_timestamp still works", session_clock({"start_timestamp": 15.0})(10) == (25.0, 0))
check("track IDs are separated per segment by a large stride", SEGMENT_TRACK_STRIDE >= 1_000_000)

# --- homography ---
# A camera looking at the ground: a known homography maps pixels to meters; noisy clicks should fit closely.
G = np.array([[0.05, 0.002, -40.0], [0.001, -0.12, 90.0], [0.0, 0.0011, 1.0]])
px = rng.uniform([100, 500], [1800, 1050], size=(10, 2))
ground = _project(G, px) + rng.normal(0, 0.05, (10, 2))
fit = fit_homography(px, ground)
check("least-squares fit reproduces the ground mapping", np.abs(_project(fit, px) - ground).max() < 0.3)
check("points below the horizon are on the ground; sky above it is not",
      in_front_of_horizon(fit, px, np.array([[960, 900]]))[0] and not in_front_of_horizon(fit, px, np.array([[960, -2000]]))[0])

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
