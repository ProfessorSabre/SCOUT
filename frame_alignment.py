import argparse
from pathlib import Path

import cv2
import numpy as np

# SCOUT camera-motion compensation. A calibration maps ONE image's pixels to
# the ground, so it is only valid while the camera points exactly where it did
# in that image. A camera that drifts (a phone propped on a window sill, a
# pole-mounted camera swaying in wind) slowly moves every pixel away from the
# calibrated position. This module lines every video frame up with one fixed
# reference image per camera -- the image the calibration points were clicked
# on -- so that detections can be converted into that image's pixels before
# homography_middleware.py maps them to the ground.
#
# Matching is done on the static scene (paving, benches, lamp posts,
# buildings); people and vehicles move between frames and are rejected as
# outliers. A camera that rotates in place relates any two of its frames by an
# exact homography, and the small hand movement of a propped phone (a few cm,
# against a scene tens of meters away) adds negligible parallax.

# Alignment is computed every ALIGN_INTERVAL_S and interpolated between. On the
# 2026-11-09 Bell Tower test the phones drifted smoothly at up to ~5 px/s
# (frame-to-frame shift median 0.06-0.36 px, no shake), so a half-second
# interval with linear interpolation stays well under a pixel of error.
ALIGN_INTERVAL_S = 0.5
# Feature matching runs on a copy scaled to this width: plenty of detail for
# matching a static scene, and ~4x faster than full 1080p.
WORK_WIDTH = 960
SIFT_FEATURES = 4000
RATIO_TEST = 0.75          # Lowe's ratio test: keep a match only if clearly better than the runner-up
RANSAC_REPROJ_PX = 2.0     # at WORK_WIDTH, i.e. ~4 px at 1080p: static points must agree this closely
MIN_INLIERS = 40           # fewer agreeing static points than this and the alignment isn't trusted
# A frame further than this from any trusted alignment gets no position at all
# (blank coordinates) rather than a guessed one.
MAX_GAP_S = 3.0
# The reference image is chosen from frames sampled this often across the video.
REFERENCE_SAMPLE_S = 10.0


def _gray_small(image: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = min(1.0, WORK_WIDTH / gray.shape[1])
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray, scale


class FrameAligner:
    """Finds the homography taking a frame's pixels to the reference image's."""

    def __init__(self, reference: np.ndarray):
        self.sift = cv2.SIFT_create(SIFT_FEATURES)
        # Approximate nearest-neighbour matching (kd-trees): ~3x faster than
        # brute force with no loss of inliers at this feature count.
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=32))
        gray, self.ref_scale = _gray_small(reference)
        self.ref_kp, self.ref_desc = self.sift.detectAndCompute(gray, None)
        self.ref_shape = reference.shape[:2]

    def align(self, frame: np.ndarray) -> tuple[np.ndarray | None, int]:
        """(3x3 homography frame pixels -> reference pixels, inlier count), or
        (None, inliers) when too few static points agree."""
        gray, scale = _gray_small(frame)
        kp, desc = self.sift.detectAndCompute(gray, None)
        if desc is None or self.ref_desc is None or len(kp) < MIN_INLIERS:
            return None, 0
        pairs = self.matcher.knnMatch(desc, self.ref_desc, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO_TEST * p[1].distance]
        if len(good) < MIN_INLIERS:
            return None, len(good)
        src = np.float32([kp[m.queryIdx].pt for m in good])
        dst = np.float32([self.ref_kp[m.trainIdx].pt for m in good])
        h_small, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ_PX)
        inliers = int(mask.sum()) if mask is not None else 0
        if h_small is None or inliers < MIN_INLIERS:
            return None, inliers
        # Undo the working-copy scaling on both sides: full frame -> small frame
        # -> small reference -> full reference.
        to_small = np.diag([scale, scale, 1.0])
        from_small_ref = np.diag([1.0 / self.ref_scale, 1.0 / self.ref_scale, 1.0])
        return from_small_ref @ h_small @ to_small, inliers


class AlignmentTrack:
    """Alignments sampled through a video, applied to detections at any time."""

    def __init__(self):
        self.times, self.matrices = [], []
        self.attempts = 0
        self.inliers = []

    def add(self, timestamp: float, matrix: np.ndarray | None, inliers: int):
        self.attempts += 1
        if matrix is not None:
            self.times.append(timestamp)
            self.matrices.append(matrix)
            self.inliers.append(inliers)

    def map_points(self, timestamps: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Each point (in its own frame's pixels) in reference pixels: mapped
        with the alignments just before and after its frame, blended by time.
        NaN where no trusted alignment is within MAX_GAP_S."""
        timestamps = np.asarray(timestamps, dtype=np.float64)
        out = np.full((len(timestamps), 2), np.nan)
        if not self.times or len(timestamps) == 0:
            return out
        ts = np.asarray(self.times)
        hs = np.stack(self.matrices)
        # The samples just before and after each timestamp; before the first
        # sample or after the last one, both are that end sample.
        after = np.searchsorted(ts, timestamps)
        lo = np.clip(after - 1, 0, len(ts) - 1)
        hi = np.clip(after, 0, len(ts) - 1)

        homog = np.c_[np.asarray(points, dtype=np.float64), np.ones(len(points))]

        def apply(idx):
            p = np.einsum("nij,nj->ni", hs[idx], homog)
            return p[:, :2] / p[:, 2:3]

        p_lo, p_hi = apply(lo), apply(hi)
        span = ts[hi] - ts[lo]
        w = np.where(span > 0, (timestamps - ts[lo]) / np.where(span > 0, span, 1.0), 0.0)
        blended = p_lo + np.clip(w, 0.0, 1.0)[:, None] * (p_hi - p_lo)
        # Across a short gap every frame is within MAX_GAP_S of an end, so
        # blending is safe. Across a long one (alignment failed for a while),
        # a frame keeps the alignment of whichever end is within MAX_GAP_S,
        # and gets no position if neither is.
        from_lo, to_hi = np.abs(timestamps - ts[lo]), np.abs(ts[hi] - timestamps)
        short = span <= 2 * MAX_GAP_S
        use_lo = ~short & (from_lo <= MAX_GAP_S) & (from_lo <= to_hi)
        use_hi = ~short & ~use_lo & (to_hi <= MAX_GAP_S)
        short &= np.minimum(from_lo, to_hi) <= MAX_GAP_S  # beyond either end of the whole track
        out[short] = blended[short]
        out[use_lo] = p_lo[use_lo]
        out[use_hi] = p_hi[use_hi]
        return out

    def summary(self) -> str:
        if not self.times:
            return f"alignment: 0 of {self.attempts} samples aligned -- no positions written"
        return (f"alignment: {len(self.times)} of {self.attempts} samples aligned "
                f"(inliers median {int(np.median(self.inliers))}, min {min(self.inliers)})")


def center_offsets(matrices: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Where each frame's center lands in the reference image, relative to the
    reference's own center: how far the camera points away from it, in px."""
    h, w = shape
    c = np.array([w / 2.0, h / 2.0, 1.0])
    p = matrices @ c
    return p[:, :2] / p[:, 2:3] - c[:2]


def pick_reference(video_path: Path, sample_seconds: float = REFERENCE_SAMPLE_S) -> tuple[np.ndarray, dict]:
    """The frame closest to where the camera pointed on average, so no part of
    the recording is far from the calibrated view. Every sampled frame is
    checked against the chosen one, which shows whether one reference covers
    the whole recording."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * sample_seconds)))
    frames, idx = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                frames.append((idx / fps, frame))
        idx += 1
    cap.release()
    if not frames:
        raise SystemExit(f"Could not read frames from {video_path}")

    # Where every sample points, measured against the first sample.
    first = FrameAligner(frames[0][1])
    aligned = [(t, f, first.align(f)[0]) for t, f in frames]
    usable = [(t, f, m) for t, f, m in aligned if m is not None]
    offsets = center_offsets(np.stack([m for _, _, m in usable]), frames[0][1].shape[:2])
    target = np.median(offsets, axis=0)
    best = int(np.argmin(np.linalg.norm(offsets - target, axis=1)))
    ref_time, reference = usable[best][0], usable[best][1]

    # Check the choice: every sample against the chosen reference.
    chosen = FrameAligner(reference)
    results = [chosen.align(f) for _, f in frames]
    ok_matrices = [m for m, _ in results if m is not None]
    shifts = np.linalg.norm(center_offsets(np.stack(ok_matrices), reference.shape[:2]), axis=1) if ok_matrices else np.array([])
    report = {
        "video": video_path.name,
        "reference_time_s": round(ref_time, 2),
        "frame_size": [int(reference.shape[1]), int(reference.shape[0])],
        "samples": len(frames),
        "samples_aligned": len(ok_matrices),
        "min_inliers": min((n for _, n in results), default=0),
        "max_shift_from_reference_px": round(float(shifts.max()), 1) if shifts.size else None,
        "median_shift_from_reference_px": round(float(np.median(shifts)), 1) if shifts.size else None,
    }
    return reference, report


def main():
    parser = argparse.ArgumentParser(description="SCOUT camera-motion compensation")
    sub = parser.add_subparsers(dest="command", required=True)
    ref = sub.add_parser("reference", help="Choose and save a camera's reference image (the one to calibrate on)")
    ref.add_argument("--video", required=True)
    ref.add_argument("--output", required=True, help="PNG path, e.g. calibrations/reference_frames/<site_id>/<camera_id>.png")
    ref.add_argument("--sample-seconds", type=float, default=REFERENCE_SAMPLE_S)
    args = parser.parse_args()

    if args.command == "reference":
        reference, report = pick_reference(Path(args.video), args.sample_seconds)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), reference)  # PNG: lossless, so calibration clicks land on the exact pixels
        for k, v in report.items():
            print(f"  {k}: {v}")
        if report["samples_aligned"] < report["samples"]:
            print(f"WARNING: {report['samples'] - report['samples_aligned']} sampled frames could not be aligned to this "
                  f"reference -- detections from those stretches will have blank positions")
        print(f"Reference frame saved to {out}")


if __name__ == "__main__":
    main()
