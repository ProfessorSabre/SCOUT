import argparse

import numpy as np
import pandas as pd

# SCOUT clock sync: finds the time offset between two cameras from the people
# they both see. Where two cameras' views overlap, the same people stand on the
# same ground at the same moments, so sliding one camera's detections over the
# other's in time, the number of (moment, ground cell) pairs both cameras
# occupy peaks at the true offset. Needed when a recording has lost its
# capture time (a video re-encoded by a messaging app keeps none), and useful
# as a check on deployed cameras' clocks.
#
# Inputs are two transformed per-camera CSVs (homography_middleware.py
# transform output); only person rows with a ground position are used. The
# result is the number of seconds to ADD to the second camera's timestamps.

# Ground cells: 2 m, the scale at which two calibrated cameras agree on where
# one person stands (the fusion stage hands people off within 1.5 m).
CELL_M = 2.0
# Time resolution of the search. Walking people cross a 2 m cell in ~1.5 s, so
# the peak is a few bins wide; its center is refined by a parabola fit.
BIN_S = 0.25
# A cell counts toward the score only if both cameras put people in it at some
# point: cells outside the shared area would only add noise.
MIN_DETECTIONS_PER_CELL = 20
# The best offset is reported as reliable only if its score stands this many
# times above the typical score of wrong offsets (median + spread).
MIN_PEAK_RATIO = 3.0


def occupancy(df: pd.DataFrame, cells: dict, t0: float, n_bins: int) -> np.ndarray:
    """Per time bin and shared cell: how many detections fall there."""
    out = np.zeros((n_bins, len(cells)))
    keys = list(zip(np.floor(df["Local_X_Meters"] / CELL_M).astype(int), np.floor(df["Local_Y_Meters"] / CELL_M).astype(int)))
    col = np.array([cells.get(k, -1) for k in keys])
    row = np.floor((df["Timestamp"].to_numpy() - t0) / BIN_S).astype(int)
    keep = (col >= 0) & (row >= 0) & (row < n_bins)
    np.add.at(out, (row[keep], col[keep]), 1.0)
    return np.minimum(out, 1.0)  # presence, not count: a crowd shouldn't outweigh the timing of many single passers


def estimate_offset(ref: pd.DataFrame, other: pd.DataFrame) -> dict:
    ref = ref[(ref["Class"] == "person") & ref["Local_X_Meters"].notna()]
    other = other[(other["Class"] == "person") & other["Local_X_Meters"].notna()]

    def cell_counts(df):
        return pd.Series(1, index=pd.MultiIndex.from_arrays(
            [np.floor(df["Local_X_Meters"] / CELL_M).astype(int), np.floor(df["Local_Y_Meters"] / CELL_M).astype(int)])).groupby(level=[0, 1]).size()

    a, b = cell_counts(ref), cell_counts(other)
    shared = a.index.intersection(b.index)
    shared = [k for k in shared if a[k] >= MIN_DETECTIONS_PER_CELL and b[k] >= MIN_DETECTIONS_PER_CELL]
    if not shared:
        raise SystemExit("The two cameras share no ground cells -- their views don't overlap, so their clocks can't be compared")
    cells = {k: i for i, k in enumerate(shared)}

    t0_ref, t0_other = ref["Timestamp"].min(), other["Timestamp"].min()
    n_ref = int(np.ceil((ref["Timestamp"].max() - t0_ref) / BIN_S)) + 1
    n_other = int(np.ceil((other["Timestamp"].max() - t0_other) / BIN_S)) + 1
    occ_ref, occ_other = occupancy(ref, cells, t0_ref, n_ref), occupancy(other, cells, t0_other, n_other)

    # Cross-correlate every shared cell's presence over time and sum (FFT):
    # score[lag] = how many (moment, cell) pairs line up when the other camera
    # is shifted by lag bins.
    n = n_ref + n_other
    f = np.fft.rfft(occ_ref, n, axis=0) * np.conj(np.fft.rfft(occ_other, n, axis=0))
    score = np.fft.irfft(f, n, axis=0).sum(axis=1)
    lags = np.arange(n)
    lags = np.where(lags > n_ref, lags - n, lags)  # negative lags wrap to the end
    order = np.argsort(lags)
    lags, score = lags[order], score[order]

    best = int(np.argmax(score))
    # Parabola through the peak and its neighbors: sub-bin center.
    if 0 < best < len(score) - 1:
        y0, y1, y2 = score[best - 1], score[best], score[best + 1]
        denom = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        frac = 0.0
    lag_s = (lags[best] + frac) * BIN_S
    offset = t0_ref + lag_s - t0_other

    # How distinct the peak is: against offsets more than 10 s away from it.
    away = np.abs(lags - lags[best]) * BIN_S > 10
    bg = score[away]
    spread = np.median(np.abs(bg - np.median(bg))) or 1e-9
    ratio = (score[best] - np.median(bg)) / (spread * 1.4826)
    runner_up = int(np.argmax(np.where(away, score, -np.inf)))
    return {
        "offset_seconds": round(float(offset), 2),
        "shared_cells": len(cells),
        "shared_area_m2": len(cells) * CELL_M ** 2,
        "peak_score": round(float(score[best]), 1),
        "background_median": round(float(np.median(bg)), 1),
        "peak_strength_sigma": round(float(ratio), 1),
        "runner_up_offset_seconds": round(float(t0_ref + lags[runner_up] * BIN_S - t0_other), 2),
        "runner_up_score": round(float(score[runner_up]), 1),
        "reliable": bool(score[best] >= MIN_PEAK_RATIO * max(np.median(bg), 1.0) and score[best] > 1.5 * score[runner_up]),
    }


def main():
    parser = argparse.ArgumentParser(description="SCOUT clock sync: time offset between two overlapping cameras")
    parser.add_argument("--reference-csv", required=True, help="Transformed CSV of a camera whose timestamps are trusted")
    parser.add_argument("--csv", required=True, help="Transformed CSV of the camera to sync")
    args = parser.parse_args()
    result = estimate_offset(pd.read_csv(args.reference_csv), pd.read_csv(args.csv))
    for k, v in result.items():
        print(f"  {k}: {v}")
    verdict = "reliable" if result["reliable"] else "NOT reliable -- check the overlap and both calibrations"
    print(f"Add {result['offset_seconds']} s to the second camera's timestamps ({verdict})")


if __name__ == "__main__":
    main()
