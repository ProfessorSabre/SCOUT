"""Synthetic checks for mcmt_fusion.py's within-camera track stitching and
composite gap filling. Run from the project root:  python tests/test_stitching.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mcmt_fusion as mf  # noqa: E402

FPS = 30.0
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))


def track(cam, tid, t0, t1, x0, y0, vx=0.0, vy=0.0):
    ts = np.arange(t0, t1 + 1e-9, 1 / FPS)
    return pd.DataFrame({"Camera": cam, "Timestamp": np.round(ts, 2), "Track_ID": tid, "Class": "person",
                         "Local_X_Meters": x0 + vx * (ts - t0), "Local_Y_Meters": y0 + vy * (ts - t0)})


def resolved_for(df):
    keys = sorted({(c, int(t)) for c, t in zip(df.Camera, df.Track_ID)})
    return {k: {"global_id": f"Global_ID_{i + 1:04d}", "confidence": None} for i, k in enumerate(keys)}


def gid(res, cam, tid):
    return res[(cam, tid)]["global_id"]


# 1. A walker (1.4 m/s) lost for 0.6 s, picked up again on the same line -> one identity.
df = pd.concat([track("c1", 1, 0, 5, 0, 0, vx=1.4), track("c1", 2, 5.6, 9, 0 + 1.4 * 5.6, 0, vx=1.4)])
res = resolved_for(df); mf.stitch_track_breaks(df, res)
check("walker lost for 0.6 s is stitched", gid(res, "c1", 1) == gid(res, "c1", 2))

# 2. Same, but lost for 1.5 s (> STITCH_MAX_GAP_S) -> left apart.
df = pd.concat([track("c1", 1, 0, 5, 0, 0, vx=1.4), track("c1", 2, 6.5, 9, 1.4 * 6.5, 0, vx=1.4)])
res = resolved_for(df); mf.stitch_track_breaks(df, res)
check("gap longer than 1 s is not stitched", gid(res, "c1", 1) != gid(res, "c1", 2))

# 3. A new track starting 5 m off the walker's heading after 0.5 s -> left apart.
df = pd.concat([track("c1", 1, 0, 5, 0, 0, vx=1.4), track("c1", 2, 5.5, 9, 1.4 * 5.5, 5.0, vx=1.4)])
res = resolved_for(df); mf.stitch_track_breaks(df, res)
check("restart off the predicted path is not stitched", gid(res, "c1", 1) != gid(res, "c1", 2))

# 4. A cyclist at 6 m/s lost for 0.8 s reappears 4.8 m on: only the velocity prediction allows it
#    (a fixed walking allowance would be 1 + 3 x 0.8 = 3.4 m).
df = pd.concat([track("c1", 1, 0, 3, 0, 0, vx=6.0), track("c1", 2, 3.8, 6, 6.0 * 3.8, 0, vx=6.0)])
res = resolved_for(df); mf.stitch_track_breaks(df, res)
check("cyclist stitched through the velocity prediction", gid(res, "c1", 1) == gid(res, "c1", 2))

# 5. Two people walking side by side 1.2 m apart, both lost at the same moment and found again:
#    each continues as themselves (no swap).
df = pd.concat([track("c1", 1, 0, 4, 0, 0, vx=1.3), track("c1", 2, 0, 4, 0, 1.2, vx=1.3),
                track("c1", 3, 4.5, 8, 1.3 * 4.5, 0, vx=1.3), track("c1", 4, 4.5, 8, 1.3 * 4.5, 1.2, vx=1.3)])
res = resolved_for(df); mf.stitch_track_breaks(df, res)
check("side-by-side pair keeps their own identities",
      gid(res, "c1", 1) == gid(res, "c1", 3) and gid(res, "c1", 2) == gid(res, "c1", 4) and gid(res, "c1", 1) != gid(res, "c1", 2))

# 6. A stitch that would put one identity on two of a camera's tracks at once is refused:
#    track 1's identity already owns track 3, which runs at the same time as track 2.
df = pd.concat([track("c1", 1, 0, 3, 0, 0, vx=1.0), track("c1", 2, 3.5, 6, 3.5, 0, vx=1.0), track("c1", 3, 4, 7, 20, 20)])
res = resolved_for(df); res[("c1", 3)]["global_id"] = gid(res, "c1", 1)
mf.stitch_track_breaks(df, res)
check("stitch refused when it would duplicate an identity in one camera", gid(res, "c1", 2) != gid(res, "c1", 1))

# 7. Stitching never crosses cameras.
df = pd.concat([track("c1", 1, 0, 3, 0, 0, vx=1.0), track("c2", 1, 3.3, 6, 3.3, 0, vx=1.0)])
res = resolved_for(df); mf.stitch_track_breaks(df, res)
check("tracks from different cameras are not stitched", gid(res, "c1", 1) != gid(res, "c2", 1))

# 8. A standing person with too short a history for a velocity, re-found 0.3 s later 0.4 m away -> stitched.
df = pd.concat([track("c1", 1, 0, 0.1, 5, 5), track("c1", 2, 0.4, 3, 5.4, 5)])
res = resolved_for(df); mf.stitch_track_breaks(df, res)
check("short track without a velocity uses the walking allowance", gid(res, "c1", 1) == gid(res, "c1", 2))

# 9-11. Composite gap filling.
b = 0.5
rows = []
for t in list(np.arange(0, 5.01, b)) + list(np.arange(6.5, 9.01, b)) + list(np.arange(12, 14.01, b)):   # gaps: 1.5 s, 3 s
    rows.append({"Camera": "c1", "Timestamp": round(t, 2), "Track_ID": 1, "Class": "person", "Global_ID": "Global_ID_0001",
                 "Local_X_Meters": 2.0 * t, "Local_Y_Meters": 0.0, "Longitude": -86.0 + t * 1e-5, "Latitude": 40.0,
                 "Foot_X": 100.0, "Foot_Y": 200.0, "Center_Y": 150.0, "_precision": 1.0})
comp = mf.build_composite_csv(pd.DataFrame(rows), b)
filled = comp[comp.Interpolated]
check("1.5 s gap filled with interpolated rows", sorted(filled.Timestamp.tolist()) == [5.5, 6.0], str(filled.Timestamp.tolist()))
check("interpolated positions lie on the straight line", np.allclose(filled.Local_X_Meters, 2.0 * filled.Timestamp))
check("3 s gap left unfilled; real rows marked False; pixels blank on filled rows",
      not ((comp.Timestamp > 9) & (comp.Timestamp < 12)).any() and (~comp[~comp.Interpolated].Interpolated).all()
      and filled.Foot_X.isna().all())

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
