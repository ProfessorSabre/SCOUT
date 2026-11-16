"""Synthetic checks for mcmt_fusion.py's resolve_simultaneous_conflicts: one
Global_ID can't be in two cameras at the same moment far apart. Run from the
project root:  python tests/test_conflicts.py"""
import itertools
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


def one_identity(df, confidences):
    keys = sorted({(c, int(t)) for c, t in zip(df.Camera, df.Track_ID)})
    return {k: {"global_id": "Global_ID_0001", "confidence": confidences.get(k)} for k in keys}


def run(df, res):
    counter = itertools.count(100)
    return mf.resolve_simultaneous_conflicts(df, res, 0.5, lambda: f"Global_ID_{next(counter):04d}")


def gid(res, cam, tid):
    return res[(cam, tid)]["global_id"]


# 1. Same moment, 40 m apart, confidences 0.9 vs 0.82, no earlier sighting -> the 0.82 track loses the ID.
df = pd.concat([track("c1", 1, 0, 10, 0, 0, vx=1.3), track("c2", 2, 2, 8, 40, 0, vx=-1.3)])
res = one_identity(df, {("c1", 1): 0.9, ("c2", 2): 0.82}); counts = run(df, res)
check("lower-confidence track loses the identity", gid(res, "c1", 1) == "Global_ID_0001" and gid(res, "c2", 2) != "Global_ID_0001")
check("detached track's confidence is cleared", res[("c2", 2)]["confidence"] is None)

# 2. The higher-confidence track (0.95) can't be reached from where the identity was last seen
#    (c3 at x=0 until t=10; c2 appears at x=200 at t=11 -> 200 m in 1 s), the lower one (0.85) can -> the higher one loses it.
df = pd.concat([track("c3", 9, 0, 10, 0, 0), track("c1", 1, 11, 20, 1, 0, vx=1.3), track("c2", 2, 11, 20, 200, 0)])
res = one_identity(df, {("c1", 1): 0.85, ("c2", 2): 0.95}); counts = run(df, res)
check("unreachable higher-confidence track loses the identity",
      gid(res, "c1", 1) == "Global_ID_0001" and gid(res, "c2", 2) != "Global_ID_0001" and counts["kept_reachable"] == 1)

# 3. Both reachable from the last sighting -> confidence decides.
df = pd.concat([track("c3", 9, 0, 10, 0, 0), track("c1", 1, 30, 40, 5, 0), track("c2", 2, 30, 40, 60, 0)])
res = one_identity(df, {("c1", 1): 0.85, ("c2", 2): 0.95}); run(df, res)
check("both reachable: higher confidence keeps it", gid(res, "c2", 2) == "Global_ID_0001" and gid(res, "c1", 1) != "Global_ID_0001")

# 4. A genuine overlap hand-off (two cameras, same ground within 1 m) is left alone.
df = pd.concat([track("c1", 1, 0, 10, 0, 0, vx=1.3), track("c2", 2, 3, 9, 3 * 1.3 + 0.8, 0, vx=1.3)])
res = one_identity(df, {}); counts = run(df, res)
check("same-ground overlap is not a conflict", gid(res, "c1", 1) == gid(res, "c2", 2) and counts["tracks_detached"] == 0)

# 5. The losing track takes the tracks stitched to it in its camera along, under one new ID;
#    a separate re-entry in that camera much later stays.
df = pd.concat([track("c1", 1, 0, 20, 0, 0), track("c2", 2, 5, 8, 50, 0), track("c2", 3, 8.6, 12, 50, 0),
                track("c2", 4, 60, 70, 50, 0)])
res = one_identity(df, {("c1", 1): 0.95, ("c2", 2): 0.80, ("c2", 3): 0.80, ("c2", 4): 0.88}); run(df, res)
check("stitched continuation leaves with the losing track", gid(res, "c2", 2) == gid(res, "c2", 3) != "Global_ID_0001")
check("unrelated later re-entry keeps the identity", gid(res, "c2", 4) == "Global_ID_0001")

# 6. No score on either (first sighting / joined by position): the track that started first keeps it.
df = pd.concat([track("c1", 1, 0, 10, 0, 0), track("c2", 2, 4, 10, 30, 0)])
res = one_identity(df, {}); run(df, res)
check("tie: earlier track keeps the identity", gid(res, "c1", 1) == "Global_ID_0001" and gid(res, "c2", 2) != "Global_ID_0001")

# 7. Three cameras chained: c1 and c2 far apart, c2 and c3 far apart, c1 and c3 together -> only c2 leaves.
df = pd.concat([track("c1", 1, 0, 10, 0, 0), track("c3", 3, 0, 10, 0.5, 0), track("c2", 2, 2, 8, 40, 0)])
res = one_identity(df, {("c1", 1): 0.9, ("c3", 3): 0.9, ("c2", 2): 0.8}); counts = run(df, res)
check("one detachment settles a three-camera conflict",
      gid(res, "c1", 1) == gid(res, "c3", 3) == "Global_ID_0001" and gid(res, "c2", 2) != "Global_ID_0001"
      and counts["tracks_detached"] == 1)

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
