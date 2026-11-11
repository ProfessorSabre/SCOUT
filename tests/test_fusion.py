"""Synthetic checks for mcmt_fusion.py: certain-stranger pairs, threshold
calibration, overlap hand-off by position, travel mode, vehicles, composite,
counts and the duplicate check. Run: python tests/test_fusion.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mcmt_fusion as mf  # noqa: E402

FPS = 10.0
D = 32
rng = np.random.default_rng(3)
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))


def unit(v):
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


def track(cam, tid, cls, t0, t1, x0, y0, vx=0.0, vy=0.0, px0=100.0, py0=500.0, h=120.0):
    ts = np.round(np.arange(t0, t1 + 1e-9, 1 / FPS), 2)
    return pd.DataFrame({"Camera": cam, "Timestamp": ts, "Track_ID": tid, "Class": cls,
                         "Local_X_Meters": x0 + vx * (ts - t0), "Local_Y_Meters": y0 + vy * (ts - t0),
                         "Longitude": -86.9, "Latitude": 40.4,
                         "Foot_X": px0 + 10 * vx * (ts - t0), "Foot_Y": py0, "Center_Y": py0 - h / 2})


# --- candidates and eligibility ---
df = pd.concat([track("c1", 1, "person", 0, 5, 0, 0, h=120), track("c1", 2, "person", 0, 5, 10, 0, h=30)])
fps_ = {("c1", 1): {"vectors": unit(rng.normal(size=(6, D))), "heights": np.full(6, 120.0), "timestamps": np.linspace(0, 5, 6)},
        ("c1", 2): {"vectors": unit(rng.normal(size=(6, D))), "heights": np.full(6, 30.0), "timestamps": np.linspace(0, 5, 6)}}
cands = {(c["camera_id"], c["track_id"]): c for c in mf.build_candidates(df, fps_, 60, 5)}
check("tracks with fingerprints >= 60 px are eligible, smaller ones are not",
      cands[("c1", 1)]["eligible"] and not cands[("c1", 2)]["eligible"])
check("a candidate uses at most --fingerprints-used views", len(cands[("c1", 1)]["exemplars"]) <= 5)

# --- certain strangers ---
a = track("c1", 1, "person", 0, 5, 0, 0, px0=100); b = track("c1", 2, "person", 0, 5, 5, 0, px0=400)       # apart, same frames
flick = track("c1", 3, "person", 5.5, 9, 0, 0, px0=100)                                                      # never shares a frame
double = track("c1", 4, "person", 0, 5, 0, 0, px0=130)                                                       # doubled box on track 1
df = pd.concat([a, b, flick, double])
cands = [{"camera_id": "c1", "track_id": t, "timestamp": g.Timestamp.min(), "exit_timestamp": g.Timestamp.max(),
          "eligible": True, "exemplars": unit(rng.normal(size=(1, D)))} for t, g in df.groupby("Track_ID")]
by_id = {c["track_id"]: c for c in cands}
n_pairs = lambda ids: len(mf.stranger_pairs([by_id[i] for i in ids], df[df.Track_ID.isin(ids)]))
check("certain strangers: two people apart in the same frames count; a flickering ID and a doubled box don't",
      n_pairs([1, 2]) == 1 and n_pairs([1, 3]) == 0 and n_pairs([1, 4]) == 0,
      f"apart {n_pairs([1, 2])}, flicker {n_pairs([1, 3])}, doubled {n_pairs([1, 4])}")

# --- threshold calibration ---
n = 40
cands = [{"camera_id": "c1", "track_id": i, "timestamp": 0.0, "exit_timestamp": 10.0, "eligible": True,
          "exemplars": unit(rng.normal(size=(1, D)))} for i in range(n)]
cal_df = pd.concat([track("c1", i, "person", 0, 10, 0, 0, px0=300 * i) for i in range(n)])
rep = mf.calibrate_thresholds(cands, cal_df, "auto", 0.01, 3600)
sims = np.array([(cands[i]["exemplars"] @ cands[j]["exemplars"].T).item() for i in range(n) for j in range(i + 1, n)])
check("auto bar lets through about 1% of known strangers", abs((sims >= rep["run_threshold"]).mean() - 0.01) < 0.01,
      f"{(sims >= rep['run_threshold']).mean():.3f}")
few = mf.calibrate_thresholds(cands[:5], cal_df, "auto", 0.01, 3600)
check("too few stranger pairs -> fallback bar, and the report says so", "fallback" in few["run_threshold_source"])
fixed = mf.calibrate_thresholds(cands, cal_df, "0.7", 0.01, 3600)
check("a fixed bar is applied as given", fixed["mode"] == "fixed" and all(c["threshold"] == 0.7 for c in cands))


# --- overlap hand-off by position ---
def resolved_for(df):
    keys = sorted({(c, int(t)) for c, t in zip(df.Camera, df.Track_ID)})
    return {k: {"global_id": f"Global_ID_{i + 1:04d}", "confidence": None} for i, k in enumerate(keys)}


df = pd.concat([track("cA", 1, "person", 0, 5, 0, 0, vx=1.0), track("cB", 1, "person", 0, 5, 0.6, 0.3, vx=1.0)])
res = resolved_for(df); mf.merge_co_sightings(df, res, 0.5)
check("two cameras, one spot, same moments -> one Global_ID", res[("cA", 1)]["global_id"] == res[("cB", 1)]["global_id"])
df = pd.concat([track("cA", 1, "person", 0, 5, 0, 0, vx=1.0), track("cB", 1, "person", 0, 5, 8, 0, vx=1.0)])
res = resolved_for(df); mf.merge_co_sightings(df, res, 0.5)
check("two cameras, people 8 m apart -> kept separate", res[("cA", 1)]["global_id"] != res[("cB", 1)]["global_id"])
df = pd.concat([track("cA", 1, "person", 0, 1, 0, 0), track("cB", 1, "person", 0, 1, 0.3, 0)])
res = resolved_for(df); mf.merge_co_sightings(df, res, 0.5)
check("on one spot for only 1 s -> kept separate", res[("cA", 1)]["global_id"] != res[("cB", 1)]["global_id"])

# --- travel mode ---
mdf = pd.concat([track("c1", 10, "person", 0, 6, 0, 0, vx=4.0), track("c1", 11, "bicycle", 0, 6, 0.6, 0, vx=4.0),     # cyclist
                 track("c1", 12, "person", 0, 6, 50, 0), track("c1", 13, "bicycle", 0, 6, 50.8, 0),                   # beside parked bike
                 track("c1", 14, "person", 0, 6, 100, 0, vx=1.3), track("c1", 15, "bicycle", 0, 6, 103, 0),           # walks past a bike
                 track("c1", 16, "bicycle", 0, 6, 200, 0, vx=5.0),                                                    # unridden moving bike
                 track("c1", 20, "person", 0, 6, 300, 0, vx=1.5), track("c1", 21, "wheelchair", 0, 6, 300.5, 0, vx=1.5)]).reset_index(drop=True)
mode, stats = mf.associate_modes(mdf)
mdf["Mode"] = mode
late = mdf[mdf.Timestamp >= 2.0]
check("person moving with a bike -> bicycle", (late[late.Track_ID == 10].Mode == "bicycle").all())
check("person standing next to a parked bike -> on foot", (mdf[mdf.Track_ID == 12].Mode == mf.ON_FOOT).all())
check("person walking past a parked bike -> on foot", (mdf[mdf.Track_ID == 14].Mode == mf.ON_FOOT).all())
check("moving bike with no detected rider is reported", stats["moving_rideable_rows_without_rider"] > 0)
check("wheelchair is deferred: a person moving with one stays on foot", (mdf[mdf.Track_ID == 20].Mode == mf.ON_FOOT).all())

# --- vehicles ---
v = pd.DataFrame([
    {"Camera": "c1", "Timestamp": 1.0, "Class": "bicycle", "Local_X_Meters": 0.0, "Local_Y_Meters": 0.0, "Longitude": 0, "Latitude": 0, "_precision": 50},
    {"Camera": "c2", "Timestamp": 1.1, "Class": "bicycle", "Local_X_Meters": 1.0, "Local_Y_Meters": 0.0, "Longitude": 0, "Latitude": 0, "_precision": 90},
    {"Camera": "c1", "Timestamp": 1.0, "Class": "bicycle", "Local_X_Meters": 1.5, "Local_Y_Meters": 0.0, "Longitude": 0, "Latitude": 0, "_precision": 60},
    {"Camera": "c1", "Timestamp": 1.0, "Class": "vehicle", "Local_X_Meters": 30.0, "Local_Y_Meters": 0.0, "Longitude": 0, "Latitude": 0, "_precision": 99},
    {"Camera": "c1", "Timestamp": 1.0, "Class": "wheelchair", "Local_X_Meters": 60.0, "Local_Y_Meters": 0.0, "Longitude": 0, "Latitude": 0, "_precision": 99},
])
veh = mf.build_vehicles_csv(v, 0.5)
bikes = veh[veh.Class == "bicycle"]
check("same bike seen by two cameras at once is kept once, from the more precise camera",
      len(bikes) == 2 and "c2" in set(bikes.Source_Camera))
check("two bikes side by side in ONE camera stay two", len(bikes) == 2 and (bikes.Source_Camera == "c1").sum() == 1)
check("wheelchair is deferred: left out of the vehicles file", "wheelchair" not in set(veh.Class))

# --- composite and counts ---
p = pd.DataFrame([
    {"Camera": "c1", "Timestamp": 1.0, "Track_ID": 1, "Class": "person", "Global_ID": "Global_ID_0001", "Local_X_Meters": 0.0, "Local_Y_Meters": 0.0,
     "Longitude": 0, "Latitude": 0, "_precision": 40, "Mode": mf.ON_FOOT},
    {"Camera": "c2", "Timestamp": 1.1, "Track_ID": 1, "Class": "person", "Global_ID": "Global_ID_0001", "Local_X_Meters": 0.4, "Local_Y_Meters": 0.0,
     "Longitude": 0, "Latitude": 0, "_precision": 140, "Mode": mf.ON_FOOT},
])
comp = mf.build_composite_csv(p, 0.5)
check("composite keeps the most precise camera's position", len(comp) == 1 and comp.Camera.iloc[0] == "c2")
counts = mf.build_counts_csv(comp, veh, 0, 2, 0.5, 60)
peak = counts.set_index("Category").Peak_Present.to_dict()
check("counts: peak present per category", peak.get("person (on foot)") == 1 and peak.get("bicycle") == 2, str(peak))

# --- duplicate check ---
dup = pd.DataFrame({"Camera": "c1", "Timestamp": [1.0, 1.0], "Track_ID": [1, 2], "Class": "person", "Global_ID": "Global_ID_0001"})
check("a planted same-camera duplicate identity is detected", mf.same_camera_duplicates(dup) == 1)

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
