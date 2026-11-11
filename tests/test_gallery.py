"""Synthetic checks for reid/gallery.py (MCMTGallery): appearance + physical
plausibility, windowed Hungarian assignment, new-identity clustering,
retention, same-camera exclusivity. Run: python tests/test_gallery.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reid.gallery import MCMTGallery, set_similarity  # noqa: E402

D = 64
rng = np.random.default_rng(7)
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def person():
    return unit(rng.normal(size=D))


def like(v, noise=0.15):
    return unit(v + noise * unit(rng.normal(size=D)))


def cand(cam, tid, t, x, y, vec, **extra):
    c = {"camera_id": cam, "track_id": tid, "timestamp": t, "x": x, "y": y, "exemplars": np.atleast_2d(vec)}
    c.update(extra)
    return c


def gallery(**kw):
    kw.setdefault("feature_dim", D)
    kw.setdefault("similarity_threshold", 0.8)
    return MCMTGallery(**kw)


# 1. Same person, walking pace between cameras -> same identity.
g = gallery(); p = person()
a = g.resolve_window([cand("c1", 1, 0, 0, 0, p)])[0]
b = g.resolve_window([cand("c2", 1, 20, 25, 0, like(p))])[0]
check("same person at walking speed keeps one identity", a["global_id"] == b["global_id"])

# 2. Same appearance, physically impossible displacement (500 m in 2 s) -> new identity.
g = gallery(); p = person()
a = g.resolve_window([cand("c1", 1, 0, 0, 0, p)])[0]
b = g.resolve_window([cand("c2", 1, 2, 500, 0, p)])[0]
check("impossible speed rejected despite identical appearance", a["global_id"] != b["global_id"])

# 3. Different-looking person nearby -> new identity.
g = gallery()
a = g.resolve_window([cand("c1", 1, 0, 0, 0, person())])[0]
b = g.resolve_window([cand("c1", 2, 5, 2, 0, person())])[0]
check("different appearance kept separate", a["global_id"] != b["global_id"])

# 4. Two overlapping cameras see one brand-new person at the same moment -> one identity.
g = gallery(); p = person()
r = g.resolve_window([cand("c1", 1, 10, 5, 5, p), cand("c2", 1, 10.2, 5.5, 5, like(p))])
check("simultaneous new sighting in two cameras -> one identity", r[0]["global_id"] == r[1]["global_id"])

# 5. Two different people at the same moment, close together -> two identities.
g = gallery()
r = g.resolve_window([cand("c1", 1, 10, 5, 5, person()), cand("c2", 1, 10, 5.5, 5, person())])
check("two different people at once stay two", r[0]["global_id"] != r[1]["global_id"])

# 6. Batch (Hungarian) assignment beats greedy: A resembles both P and Q a little more than
#    the bar; B only resembles P. Greedy would give P to A (higher score) and strand B.
g = gallery(similarity_threshold=0.5)
P, Q = person(), person()
gp = g.resolve_window([cand("c1", 1, 0, 0, 0, P)])[0]["global_id"]
gq = g.resolve_window([cand("c1", 2, 0.5, 30, 0, Q)])[0]["global_id"]
A = unit(0.75 * P + 0.66 * Q)          # ~0.75 to P, ~0.66 to Q
B = like(P, 0.9)                        # ~0.70 to P, ~0 to Q
r = g.resolve_window([cand("c2", 1, 20, 2, 0, A), cand("c2", 2, 20, 28, 0, B)])
check("windowed Hungarian assignment beats greedy matching",
      r[0]["global_id"] == gq and r[1]["global_id"] == gp, f"{r}")

# 7. Retention: an identity unseen for longer than retention_seconds is no longer matchable.
g = gallery(retention_seconds=60); p = person()
a = g.resolve_window([cand("c1", 1, 0, 0, 0, p)])[0]
b = g.resolve_window([cand("c1", 2, 61, 1, 0, p)])[0]
c = gallery(retention_seconds=60)
a2 = c.resolve_window([cand("c1", 1, 0, 0, 0, p)])[0]
b2 = c.resolve_window([cand("c1", 2, 59, 1, 0, p)])[0]
check("retention boundary: 61 s gap not matched, 59 s gap matched",
      a["global_id"] != b["global_id"] and a2["global_id"] == b2["global_id"])

# 8. Same-camera exclusivity: a track that starts before the identity's current track in the
#    same camera has ended can't take that identity.
g = gallery(); p = person()
a = g.resolve_window([cand("c1", 1, 0, 0, 0, p, exit_timestamp=30, exit_x=10, exit_y=0)])[0]
b = g.resolve_window([cand("c1", 2, 20, 9, 0, p)])[0]
check("one identity is never on two overlapping tracks in one camera", a["global_id"] != b["global_id"])

# 9. Per-candidate thresholds: similarity 0.80 passes a 0.75 bar and fails a 0.85 bar.
p = person(); r = rng.normal(size=D); r = unit(r - (r @ p) * p); q = unit(0.8 * p + 0.6 * r)   # exactly 0.80 similar
sim = float(p @ q)
g = gallery(); ga = g.resolve_window([cand("c1", 1, 0, 0, 0, p)])[0]["global_id"]
pass_ = g.resolve_window([cand("c1", 2, 10, 2, 0, q, threshold=0.75)])[0]["global_id"] == ga
g = gallery(); gb = g.resolve_window([cand("c1", 1, 0, 0, 0, p)])[0]["global_id"]
fail_ = g.resolve_window([cand("c1", 2, 10, 2, 0, q, threshold=0.85)])[0]["global_id"] != gb
check("each candidate's own similarity bar is applied", pass_ and fail_ and 0.78 < sim < 0.82, f"sim {sim:.3f}")

# 10. Best-view rule: two fingerprint sets match on their most similar pair of views.
front, side, back = person(), person(), person()
check("set similarity uses the best-matching pair of views",
      abs(set_similarity(np.vstack([front, side]), np.vstack([back, like(side, 0.05)])) - float(side @ like(side, 0.05))) < 0.05)

# 11. Exemplar cap: an identity never stores more than max_exemplars views.
g = gallery(max_exemplars=5); p = person()
for k in range(8):
    g.resolve_window([cand("c1", k + 1, 10.0 * k, 0.5 * k, 0, np.vstack([like(p, 0.1) for _ in range(3)]))])
check("identity keeps at most max_exemplars fingerprints", max(len(e) for e in g._exemplars) <= 5)

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
