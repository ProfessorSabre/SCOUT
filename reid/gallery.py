import numpy as np
from scipy.optimize import linear_sum_assignment

# Cross-camera identity resolution: multi-camera multi-target (MCMT) matching.
#
# One Global_ID per real person across every camera at a site, decided from
# two kinds of evidence together:
#   - appearance: each identity keeps a SET of fingerprints (embeddings) from
#     its varied views, and a new sighting is compared against whichever of
#     them it matches best. Keeping the views separate -- rather than averaging
#     them into one -- lets someone first seen from behind be recognized later
#     from the front, once both views are on file. Fingerprints are numbers,
#     not images; no picture of anyone is kept.
#   - physical plausibility, in real-world meters (Local_X/Y_Meters from
#     homography_middleware.py transform): a match is ruled out if the person
#     would have had to move faster than anyone plausibly can, or if the
#     identity is already on another track in the same camera at that moment.
# Matches are resolved in time-windowed batches with the Hungarian algorithm
# (scipy.optimize.linear_sum_assignment), so a window's candidates and
# identities get the globally best pairing rather than first-come-first-served.
#
# History, briefly (see Claude.md for detail): a greedy, appearance-only,
# one-vector-per-identity matcher merged 8,195 Auburn tracks into 85 IDs; a
# later version that REPLACED an identity's vector on every match let
# identities chain from person to person at Founders Square (2026-10-06).

# Upper bound for "could the same person have gotten from there to here".
# Deliberately one ceiling for everyone, not a speed per travel mode: people
# switch modes, and a cyclist may ride at walking pace one minute and 20+ mph
# (~9 m/s) the next. 15 m/s (~34 mph) covers fast cycling and e-scooters; the
# check only rules out the physically impossible, never classifies anyone.
# The cost of a generous ceiling: for a pedestrian, more candidates stay in
# play, so appearance and the calibrated similarity bar do more of the work.
DEFAULT_MAX_SPEED_MPS = 15.0

# Two detections this close together, at nearly the same instant, are treated
# as candidates for "same physical person, seen by two overlapping cameras
# right now" rather than "same person who traveled here." Roughly a person's
# own footprint plus typical calibration error, since two independent camera
# calibrations never place the same real point at an identical (x, y).
DEFAULT_OVERLAP_RADIUS_METERS = 3.0

# Candidates from different cameras are only eligible for the same-instant
# overlap merge (above) if their timestamps are this close; beyond it, the
# ordinary travel-time speed gate applies instead.
DEFAULT_SIMULTANEITY_WINDOW_SECONDS = 1.0

# How long an identity stays matchable after it was last seen. Long enough
# that someone briefly out of view (behind a tree, sitting partly occluded on
# a bench) is still the same person when they reappear -- which dwell-time
# analysis depends on. Past roughly a minute the speed gate allows anywhere on
# a plaza-sized site, so a long-gap match rests on appearance alone.
DEFAULT_RETENTION_SECONDS = 300.0

# Used only when a run can't calibrate its own bar (see mcmt_fusion.py,
# --similarity-threshold auto): the appearance similarity a candidate must
# reach to be matched to an existing identity rather than start a new one.
DEFAULT_SIMILARITY_THRESHOLD = 0.80

# Fingerprints kept per identity. Beyond this, the most redundant one (the
# closest duplicate of another already kept) is dropped, so the set stays
# varied -- several views -- rather than filling up with near-copies.
DEFAULT_MAX_EXEMPLARS = 40

# Sentinel cost for a physically-implausible pairing (Hungarian assignment
# needs a large finite number, not np.inf, to stay numerically well-behaved).
IMPLAUSIBLE_COST = 1e6


def exemplars_of(candidate: dict) -> np.ndarray:
    """A candidate's fingerprints as a (k, D) array: `exemplars` if given,
    otherwise its single `vector`."""
    if candidate.get("exemplars") is not None:
        return np.asarray(candidate["exemplars"], dtype=np.float32)
    return np.asarray(candidate["vector"], dtype=np.float32).reshape(1, -1)


def set_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Best match between two fingerprint sets: the most similar pair of views.
    Fingerprints are L2-normalized, so the dot product is cosine similarity."""
    return float((a @ b.T).max())


class MCMTGallery:
    """Cross-camera person identity gallery for one observation session.

    Call resolve_window() once per time-sorted batch of candidates. Each
    candidate is a dict with: camera_id, track_id, timestamp/x/y (entry point,
    seconds and Local meters), exemplars ((k, D) L2-normalized fingerprints)
    or a single vector, and optionally exit_timestamp/exit_x/exit_y and a
    per-candidate `threshold` (the similarity bar for that candidate's time).

    Returns one dict per candidate, in order: global_id, confidence (the
    best-matching similarity, or for brand-new identities None/mean pairwise
    similarity of a same-instant cross-camera cluster).
    """

    def __init__(self, feature_dim: int = 512, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                 max_speed_mps: float = DEFAULT_MAX_SPEED_MPS, overlap_radius_meters: float = DEFAULT_OVERLAP_RADIUS_METERS,
                 simultaneity_window_seconds: float = DEFAULT_SIMULTANEITY_WINDOW_SECONDS,
                 retention_seconds: float = DEFAULT_RETENTION_SECONDS, max_exemplars: int = DEFAULT_MAX_EXEMPLARS):
        self.feature_dim = feature_dim
        self.similarity_threshold = similarity_threshold
        self.max_speed_mps = max_speed_mps
        self.overlap_radius_meters = overlap_radius_meters
        self.simultaneity_window_seconds = simultaneity_window_seconds
        self.retention_seconds = retention_seconds
        self.max_exemplars = max_exemplars

        # One entry per known Global_ID:
        #   _exemplars: its fingerprint set
        #   _x/_y/_timestamp/_camera_id: its most recent sighting in any camera
        #   _last_exit_by_camera: when its latest track in each camera ended, so
        #     one identity can never be on two overlapping tracks in one camera
        self._exemplars = []
        self._x = np.zeros(0, dtype=np.float64)
        self._y = np.zeros(0, dtype=np.float64)
        self._timestamp = np.zeros(0, dtype=np.float64)
        self._camera_id = []
        self._last_exit_by_camera = []
        self.global_ids = []
        self._index = {}
        self._next_global_id = 1

    def resolve_window(self, candidates: list[dict]) -> list[dict]:
        """The entry point (timestamp/x/y) is what gets matched -- "could this
        person have arrived here from where an identity was last seen" -- and
        the exit point is what the identity is left at for whoever comes next.
        Without exit fields, the entry point is used for both."""
        if not candidates:
            return []

        n = len(candidates)
        results = [None] * n
        sets = [exemplars_of(c) for c in candidates]
        thresholds = np.array([c.get("threshold", self.similarity_threshold) for c in candidates], dtype=np.float64)

        # Only identities seen within the retention window are matchable, so
        # the assignment problem scales with recent activity, not with every
        # identity ever seen in the session.
        earliest = min(c["timestamp"] for c in candidates)
        active = np.flatnonzero(earliest - self._timestamp <= self.retention_seconds)
        if len(active):
            cost_matrix = self._build_cost_matrix(candidates, sets, active)
            # One dummy "new identity" column per candidate: a candidate only
            # takes a real identity when that beats starting a new one, i.e.
            # when its similarity clears that candidate's own bar.
            dummy_block = np.repeat((1.0 - thresholds).reshape(n, 1), n, axis=1)
            full_cost = np.hstack([cost_matrix, dummy_block])
            row_idx, col_idx = linear_sum_assignment(full_cost)

            # Each identity column can go to at most one candidate per window,
            # so when two simultaneous candidates in one camera want the same
            # identity, the higher-confidence (lower-cost) one gets it.
            for i, col in zip(row_idx, col_idx):
                cost = full_cost[i, col]
                if col < len(active) and cost < IMPLAUSIBLE_COST and cost <= 1.0 - thresholds[i]:
                    results[i] = {"global_id": self.global_ids[active[col]], "confidence": round(float(1.0 - cost), 3)}

        new_candidate_indices = [i for i in range(n) if results[i] is None]
        if new_candidate_indices:
            self._resolve_new_identities(candidates, sets, thresholds, new_candidate_indices, results)

        for i, c in enumerate(candidates):
            self._record(results[i]["global_id"], sets[i],
                         c.get("exit_x", c["x"]), c.get("exit_y", c["y"]),
                         c.get("exit_timestamp", c["timestamp"]), c["camera_id"])
        return results

    def _build_cost_matrix(self, candidates: list[dict], sets: list[np.ndarray], active: np.ndarray) -> np.ndarray:
        n, m = len(candidates), len(active)

        # Appearance: best-matching view pair between each candidate and each
        # identity, computed in one pass over all active identities' fingerprints.
        stacked = np.vstack([self._exemplars[k] for k in active])
        starts = np.cumsum([0] + [len(self._exemplars[k]) for k in active[:-1]])
        similarity = np.vstack([np.maximum.reduceat((s @ stacked.T).max(axis=0), starts) for s in sets])
        appearance_cost = 1.0 - similarity

        cand_x = np.array([c["x"] for c in candidates]).reshape(n, 1)
        cand_y = np.array([c["y"] for c in candidates]).reshape(n, 1)
        cand_t = np.array([c["timestamp"] for c in candidates]).reshape(n, 1)
        gal_t = self._timestamp[active].reshape(1, m)
        dist = np.hypot(cand_x - self._x[active].reshape(1, m), cand_y - self._y[active].reshape(1, m))
        # Unsigned on purpose: a candidate in ANOTHER camera may start before
        # the identity's last sighting ended (one person visible to two
        # cameras at once), and the distance a person can cover is bounded by
        # the time between the two sightings either way. Same-camera overlap
        # is a different matter, handled below.
        dt = np.abs(cand_t - gal_t)

        # Required speed to have traveled between the two sightings. dt==0
        # (same-instant, different camera) is handled by the overlap-radius
        # check instead of a division that would blow up to infinite speed.
        with np.errstate(divide="ignore", invalid="ignore"):
            required_speed = np.where(dt > 0, dist / np.maximum(dt, 1e-6), np.inf)

        implausible = (dt <= self.simultaneity_window_seconds) & (dist > self.overlap_radius_meters)
        implausible |= (dt > self.simultaneity_window_seconds) & (required_speed > self.max_speed_mps)
        implausible |= (cand_t - gal_t) > self.retention_seconds

        # One identity, one track per camera at any moment: a candidate can't
        # take an identity whose latest track in the same camera hasn't ended
        # by the time this one starts.
        for i, c in enumerate(candidates):
            for j, k in enumerate(active):
                last_exit = self._last_exit_by_camera[k].get(c["camera_id"])
                if last_exit is not None and c["timestamp"] <= last_exit:
                    implausible[i, j] = True

        return np.where(implausible, IMPLAUSIBLE_COST, appearance_cost)

    def _resolve_new_identities(self, candidates, sets, thresholds, indices, results):
        """Second pass: cluster candidates that got no plausible gallery match
        against EACH OTHER, so two overlapping cameras seeing the same
        brand-new person at once merge into one Global_ID instead of two."""
        parent = {i: i for i in indices}
        cluster_cameras = {i: {candidates[i]["camera_id"]} for i in indices}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        pairs = []
        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                i, j = indices[a_pos], indices[b_pos]
                ci, cj = candidates[i], candidates[j]
                if ci["camera_id"] == cj["camera_id"]:
                    continue  # same camera can't be simultaneous sightings of one person
                if abs(ci["timestamp"] - cj["timestamp"]) > self.simultaneity_window_seconds:
                    continue
                if np.hypot(ci["x"] - cj["x"], ci["y"] - cj["y"]) > self.overlap_radius_meters:
                    continue
                similarity = set_similarity(sets[i], sets[j])
                if similarity >= max(thresholds[i], thresholds[j]):
                    pairs.append((similarity, i, j))

        # Strongest pairs first, and never let a cluster hold two tracks from
        # the same camera: A(cam1)~B(cam2) plus B~C(cam1) would otherwise put
        # two simultaneous cam1 tracks under one identity. The higher-
        # confidence link wins; the weaker one is left as a separate identity.
        for _, i, j in sorted(pairs, reverse=True):
            ri, rj = find(i), find(j)
            if ri != rj and not (cluster_cameras[ri] & cluster_cameras[rj]):
                parent[ri] = rj
                cluster_cameras[rj] |= cluster_cameras.pop(ri)

        clusters = {}
        for i in indices:
            clusters.setdefault(find(i), []).append(i)

        for members in clusters.values():
            new_id = self._mint_global_id()
            if len(members) == 1:
                confidence = None  # nothing to compare a genuinely first sighting against
            else:
                sims = [set_similarity(sets[a], sets[b]) for x, a in enumerate(members) for b in members[x + 1:]]
                confidence = round(float(np.mean(sims)), 3)
            for i in members:
                results[i] = {"global_id": new_id, "confidence": confidence}

    def _mint_global_id(self) -> str:
        new_id = f"Global_ID_{self._next_global_id:04d}"
        self._next_global_id += 1
        return new_id

    def mint_unlinked_id(self) -> str:
        """A Global_ID for a track that is deliberately not matched on
        appearance (too small in frame, or no usable fingerprint). It never
        enters the gallery, so nothing can match to it later either."""
        return self._mint_global_id()

    def _record(self, global_id: str, exemplars: np.ndarray, x: float, y: float, timestamp: float, camera_id: str):
        k = self._index.get(global_id)
        if k is None:
            k = len(self.global_ids)
            self._index[global_id] = k
            self.global_ids.append(global_id)
            self._exemplars.append(np.asarray(exemplars, dtype=np.float32))
            self._x = np.append(self._x, x)
            self._y = np.append(self._y, y)
            self._timestamp = np.append(self._timestamp, timestamp)
            self._camera_id.append(camera_id)
            self._last_exit_by_camera.append({})
        else:
            self._exemplars[k] = np.vstack([self._exemplars[k], exemplars])
            # Several tracks of one identity can be committed in one window;
            # "last seen" is the latest of them, not whichever is recorded last.
            if timestamp >= self._timestamp[k]:
                self._x[k], self._y[k], self._timestamp[k], self._camera_id[k] = x, y, timestamp, camera_id
        self._exemplars[k] = self._prune(self._exemplars[k])
        previous = self._last_exit_by_camera[k].get(camera_id, -np.inf)
        self._last_exit_by_camera[k][camera_id] = max(previous, timestamp)

    def _prune(self, exemplars: np.ndarray) -> np.ndarray:
        """Keep the set varied: while over the cap, drop the fingerprint that
        most closely duplicates another one already kept."""
        if len(exemplars) <= self.max_exemplars:
            return exemplars
        keep = list(range(len(exemplars)))
        sim = exemplars @ exemplars.T
        np.fill_diagonal(sim, -np.inf)
        while len(keep) > self.max_exemplars:
            sub = sim[np.ix_(keep, keep)]
            keep.pop(int(np.argmax(sub.max(axis=1))))
        return exemplars[keep]
