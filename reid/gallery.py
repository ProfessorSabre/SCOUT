import numpy as np
from scipy.optimize import linear_sum_assignment

# Cross-camera identity resolution: multi-camera multi-target (MCMT) matching.
#
# Supersedes the earlier ReIDGallery, which matched one track at a time,
# greedily, on appearance alone. That approach collapsed at production scale:
# a full real run merged 8,195 distinct local person tracks into just 85
# Global_IDs, with the single largest absorbing 3,330 tracks (12.3% of every
# person detection across both videos) -- overwhelmingly likely hundreds of
# different real people incorrectly merged, not one person seen that often.
# Two root causes: (1) small/distant detections don't carry enough real pixel
# information for appearance alone to be reliable, and (2) greedy one-at-a-time
# matching has no way to express "this candidate is a good appearance match,
# but 40 meters away one second ago -- physically impossible for a person."
#
# This module fixes both by:
#   - requiring real-world coordinates (Local_X_Meters/Local_Y_Meters from
#     homography_middleware.py transform) so spatial-temporal plausibility can
#     be checked in actual meters, not raw pixels or lat/long degrees (degrees
#     aren't a constant real-world distance and would make a speed cutoff
#     meaningless)
#   - resolving matches in time-windowed BATCHES via the Hungarian algorithm
#     (scipy.optimize.linear_sum_assignment) rather than greedily one at a
#     time, so a window with several plausible candidates and several gallery
#     entries finds the globally optimal assignment instead of whichever
#     candidate happened to be processed first
#   - a hard speed-based plausibility gate: a candidate can never be matched
#     to a gallery entry that would require the same person to have traveled
#     faster than a person physically can, no matter how strong the appearance
#     similarity looks
#   - a second pass that clusters brand-new-identity candidates against each
#     other within the same window, to catch the case where two overlapping
#     cameras both see a brand-new (not-yet-in-gallery) person at the same
#     instant -- without this, that person would silently get two different
#     Global_IDs, one per camera, defeating the entire point of cross-camera
#     identity
#
# Confidence scores are returned per resolution, not discarded, since the
# real-world deployment spec calls for confidence as a first-class output
# column (see Claude.md, Architectural Decisions).

# A person cannot run faster than roughly this without it being a data/
# calibration error, not a real observation -- literature/informal-athletics
# sprinting speeds top out well above this, but SCOUT is observing ordinary
# pedestrian site behavior, and a generous 4.0 m/s (~14.4 km/h, a fast jog)
# gives real people margin without letting a distant, differently-dressed
# stranger get matched just because the timing happens to work out.
DEFAULT_MAX_SPEED_MPS = 4.0

# Two detections this close together, at nearly the same instant, are treated
# as candidates for "same physical person, seen by two overlapping cameras
# right now" rather than "same person who traveled here." Set to roughly a
# person's own footprint plus typical homography reprojection error (Auburn's
# real calibration measured ~0.75m mean / ~3.76m max reprojection error --
# see Claude.md/Development History) rather than zero, since two independent
# camera calibrations will never place the same real point at an identical
# (x, y).
DEFAULT_OVERLAP_RADIUS_METERS = 3.0

# Candidates from different cameras are only eligible for the same-instant
# overlap merge (above) if their timestamps are this close; beyond it, the
# ordinary travel-time speed gate applies instead.
DEFAULT_SIMULTANEITY_WINDOW_SECONDS = 1.0

# Cosine similarity is a lower-is-better cost (1 - similarity) inside the
# assignment matrix. A candidate is only matched to an existing Global_ID if
# its cost is at least this good; otherwise Hungarian assignment is free to
# assign it to a dummy "new identity" column instead of forcing a bad match.
DEFAULT_SIMILARITY_THRESHOLD = 0.55

# Cost assigned to every dummy "new identity" column, so a candidate is only
# matched to a real gallery entry when that match is genuinely better than
# just starting a new identity. Kept equal to the same-cost boundary implied
# by DEFAULT_SIMILARITY_THRESHOLD so both routes are judged on one consistent
# scale.
NEW_IDENTITY_COST = 1.0 - DEFAULT_SIMILARITY_THRESHOLD

# Sentinel cost for a physically-implausible pairing (Hungarian assignment
# needs a large finite number, not np.inf, to stay numerically well-behaved).
IMPLAUSIBLE_COST = 1e6


class MCMTGallery:
    """Cross-camera person identity gallery for one observation session.

    Call resolve_window() once per time-sorted batch of candidate person
    detections (already homography-transformed to real-world meters). Each
    candidate is a dict with keys: camera_id, track_id, timestamp (seconds),
    x, y (Local_X_Meters/Local_Y_Meters), vector (L2-normalized embedding).

    Returns one result dict per candidate, in the same order, with keys:
    global_id, confidence (0-1, or None for a brand-new identity with nothing
    to compare against).
    """

    def __init__(self, feature_dim: int = 512, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                 max_speed_mps: float = DEFAULT_MAX_SPEED_MPS, overlap_radius_meters: float = DEFAULT_OVERLAP_RADIUS_METERS,
                 simultaneity_window_seconds: float = DEFAULT_SIMULTANEITY_WINDOW_SECONDS):
        self.feature_dim = feature_dim
        self.similarity_threshold = similarity_threshold
        self.max_speed_mps = max_speed_mps
        self.overlap_radius_meters = overlap_radius_meters
        self.simultaneity_window_seconds = simultaneity_window_seconds
        self.new_identity_cost = 1.0 - similarity_threshold

        # One row per known Global_ID; state is always the person's most
        # recent sighting, since plausibility is judged against "where were
        # they last seen," not their full history.
        self._vectors = np.zeros((0, feature_dim), dtype=np.float32)
        self._x = np.zeros(0, dtype=np.float64)
        self._y = np.zeros(0, dtype=np.float64)
        self._timestamp = np.zeros(0, dtype=np.float64)
        self._camera_id = []
        self.global_ids = []
        self._next_global_id = 1

    def resolve_window(self, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        n = len(candidates)
        results = [None] * n
        vectors = np.stack([np.asarray(c["vector"], dtype=np.float32) for c in candidates])

        n_gallery = len(self.global_ids)
        if n_gallery > 0:
            cost_matrix = self._build_cost_matrix(candidates, vectors)
            # Pad with one dummy "new identity" column per candidate, so
            # Hungarian assignment can route a candidate to a fresh identity
            # instead of being forced into the least-bad real match.
            dummy_block = np.full((n, n), self.new_identity_cost, dtype=np.float64)
            full_cost = np.hstack([cost_matrix, dummy_block])
            row_idx, col_idx = linear_sum_assignment(full_cost)

            assignment = {r: c for r, c in zip(row_idx, col_idx)}
            for i in range(n):
                col = assignment[i]
                cost = full_cost[i, col]
                if col < n_gallery and cost < IMPLAUSIBLE_COST:
                    results[i] = {"global_id": self.global_ids[col], "confidence": round(float(1.0 - cost), 3)}
        # else: no gallery yet, every candidate falls through to "new identity" below

        new_candidate_indices = [i for i in range(n) if results[i] is None]
        if new_candidate_indices:
            self._resolve_new_identities(candidates, vectors, new_candidate_indices, results)

        for i in range(n):
            c = candidates[i]
            self.update_last_known(results[i]["global_id"], vectors[i], c["x"], c["y"], c["timestamp"], c["camera_id"])

        return results

    def _build_cost_matrix(self, candidates: list[dict], vectors: np.ndarray) -> np.ndarray:
        n = len(candidates)
        n_gallery = len(self.global_ids)

        # cosine similarity == dot product, since all stored/query vectors are L2-normalized
        similarity = vectors @ self._vectors.T
        appearance_cost = 1.0 - similarity

        cand_x = np.array([c["x"] for c in candidates]).reshape(n, 1)
        cand_y = np.array([c["y"] for c in candidates]).reshape(n, 1)
        cand_t = np.array([c["timestamp"] for c in candidates]).reshape(n, 1)
        dist = np.sqrt((cand_x - self._x.reshape(1, n_gallery)) ** 2 + (cand_y - self._y.reshape(1, n_gallery)) ** 2)
        dt = np.abs(cand_t - self._timestamp.reshape(1, n_gallery))

        # Required speed to have traveled from the gallery entry's last known
        # position to this candidate's position in the elapsed time. dt==0
        # (same-instant, different camera) is handled by the overlap-radius
        # check instead of a division that would blow up to infinite speed.
        with np.errstate(divide="ignore", invalid="ignore"):
            required_speed = np.where(dt > 0, dist / np.maximum(dt, 1e-6), np.inf)

        implausible = (dt <= self.simultaneity_window_seconds) & (dist > self.overlap_radius_meters)
        implausible |= (dt > self.simultaneity_window_seconds) & (required_speed > self.max_speed_mps)

        cost = np.where(implausible, IMPLAUSIBLE_COST, appearance_cost)
        return cost

    def _resolve_new_identities(self, candidates, vectors, indices, results):
        """Second pass: cluster candidates that got no plausible gallery match
        against EACH OTHER, so two overlapping cameras seeing the same
        brand-new person at once merge into one Global_ID instead of two."""
        parent = {i: i for i in indices}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                i, j = indices[a_pos], indices[b_pos]
                ci, cj = candidates[i], candidates[j]
                if ci["camera_id"] == cj["camera_id"]:
                    continue  # same camera can't be simultaneous sightings of one person
                dt = abs(ci["timestamp"] - cj["timestamp"])
                if dt > self.simultaneity_window_seconds:
                    continue
                dist = ((ci["x"] - cj["x"]) ** 2 + (ci["y"] - cj["y"]) ** 2) ** 0.5
                if dist > self.overlap_radius_meters:
                    continue
                similarity = float(vectors[i] @ vectors[j])
                if similarity >= self.similarity_threshold:
                    union(i, j)

        clusters = {}
        for i in indices:
            clusters.setdefault(find(i), []).append(i)

        for members in clusters.values():
            new_id = self._mint_global_id()
            if len(members) == 1:
                confidence = None  # nothing to compare a genuinely first sighting against
            else:
                sims = []
                for a in range(len(members)):
                    for b in range(a + 1, len(members)):
                        sims.append(float(vectors[members[a]] @ vectors[members[b]]))
                confidence = round(float(np.mean(sims)), 3)
            for i in members:
                results[i] = {"global_id": new_id, "confidence": confidence}

    def _mint_global_id(self) -> str:
        new_id = f"Global_ID_{self._next_global_id:04d}"
        self._next_global_id += 1
        return new_id

    def update_last_known(self, global_id: str, vector: np.ndarray, x: float, y: float, timestamp: float, camera_id: str):
        """Overwrites a Global_ID's last-known sighting. resolve_window() calls
        this itself using the candidate's match point, but callers that track
        entry/exit separately (a track's FIRST point is the right query for
        "could this plausibly have arrived from elsewhere," while its LAST
        point is the right state to leave behind for whoever is matched
        against next) should call this again afterward with the track's exit
        point -- see mcmt_fusion.py."""
        if global_id in self.global_ids:
            idx = self.global_ids.index(global_id)
            self._vectors[idx] = vector
            self._x[idx] = x
            self._y[idx] = y
            self._timestamp[idx] = timestamp
            self._camera_id[idx] = camera_id
        else:
            self._vectors = np.vstack([self._vectors, vector.reshape(1, -1)]) if len(self.global_ids) else vector.reshape(1, -1)
            self._x = np.append(self._x, x)
            self._y = np.append(self._y, y)
            self._timestamp = np.append(self._timestamp, timestamp)
            self._camera_id.append(camera_id)
            self.global_ids.append(global_id)
