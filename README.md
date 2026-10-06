# SCOUT — Spatial Context & Observation Utility Tool

*This is the `main` branch: validated against this project's actual NVIDIA workstation + Jetson Orin Nano deployment target. If you're running different GPU hardware (AMD or Apple Silicon), the [`gpu-agnostic` branch](https://github.com/ProfessorSabre/SCOUT/tree/gpu-agnostic) auto-detects and supports those instead of assuming NVIDIA.*

SCOUT is a computer-vision site-observation system built by the SPACE Lab (Spatial Prediction & Adaptive Contexts Engine), Purdue University Department of Horticulture and Landscape Architecture. It uses fixed, oblique-angle cameras to detect, track, and geolocate pedestrians and micro-mobility users (bicycles, scooters, skateboards, golf carts) in the built environment, producing timestamped, real-world-coordinate movement data for empirical site analysis, GIS, and design-simulation research.

SCOUT is an **observation tool**, not a simulator. It answers "what actually happened at this site" from real video, as ground truth for downstream analysis (dwell time, route choice, occupancy) or for validating a separate simulation against reality. It does not generate predictions, run simulations, or infer behavior it didn't observe.

## What it does

1. **Detect & track** (`run_inference.py`) — a fine-tuned YOLOv11 model detects people and micro-mobility classes in video; Ultralytics' built-in ByteTrack keeps a persistent ID per object within one camera's view. Optionally extracts up to 20 appearance fingerprints per person track (numeric vectors from the clearest views across the track; no image of anyone is ever written to disk).
2. **Calibrate & georeference** (`homography_middleware.py`, `scout_calibration.html`) — a least-squares homography fit over 6+ real-world reference points converts pixel coordinates to real-world meters and GPS coordinates, reporting each point's leave-one-out error in meters so a bad click is visible. The browser-based calibration tool (`scout_calibration.html`) is a single self-contained HTML file — no backend, no install — for clicking video-frame (or still-image) points against a choice of aerial imagery. Detections above the ground plane's horizon get no coordinates rather than impossible ones.
3. **Fuse the site** (`mcmt_fusion.py`) — works with one camera or many. Resolves one identity per person across the site: by ground position where camera views overlap, and by appearance plus physical plausibility (nobody moves faster than 15 m/s) elsewhere, solved as a batch assignment problem rather than greedy matching. The appearance threshold is calibrated by every run from pairs of people it knows are different, per hour of footage. Also attributes travel mode (on foot, or riding a bicycle/scooter/skateboard, from co-movement), keeps vehicles as anonymous positions, and counts what's present over time. Every run writes a coverage map and a manifest of its settings.

## Repository layout

```
run_inference.py           Detection + tracking, per-camera CSV + embedding output
homography_middleware.py   Pixel -> real-world coordinate calibration and transform
mcmt_fusion.py              Site fusion: identities, travel modes, vehicles, counts
coverage_map.py            Coverage map (written by every fusion run; also runs standalone)
scout_calibration.html     Standalone browser calibration tool (open directly, no server)
train_master_model.py      YOLOv11 training script
reid/                      Person re-identification (embedding model + cross-camera gallery)
weights/                   Trained detection model weights (best.pt)
calibrations/              Per-site calibration registry (includes a real worked example)
datasets/PBED/             Training dataset config (images sourced separately, see its README)
frame_extraction_tools/    Video-to-frames utilities for building/expanding training data
```

## Getting started

See [SETUP.md](SETUP.md) for environment setup (Python version, PyTorch/CUDA install, dependencies) and a quick smoke test.

Typical workflow, once set up (the same steps for one camera or many):

```bash
# 1. Detect and track people/objects in each camera's video, extracting appearance fingerprints
python run_inference.py --source path/to/videos/ --camera-map camera_map.json --extract-embeddings --output-dir out/

# 2. Calibrate each camera once (scout_calibration.html -> calibrate-from-points), then apply it
python homography_middleware.py calibrate-from-points --points-json calibration_points.json
python homography_middleware.py transform --tracking-csv out/<camera_id>_tracking.csv --site-id your_site --output out/<camera_id>_movement.csv

# 3. Fuse the site
python mcmt_fusion.py --tracking-csvs out/*_movement.csv --embeddings out/*_embeddings.npz --site-id your_site --output-dir out/
```

Step 3 writes, into `--output-dir`: `full_site_composite.csv` (one row per person per half second; the file GIS/Grasshopper/Blender should read), `full_site_vehicles.csv`, `counts_over_time.csv`, `coverage_map.png`, `run_manifest.json`, and one `<camera_id>_fused.csv` per camera (the audit layer). Each stage's output feeds the next; see the comments at the top of each script for the full detail on what it reads/writes and why.

## Design notes worth knowing before you deploy

- **Real-world coordinates, not pixels.** Cross-camera matching and any distance/speed reasoning always happens in real-world meters (post-calibration), never raw pixels or raw lat/long degrees — a degree of longitude isn't a constant real-world distance.
- **Person-only identity.** Re-identification is scoped to the `person` class. Bicycles, vehicles, etc. are kept as anonymous positions over time (a vehicle seen by two overlapping cameras at once is kept once) plus counts per interval, never an identity, and no image content such as license plates is retained. A person's travel mode is attributed to the person: within 2 m of a bicycle/scooter/skateboard and moving with it at matching speed and direction means riding it. There are no speed categories.
- **Zero-PII by construction.** Appearance fingerprints are numeric vectors computed in memory; no frame or crop of a person is written to disk.
- **Overlap is where hand-off is reliable.** Where camera views overlap, people are handed between cameras by ground position (same spot within 1.5 m for 2+ s), which depends on calibration accuracy, not appearance. Across gaps between views, identity rests on appearance and should be expected to fragment.
- **Confidence is a first-class output.** Every appearance match carries a similarity/confidence score in the output CSV, not just a bare identity assignment.
- **Reproducible runs.** The appearance threshold is calibrated per run (per hour of footage) from pairs of people known to be different; `run_manifest.json` records every setting, input and calibrated threshold.
- **Camera/site IDs are permanent.** Every physical camera gets a `site_id` + `camera_id` assigned once and never reused after a physical move — recalibrate under a new `camera_id` instead. This is the join key between a camera's saved calibration and every tracking run against it, ever.

## Known limitations

- Cross-camera fusion has been tested on one real three-camera recording (handheld phones, farther away and at shallower angles than the deployment spec). There, position-based hand-off worked between well-calibrated overlapping cameras, but appearance did not distinguish the same person seen from a second camera from a stranger, so identity across gaps between views is not yet reliable. Missed matches have not been measured against hand-traced ground truth.
- People less than 60 px tall aren't matched on appearance (there isn't enough pixel information for any embedding model); they're kept and counted in every output, unflagged, but each track gets its own identity.
- Riders seen from above are often detected only as the bicycle/scooter, so travel mode undercounts riders until the detector is retrained with riders labeled as people.
- The `wheelchair` detection class is known to be weak (confirmed hallucinations and confusion with bicycles) and is being rebuilt from better training data. Until then `mcmt_fusion.py` leaves it out of travel mode, the vehicles file and the counts (`DEFERRED_CLASSES`); raw detections stay in the per-camera fused CSVs.
- No edge-deployment (Jetson-class hardware, live camera feeds, network publishing) code exists yet — this repo is the core detection/tracking/calibration/fusion pipeline, run as batch processing over recorded video.

## License

GPL-3.0 — see [LICENSE](LICENSE).
