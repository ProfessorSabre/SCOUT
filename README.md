# SCOUT — Spatial Context & Observation Utility Tool

*This is the `gpu-agnostic` branch: auto-detects NVIDIA (CUDA), AMD (ROCm/DirectML), and Apple Silicon (MPS) instead of assuming NVIDIA. If you're running the project's own NVIDIA workstation + Jetson Orin Nano deployment hardware, the [`main` branch](https://github.com/ProfessorSabre/SCOUT/tree/main) is the version actually validated against that setup.*

SCOUT is a computer-vision site-observation system built by the SPACE Lab (Spatial Prediction & Adaptive Contexts Engine), Purdue University Department of Horticulture and Landscape Architecture. It uses fixed, oblique-angle cameras to detect, track, and geolocate pedestrians and micro-mobility users (bicycles, scooters, skateboards, golf carts) in the built environment, producing timestamped, real-world-coordinate movement data for empirical site analysis, GIS, and design-simulation research.

SCOUT is an **observation tool**, not a simulator. It answers "what actually happened at this site" from real video, as ground truth for downstream analysis (dwell time, route choice, occupancy) or for validating a separate simulation against reality. It does not generate predictions, run simulations, or infer behavior it didn't observe.

## What it does

1. **Detect & track** (`run_inference.py`) — a fine-tuned YOLOv11 model detects people and micro-mobility classes in video; Ultralytics' built-in ByteTrack keeps a persistent ID per object within one camera's view. Optionally extracts a per-person appearance embedding for cross-camera identity (below).
2. **Calibrate & georeference** (`homography_middleware.py`, `scout_calibration.html`) — a homography fit from real-world reference points converts pixel coordinates to real-world meters and GPS coordinates. The browser-based calibration tool (`scout_calibration.html`) is a single self-contained HTML file — no backend, no install — for clicking video-frame points against a live satellite map.
3. **Fuse across cameras** (`mcmt_fusion.py`) — for multi-camera deployments, resolves a single persistent identity per real person across overlapping camera views, using combined appearance similarity and physical-plausibility checks (a person can't teleport), solved as a batch assignment problem rather than greedy matching. Produces one composite, site-wide movement dataset.

## Repository layout

```
run_inference.py           Detection + tracking, per-camera CSV + embedding output
homography_middleware.py   Pixel -> real-world coordinate calibration and transform
mcmt_fusion.py              Cross-camera identity resolution and fusion
scout_calibration.html     Standalone browser calibration tool (open directly, no server)
train_master_model.py      YOLOv11 training script
reid/                      Person re-identification (embedding model + cross-camera gallery)
weights/                   Trained detection model weights (best.pt)
calibrations/              Per-site calibration registry (includes a real worked example)
datasets/PBED/             Training dataset config (images sourced separately, see its README)
frame_extraction_tools/    Video-to-frames utilities for building/expanding training data
```

## Getting started

See [SETUP.md](SETUP.md) for environment setup (Python version, PyTorch install for your GPU vendor, dependencies) and a quick smoke test.

Typical single-camera workflow, once set up:

```bash
# 1. Detect and track people/objects in your video, extracting Re-ID embeddings
python run_inference.py --source path/to/video.mp4 --extract-embeddings --output-dir out/

# 2. Calibrate your camera (see scout_calibration.html) once, then apply it
python homography_middleware.py transform --tracking-csv out/<camera_id>_tracking.csv --site-id your_site --output out/<camera_id>_movement.csv

# 3. (Multi-camera only) Fuse identities across every camera at the site
python mcmt_fusion.py --tracking-csvs out/*_movement.csv --embeddings out/*_embeddings.npz --output-dir out/ --composite-output out/site_composite.csv
```

Each stage's output feeds the next; see the docstring-style comments at the top of each script for the full detail on what it reads/writes and why.

## Design notes worth knowing before you deploy

- **Real-world coordinates, not pixels.** Cross-camera matching and any distance/speed reasoning always happens in real-world meters (post-calibration), never raw pixels or raw lat/long degrees — a degree of longitude isn't a constant real-world distance.
- **Person-only cross-camera identity.** Re-identification across cameras is scoped to the `person` class. Other classes (bicycle, vehicle, etc.) only need per-camera tracking, not cross-camera identity, for density/modality analysis.
- **Confidence is a first-class output.** Every cross-camera match carries a similarity/confidence score in the output CSV, not just a bare identity assignment.
- **Camera/site IDs are permanent.** Every physical camera gets a `site_id` + `camera_id` assigned once and never reused after a physical move — recalibrate under a new `camera_id` instead. This is the join key between a camera's saved calibration and every tracking run against it, ever.

## Known limitations

- Cross-camera fusion (`mcmt_fusion.py`) has been validated with synthetic scenario tests and on real single-camera footage, but not yet against genuine multi-camera handoffs — the physical-plausibility thresholds (max walking speed, camera-overlap radius) are literature-informed defaults, not calibrated against real multi-camera ground truth.
- Re-identification accuracy degrades meaningfully for small/distant detections (there's not enough real pixel information in a ~20px-wide crop for any embedding model to extract reliable identity) — this is a property of the input, not a fixable model limitation.
- The `wheelchair` detection class is known to be weak (confirmed hallucinations and confusion with bicycles) and is being rebuilt from better training data.
- No edge-deployment (Jetson-class hardware, live camera feeds, network publishing) code exists yet — this repo is the core detection/tracking/calibration/fusion pipeline, run as batch processing over recorded video.

## License

GPL-3.0 — see [LICENSE](LICENSE).
