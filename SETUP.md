# SCOUT — Setup Guide

This covers getting SCOUT running from a bare machine. For what the system does and how its pieces fit together, see [README.md](README.md); this file is just "how do I get it running."

## Prerequisites

- **Python 3.11.** SCOUT was built and tested against 3.11 specifically; other 3.x versions are untested.
- **An NVIDIA GPU with CUDA 12.8-compatible drivers**, for real-time-capable detection/tracking/embedding speed. A CPU-only setup works (see below) but inference and Re-ID embedding extraction will be much slower — fine for a quick test, not for processing real footage at scale.
- **Windows or Linux.** This has only been run on Windows so far; paths in the codebase use `pathlib`/forward slashes throughout, so Linux should work but hasn't been verified.

## 1. Get the code

Clone or copy this repository to your machine.

## 2. Create the virtual environment

```bash
python -m venv .venv
```

Activate it:
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- Windows (cmd): `.venv\Scripts\activate.bat`
- Linux/macOS: `source .venv/bin/activate`

## 3. Install PyTorch first, from PyTorch's own index

`requirements.txt` pins CUDA 12.8 builds of `torch`/`torchvision`, which are **not** on the default PyPI index — installing `requirements.txt` directly without this step will fail to resolve those two lines.

```bash
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

**No NVIDIA GPU, or a different CUDA version?** Skip the pinned versions above and instead install whatever build matches your hardware from [pytorch.org's install selector](https://pytorch.org/get-started/locally/) — for CPU-only: `pip install torch torchvision`. Then remove the two `torch`/`torchvision` lines from `requirements.txt` before the next step, since they'd otherwise conflict.

## 4. Install everything else

```bash
pip install -r requirements.txt
```

Everything past the `torch`/`torchvision` lines installs normally from PyPI — no other package in this project has hit a native-binary or CUDA-index complication.

## 5. Verify the install

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available: True` confirms the GPU path is live. `False` means you're on CPU-only (expected if you followed the CPU-only branch in step 3, a problem otherwise).

## 6. Model weights

- **`weights/best.pt`** — the trained YOLOv11 detection model — is already included in this repo (see `weights/README.md`).
- **`reid/weights/osnet_ain_x1_0_msmt17.pth`** — the Re-ID embedding model — is **not** included (third-party checkpoint, redistribution terms unverified). Only needed if you're using cross-camera identity (`--extract-embeddings` / `mcmt_fusion.py`); detection and tracking alone work without it. See `reid/weights/DOWNLOAD_WEIGHTS.md` for the exact file and source to download.

## 7. Quick smoke test

Run detection+tracking on a short slice of any video you have, without committing to a full run:

```bash
python run_inference.py --source path/to/a/video.mp4 --max-frames 300 --output-dir test_output/
```

A `test_output/<camera_id>_tracking.csv` with `Camera, Timestamp, Track_ID, Class, Center_X, Center_Y, Foot_X, Foot_Y` rows means the core pipeline is working end to end. From there, see [README.md](README.md) for the full pipeline (calibration, Re-ID embedding extraction, and cross-camera fusion via `mcmt_fusion.py`) and how each stage's output feeds the next.

## Regenerating requirements.txt

If you add or upgrade a package, regenerate the pin list from the real environment rather than hand-editing version numbers:

```bash
pip freeze
```

...and update `requirements.txt` to match, keeping the `torch`/`torchvision`-first-with-its-own-index-URL note intact at the top.
