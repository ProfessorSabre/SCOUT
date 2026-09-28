# SCOUT — Setup Guide

This covers getting SCOUT running from a bare machine. For what the system does and how its pieces fit together, see [README.md](README.md); this file is just "how do I get it running."

## Prerequisites

- **Python 3.11.** SCOUT was built and tested against 3.11 specifically; other 3.x versions are untested.
- **A GPU is recommended** for real-time-capable detection/tracking/embedding speed. NVIDIA (CUDA), AMD (ROCm on Linux/WSL2; experimental DirectML on Windows), and Apple Silicon (MPS) are all supported. A CPU-only setup works too (see below) but inference and Re-ID embedding extraction will be much slower — fine for a quick test, not for processing real footage at scale.
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

## 3. Install PyTorch first, matching your hardware

`torch`/`torchvision` are not pinned in `requirements.txt` (see the comment there) because the correct build depends on your GPU vendor. Pick the section below that matches your hardware, then continue to step 4.

### NVIDIA (CUDA)

This is the hardware SCOUT has been developed and validated on.

```bash
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Requires CUDA 12.8-compatible drivers. For a different CUDA version, use [pytorch.org's install selector](https://pytorch.org/get-started/locally/) instead of the command above.

### AMD (Linux / WSL2) — ROCm

Mature, well-supported. Use [pytorch.org's install selector](https://pytorch.org/get-started/locally/), select Linux + ROCm, and run the command it gives you — a specific version isn't hardcoded here since ROCm's supported wheel tags change over time and a stale pinned command would be more misleading than useful. Not verified against this project's own test footage; if you hit issues, they're worth reporting.

### AMD (Windows) — experimental / best-effort

Native ROCm is not available on Windows. The best-effort path is [`torch-directml`](https://github.com/microsoft/DirectML), which routes PyTorch through Microsoft's DirectML rather than a vendor-native backend. This is **not** a fully supported or verified path here — expect rough edges, and expect some ops or performance characteristics to differ from CUDA/ROCm. If reliability matters more than convenience, WSL2 + the Linux/ROCm path above is the more mature option on the same machine.

### Apple Silicon (MPS)

```bash
pip install torch torchvision
```

PyTorch's MPS backend is built into the standard macOS wheel — no special index needed. Not verified against this project's own test footage.

### CPU-only (any platform)

```bash
pip install torch torchvision
```

Works everywhere but inference and Re-ID embedding extraction will be much slower — fine for a quick test, not for processing real footage at scale.

## 4. Install everything else

```bash
pip install -r requirements.txt
```

Everything past the `torch`/`torchvision` lines installs normally from PyPI — no other package in this project has hit a native-binary or CUDA-index complication.

## 5. Verify the install

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('MPS available:', torch.backends.mps.is_available())"
```

One of these should print `True` if you installed a GPU build in step 3 (`CUDA available` covers both NVIDIA and AMD/ROCm, since ROCm builds report through the same `torch.cuda` namespace; `MPS available` covers Apple Silicon). Both `False` means you're on CPU-only (expected if you followed the CPU-only branch in step 3, or the DirectML branch, which doesn't report through either flag, a problem otherwise).

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

...and update `requirements.txt` to match, but strip the `torch==...`/`torchvision==...` lines back out before committing -- those stay unpinned here on purpose (see the comment at the top of `requirements.txt`), since the correct build is platform-specific and `pip freeze` will only capture whichever one happens to be installed on your machine.
