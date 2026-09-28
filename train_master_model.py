from ultralytics import YOLO

def main():
    # V3: upgrading from yolo11s (Small) to yolo11m (Medium) for the heavyweight
    # model. Swap to "yolo11l.pt" for Large if m's accuracy isn't enough --
    # l costs meaningfully more VRAM/train time, so start with m first.
    model = YOLO("yolo11m.pt")

    # The original batch=16 was tuned for an RTX 3080 (12GB) running yolo11s.
    # This is now training a bigger model (m, not s) on a different GPU (RTX
    # 5090, 24GB) -- neither assumption carries over automatically. 16 is a
    # reasonable starting point (2x the VRAM roughly offsetting the bigger
    # model), but it hasn't been empirically verified on this GPU/model combo
    # yet, and batch size will need retuning again on hardware with a
    # different memory budget (AMD/Apple Silicon memory sizes vary
    # independently of this history). Watch for OOM on the first run and
    # adjust; don't trust this number blindly on any GPU.
    results = model.train(
        data="datasets/PBED/data.yaml",
        epochs=50,
        # Training at 640 while run_inference.py's actual inference runs at
        # imgsz=1280 (see Claude.md) is a real train/inference resolution
        # mismatch worth deciding on deliberately -- left at 640 here since
        # bumping it to 1280 substantially increases training VRAM/time cost
        # and hasn't been discussed. Revisit if V3 accuracy disappoints.
        imgsz=640,
        batch=16,
        # No explicit device= here: Ultralytics auto-selects cuda -> mps ->
        # cpu when omitted, matching run_inference.py's model.track() call,
        # which already relies on the same auto-selection to stay
        # vendor-agnostic rather than hardcoding a CUDA device index.
        project="Site_Analyzer_Runs",
        name="master_7class_v3"  # 7 classes as of the wheelchair-removal decision, not 9
    )

if __name__ == '__main__':
    main()
