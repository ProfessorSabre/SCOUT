# Detection model weights

`best.pt` is the trained YOLOv11s detection model ("Version 2"), trained on the PBED dataset (`../datasets/PBED/`) for oblique-angle site-observation footage. 8 classes: `bicycle, bus, golfcart, person, scooter, skateboard, vehicle, wheelchair` (`wheelchair` is a known weak class -- see the main README's Known Limitations). mAP50 ≈ 0.795 on the validation split.

This is the only pretrained weight file included directly in this repo -- it's the project's own trained output. The separate Re-ID embedding model's weights are a third-party checkpoint and are documented (with a download link) at `../reid/weights/DOWNLOAD_WEIGHTS.md` instead.
