# PBED training dataset (config only -- images not included)

`data.yaml` is the real Roboflow export config for the dataset this project's detection model was trained on -- class list, split layout, and augmentation-source metadata. The actual images/labels are **not** included in this repo (2,170+ images, too large for a source repo and better consumed directly from the source).

The dataset is published on Roboflow Universe under **CC BY 4.0**:
https://universe.roboflow.com/pbed-purdue-built-environment-database/siteanalyzer-9class/dataset/2

To train or fine-tune, download the dataset (YOLO format) from that URL into `train/`, `valid/`, `test/` folders alongside this `data.yaml`, matching the paths it already specifies.
