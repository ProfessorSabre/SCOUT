# Frame extraction tools

Two related but distinct video-to-frames utilities, used to prepare training data for the detection model (`../train_master_model.py`). Both expect to be run from the repo root (e.g. `python frame_extraction_tools/extract_frames.py`) so their default relative paths resolve correctly.

**`extract_frames.py`** -- general-purpose frame extraction for labeling. Pulls one frame every few seconds from every video under a source folder (recursing into subfolders) and saves them as images ready to drag into a labeling tool (e.g. Roboflow). Use this to build a labeling backlog from new footage.

**`mine_negatives.py`** -- finds *candidate* "nothing here" frames (zero detections) for negative training data, using the current trained model to scan video and flag empty-looking frames. Candidates only -- always requires human visual review before upload, since an automated "no detections" signal is not the same as "nothing is actually there" (a real, confirmed failure mode found during this project's own negative-mining pass: two source videos had 100% of their flagged "empty" frames actually containing missed people). Also supports `--target-class` (find frames missing one specific class) and `--soft-confidence-threshold` (catch persistent low-confidence false positives) for scenes that are never fully empty.
