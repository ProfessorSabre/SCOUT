# OSNet-AIN weights (not included in this repo)

Cross-camera person re-identification (`reid/embedder.py`) needs one pretrained checkpoint that isn't bundled here, since it's a third-party file whose redistribution terms weren't verified for this repo:

**File:** `osnet_ain_x1_0_msmt17.pth`
**Source:** the original author's checkpoint, hosted at [huggingface.co/kaiyangzhou/osnet](https://huggingface.co/kaiyangzhou/osnet)

Download it and place it at:

```
reid/weights/osnet_ain_x1_0_msmt17.pth
```

`reid/embedder.py` expects exactly that path (relative to the `reid/` folder) by default.

Note: the PyPI package literally named `torchreid` is an unofficial third-party repackaging, not the original research group's code. This project doesn't depend on that package at all -- only the standalone `reid/osnet_ain.py` architecture file (extracted from the original repo, zero dependencies beyond PyTorch) plus this checkpoint, loaded directly.
