from pathlib import Path

import cv2
import numpy as np
import torch

from osnet_ain import osnet_ain_x1_0

# OSNet-AIN (MSMT17-trained) person appearance embedder. Extracted as a
# standalone architecture file (see osnet_ain.py) rather than depending on
# the full torchreid package, whose setup.py drags in an unrelated
# dependency chain (numpy -> Cython -> gdown) just to read a version string.
# Weights are the original author's own checkpoint from huggingface.co/kaiyangzhou/osnet
# (MSMT17-trained, "AIN" = instance-normalization variant chosen specifically
# for better cross-domain/cross-camera generalization than the plain variant).

WEIGHTS_PATH = Path(__file__).parent / "weights" / "osnet_ain_x1_0_msmt17.pth"
NUM_CLASSES_IN_CHECKPOINT = 4101  # MSMT17 identity count; only the 512-d feature output before this head is used
INPUT_SIZE = (256, 128)  # (height, width), OSNet's expected input
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PersonEmbedder:
    def __init__(self, weights_path: Path = WEIGHTS_PATH, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = osnet_ain_x1_0(num_classes=NUM_CLASSES_IN_CHECKPOINT, pretrained=False)

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in checkpoint.items()}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"OSNet weights did not load cleanly: missing={missing}, unexpected={unexpected}")

        self.model.eval().to(self.device)

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(crop_rgb, (INPUT_SIZE[1], INPUT_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return normalized.transpose(2, 0, 1)  # HWC -> CHW

    @torch.no_grad()
    def embed(self, crops_bgr: list) -> np.ndarray:
        """crops_bgr: list of HxWx3 BGR uint8 arrays (as read by cv2). Returns
        an (N, 512) L2-normalized float32 array, ready for cosine-similarity
        (inner product) search against the gallery."""
        if not crops_bgr:
            return np.zeros((0, 512), dtype=np.float32)

        batch = np.stack([self._preprocess(c) for c in crops_bgr])
        tensor = torch.from_numpy(batch).to(self.device)
        features = self.model(tensor)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().numpy().astype(np.float32)
