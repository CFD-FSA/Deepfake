from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def block_dct_descriptor(image: Image.Image, block_size: int = 8) -> np.ndarray:
    """Compact secondary frequency descriptor for the optional frequency ablation.

    This is intentionally secondary to CFD-FSA. It is not the proposed core method.
    """
    gray = np.asarray(image.convert("L").resize((224, 224)), dtype=np.float32) / 255.0
    h, w = gray.shape
    values = []
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = gray[y:y+block_size, x:x+block_size]
            dct = cv2.dct(block)
            values.append(np.abs(dct).reshape(-1))
    return np.mean(np.stack(values), axis=0).astype(np.float32)
