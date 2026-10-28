from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def save_yaml(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(
        p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)
    )

class FeatureStore:
    """Loads a cached split and provides metadata-to-feature lookup."""

    def __init__(self, path: str | Path):
        payload = torch.load(path, map_location="cpu")
        self.clip = payload["clip_features"].float()
        self.content = payload["content_features"].float()
        self.frequency = payload.get("frequency_features")
        if self.frequency is not None:
            self.frequency = self.frequency.float()
        self.metadata = pd.DataFrame(payload["metadata"])
        if "face_path" not in self.metadata.columns:
            raise ValueError(f"Cached feature file has no face_path metadata: {path}")
        feature_lengths = {
            "clip_features": len(self.clip),
            "content_features": len(self.content),
        }
        if self.frequency is not None:
            feature_lengths["frequency_features"] = len(self.frequency)
        if len(set(feature_lengths.values())) != 1 or next(iter(feature_lengths.values())) != len(self.metadata):
            raise ValueError(
                f"Cached feature length mismatch in {path}: "
                f"metadata={len(self.metadata)}, {feature_lengths}. "
                "Re-run build_content_cache.py to regenerate the cache."
            )
        self.index = {str(p): i for i, p in enumerate(self.metadata["face_path"])}

    def get(self, rows):
        if isinstance(rows, pd.DataFrame):
            paths = rows["face_path"].astype(str).tolist()
        else:
            paths = [str(rows)]
        indexes = [self.index[p] for p in paths]
        freq = self.frequency[indexes] if self.frequency is not None else None
        return self.clip[indexes], self.content[indexes], freq
