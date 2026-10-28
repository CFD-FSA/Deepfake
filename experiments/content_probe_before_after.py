from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from src.models import CFD_FSA
from src.utils import FeatureStore, get_device, load_yaml


def cosine_mean(pred, target):
    pred = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    target = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(pred * target, axis=1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage_d.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    device = get_device(cfg.get("device", "auto"))
    root = Path(cfg["paths"]["feature_root"])
    train_store = FeatureStore(root / "train.pt"); test_store = FeatureStore(root / "test.pt")

    payload = torch.load(args.checkpoint, map_location=device)
    model = CFD_FSA(
        train_store.clip.shape[1],
        cfg["model"].get("forensic_dim", 128),
        train_store.content.shape[1],
        cfg["model"].get("grl_lambda", 1.0),
        cfg["model"].get("adapter_dropout", 0.1),
        use_frequency=cfg["model"].get("use_frequency_branch", False),
    ).to(device)
    model.load_state_dict(payload.get("model", payload))
    model.eval()

    x_train = train_store.clip.numpy(); y_content = train_store.content.numpy()
    x_test = test_store.clip.numpy(); y_test_content = test_store.content.numpy()
    before = Ridge(alpha=1.0).fit(x_train, y_content)
    before_pred = before.predict(x_test)

    with torch.no_grad():
        z_train = model.fba.forensic_embedding(train_store.clip.to(device)).cpu().numpy()
        z_test = model.fba.forensic_embedding(test_store.clip.to(device)).cpu().numpy()
    after = Ridge(alpha=1.0).fit(z_train, y_content)
    after_pred = after.predict(z_test)

    result = pd.DataFrame([{
        "content_cosine_before": cosine_mean(before_pred, y_test_content),
        "content_cosine_after": cosine_mean(after_pred, y_test_content),
    }])
    out = Path(cfg["paths"]["results_root"]); out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "content_probe_before_after.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
