from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.metrics import binary_metrics
from src.models import CFD_FSA
from src.protonet import prototypical_logits
from src.sampling import sample_support, validate_no_source_overlap
from src.utils import FeatureStore, get_device, load_yaml, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage_d.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-features", required=True, help="Celeb-DF-v2 cached test feature file")
    parser.add_argument("--target-manifest", required=True, help="Celeb-DF-v2 cached metadata CSV")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = get_device(cfg.get("device", "auto")); set_seed(cfg.get("seed", 42))
    root = Path(cfg["paths"]["feature_root"])
    train_store = FeatureStore(root / "train.pt")
    target_store = FeatureStore(args.target_features)
    train_df, target_df = train_store.metadata, target_store.metadata

    clip_dim, content_dim = train_store.clip.shape[1], train_store.content.shape[1]
    model = CFD_FSA(
        clip_dim,
        cfg["model"].get("forensic_dim", 128),
        content_dim,
        cfg["model"].get("grl_lambda", 1.0),
        cfg["model"].get("adapter_dropout", 0.1),
        use_frequency=cfg["model"].get("use_frequency_branch", False),
    ).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    model.eval()

    k = int(payload.get("k", cfg["few_shot"]["k_values"][0]))
    support = sample_support(train_df, k, cfg.get("seed", 42), fake_manipulations=cfg["experiment"]["adaptation_manipulations"], split="train")
    query = target_df.copy()
    validate_no_source_overlap(support, query)

    s_clip, _, _ = train_store.get(support)
    q_clip, _, _ = target_store.get(query)
    labels = torch.tensor(support.label.to_numpy(), dtype=torch.long, device=device)
    with torch.no_grad():
        s_z = model.forensic(s_clip.to(device))
        q_z = model.forensic(q_clip.to(device))
        logits, _ = prototypical_logits(s_z, labels, q_z, cfg["model"].get("temperature", 0.07))
        scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    metrics = binary_metrics(query.label.to_numpy(), scores)
    out = Path("results/cross_dataset"); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out / "celebd_v2_results.csv", index=False)
    print("Celeb-DF-v2:", metrics)


if __name__ == "__main__":
    main()
