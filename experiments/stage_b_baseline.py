from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from src.metrics import binary_metrics
from src.protonet import prototypical_loss, prototypical_logits
from src.sampling import build_query_set, sample_support, validate_no_source_overlap
from src.utils import FeatureStore, get_device, load_yaml, save_json, set_seed


def run_episode(store_train, train_df, store_test, test_df, k, seed, adaptation, held_out, temperature, query_per_class, device):
    support = sample_support(train_df, k, seed, fake_manipulations=adaptation, split="train")
    query = build_query_set(test_df, adaptation_manipulations=adaptation, held_out_manipulations=held_out, split="test")
    validate_no_source_overlap(support, query)
    # Limit query for fast episode evaluation while keeping both classes.
    rng = np.random.default_rng(seed)
    q_real = query[query.label.eq(0)]
    q_fake = query[query.label.eq(1)]
    if len(q_real) > query_per_class:
        q_real = q_real.iloc[rng.choice(len(q_real), query_per_class, replace=False)]
    if len(q_fake) > query_per_class:
        q_fake = q_fake.iloc[rng.choice(len(q_fake), query_per_class, replace=False)]
    query = pd.concat([q_real, q_fake], ignore_index=True)

    s_clip, _, _ = store_train.get(support)
    q_clip, _, _ = store_test.get(query)
    s_clip, q_clip = s_clip.to(device), q_clip.to(device)
    s_labels = torch.tensor(support.label.to_numpy(), dtype=torch.long, device=device)
    q_labels = torch.tensor(query.label.to_numpy(), dtype=torch.long, device=device)
    logits, _ = prototypical_logits(s_clip, s_labels, q_clip, temperature)
    scores = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    return binary_metrics(q_labels.cpu().numpy(), scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage_b.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    device = get_device(cfg.get("device", "auto"))
    set_seed(cfg.get("seed", 42))

    train_store = FeatureStore(Path(cfg["paths"]["feature_root"]) / "train.pt")
    test_store = FeatureStore(Path(cfg["paths"]["feature_root"]) / "test.pt")
    train_df = train_store.metadata
    test_df = test_store.metadata
    out_dir = Path(cfg["paths"]["results_root"]); out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for k in cfg["few_shot"]["k_values"]:
        for seed in cfg["few_shot"]["support_seeds"]:
            metrics = run_episode(
                train_store, train_df, test_store, test_df,
                k, seed,
                cfg["experiment"]["adaptation_manipulations"],
                cfg["experiment"]["held_out_manipulations"],
                cfg["model"].get("temperature", 0.07),
                cfg["training"].get("query_per_class", 16), device,
            )
            rows.append({"stage": "B", "k": k, "seed": seed, **metrics})
            print(f"k={k} seed={seed}: AUC={metrics['auc']:.4f}")

    pd.DataFrame(rows).to_csv(out_dir / "stage_b_results.csv", index=False)


if __name__ == "__main__":
    main()
