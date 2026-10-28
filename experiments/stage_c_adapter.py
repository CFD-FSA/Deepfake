from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import optim

from src.metrics import binary_metrics
from src.models import ForensicBottleneckAdapter
from src.protonet import prototypical_loss, prototypical_logits
from src.sampling import build_query_set, sample_support, validate_no_source_overlap
from src.utils import FeatureStore, get_device, load_yaml, set_seed, count_parameters


def train_adapter(model, train_store, train_df, cfg, adaptation, k, device):
    opt = optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"].get("weight_decay", 0.0))
    model.train()
    for epoch in range(cfg["training"]["epochs"]):
        for episode in range(cfg["training"]["episodes_per_epoch"]):
            seed = cfg["seed"] + epoch * 10000 + episode
            support = sample_support(train_df, k, seed, fake_manipulations=adaptation, split="train")
            # Draw a query from the same adaptation distribution for adaptation only.
            query = sample_support(train_df, cfg["training"].get("query_per_class", 16), seed + 1, fake_manipulations=adaptation, split="train")
            s_clip, _, s_freq = train_store.get(support)
            q_clip, _, q_freq = train_store.get(query)
            s_clip, q_clip = s_clip.to(device), q_clip.to(device)
            s_labels = torch.tensor(support.label.to_numpy(), dtype=torch.long, device=device)
            q_labels = torch.tensor(query.label.to_numpy(), dtype=torch.long, device=device)
            s_z = model(s_clip)
            q_z = model(q_clip)
            loss, _, _ = prototypical_loss(s_z, s_labels, q_z, q_labels, cfg["model"].get("temperature", 0.07))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


def evaluate(model, train_store, train_df, test_store, test_df, cfg, adaptation, held_out, k, seed, device):
    model.eval()
    support = sample_support(train_df, k, seed, fake_manipulations=adaptation, split="train")
    query = build_query_set(test_df, held_out_manipulations=held_out, split="test")
    validate_no_source_overlap(support, query)
    rng = np.random.default_rng(seed)
    qr = query[query.label.eq(0)]; qf = query[query.label.eq(1)]
    qn = cfg["training"].get("query_per_class", 64)
    if len(qr) > qn: qr = qr.iloc[rng.choice(len(qr), qn, replace=False)]
    if len(qf) > qn: qf = qf.iloc[rng.choice(len(qf), qn, replace=False)]
    query = pd.concat([qr, qf], ignore_index=True)
    s_clip, _, s_freq = train_store.get(support); q_clip, _, q_freq = test_store.get(query)
    with torch.no_grad():
        s_z = model(s_clip.to(device)); q_z = model(q_clip.to(device))
        s_labels = torch.tensor(support.label.to_numpy(), dtype=torch.long, device=device)
        logits, _ = prototypical_logits(s_z, s_labels, q_z, cfg["model"].get("temperature", 0.07))
        scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return binary_metrics(query.label.to_numpy(), scores)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/stage_c.yaml"); args = parser.parse_args()
    cfg = load_yaml(args.config); device = get_device(cfg.get("device", "auto")); set_seed(cfg.get("seed", 42))
    train_store = FeatureStore(Path(cfg["paths"]["feature_root"]) / "train.pt")
    test_store = FeatureStore(Path(cfg["paths"]["feature_root"]) / "test.pt")
    train_df, test_df = train_store.metadata, test_store.metadata
    clip_dim = train_store.clip.shape[1]
    results = []
    for k in cfg["few_shot"]["k_values"]:
        model = ForensicBottleneckAdapter(clip_dim, cfg["model"].get("forensic_dim", 128), cfg["model"].get("adapter_dropout", 0.1)).to(device)
        model = train_adapter(model, train_store, train_df, cfg, cfg["experiment"]["adaptation_manipulations"], k, device)
        for seed in cfg["few_shot"]["support_seeds"]:
            metrics = evaluate(model, train_store, train_df, test_store, test_df, cfg, cfg["experiment"]["adaptation_manipulations"], cfg["experiment"]["held_out_manipulations"], k, seed, device)
            results.append({"stage": "C", "k": k, "seed": seed, "trainable_params": count_parameters(model, True), **metrics})
    out = Path(cfg["paths"]["results_root"]); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / "stage_c_results.csv", index=False)


if __name__ == "__main__":
    main()
