from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.utils import FeatureStore, get_device, load_yaml, set_seed
from src.sampling import sample_support, build_query_set, validate_no_source_overlap
from src.protonet import prototypical_logits
from src.metrics import binary_metrics
from src.models import CFD_FSA
from experiments.stage_d_cfd_fsa import train_cfd_fsa


def evaluate_direction(model, train_store, test_store, train_df, test_df, adaptation, held_out, k, seed, cfg, device):
    support = sample_support(train_df, k, seed, fake_manipulations=adaptation, split="train")
    query = build_query_set(test_df, held_out_manipulations=held_out, split="test")
    validate_no_source_overlap(support, query)
    s_clip, _, _ = train_store.get(support); q_clip, _, _ = test_store.get(query)
    with torch.no_grad():
        s_z = model.fba.forensic_embedding(s_clip.to(device))
        q_z = model.fba.forensic_embedding(q_clip.to(device))
        labels = torch.tensor(support.label.to_numpy(), dtype=torch.long, device=device)
        logits, _ = prototypical_logits(s_z, labels, q_z, cfg["model"].get("temperature", 0.07))
        scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return binary_metrics(query.label.to_numpy(), scores)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/stage_d.yaml"); args = parser.parse_args()
    cfg = load_yaml(args.config); device = get_device(cfg.get("device", "auto")); set_seed(cfg.get("seed", 42))
    root = Path(cfg["paths"]["feature_root"])
    train_store = FeatureStore(root / "train.pt"); test_store = FeatureStore(root / "test.pt")
    train_df, test_df = train_store.metadata, test_store.metadata
    clip_dim, content_dim = train_store.clip.shape[1], train_store.content.shape[1]
    directions = [
        (["DF", "F2F"], ["FS", "NT"]),
        (["FS", "NT"], ["DF", "F2F"]),
    ]
    rows = []
    for adaptation, held_out in directions:
        for k in cfg["few_shot"]["k_values"]:
            # Clone a fresh model for every direction/K.
            model = CFD_FSA(
                clip_dim,
                cfg["model"].get("forensic_dim", 128),
                content_dim,
                cfg["model"].get("grl_lambda", 1.0),
                cfg["model"].get("adapter_dropout", 0.1),
            ).to(device)
            model = train_cfd_fsa(model, train_store, train_df, cfg, adaptation, k, device, acs_enabled=True)
            for seed in cfg["few_shot"]["support_seeds"]:
                metrics = evaluate_direction(model, train_store, test_store, train_df, test_df, adaptation, held_out, k, seed, cfg, device)
                rows.append({"adaptation": "+".join(adaptation), "held_out": "+".join(held_out), "k": k, "seed": seed, **metrics})
                print(f"{adaptation} -> {held_out}, K={k}, seed={seed}, AUC={metrics['auc']:.4f}")
    out = Path("results/cross_manipulation"); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "cross_manipulation_results.csv", index=False)


if __name__ == "__main__":
    main()
