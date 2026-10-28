from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim

from src.metrics import binary_metrics
from src.models import ForensicBottleneckAdapter, StandardForensicClassifier
from src.protonet import cosine_logits
from src.sampling import build_query_set, sample_support, validate_no_source_overlap
from src.utils import FeatureStore, get_device, load_yaml, set_seed

ABLATIONS = {
    "D_no_bottleneck": ["python", "experiments/stage_d_cfd_fsa.py", "--no-bottleneck"],
    "F_no_acs": ["python", "experiments/stage_d_cfd_fsa.py", "--no-acs"],
    "G_lambda_0.1": ["python", "experiments/stage_d_cfd_fsa.py", "--grl-lambda", "0.1"],
    "G_lambda_0.5": ["python", "experiments/stage_d_cfd_fsa.py", "--grl-lambda", "0.5"],
    "G_lambda_2.0": ["python", "experiments/stage_d_cfd_fsa.py", "--grl-lambda", "2.0"],
    "H_dim_64": ["python", "experiments/stage_d_cfd_fsa.py", "--forensic-dim", "64"],
    "H_dim_256": ["python", "experiments/stage_d_cfd_fsa.py", "--forensic-dim", "256"],
}


def standard_classifier_ablation(config_path: str):
    cfg = load_yaml(config_path)
    device = get_device(cfg.get("device", "auto")); set_seed(cfg.get("seed", 42))
    root = Path(cfg["paths"]["feature_root"])
    train_store = FeatureStore(root / "train.pt"); test_store = FeatureStore(root / "test.pt")
    train_df, test_df = train_store.metadata, test_store.metadata
    results = []

    for k in cfg["few_shot"]["k_values"]:
        for seed in cfg["few_shot"]["support_seeds"]:
            support = sample_support(train_df, k, seed, fake_manipulations=cfg["experiment"]["adaptation_manipulations"], split="train")
            query = build_query_set(test_df, held_out_manipulations=cfg["experiment"]["held_out_manipulations"], split="test")
            validate_no_source_overlap(support, query)
            s_clip, _, _ = train_store.get(support); q_clip, _, _ = test_store.get(query)
            s_y = torch.tensor(support.label.to_numpy(), dtype=torch.long, device=device)
            q_y = torch.tensor(query.label.to_numpy(), dtype=torch.long, device=device)

            fba = ForensicBottleneckAdapter(train_store.clip.shape[1], cfg["model"].get("forensic_dim", 128)).to(device)
            clf = StandardForensicClassifier(cfg["model"].get("forensic_dim", 128)).to(device)
            opt = optim.AdamW(list(fba.parameters()) + list(clf.parameters()), lr=cfg["training"]["learning_rate"])
            s_clip = s_clip.to(device); q_clip = q_clip.to(device)
            for _ in range(cfg["training"]["epochs"]):
                fba.train(); clf.train()
                z = fba.forensic_embedding(s_clip)
                loss = F.cross_entropy(clf(z), s_y)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            fba.eval(); clf.eval()
            with torch.no_grad():
                score = torch.softmax(clf(fba.forensic_embedding(q_clip)), dim=1)[:, 1].cpu().numpy()
            metrics = binary_metrics(q_y.cpu().numpy(), score)
            results.append({"ablation": "E_standard_classifier", "k": k, "seed": seed, **metrics})
            print(f"E standard classifier K={k} seed={seed} AUC={metrics['auc']:.4f}")

    out = Path("results/ablations"); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / "E_standard_classifier.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage_d.yaml")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.list:
        for name, cmd in ABLATIONS.items():
            print(name, "::", " ".join(cmd))
        print("E_standard_classifier :: implemented in this script")
        print("I_content_targets :: provide an alternate aligned content-target cache")
        print("J_before_after :: python experiments/content_probe_before_after.py --checkpoint <checkpoint>")
        print("K_independent_forensic_probe :: Stage A")
        print("L_frequency_on :: set model.use_frequency_branch=true")
        return
    if args.name == "E_standard_classifier":
        standard_classifier_ablation(args.config)
        return
    if not args.name or args.name not in ABLATIONS:
        raise SystemExit("Choose --list, E_standard_classifier, or a valid ablation name.")
    cmd = ABLATIONS[args.name]
    print("Running:", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
