from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.metrics import binary_metrics
from src.sampling import sample_support
from src.utils import FeatureStore, load_yaml, set_seed


def sample_forensic_support(df, k, seed, adaptations):
    return sample_support(df, k, seed, fake_manipulations=adaptations, split="train")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/stage_a.yaml"); args = parser.parse_args()
    cfg = load_yaml(args.config); set_seed(cfg.get("seed", 42))
    root = Path(cfg["paths"]["feature_root"])
    train_store = FeatureStore(root / "train.pt")
    test_store = FeatureStore(root / "test.pt")
    train_df, test_df = train_store.metadata, test_store.metadata

    adaptations = ["DF", "F2F"]
    rows = []
    all_x = train_store.clip.numpy()
    all_y = train_df.label.to_numpy().astype(int)

    # Open-set-safe continuous content probe: predict the frozen face-recognition
    # embedding, not a closed-set identity ID.
    for k in cfg["stage_a"]["k_values"]:
        for seed in cfg["stage_a"]["seeds"]:
            try:
                support = sample_forensic_support(train_df, k, seed, adaptations)
            except ValueError as exc:
                print(f"Skipping K={k}: {exc}")
                continue
            sx, sc, _ = train_store.get(support)
            tx, tc, _ = test_store.get(test_df)
            sx, sc, tx, tc = sx.numpy(), sc.numpy(), tx.numpy(), tc.numpy()

            alpha = 1.0
            content_probe = Ridge(alpha=alpha)
            content_probe.fit(sx, sc)
            pred_content = content_probe.predict(tx)
            pred_content /= np.linalg.norm(pred_content, axis=1, keepdims=True) + 1e-8
            target_content = tc / (np.linalg.norm(tc, axis=1, keepdims=True) + 1e-8)
            content_cos = float(np.mean(np.sum(pred_content * target_content, axis=1)))

            forensic_probe = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=2000, class_weight="balanced"))
            labels = support.label.to_numpy().astype(int)
            forensic_probe.fit(sx, labels)
            scores = forensic_probe.predict_proba(tx)[:, 1]
            fm = binary_metrics(test_df.label.to_numpy().astype(int), scores)

            rows.append({
                "stage": "A",
                "k": k,
                "seed": seed,
                "content_cosine": content_cos,
                **{f"forensic_{name}": value for name, value in fm.items()},
            })
            print(f"K={k} seed={seed}: content_cos={content_cos:.4f} forensic_auc={fm['auc']:.4f}")

    out = Path(cfg["paths"]["results_root"]); out.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(rows)
    result_df.to_csv(out / "stage_a_results.csv", index=False)
    summary = result_df.groupby("k").agg({"content_cosine": ["mean", "std"], "forensic_auc": ["mean", "std"]})
    summary.to_csv(out / "stage_a_summary.csv")


if __name__ == "__main__":
    main()
