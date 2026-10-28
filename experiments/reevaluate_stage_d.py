from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from experiments.stage_d_cfd_fsa import make_model, evaluate_cfd
from src.utils import FeatureStore, get_device, set_seed

CHECKPOINT = Path(r"D:\CFD-FSA-data\checkpoints\stage_d_all_60\k20.pt")
OUTPUT = Path(r"D:\CFD-FSA-data\results\stage_d_all_60\stage_d_results_eer_fixed.csv")


def main():
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    cfg = checkpoint["config"]
    k = int(checkpoint["k"])

    device = get_device(cfg.get("device", "auto"))
    set_seed(cfg.get("seed", 42))

    feature_root = Path(cfg["paths"]["feature_root"])

    train_store = FeatureStore(feature_root / "train.pt")
    test_store = FeatureStore(feature_root / "test.pt")

    train_df = train_store.metadata
    test_df = test_store.metadata

    clip_dim = train_store.clip.shape[1]
    content_dim = train_store.content.shape[1]

    adaptation = cfg["experiment"]["adaptation_manipulations"]
    held_out = cfg["experiment"]["held_out_manipulations"]

    model = make_model(
        cfg,
        clip_dim,
        content_dim,
        device,
    )

    model.load_state_dict(checkpoint["model"])
    model.eval()

    print("Checkpoint:", CHECKPOINT)
    print("Device:", device)
    print("K:", k)
    print("GRL lambda:", cfg["model"]["grl_lambda"])
    print("Content loss weight:", cfg["training"]["content_loss_weight"])
    print()

    results = []

    for seed in cfg["few_shot"]["support_seeds"]:
        metrics = evaluate_cfd(
            model,
            train_store,
            train_df,
            test_store,
            test_df,
            cfg,
            adaptation,
            held_out,
            k,
            int(seed),
            device,
        )

        row = {
            "stage": "D",
            "k": k,
            "seed": int(seed),
            "grl_lambda": cfg["model"].get("grl_lambda", 0.0),
            "forensic_dim": cfg["model"].get("forensic_dim", clip_dim),
            **metrics,
        }

        results.append(row)

        print(
            f"seed={seed}: "
            f"AUC={metrics['auc']:.4f} "
            f"Accuracy={metrics['accuracy']:.4f} "
            f"EER={metrics['eer']:.4f} "
            f"TAR@1%={metrics['tar_at_far_1pct']:.4f} "
            f"F1={metrics['f1']:.4f}"
        )

    df = pd.DataFrame(results)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
