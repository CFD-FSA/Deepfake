from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.models import CFD_FSA
from src.utils import FeatureStore, get_device, load_yaml


def run_tsne(x: np.ndarray, seed: int = 42) -> np.ndarray:
    x = StandardScaler().fit_transform(x)

    perplexity = min(30, max(5, (len(x) - 1) // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )

    return tsne.fit_transform(x)


def plot_embedding(embedding, labels, title, output_path):
    plt.figure(figsize=(9, 7))

    labels = np.asarray(labels)

    unique_labels = pd.unique(labels)

    for label in unique_labels:
        mask = labels == label
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=10,
            alpha=0.65,
            label=str(label),
        )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/stage_d.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
    )

    args = parser.parse_args()

    cfg = load_yaml(args.config)

    device = get_device(
        cfg.get("device", "auto")
    )

    root = Path(
        cfg["paths"]["feature_root"]
    )

    test_store = FeatureStore(
        root / "test.pt"
    )

    # -------------------------------------------------
    # Load test data
    # -------------------------------------------------

    clip = test_store.clip
    freq = test_store.frequency
    metadata = test_store.metadata.copy()

    n = len(metadata)

    rng = np.random.default_rng(
        cfg.get("seed", 42)
    )

    if n > args.max_samples:
        indices = np.sort(
            rng.choice(
                n,
                size=args.max_samples,
                replace=False,
            )
        )
    else:
        indices = np.arange(n)

    clip = clip[indices]

    if freq is not None:
        freq = freq[indices]

    metadata = metadata.iloc[indices].reset_index(drop=True)

    # -------------------------------------------------
    # Load trained CFD-FSA
    # -------------------------------------------------

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model = CFD_FSA(
        test_store.clip.shape[1],
        cfg["model"].get("forensic_dim", 128),
        test_store.content.shape[1],
        cfg["model"].get("grl_lambda", 1.0),
        cfg["model"].get("adapter_dropout", 0.1),
        use_frequency=cfg["model"].get(
            "use_frequency_branch",
            False,
        ),
    ).to(device)

    state_dict = checkpoint.get(
        "model",
        checkpoint,
    )

    model.load_state_dict(state_dict)

    model.eval()

    # -------------------------------------------------
    # Extract representations
    # -------------------------------------------------

    with torch.no_grad():

        clip_device = clip.to(device)

        freq_device = (
            freq.to(device)
            if freq is not None
            else None
        )

        # CLIP representation
        clip_repr = (
            clip_device
            .cpu()
            .numpy()
        )

        # FBA-only representation
        fba_repr = (
            model.fba.forensic_embedding(
                clip_device
            )
            .cpu()
            .numpy()
        )

        # Full CFD-FSA forensic representation
        forensic_repr = (
            model.forensic(
                clip_device,
                freq_device,
            )
            .cpu()
            .numpy()
        )

    # -------------------------------------------------
    # Select labels
    # -------------------------------------------------

    if "label" in metadata.columns:
        labels = metadata["label"].to_numpy()

    elif "manipulation" in metadata.columns:
        labels = metadata["manipulation"].astype(str).to_numpy()

    else:
        labels = np.zeros(
            len(metadata),
            dtype=int,
        )

    # -------------------------------------------------
    # Output directory
    # -------------------------------------------------

    out = Path(
        cfg["paths"]["results_root"]
    ) / "representation"

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------
    # t-SNE
    # -------------------------------------------------

    print("Running t-SNE for CLIP...")
    clip_2d = run_tsne(
        clip_repr,
        cfg.get("seed", 42),
    )

    print("Running t-SNE for FBA...")
    fba_2d = run_tsne(
        fba_repr,
        cfg.get("seed", 42),
    )

    print("Running t-SNE for CFD-FSA...")
    forensic_2d = run_tsne(
        forensic_repr,
        cfg.get("seed", 42),
    )

    # -------------------------------------------------
    # Save embeddings
    # -------------------------------------------------

    np.save(
        out / "clip_tsne.npy",
        clip_2d,
    )

    np.save(
        out / "fba_tsne.npy",
        fba_2d,
    )

    np.save(
        out / "cfd_fsa_tsne.npy",
        forensic_2d,
    )

    metadata.to_csv(
        out / "metadata.csv",
        index=False,
    )

    # -------------------------------------------------
    # Create figures
    # -------------------------------------------------

    plot_embedding(
        clip_2d,
        labels,
        "CLIP Representation",
        out / "clip_tsne.png",
    )

    plot_embedding(
        fba_2d,
        labels,
        "FBA Representation",
        out / "fba_tsne.png",
    )

    plot_embedding(
        forensic_2d,
        labels,
        "CFD-FSA Forensic Representation",
        out / "cfd_fsa_tsne.png",
    )

    print()
    print("Representation analysis complete.")
    print(f"Results saved to: {out}")


if __name__ == "__main__":
    main()