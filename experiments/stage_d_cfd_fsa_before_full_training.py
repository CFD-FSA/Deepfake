from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim

from src.metrics import binary_metrics
from src.models import AdversarialContentSuppressor, CFD_FSA, ForensicBottleneckAdapter, GradientReversal, GradientReversalFunction, IdentityForensicAdapter
from src.protonet import prototypical_loss, prototypical_logits
from src.sampling import build_query_set, sample_support, validate_no_source_overlap
from src.utils import FeatureStore, count_parameters, get_device, load_yaml, save_json, set_seed


def content_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()


def train_cfd_fsa(
    model: CFD_FSA,
    train_store: FeatureStore,
    train_df: pd.DataFrame,
    cfg: dict,
    adaptation: list[str],
    k: int,
    device: torch.device,
    acs_enabled: bool = True,
):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(params, lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"].get("weight_decay", 0.0))
    model.train()

    for epoch in range(cfg["training"]["epochs"]):
        for episode in range(cfg["training"]["episodes_per_epoch"]):
            seed = int(cfg["seed"]) + epoch * 10000 + episode
            support = sample_support(train_df, k, seed, fake_manipulations=adaptation, split="train")
            qk = cfg["training"].get("query_per_class", 16)
            query = sample_support(train_df, qk, seed + 17, fake_manipulations=adaptation, split="train")

            s_clip, s_content, s_freq = train_store.get(support)
            q_clip, q_content, q_freq = train_store.get(query)
            s_clip, q_clip = s_clip.to(device), q_clip.to(device)
            s_content, q_content = s_content.to(device), q_content.to(device)
            if s_freq is not None:
                s_freq, q_freq = s_freq.to(device), q_freq.to(device)
            s_labels = torch.tensor(support.label.to_numpy(), dtype=torch.long, device=device)
            q_labels = torch.tensor(query.label.to_numpy(), dtype=torch.long, device=device)

            s_z = model.forensic(s_clip, s_freq if model.use_frequency else None)
            q_z = model.forensic(q_clip, q_freq if model.use_frequency else None)
            forensic_loss, _, _ = prototypical_loss(
                s_z, s_labels, q_z, q_labels, cfg["model"].get("temperature", 0.07)
            )

            total = forensic_loss
            if acs_enabled:
                s_pred = model.acs(s_z)
                q_pred = model.acs(q_z)
                content_loss = content_cosine_loss(
                    torch.cat([s_pred, q_pred], dim=0),
                    torch.cat([s_content, q_content], dim=0),
                )
                total = total + cfg["training"].get("content_loss_weight", 1.0) * content_loss

            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["training"].get("grad_clip", 1.0))
            opt.step()

    return model


def evaluate_cfd(
    model: CFD_FSA,
    train_store: FeatureStore,
    train_df: pd.DataFrame,
    test_store: FeatureStore,
    test_df: pd.DataFrame,
    cfg: dict,
    adaptation: list[str],
    held_out: list[str],
    k: int,
    seed: int,
    device: torch.device,
):
    model.eval()
    support = sample_support(train_df, k, seed, fake_manipulations=adaptation, split="train")
    query = build_query_set(test_df, held_out_manipulations=held_out, split="test")
    validate_no_source_overlap(support, query)

    rng = np.random.default_rng(seed)
    qn = cfg["training"].get("query_per_class", 64)
    real, fake = query[query.label.eq(0)], query[query.label.eq(1)]
    if len(real) > qn: real = real.iloc[rng.choice(len(real), qn, replace=False)]
    if len(fake) > qn: fake = fake.iloc[rng.choice(len(fake), qn, replace=False)]
    query = pd.concat([real, fake], ignore_index=True)

    s_clip, s_content, s_freq = train_store.get(support)
    q_clip, q_content, q_freq = test_store.get(query)
    s_labels = torch.tensor(support.label.to_numpy(), dtype=torch.long, device=device)

    with torch.no_grad():
        sf = s_freq.to(device) if (model.use_frequency and s_freq is not None) else None
        qf = q_freq.to(device) if (model.use_frequency and q_freq is not None) else None
        s_z = model.forensic(s_clip.to(device), sf)
        q_z = model.forensic(q_clip.to(device), qf)
        logits, _ = prototypical_logits(s_z, s_labels, q_z, cfg["model"].get("temperature", 0.07))
        scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        content_pred = model.acs(s_z) if hasattr(model, "acs") else None
        if content_pred is not None:
            # Diagnostics: how predictive of the content target the learned forensic space remains.
            content_cos = float(F.cosine_similarity(content_pred.cpu(), s_content, dim=-1).mean())
        else:
            content_cos = float("nan")

    metrics = binary_metrics(query.label.to_numpy(), scores)
    metrics["support_content_cosine"] = content_cos
    return metrics


def make_model(cfg, clip_dim, content_dim, device, no_acs=False, no_bottleneck=False):
    forensic_dim = clip_dim if no_bottleneck else cfg["model"].get("forensic_dim", 128)
    use_frequency = bool(cfg["model"].get("use_frequency_branch", False))
    if no_bottleneck:
        # A trainable identity mapping keeps the interface consistent; ACS works directly in CLIP space.
        model = CFD_FSA(clip_dim, clip_dim, content_dim, cfg["model"].get("grl_lambda", 1.0), cfg["model"].get("adapter_dropout", 0.1), use_frequency=use_frequency)
        model.fba = IdentityForensicAdapter()
    else:
        model = CFD_FSA(clip_dim, forensic_dim, content_dim, cfg["model"].get("grl_lambda", 1.0), cfg["model"].get("adapter_dropout", 0.1), use_frequency=use_frequency)
    if no_acs:
        model.acs = AdversarialContentSuppressor(forensic_dim, content_dim, grl_lambda=0.0)
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage_d.yaml")
    parser.add_argument("--grl-lambda", type=float, default=None)
    parser.add_argument("--forensic-dim", type=int, default=None)
    parser.add_argument("--no-acs", action="store_true", help="FBA + ProtoNet without content suppression")
    parser.add_argument("--no-bottleneck", action="store_true", help="ACS applied without the compact FBA bottleneck")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.grl_lambda is not None:
        cfg["model"]["grl_lambda"] = args.grl_lambda
    if args.forensic_dim is not None:
        cfg["model"]["forensic_dim"] = args.forensic_dim

    device = get_device(cfg.get("device", "auto")); set_seed(cfg.get("seed", 42))
    feature_root = Path(cfg["paths"]["feature_root"])
    train_store = FeatureStore(feature_root / "train.pt")
    test_store = FeatureStore(feature_root / "test.pt")
    train_df, test_df = train_store.metadata, test_store.metadata
    clip_dim = train_store.clip.shape[1]
    content_dim = train_store.content.shape[1]

    adaptation = cfg["experiment"]["adaptation_manipulations"]
    held_out = cfg["experiment"]["held_out_manipulations"]

    results = []
    for k in cfg["few_shot"]["k_values"]:
        model = make_model(cfg, clip_dim, content_dim, device, args.no_acs, args.no_bottleneck)
        if args.no_acs:
            for p in model.acs.parameters():
                p.requires_grad = False
        model = train_cfd_fsa(model, train_store, train_df, cfg, adaptation, k, device, acs_enabled=not args.no_acs)
        ckpt_dir = Path(cfg["paths"].get("checkpoint_root", "checkpoints/stage_d")); ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "config": cfg, "k": k}, ckpt_dir / f"k{k}.pt")
        for seed in cfg["few_shot"]["support_seeds"]:
            metrics = evaluate_cfd(model, train_store, train_df, test_store, test_df, cfg, adaptation, held_out, k, seed, device)
            if args.no_acs:
                metrics["support_content_cosine"] = float("nan")
            results.append({
                "stage": "D" if not args.no_acs else "F",
                "k": k,
                "seed": seed,
                "grl_lambda": cfg["model"].get("grl_lambda", 0.0),
                "forensic_dim": cfg["model"].get("forensic_dim", clip_dim),
                "trainable_params": count_parameters(model, True),
                **metrics,
            })
            print(f"k={k} seed={seed}: AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f}")

    out_dir = Path(cfg["paths"]["results_root"]); out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / ("stage_d_results.csv" if not args.no_acs else "fba_protonet_results.csv"), index=False)


if __name__ == "__main__":
    main()
