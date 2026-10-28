from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim

from src.metrics import binary_metrics
from src.models import AdversarialContentSuppressor, CFD_FSA, ForensicBottleneckAdapter, GradientReversal, GradientReversalFunction, IdentityForensicAdapter
from src.protonet import prototypical_loss, prototypical_logits
from src.sampling import build_query_set, sample_support, sample_query_disjoint, validate_no_source_overlap
from src.utils import FeatureStore, count_parameters, get_device, load_yaml, save_json, set_seed


def content_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()


def orthogonal_residual_loss(z_residual: torch.Tensor, z_clip: torch.Tensor) -> torch.Tensor:
    """Novelty 2: Orthogonal Residual Constraint (ORC).

    Penalizes collinearity between the learned forensic residual and the frozen CLIP embedding:
    L_ortho = mean(|cos(z_residual, z_clip)|)
    """
    z_res_norm = F.normalize(z_residual, dim=-1)
    z_clip_norm = F.normalize(z_clip, dim=-1)
    return torch.abs((z_res_norm * z_clip_norm).sum(dim=-1)).mean()


def augment_features(
    features: torch.Tensor,
    noise_std: float = 0.0,
    feature_dropout: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    """Feature-level data augmentation (noise + dropout) for cached embeddings."""
    if not training or (noise_std <= 0.0 and feature_dropout <= 0.0):
        return features
    out = features
    if feature_dropout > 0.0:
        out = F.dropout(out, p=feature_dropout, training=True)
    if noise_std > 0.0:
        out = out + torch.randn_like(out) * noise_std
    return F.normalize(out, dim=-1)


def train_cfd_fsa(
    model: CFD_FSA,
    train_store: FeatureStore,
    train_df: pd.DataFrame,
    cfg: dict,
    adaptation: list[str],
    k: int,
    device: torch.device,
    acs_enabled: bool = True,
    val_store: FeatureStore | None = None,
    val_df: pd.DataFrame | None = None,
    held_out: list[str] | None = None,
    train_seed: int = 42,
):
    params = [p for p in model.parameters() if p.requires_grad]

    opt = optim.AdamW(
        params,
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 0.0),
    )

    use_amp = (
        bool(cfg["training"].get("amp", False))
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    epochs = int(cfg["training"]["epochs"])
    episodes_per_epoch = int(cfg["training"]["episodes_per_epoch"])
    query_per_class = int(cfg["training"].get("query_per_class", 64))
    temperature = float(cfg["model"].get("temperature", 0.07))
    content_loss_weight = float(
        cfg["training"].get("content_loss_weight", 1.0)
    )
    ortho_loss_weight = float(
        cfg["training"].get("ortho_loss_weight", 0.0)
    )
    grad_clip = float(
        cfg["training"].get("grad_clip", 1.0)
    )
    feat_noise = float(cfg["training"].get("feature_noise_std", 0.0))
    feat_drop = float(cfg["training"].get("feature_dropout", 0.0))

    # -- Cosine LR scheduler with linear warmup -------------------------
    total_steps = epochs * episodes_per_epoch
    warmup_steps = max(1, int(total_steps * cfg["training"].get("warmup_fraction", 0.05)))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    # -------------------------------------------------------------------

    # -- GRL lambda warmup / scheduling ---------------------------------
    target_grl_lambda = float(cfg["model"].get("grl_lambda", 1.0))
    grl_schedule = cfg["training"].get("grl_schedule", "none")
    grl_warmup_fraction = float(cfg["training"].get("grl_warmup_fraction", 0.15))
    grl_warmup_steps = max(1, int(total_steps * grl_warmup_fraction))
    # -------------------------------------------------------------------

    # -- Validation-based model selection & Early Stopping --------------
    best_val_auc = -1.0
    best_state = None
    val_every = max(1, cfg["training"].get("val_every_n_epochs", 5))
    val_enabled = val_store is not None and val_df is not None
    early_stopping_patience = int(cfg["training"].get("early_stopping_patience", 0))
    patience_counter = 0
    # -------------------------------------------------------------------

    model.train()

    print(
        f"Training: epochs={epochs}, "
        f"episodes/epoch={episodes_per_epoch}, "
        f"K={k}, query/class={query_per_class}, "
        f"warmup={warmup_steps}/{total_steps} steps, "
        f"GRL schedule={grl_schedule} (target={target_grl_lambda}), "
        f"ORC weight={ortho_loss_weight}, "
        f"MSPM={'enabled' if getattr(model, 'use_mspm', False) else 'disabled'}, "
        f"val_every={val_every if val_enabled else 'disabled'}, "
        f"patience={early_stopping_patience or 'disabled'}, "
        f"AMP={use_amp}, device={device}"
    )

    for epoch in range(epochs):
        epoch_loss = 0.0

        for episode in range(episodes_per_epoch):
            seed = int(train_seed) + epoch * 10000 + episode

            support = sample_support(
                train_df,
                k,
                seed,
                fake_manipulations=adaptation,
                split="train",
            )

            query = sample_query_disjoint(
                train_df,
                query_per_class=query_per_class,
                seed=seed + 17,
                support=support,
                fake_manipulations=adaptation,
                split="train",
            )

            validate_no_source_overlap(support, query)

            s_clip, s_content, s_freq = train_store.get(support)
            q_clip, q_content, q_freq = train_store.get(query)

            s_clip = augment_features(s_clip.to(device, non_blocking=True), feat_noise, feat_drop, training=True)
            q_clip = augment_features(q_clip.to(device, non_blocking=True), feat_noise, feat_drop, training=True)

            s_content = s_content.to(device, non_blocking=True)
            q_content = q_content.to(device, non_blocking=True)

            if s_freq is not None:
                s_freq = s_freq.to(device, non_blocking=True)
                q_freq = q_freq.to(device, non_blocking=True)

            s_labels = torch.tensor(
                support.label.to_numpy(),
                dtype=torch.long,
                device=device,
            )

            q_labels = torch.tensor(
                query.label.to_numpy(),
                dtype=torch.long,
                device=device,
            )

            # Update GRL lambda according to schedule
            global_step = epoch * episodes_per_epoch + episode
            if acs_enabled and hasattr(model, "acs"):
                if grl_schedule == "linear":
                    cur_grl = target_grl_lambda * min(1.0, float(global_step) / float(grl_warmup_steps))
                    model.acs.set_lambda(cur_grl)
                elif grl_schedule == "dann":
                    p = float(global_step) / float(max(1, total_steps))
                    cur_grl = target_grl_lambda * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
                    model.acs.set_lambda(cur_grl)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                s_z = model.forensic(
                    s_clip,
                    s_freq if model.use_frequency else None,
                )

                q_z = model.forensic(
                    q_clip,
                    q_freq if model.use_frequency else None,
                )

                cur_temp = model.get_temperature() if hasattr(model, "get_temperature") else temperature
                enricher = model.enrich_prototypes if hasattr(model, "enrich_prototypes") else None

                forensic_loss, _, _ = prototypical_loss(
                    s_z,
                    s_labels,
                    q_z,
                    q_labels,
                    cur_temp,
                    prototype_enricher=enricher,
                )

                total = forensic_loss

                if acs_enabled:
                    s_pred = model.acs(s_z)
                    q_pred = model.acs(q_z)

                    content_loss = content_cosine_loss(
                        torch.cat([s_pred, q_pred], dim=0),
                        torch.cat([s_content, q_content], dim=0),
                    )

                    total = (
                        total
                        + content_loss_weight * content_loss
                    )

                if ortho_loss_weight > 0.0 and hasattr(model, "residual_projection"):
                    s_res = model.residual_projection(s_z)
                    q_res = model.residual_projection(q_z)
                    ortho_loss = orthogonal_residual_loss(
                        torch.cat([s_res, q_res], dim=0),
                        torch.cat([s_clip, q_clip], dim=0),
                    )
                    total = total + ortho_loss_weight * ortho_loss

            scaler.scale(total).backward()

            scaler.unscale_(opt)

            torch.nn.utils.clip_grad_norm_(
                params,
                grad_clip,
            )

            scaler.step(opt)
            scaler.update()
            scheduler.step()

            epoch_loss += float(total.detach().item())

        mean_loss = epoch_loss / episodes_per_epoch
        cur_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"mean loss: {mean_loss:.6f} | "
            f"lr: {cur_lr:.2e} | "
            f"AMP: {use_amp}"
        )

        # -- Validation checkpoint selection & Early Stopping -----------
        if val_enabled and (epoch + 1) % val_every == 0:
            val_metrics = evaluate_cfd(
                model, train_store, train_df,
                val_store, val_df, cfg,
                adaptation, held_out or [], k, seed=0, device=device,
                eval_split="val",
            )
            val_auc = val_metrics["auc"]
            temp_str = f" | temp: {val_metrics['temperature']:.3f}" if "temperature" in val_metrics else ""
            print(f"  -> val AUC: {val_auc:.4f} (best: {best_val_auc:.4f}){temp_str}")
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {n: p.clone() for n, p in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                    print(
                        f"Early stopping triggered at epoch {epoch + 1}: "
                        f"{patience_counter} consecutive val checks without improvement "
                        f"(patience={early_stopping_patience})."
                    )
                    break
            model.train()
        # ---------------------------------------------------------------

    # Restore best model if validation was used.
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best val checkpoint (AUC={best_val_auc:.4f}).")

    model.best_val_auc = float(best_val_auc)
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
    eval_split: str = "test",
):
    model.eval()
    support = sample_support(train_df, k, seed, fake_manipulations=adaptation, split="train")
    query = build_query_set(test_df, held_out_manipulations=held_out, split=eval_split)
    validate_no_source_overlap(support, query)

    # Use the FULL query/test set for robust metric estimation.
    # Previously capped at query_per_class (16–64), which made metrics
    # unreliable and caused EER to report 1.0 everywhere.

    s_clip, s_content, s_freq = train_store.get(support)
    q_clip, q_content, q_freq = test_store.get(query)
    s_labels = torch.tensor(support.label.to_numpy(), dtype=torch.long, device=device)

    with torch.no_grad():
        sf = s_freq.to(device) if (model.use_frequency and s_freq is not None) else None
        qf = q_freq.to(device) if (model.use_frequency and q_freq is not None) else None
        s_z = model.forensic(s_clip.to(device), sf)
        q_z = model.forensic(q_clip.to(device), qf)
        cur_temp = model.get_temperature().item() if hasattr(model, "get_temperature") else float(cfg["model"].get("temperature", 0.07))
        enricher = model.enrich_prototypes if hasattr(model, "enrich_prototypes") else None
        logits, _ = prototypical_logits(s_z, s_labels, q_z, cur_temp, prototype_enricher=enricher)
        scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        content_pred = model.acs(s_z) if hasattr(model, "acs") else None
        if content_pred is not None:
            # Diagnostics: how predictive of the content target the learned forensic space remains.
            content_cos = float(F.cosine_similarity(content_pred.cpu(), s_content, dim=-1).mean())
        else:
            content_cos = float("nan")

    metrics = binary_metrics(query.label.to_numpy(), scores)
    metrics["support_content_cosine"] = content_cos
    metrics["temperature"] = cur_temp
    return metrics


def make_model(cfg, clip_dim, content_dim, device, no_acs=False, no_bottleneck=False):
    forensic_dim = clip_dim if no_bottleneck else cfg["model"].get("forensic_dim", 128)
    use_frequency = bool(cfg["model"].get("use_frequency_branch", False))
    initial_temp = float(cfg["model"].get("temperature", 0.07))
    learnable_temp = bool(cfg["model"].get("learnable_temperature", False))
    use_mspm = bool(cfg["model"].get("use_mspm", False))
    mspm_slots = int(cfg["model"].get("mspm_slots", 16))
    if no_bottleneck:
        # A trainable identity mapping keeps the interface consistent; ACS works directly in CLIP space.
        model = CFD_FSA(
            clip_dim, clip_dim, content_dim,
            cfg["model"].get("grl_lambda", 1.0),
            cfg["model"].get("adapter_dropout", 0.1),
            use_frequency=use_frequency,
            temperature=initial_temp,
            learnable_temperature=learnable_temp,
            use_mspm=use_mspm,
            mspm_slots=mspm_slots,
        )
        model.fba = IdentityForensicAdapter()
    else:
        model = CFD_FSA(
            clip_dim, forensic_dim, content_dim,
            cfg["model"].get("grl_lambda", 1.0),
            cfg["model"].get("adapter_dropout", 0.1),
            use_frequency=use_frequency,
            temperature=initial_temp,
            learnable_temperature=learnable_temp,
            use_mspm=use_mspm,
            mspm_slots=mspm_slots,
        )
    if no_acs:
        model.acs = AdversarialContentSuppressor(forensic_dim, content_dim, grl_lambda=0.0)
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage_d.yaml")
    parser.add_argument("--grl-lambda", type=float, default=None)
    parser.add_argument("--forensic-dim", type=int, default=None)
    parser.add_argument("--ortho-weight", type=float, default=None, help="Orthogonal Residual Constraint weight (Novelty 2)")
    parser.add_argument("--use-mspm", action="store_true", help="Enable Manipulation-Selective Prototype Memory (Novelty 3)")
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        choices=["baseline", "ais_only", "orc_only", "no_mspm", "no_orc", "no_ais", "full"],
        help="Ablation variant: baseline (ProtoNet), ais_only, orc_only, no_mspm (AIS+ORC), no_orc (AIS+MSPM), no_ais (ORC+MSPM), or full (Full CFD-FSA)",
    )
    parser.add_argument("--training-seeds", type=int, nargs="+", default=None, help="One or more training seeds to train distinct models")
    parser.add_argument("--no-acs", action="store_true", help="FBA + ProtoNet without content suppression")
    parser.add_argument("--no-bottleneck", action="store_true", help="ACS applied without the compact FBA bottleneck")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.grl_lambda is not None:
        cfg["model"]["grl_lambda"] = args.grl_lambda
    if args.forensic_dim is not None:
        cfg["model"]["forensic_dim"] = args.forensic_dim
    if args.ortho_weight is not None:
        cfg["training"]["ortho_loss_weight"] = args.ortho_weight
    if args.use_mspm:
        cfg["model"]["use_mspm"] = True

    # Determine ablation variant
    if args.ablation:
        variant = args.ablation.strip().lower()
        if variant in ["baseline", "protonet"]:
            variant = "baseline"
            cfg["model"]["use_mspm"] = False
            cfg["training"]["ortho_loss_weight"] = 0.0
            args.no_acs = True
        elif variant == "ais_only":
            cfg["model"]["use_mspm"] = False
            cfg["training"]["ortho_loss_weight"] = 0.0
            args.no_acs = False
        elif variant == "orc_only":
            cfg["model"]["use_mspm"] = False
            if float(cfg["training"].get("ortho_loss_weight", 0.0)) == 0.0:
                cfg["training"]["ortho_loss_weight"] = 0.05
            args.no_acs = True
        elif variant == "no_mspm":
            cfg["model"]["use_mspm"] = False
            if float(cfg["training"].get("ortho_loss_weight", 0.0)) == 0.0:
                cfg["training"]["ortho_loss_weight"] = 0.05
            args.no_acs = False
        elif variant == "no_orc":
            cfg["model"]["use_mspm"] = True
            cfg["training"]["ortho_loss_weight"] = 0.0
            args.no_acs = False
        elif variant == "no_ais":
            cfg["model"]["use_mspm"] = True
            if float(cfg["training"].get("ortho_loss_weight", 0.0)) == 0.0:
                cfg["training"]["ortho_loss_weight"] = 0.05
            args.no_acs = True
        elif variant == "full":
            cfg["model"]["use_mspm"] = True
            if float(cfg["training"].get("ortho_loss_weight", 0.0)) == 0.0:
                cfg["training"]["ortho_loss_weight"] = 0.05
            args.no_acs = False
    else:
        if args.no_acs and float(cfg["training"].get("ortho_loss_weight", 0.0)) == 0.0 and not bool(cfg["model"].get("use_mspm", False)):
            variant = "baseline"
        elif args.no_acs and float(cfg["training"].get("ortho_loss_weight", 0.0)) == 0.0:
            variant = "no_ais"
        elif float(cfg["training"].get("ortho_loss_weight", 0.0)) == 0.0 and not bool(cfg["model"].get("use_mspm", False)):
            variant = "ais_only"
        elif args.no_acs:
            variant = "no_ais"
        elif float(cfg["training"].get("ortho_loss_weight", 0.0)) == 0.0:
            variant = "no_orc"
        elif not bool(cfg["model"].get("use_mspm", False)):
            variant = "no_mspm"
        else:
            variant = "full"

    device = get_device(cfg.get("device", "auto"))
    feature_root = Path(cfg["paths"]["feature_root"])
    train_store = FeatureStore(feature_root / "train.pt")
    test_store = FeatureStore(feature_root / "test.pt")

    # Load validation features for best-checkpoint selection.
    val_path = feature_root / "val.pt"
    val_store = FeatureStore(val_path) if val_path.exists() else None
    val_df = val_store.metadata if val_store is not None else None
    if val_store is not None:
        print(f"Loaded val set: {len(val_df)} samples -> best-checkpoint selection enabled.")
    else:
        print("WARNING: val.pt not found; skipping validation-based model selection.")

    train_df, test_df = train_store.metadata, test_store.metadata
    clip_dim = train_store.clip.shape[1]
    content_dim = train_store.content.shape[1]

    adaptation = cfg["experiment"]["adaptation_manipulations"]
    held_out = cfg["experiment"]["held_out_manipulations"]

    # Retrieve training seeds: CLI flag -> config few_shot.training_seeds -> single seed
    raw_train_seeds = (
        args.training_seeds
        or cfg.get("training_seeds")
        or cfg.get("few_shot", {}).get("training_seeds")
    )
    if not raw_train_seeds:
        raw_train_seeds = [cfg.get("seed", 42)]
    train_seeds = [int(s) for s in raw_train_seeds]

    ckpt_dir = Path(cfg["paths"].get("checkpoint_root", "checkpoints/stage_d"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results = []
    model_summaries = []

    for k in cfg["few_shot"]["k_values"]:
        best_overall_val_auc = -1.0
        best_overall_seed = None

        for t_seed in train_seeds:
            print(f"\n{'='*75}")
            print(f"Training Model: K={k} | Training Seed={t_seed}")
            print(f"{'='*75}")
            set_seed(t_seed)

            model = make_model(cfg, clip_dim, content_dim, device, args.no_acs, args.no_bottleneck)
            if args.no_acs:
                for p in model.acs.parameters():
                    p.requires_grad = False

            model = train_cfd_fsa(
                model, train_store, train_df, cfg, adaptation, k, device,
                acs_enabled=not args.no_acs,
                val_store=val_store, val_df=val_df, held_out=held_out,
                train_seed=t_seed,
            )
            val_auc = getattr(model, "best_val_auc", -1.0)

            # Evaluate this distinct model across support evaluation seeds
            support_seeds = cfg["few_shot"].get("support_seeds", [0])
            eval_metrics_list = []
            for s_seed in support_seeds:
                metrics = evaluate_cfd(model, train_store, train_df, test_store, test_df, cfg, adaptation, held_out, k, s_seed, device)
                if args.no_acs:
                    metrics["support_content_cosine"] = float("nan")
                eval_metrics_list.append(metrics)
                results.append({
                    "stage": "D" if not args.no_acs else "F",
                    "k": k,
                    "train_seed": t_seed,
                    "support_seed": s_seed,
                    "val_auc": val_auc,
                    "grl_lambda": cfg["model"].get("grl_lambda", 0.0),
                    "forensic_dim": cfg["model"].get("forensic_dim", clip_dim),
                    "trainable_params": count_parameters(model, True),
                    **metrics,
                })

            mean_test_auc = float(np.mean([m["auc"] for m in eval_metrics_list]))
            mean_test_eer = float(np.mean([m["eer"] for m in eval_metrics_list]))
            mean_test_f1 = float(np.mean([m["f1"] for m in eval_metrics_list]))

            # Save distinct model checkpoint for this training seed & variant
            seed_ckpt_name = f"k{k}_{variant}_seed{t_seed}.pt"
            seed_ckpt_path = ckpt_dir / seed_ckpt_name
            torch.save({
                "model": model.state_dict(),
                "config": cfg,
                "k": k,
                "variant": variant,
                "train_seed": t_seed,
                "val_auc": val_auc,
                "test_auc": mean_test_auc,
                "test_eer": mean_test_eer,
                "test_f1": mean_test_f1,
            }, seed_ckpt_path)

            model_summaries.append({
                "variant": variant,
                "k": k,
                "train_seed": t_seed,
                "val_auc": round(val_auc, 4),
                "test_auc": round(mean_test_auc, 4),
                "test_eer": round(mean_test_eer, 4),
                "test_f1": round(mean_test_f1, 4),
                "checkpoint": seed_ckpt_name,
            })

            print(
                f"[Model Finished] K={k} [{variant}] Seed={t_seed} -> "
                f"Val AUC: {val_auc:.4f} | Test AUC: {mean_test_auc:.4f} | "
                f"Test EER: {mean_test_eer:.4f} | Saved: {seed_ckpt_name}"
            )

            # Track and save best overall checkpoint for this K and variant
            if val_auc > best_overall_val_auc or best_overall_val_auc < 0:
                best_overall_val_auc = val_auc
                best_overall_seed = t_seed
                best_payload = {
                    "model": model.state_dict(),
                    "config": cfg,
                    "k": k,
                    "variant": variant,
                    "train_seed": t_seed,
                    "val_auc": val_auc,
                    "test_auc": mean_test_auc,
                    "test_eer": mean_test_eer,
                    "test_f1": mean_test_f1,
                }
                # Variant-specific primary checkpoint
                torch.save(best_payload, ckpt_dir / f"k{k}_{variant}.pt")
                if variant == "full":
                    torch.save(best_payload, ckpt_dir / f"k{k}.pt")
                    torch.save(best_payload, ckpt_dir / f"k{k}_best.pt")

    out_dir = Path(cfg["paths"]["results_root"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save per-variant results
    pd.DataFrame(results).to_csv(
        out_dir / f"stage_d_results_{variant}.csv",
        index=False,
    )
    summary_df = pd.DataFrame(model_summaries)
    summary_df.to_csv(out_dir / f"model_summary_{variant}.csv", index=False)

    if variant == "full":
        pd.DataFrame(results).to_csv(out_dir / "stage_d_results.csv", index=False)
        summary_df.to_csv(out_dir / "model_training_summary.csv", index=False)

    # Maintain cumulative ablation_summary.csv across all runs
    cum_path = out_dir / "ablation_summary.csv"
    if cum_path.exists():
        try:
            old_df = pd.read_csv(cum_path)
            old_df = old_df[old_df["variant"] != variant]
            combined_summary = pd.concat([old_df, summary_df], ignore_index=True)
        except Exception:
            combined_summary = summary_df
    else:
        combined_summary = summary_df
    combined_summary.to_csv(cum_path, index=False)

    print("\n" + "=" * 80)
    print(f"CFD-FSA EVALUATION SUMMARY [{variant.upper()}]")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print(f"Results and checkpoints saved to: {out_dir}\n")


if __name__ == "__main__":
    main()

