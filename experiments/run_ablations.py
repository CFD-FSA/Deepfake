from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from experiments.stage_d_cfd_fsa import main as train_eval_main


def main():
    parser = argparse.ArgumentParser(description="Automated Ablation Runner for CFD-FSA")
    ALL_CHOICES = ["baseline", "ais_only", "orc_only", "no_mspm", "no_orc", "no_ais", "full"]
    parser.add_argument("--config", default="configs/stage_d_v2.yaml", help="Path to config file")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "no_mspm", "no_orc", "no_ais", "full"],
        choices=ALL_CHOICES,
        help="Ablation variants to train and evaluate",
    )
    parser.add_argument(
        "--training-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional training seeds override",
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("STARTING CFD-FSA AUTOMATED ABLATION SUITE")
    print(f"Variants to evaluate: {args.variants}")
    print("=" * 80 + "\n")

    for variant in args.variants:
        print(f"\n{'#'*80}")
        print(f"RUNNING ABLATION VARIANT: {variant.upper()}")
        print(f"{'#'*80}\n")

        # Build sys.argv for stage_d_cfd_fsa.py
        cmd_args = [
            "experiments/stage_d_cfd_fsa.py",
            "--config", args.config,
            "--ablation", variant,
        ]
        if args.training_seeds:
            cmd_args.extend(["--training-seeds"] + [str(s) for s in args.training_seeds])

        sys.argv = cmd_args
        train_eval_main()

    # Once all requested variants complete, compile master ablation table
    from src.utils import load_yaml
    cfg = load_yaml(args.config)
    results_dir = Path(cfg["paths"]["results_root"])
    ablation_summary_file = results_dir / "ablation_summary.csv"

    if ablation_summary_file.exists():
        df = pd.read_csv(ablation_summary_file)
        
        # Order variants in standard paper presentation order
        preferred_order = ["baseline", "ais_only", "orc_only", "no_mspm", "no_orc", "no_ais", "full"]
        variants_in_df = [v for v in preferred_order if v in df["variant"].values]
        # Append any other variants not in preferred_order
        for v in df["variant"].unique():
            if v not in variants_in_df:
                variants_in_df.append(v)

        table_rows = []
        for v in variants_in_df:
            sub = df[df["variant"] == v]
            if len(sub) == 0:
                continue
            auc_mean, auc_std = sub["test_auc"].mean(), sub["test_auc"].std()
            eer_mean, eer_std = sub["test_eer"].mean(), sub["test_eer"].std()
            f1_mean, f1_std = sub["test_f1"].mean(), sub["test_f1"].std()
            val_mean = sub["val_auc"].mean()
            
            ais_flag = "No" if v in ["no_ais", "baseline", "orc_only"] else "Yes"
            orc_flag = "No" if v in ["no_orc", "baseline", "ais_only"] else "Yes"
            mspm_flag = "No" if v in ["no_mspm", "baseline", "ais_only", "orc_only"] else "Yes"

            table_rows.append({
                "Variant": v,
                "AIS (Adversarial)": ais_flag,
                "ORC (Orthogonal)": orc_flag,
                "MSPM (Memory)": mspm_flag,
                "Val AUC": f"{val_mean:.4f}",
                "Test AUC": f"{auc_mean:.4f} +/- {auc_std:.4f}" if not np.isnan(auc_std) else f"{auc_mean:.4f}",
                "Test EER": f"{eer_mean:.4f} +/- {eer_std:.4f}" if not np.isnan(eer_std) else f"{eer_mean:.4f}",
                "Test F1": f"{f1_mean:.4f} +/- {f1_std:.4f}" if not np.isnan(f1_std) else f"{f1_mean:.4f}",
                "Models Count": len(sub),
            })

        table_df = pd.DataFrame(table_rows)
        table_csv_path = results_dir / "ablation_table.csv"
        table_df.to_csv(table_csv_path, index=False)

        table_md_path = results_dir / "ablation_table.md"
        with open(table_md_path, "w", encoding="utf-8") as f:
            f.write("# CFD-FSA: Table 3 Ablation Study Results\n\n")
            f.write(table_df.to_markdown(index=False))
            f.write("\n")

        print("\n" + "=" * 80)
        print("CFD-FSA FINAL COMPILATION - TABLE 3 ABLATION RESULTS")
        print("=" * 80)
        print(table_df.to_string(index=False))
        print("=" * 80)
        print(f"Ablation table written to:\n  {table_csv_path}\n  {table_md_path}\n")


if __name__ == "__main__":
    main()
