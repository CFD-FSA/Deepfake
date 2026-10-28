from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import pandas as pd

from src.data import apply_official_splits, group_split, save_manifest, add_source_groups
from src.utils import load_yaml


REQUIRED_MANIPULATIONS = {"REAL", "DF", "F2F", "FS", "NT"}


def audit(df: pd.DataFrame) -> None:
    required = ["path", "source_video_key", "label", "manipulation", "compression"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing manifest columns: {missing}")

    missing_files = [p for p in df["path"].astype(str) if not Path(p).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} manifest video paths do not exist. First examples: {missing_files[:5]}"
        )

    manipulations = set(df["manipulation"].astype(str))
    missing_categories = REQUIRED_MANIPULATIONS - manipulations
    if missing_categories:
        raise ValueError(f"Missing required manipulation categories: {sorted(missing_categories)}")

    if set(df["compression"].astype(str).str.upper()) != {"C23"}:
        raise ValueError("Initial CFD-FSA study must use C23-only videos in this manifest.")

    if not df["relative_path"].is_unique:
        duplicates = df[df["relative_path"].duplicated(keep=False)]["relative_path"].tolist()[:10]
        raise ValueError(f"Duplicate videos in manifest: {duplicates}")

    print("=== Manifest audit ===")
    print(f"Videos: {len(df)}")
    print("\nManipulations:")
    print(df["manipulation"].value_counts().sort_index().to_string())
    print("\nLabels:")
    print(df["label"].value_counts().sort_index().to_string())
    print("\nCompression:")
    print(df["compression"].value_counts().sort_index().to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--make-splits", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    manifest_path = Path(cfg["paths"]["manifest"])
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Run make_manifest_template.py first."
        )

    df = pd.read_csv(manifest_path)
    audit(df)

    if not args.make_splits:
        return

    data_cfg = cfg["data"]
    split_dir = Path(data_cfg.get("official_split_dir", "data/splits"))
    if not split_dir.is_absolute():
        split_dir = Path.cwd() / split_dir

    use_official = bool(data_cfg.get("use_official_splits", True))
    official_available = use_official and all(
        (split_dir / f"{split}.json").exists() for split in ("train", "val", "test")
    )

    if official_available:
        print(f"Using official FF++ split files from: {split_dir}")
        df = apply_official_splits(df, split_dir)
        if (df["split"] == "UNASSIGNED").any():
            unresolved = df.loc[df["split"].eq("UNASSIGNED"), "relative_path"].head(10).tolist()
            raise ValueError(
                "Some videos were not covered by official FF++ splits. "
                f"Resolve these before training. Examples: {unresolved}"
            )
    else:
        print("Official split JSONs not found; using deterministic source-video-group split.")
        df = group_split(
            df,
            float(data_cfg.get("train_fraction", 0.70)),
            float(data_cfg.get("val_fraction", 0.15)),
            int(cfg.get("seed", 42)),
        )

    # Always build/refresh source groups before the final leakage audit.
    df = add_source_groups(df)

    source_split_counts = df.groupby("source_group_key")["split"].nunique()
    leaking_groups = source_split_counts[source_split_counts > 1]
    if not leaking_groups.empty:
        examples = leaking_groups.index.tolist()[:10]
        raise AssertionError(
            f"LEAKAGE AUDIT FAILED: {len(leaking_groups)} source groups span multiple splits. "
            f"Examples: {examples}"
        )

    out_dir = manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        split_df = df[df["split"].eq(split)].copy()
        if split_df.empty:
            raise ValueError(f"Split {split!r} is empty.")
        split_df.to_csv(out_dir / f"{split}.csv", index=False)
        print(f"{split}: {len(split_df)} videos")

    # Identity overlap is reported only when a trusted identity column is actually populated.
    identity_available = (
        "identity" in df.columns
        and df["identity"].astype(str).str.strip().replace({"nan": "", "None": ""}).ne("").any()
    )
    identity_overlap = {}
    if identity_available:
        ids_by_split = {
            split: set(
                df.loc[df["split"].eq(split), "identity"]
                .astype(str)
                .str.strip()
            ) - {"", "nan", "None"}
            for split in ("train", "val", "test")
        }
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            identity_overlap[f"{a}_vs_{b}"] = len(ids_by_split[a] & ids_by_split[b])

    leakage_report = {
        "num_videos": int(len(df)),
        "split_counts": {k: int(v) for k, v in df["split"].value_counts().to_dict().items()},
        "manipulation_counts": {k: int(v) for k, v in df["manipulation"].value_counts().to_dict().items()},
        "compression_counts": {k: int(v) for k, v in df["compression"].value_counts().to_dict().items()},
        "source_groups": int(df["source_group_key"].nunique()),
        "source_group_leakage_groups": int(len(leaking_groups)),
        "identity_metadata_available": bool(identity_available),
        "identity_overlap_counts": identity_overlap,
        "split_method": "official_ffpp" if official_available else "source_group_random",
    }

    with open(out_dir / "leakage_report.json", "w", encoding="utf-8") as f:
        json.dump(leakage_report, f, indent=2)

    df.to_csv(manifest_path, index=False)
    print(f"\nUpdated manifest: {manifest_path.resolve()}")
    print(f"Leakage report: {(out_dir / 'leakage_report.json').resolve()}")
    print("Leakage audit: PASSED")


if __name__ == "__main__":
    main()
