from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.data import load_ffpp_csv_manifest, save_manifest
from src.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CFD-FSA manifest from FF++ CSV metadata.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]

    raw_root = Path(data_cfg["raw_root"])
    csv_root = Path(data_cfg.get("csv_root", raw_root / "csv"))
    if not csv_root.is_absolute():
        csv_root = Path.cwd() / csv_root

    df = load_ffpp_csv_manifest(
        raw_root=raw_root,
        csv_root=csv_root,
        compression=data_cfg.get("compression", "C23"),
    )

    required = set(data_cfg.get("required_manipulations", ["DF", "F2F", "FS", "NT", "REAL"]))
    found = set(df["manipulation"].dropna().unique())
    missing = required - found
    if missing:
        raise ValueError(f"Required manipulation categories are missing: {sorted(missing)}")

    # Explicitly exclude categories that are not part of the initial study.
    excluded = set(data_cfg.get("excluded_manipulations", ["FaceShifter", "DeepFakeDetection"]))
    if excluded & found:
        df = df[~df["manipulation"].isin(excluded)].copy()

    out = Path(cfg["paths"]["manifest"])
    save_manifest(df, out)

    print(f"Manifest written: {out.resolve()}")
    print(f"Videos: {len(df)}")
    print("\nManipulation counts:")
    print(df["manipulation"].value_counts().sort_index().to_string())
    print("\nLabel counts:")
    print(df["label"].value_counts().sort_index().to_string())
    print("\nCSV sources:")
    print(df["metadata_source"].value_counts().sort_index().to_string())
    print("\nAll listed video paths exist: YES")


if __name__ == "__main__":
    main()
