from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import importlib

from src.utils import load_yaml, get_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    packages = [
        "torch", "torchvision", "transformers", "cv2", "pandas",
        "sklearn", "scipy", "facenet_pytorch", "yaml", "PIL"
    ]
    print("=== CFD-FSA Environment Check ===")
    for name in packages:
        try:
            mod = importlib.import_module(name)
            print(f"[OK] {name}: {getattr(mod, '__version__', 'installed')}")
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")

    device = get_device(cfg.get("device", "auto"))
    print(f"Device: {device}")
    if device.type == "cuda":
        import torch
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    raw_root = Path(cfg["data"]["raw_root"]).expanduser()
    if not raw_root.is_absolute():
        raw_root = Path.cwd() / raw_root
    raw_root = raw_root.resolve()

    print(f"Dataset root: {raw_root}")
    print(f"Dataset exists: {raw_root.exists()}")

    expected_dirs = [
        "original",
        "Deepfakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
        "csv",
    ]
    for name in expected_dirs:
        p = raw_root / name
        print(f"[{ 'OK' if p.is_dir() else 'FAIL' }] {name}: {p}")

    expected_csvs = [
        "original.csv",
        "Deepfakes.csv",
        "Face2Face.csv",
        "FaceSwap.csv",
        "NeuralTextures.csv",
    ]
    csv_root = raw_root / "csv"
    for name in expected_csvs:
        p = csv_root / name
        print(f"[{ 'OK' if p.is_file() else 'FAIL' }] CSV {name}")

    excluded = [
        raw_root / "FaceShifter",
        raw_root / "DeepFakeDetection",
    ]
    print("\nExcluded from initial CFD-FSA study:")
    for p in excluded:
        print(f"  {p.name}: {'present, will be ignored' if p.exists() else 'not present'}")


if __name__ == "__main__":
    main()
