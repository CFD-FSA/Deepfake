from __future__ import annotations

import json
import warnings
import numpy as np
from pathlib import Path
from typing import Iterable

import cv2
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# These are the five categories required for the initial CFD-FSA study.
# FaceShifter and DeepFakeDetection are intentionally excluded from the
# initial experiments.
CSV_SOURCES = {
    "REAL": "original.csv",
    "DF": "Deepfakes.csv",
    "F2F": "Face2Face.csv",
    "FS": "FaceSwap.csv",
    "NT": "NeuralTextures.csv",
}

MANIPULATION_ALIASES = {
    "deepfakes": "DF",
    "df": "DF",
    "face2face": "F2F",
    "f2f": "F2F",
    "faceswap": "FS",
    "fs": "FS",
    "neuraltextures": "NT",
    "nt": "NT",
    "original": "REAL",
    "original_sequences": "REAL",
}


def normalize_manipulation(value: str) -> str:
    key = str(value).strip().replace(" ", "").replace("-", "").lower()
    return MANIPULATION_ALIASES.get(key, str(value).strip())


def _find_path_column(df: pd.DataFrame) -> str:
    candidates = [
        "File Path",
        "file_path",
        "FilePath",
        "path",
        "Path",
        "video_path",
        "Video Path",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not find a video path column. Expected one of: "
        + ", ".join(candidates)
    )


def _find_optional_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def resolve_dataset_path(raw_root: Path, relative_path: str) -> Path:
    """Resolve a CSV path relative to raw_root.

    The supplied FF++ CSVs contain entries such as
    `Deepfakes/000_003.mp4` and `original/000.mp4`.
    """
    rel = str(relative_path).strip().replace("\\", "/")
    rel = rel.lstrip("./")

    candidate = (raw_root / rel).resolve()
    if candidate.exists():
        return candidate

    # Handle a CSV accidentally containing an extra dataset-root prefix.
    parts = Path(rel).parts
    for start in range(1, len(parts)):
        candidate = (raw_root / Path(*parts[start:])).resolve()
        if candidate.exists():
            return candidate

    # Last-resort exact-name search. This is only used for malformed CSV paths.
    matches = list(raw_root.rglob(Path(rel).name))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise FileNotFoundError(
            f"CSV path is ambiguous; multiple files match {relative_path!r}: {matches[:5]}"
        )

    return (raw_root / rel).resolve()


def _infer_source_ids(video_path: Path) -> set[str]:
    """Return FF++ source IDs used for leakage grouping.

    Example:
      000.mp4      -> {000}
      000_003.mp4  -> {000, 003}

    These are used only as source-group identifiers. They are NOT treated as
    identity labels.
    """
    stem = video_path.stem
    parts = [p for p in stem.split("_") if p]
    if len(parts) >= 2 and all(p.isdigit() for p in parts[:2]):
        return {parts[0], parts[1]}
    if parts:
        return {parts[0]}
    return {stem}


def _source_video_key(relative_path: str) -> str:
    return str(relative_path).replace("\\", "/")


def load_ffpp_csv_manifest(
    raw_root: str | Path,
    csv_root: str | Path | None = None,
    compression: str = "C23",
    csv_sources: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build the video manifest from the FF++ CSV metadata.

    This function is tailored to the user's downloaded C23 structure:

        FaceForensics++_C23/
          csv/original.csv
          csv/Deepfakes.csv
          csv/Face2Face.csv
          csv/FaceSwap.csv
          csv/NeuralTextures.csv
          original/
          Deepfakes/
          Face2Face/
          FaceSwap/
          NeuralTextures/
          FaceShifter/        <- ignored initially
          DeepFakeDetection/  <- ignored initially

    The CSVs are the source of truth for labels and file paths. We do not use
    FaceShifter or DeepFakeDetection in the initial study.
    """
    raw_root = Path(raw_root).expanduser().resolve()
    if not raw_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {raw_root}")

    if csv_root is None:
        csv_root = raw_root / "csv"
    else:
        csv_root = Path(csv_root).expanduser().resolve()

    sources = csv_sources or CSV_SOURCES
    missing_csvs = [name for name in sources.values() if not (csv_root / name).exists()]
    if missing_csvs:
        raise FileNotFoundError(
            "Missing required FF++ CSV files: " + ", ".join(missing_csvs)
        )

    rows: list[dict] = []
    missing_videos: list[str] = []
    c23 = str(compression).upper()

    for manipulation, filename in sources.items():
        csv_path = csv_root / filename
        table = pd.read_csv(csv_path)
        path_col = _find_path_column(table)
        label_col = _find_optional_column(table, ["Label", "label"])
        frame_count_col = _find_optional_column(table, ["Frame Count", "frame_count"])
        width_col = _find_optional_column(table, ["Width", "width"])
        height_col = _find_optional_column(table, ["Height", "height"])
        codec_col = _find_optional_column(table, ["Codec", "codec"])
        size_col = _find_optional_column(table, ["File Size(MB)", "file_size_mb"])

        expected_label = "REAL" if manipulation == "REAL" else "FAKE"
        for _, record in table.iterrows():
            rel = str(record[path_col]).strip()
            if not rel or rel.lower() == "nan":
                continue

            video_path = resolve_dataset_path(raw_root, rel)
            if not video_path.exists():
                missing_videos.append(str(video_path))
                continue

            csv_label = str(record[label_col]).strip().upper() if label_col else expected_label
            if csv_label and csv_label != expected_label:
                warnings.warn(
                    f"Unexpected label {csv_label!r} in {csv_path.name} for {rel!r}; "
                    f"using category-derived label {expected_label!r}."
                )

            source_key = _source_video_key(rel)
            source_ids = _infer_source_ids(video_path)
            rows.append(
                {
                    "path": str(video_path),
                    "relative_path": source_key,
                    "source_video_key": source_key,
                    "video_id": "",  # Do not invent a canonical video ID.
                    "source_ids": "+".join(sorted(source_ids)),
                    "label": 0 if manipulation == "REAL" else 1,
                    "manipulation": manipulation,
                    "compression": c23,
                    "identity": "",  # Only populated if trusted metadata provides it later.
                    "csv_label": csv_label,
                    "frame_count": record[frame_count_col] if frame_count_col else None,
                    "width": record[width_col] if width_col else None,
                    "height": record[height_col] if height_col else None,
                    "codec": record[codec_col] if codec_col else None,
                    "file_size_mb": record[size_col] if size_col else None,
                    "metadata_source": csv_path.name,
                }
            )

    if not rows:
        raise RuntimeError(
            f"No videos were loaded from the required FF++ CSVs under {csv_root}"
        )

    if missing_videos:
        preview = "\n".join(missing_videos[:10])
        raise FileNotFoundError(
            f"{len(missing_videos)} CSV-listed videos could not be found.\n"
            f"First missing paths:\n{preview}"
        )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["relative_path"]).reset_index(drop=True)
    return df


def discover_videos(raw_root: str | Path, extensions: Iterable[str] = (".mp4", ".avi", ".mov", ".mkv")) -> pd.DataFrame:
    """Fallback recursive discovery for non-CSV datasets.

    The CFD-FSA FF++ pipeline should use load_ffpp_csv_manifest() instead.
    """
    raw_root = Path(raw_root).expanduser().resolve()
    if not raw_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {raw_root}")

    ext_set = {e.lower() for e in extensions}
    rows = []
    for path in raw_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in ext_set:
            continue
        rel_parts = [p.lower() for p in path.relative_to(raw_root).parts]
        manipulation = "UNKNOWN"
        for token, value in MANIPULATION_ALIASES.items():
            if token in rel_parts:
                manipulation = value
                break
        if manipulation not in {"REAL", "DF", "F2F", "FS", "NT"}:
            continue
        rel = path.relative_to(raw_root).as_posix()
        rows.append(
            {
                "path": str(path.resolve()),
                "relative_path": rel,
                "source_video_key": rel,
                "video_id": "",
                "source_ids": "+".join(sorted(_infer_source_ids(path))),
                "label": 0 if manipulation == "REAL" else 1,
                "manipulation": manipulation,
                "compression": None,
                "identity": "",
            }
        )
    if not rows:
        raise RuntimeError(f"No recognized videos found under {raw_root}.")
    return pd.DataFrame(rows).drop_duplicates(subset=["relative_path"]).reset_index(drop=True)


def load_optional_metadata(metadata_path: str | Path | None) -> pd.DataFrame | None:
    if not metadata_path:
        return None
    path = Path(metadata_path)
    if not path.exists():
        warnings.warn(f"Metadata file not found: {path}; continuing without it.")
        return None
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    return pd.read_csv(path)


def merge_metadata(discovered: pd.DataFrame, metadata: pd.DataFrame | None) -> pd.DataFrame:
    if metadata is None or metadata.empty:
        return discovered

    out = discovered.copy()
    if "relative_path" in metadata.columns:
        merged = out.merge(metadata, on="relative_path", how="left", suffixes=("", "_meta"))
    elif "path" in metadata.columns:
        merged = out.merge(metadata, on="path", how="left", suffixes=("", "_meta"))
    else:
        warnings.warn("Metadata must contain relative_path or path; no merge performed.")
        return out

    for col in ["video_id", "source_video_key", "manipulation", "compression", "identity", "label"]:
        meta_col = f"{col}_meta"
        if meta_col in merged.columns:
            mask = merged[meta_col].notna() & (merged[meta_col].astype(str) != "")
            merged.loc[mask, col] = merged.loc[mask, meta_col]
            merged.drop(columns=[meta_col], inplace=True)
    return merged


def _official_split_structure(split_file: Path):
    with open(split_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = set()
    pairs = set()
    items = data if isinstance(data, list) else [data]
    for item in items:
        if isinstance(item, str):
            ids.add(Path(item).stem)
        elif isinstance(item, (list, tuple)):
            vals = [Path(str(x)).stem for x in item]
            ids.update(vals)
            if len(vals) >= 2:
                pairs.add(tuple(sorted(vals[:2])))
        elif isinstance(item, dict):
            vals = []
            for key in ("video_id", "id", "source", "target"):
                if key in item:
                    vals.append(Path(str(item[key])).stem)
            ids.update(vals)
            if len(vals) >= 2:
                pairs.add(tuple(sorted(vals[:2])))
    return ids, pairs


def apply_official_splits(df: pd.DataFrame, split_dir: str | Path) -> pd.DataFrame:
    split_dir = Path(split_dir)
    files = {name: split_dir / f"{name}.json" for name in ("train", "val", "test")}
    if not all(p.exists() for p in files.values()):
        raise FileNotFoundError("Not all official split JSON files exist.")

    split_data = {split: _official_split_structure(path) for split, path in files.items()}
    assignments = {}
    unresolved = []

    for idx, row in df.iterrows():
        stem = Path(str(row["path"])).stem
        matched = []
        for split, (ids, pairs) in split_data.items():
            if stem in ids:
                matched.append(split)

        parts = stem.split("_")
        if len(parts) >= 2 and all(x.isdigit() for x in parts[:2]):
            pair = tuple(sorted(parts[:2]))
            for split, (ids, pairs) in split_data.items():
                if pair in pairs or all(x in ids for x in pair):
                    matched.append(split)

        matched = sorted(set(matched))
        if len(matched) == 1:
            assignments[idx] = matched[0]
        elif len(matched) > 1:
            raise ValueError(f"Video {stem!r} maps to multiple official splits: {matched}")
        else:
            unresolved.append(stem)

    out = df.copy()
    out["split"] = "UNASSIGNED"
    for idx, split in assignments.items():
        out.loc[idx, "split"] = split
    if unresolved:
        print(f"WARNING: {len(unresolved)} videos were not matched to official split JSON files.")
        print("First unmatched examples:", unresolved[:10])
    return out


def _stem_source_ids(stem: str) -> set[str]:
    parts = [p for p in stem.split("_") if p]
    if len(parts) >= 2 and all(p.isdigit() for p in parts[:2]):
        return {parts[0], parts[1]}
    return {parts[0]} if parts else {stem}


def add_source_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Create connected source-video groups for leakage-safe splitting.

    A manipulated FF++ video `000_003.mp4` connects source IDs 000 and 003.
    Connected components make sure those sources stay in the same split.
    This is a leakage-control key, not an identity label.
    """
    out = df.copy()
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    parsed: list[set[str]] = []
    for _, row in out.iterrows():
        ids = _stem_source_ids(Path(str(row["path"])).stem)
        parsed.append(ids)
        ids_list = sorted(ids)
        for x in ids_list:
            find(x)
        for x in ids_list[1:]:
            union(ids_list[0], x)

    for ids in parsed:
        for x in ids:
            find(x)

    out["source_group_key"] = [
        "source:" + "+".join(sorted({find(x) for x in ids})) for ids in parsed
    ]
    return out


def group_split(df: pd.DataFrame, train_fraction: float, val_fraction: float, seed: int) -> pd.DataFrame:
    """Deterministic source-group split with no frame/video leakage."""
    if not (0 < train_fraction < 1) or not (0 <= val_fraction < 1):
        raise ValueError("train_fraction and val_fraction must be valid fractions.")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be < 1.")

    work = add_source_groups(df)
    groups = (
        work[["source_group_key"]]
        .drop_duplicates()
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    n = len(groups)
    n_train = max(1, int(n * train_fraction))
    n_val = max(1, int(n * val_fraction))
    if n_train + n_val >= n and n >= 3:
        n_val = max(1, n - n_train - 1)

    train_keys = set(groups.iloc[:n_train]["source_group_key"])
    val_keys = set(groups.iloc[n_train:n_train + n_val]["source_group_key"])
    work["split"] = work["source_group_key"].map(
        lambda x: "train" if x in train_keys else "val" if x in val_keys else "test"
    )
    return work


def save_manifest(df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


class FaceManifestDataset(Dataset):
    """Image dataset backed by a face/frame manifest."""

    def __init__(self, manifest: pd.DataFrame | str | Path, split: str | None = None, transform=None):
        if isinstance(manifest, (str, Path)):
            df = pd.read_csv(manifest)
        else:
            df = manifest.copy()
        if split is not None:
            df = df[df["split"].eq(split)].copy()
        required = {"face_path", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["face_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "label": int(row["label"]),
            "manipulation": row.get("manipulation", ""),
            "source_video_key": row.get("source_video_key", ""),
            "source_group_key": row.get("source_group_key", ""),
            "identity": row.get("identity", ""),
            "path": row["face_path"],
        }


def extract_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    every_n_frames: int = 10,
    max_frames: int | None = None,
) -> list[Path]:
    """
    Extract frames from a video.

    If max_frames is specified, frames are sampled uniformly across the
    entire video. For example, max_frames=60 gives up to 60 frames spread
    from the beginning to the end of the video.

    If max_frames is None, the function falls back to every_n_frames.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    saved = []

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            raise RuntimeError(
                f"Could not determine frame count for: {video_path}"
            )

        # ---------------------------------------------------------
        # Uniform sampling across the entire video
        # ---------------------------------------------------------
        if max_frames is not None:
            target_count = min(int(max_frames), total_frames)

            # Exactly target_count unique frame indices distributed
            # from the first frame to the last frame.
            target_indices = np.linspace(
                0,
                total_frames - 1,
                target_count,
                dtype=int,
            )

            target_indices = np.unique(target_indices)
            target_set = set(target_indices.tolist())

            frame_idx = 0

            while True:
                ok, frame = cap.read()

                if not ok:
                    break

                if frame_idx in target_set:
                    out = output_dir / f"frame_{frame_idx:06d}.jpg"

                    success = cv2.imwrite(
                        str(out),
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                    )

                    if success:
                        saved.append(out)

                frame_idx += 1

            # Keep chronological order.
            saved.sort()

        # ---------------------------------------------------------
        # Fallback: every Nth frame
        # ---------------------------------------------------------
        else:
            frame_idx = 0
            step = max(1, int(every_n_frames))

            while True:
                ok, frame = cap.read()

                if not ok:
                    break

                if frame_idx % step == 0:
                    out = output_dir / f"frame_{frame_idx:06d}.jpg"

                    success = cv2.imwrite(
                        str(out),
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                    )

                    if success:
                        saved.append(out)

                frame_idx += 1

    finally:
        cap.release()

    return saved