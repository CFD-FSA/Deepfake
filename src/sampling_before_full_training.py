from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def sample_support(
    df: pd.DataFrame,
    k: int,
    seed: int,
    fake_manipulations: list[str] | None = None,
    split: str = "train",
) -> pd.DataFrame:
    """K real + K fake support examples, sampled by source video."""
    subset = df[df["split"].eq(split)].copy()
    if fake_manipulations:
        fake = subset[subset["manipulation"].isin(fake_manipulations)]
    else:
        fake = subset[subset["label"].eq(1)]
    real = subset[subset["label"].eq(0)]

    rng = np.random.default_rng(seed)

    def sample_class(frame: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(frame) < n:
            raise ValueError(f"Need {n} support examples but only {len(frame)} are available.")
        # Prefer one frame per source video when possible.
        groups = frame.groupby("source_video_key", group_keys=False)
        keys = list(groups.groups.keys())
        rng.shuffle(keys)
        rows = []
        for key in keys:
            part = frame[frame["source_video_key"].eq(key)]
            rows.append(part.iloc[int(rng.integers(0, len(part)))])
            if len(rows) >= n:
                break
        if len(rows) < n:
            used = {r["path"] for r in rows}
            remaining = frame[~frame["path"].isin(used)]
            extra_idx = rng.choice(len(remaining), size=n-len(rows), replace=False)
            rows.extend([remaining.iloc[int(i)] for i in extra_idx])
        return pd.DataFrame(rows)

    return pd.concat([sample_class(real, k), sample_class(fake, k)], ignore_index=True)


def build_query_set(
    df: pd.DataFrame,
    adaptation_manipulations: list[str] | None = None,
    held_out_manipulations: list[str] | None = None,
    split: str = "test",
) -> pd.DataFrame:
    subset = df[df["split"].eq(split)].copy()
    if held_out_manipulations:
        fake = subset[subset["manipulation"].isin(held_out_manipulations)]
        real = subset[subset["label"].eq(0)]
        return pd.concat([real, fake], ignore_index=True)
    if adaptation_manipulations:
        fake = subset[subset["manipulation"].isin(adaptation_manipulations)]
        real = subset[subset["label"].eq(0)]
        return pd.concat([real, fake], ignore_index=True)
    return subset


def validate_no_source_overlap(support: pd.DataFrame, query: pd.DataFrame) -> None:
    key = "source_group_key" if "source_group_key" in support.columns and "source_group_key" in query.columns else "source_video_key"
    overlap = set(support[key]) & set(query[key])
    if overlap:
        raise AssertionError(f"Source-video leakage detected for {len(overlap)} keys.")
