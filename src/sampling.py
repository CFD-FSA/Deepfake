import numpy as np
import pandas as pd
import torch


def _sample_one_per_source_group(
    frame: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    excluded_groups: set[str] | None = None,
) -> pd.DataFrame:
    """Sample one frame from each selected source group."""
    excluded_groups = excluded_groups or set()

    available = frame[
        ~frame["source_group_key"].isin(excluded_groups)
    ].copy()

    groups = available["source_group_key"].drop_duplicates().to_numpy()

    if len(groups) < n:
        raise ValueError(
            f"Need {n} unique source groups but only "
            f"{len(groups)} are available."
        )

    rng.shuffle(groups)
    selected_groups = groups[:n]

    rows = []

    for group in selected_groups:
        part = available[
            available["source_group_key"].eq(group)
        ]

        idx = int(rng.integers(0, len(part)))
        rows.append(part.iloc[idx])

    return pd.DataFrame(rows).reset_index(drop=True)


def sample_support(
    df: pd.DataFrame,
    k: int,
    seed: int,
    fake_manipulations: list[str] | None = None,
    split: str = "train",
) -> pd.DataFrame:
    """
    Sample K REAL + K FAKE support examples.

    Every selected example comes from a unique source_group_key,
    and REAL and FAKE support groups are mutually disjoint.
    """
    subset = df[df["split"].eq(split)].copy()

    if fake_manipulations:
        fake = subset[
            subset["manipulation"].isin(fake_manipulations)
            & subset["label"].eq(1)
        ]
    else:
        fake = subset[subset["label"].eq(1)]

    real = subset[subset["label"].eq(0)]

    rng = np.random.default_rng(seed)

    # Sample REAL groups first.
    real_support = _sample_one_per_source_group(
        real,
        k,
        rng,
    )

    used_groups = set(
        real_support["source_group_key"].tolist()
    )

    # Sample FAKE groups excluding all REAL support groups.
    fake_support = _sample_one_per_source_group(
        fake,
        k,
        rng,
        excluded_groups=used_groups,
    )

    support = pd.concat(
        [real_support, fake_support],
        ignore_index=True,
    )

    return support


def sample_query_disjoint(
    df: pd.DataFrame,
    query_per_class: int,
    seed: int,
    support: pd.DataFrame,
    fake_manipulations: list[str] | None = None,
    split: str = "train",
) -> pd.DataFrame:
    """
    Sample REAL + FAKE query examples.

    Query source groups are disjoint from support source groups,
    and REAL/FAKE query groups are also mutually disjoint.
    """
    subset = df[df["split"].eq(split)].copy()

    excluded_groups = set(
        support["source_group_key"].tolist()
    )

    available = subset[
        ~subset["source_group_key"].isin(excluded_groups)
    ].copy()

    if fake_manipulations:
        fake = available[
            available["manipulation"].isin(fake_manipulations)
            & available["label"].eq(1)
        ]
    else:
        fake = available[available["label"].eq(1)]

    real = available[available["label"].eq(0)]

    rng = np.random.default_rng(seed)

    # Sample FAKE groups first.
    fake_query = _sample_one_per_source_group(
        fake,
        query_per_class,
        rng,
    )

    used_groups = excluded_groups | set(
        fake_query["source_group_key"].tolist()
    )

    # Sample REAL groups excluding support and fake-query groups.
    real_query = _sample_one_per_source_group(
        real,
        query_per_class,
        rng,
        excluded_groups=used_groups,
    )

    query = pd.concat(
        [real_query, fake_query],
        ignore_index=True,
    )

    return query


def build_query_set(
    df: pd.DataFrame,
    adaptation_manipulations: list[str] | None = None,
    held_out_manipulations: list[str] | None = None,
    split: str = "test",
) -> pd.DataFrame:
    subset = df[df["split"].eq(split)].copy()

    if held_out_manipulations:
        fake = subset[
            subset["manipulation"].isin(held_out_manipulations)
            & subset["label"].eq(1)
        ]
        real = subset[subset["label"].eq(0)]
        return pd.concat([real, fake], ignore_index=True)

    if adaptation_manipulations:
        fake = subset[
            subset["manipulation"].isin(adaptation_manipulations)
            & subset["label"].eq(1)
        ]
        real = subset[subset["label"].eq(0)]
        return pd.concat([real, fake], ignore_index=True)

    return subset


def validate_no_source_overlap(
    support: pd.DataFrame,
    query: pd.DataFrame,
) -> None:
    key = (
        "source_group_key"
        if "source_group_key" in support.columns
        and "source_group_key" in query.columns
        else "source_video_key"
    )

    overlap = set(support[key]) & set(query[key])

    if overlap:
        raise AssertionError(
            f"Source-video leakage detected for {len(overlap)} keys."
        )
