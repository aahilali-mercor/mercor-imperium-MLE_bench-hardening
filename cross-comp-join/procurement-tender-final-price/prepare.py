"""
Build public/ and private/ splits for procurement final-price prediction.

Expected raw layout (file names only — no competition branding required):
  raw/tenders_source_a.csv
  raw/tenders_source_b.csv

Both CSVs must contain Procurement_ID. Source B must contain Final_Price_EUR
(and optionally Estimated_Price_EUR). Source A may contain saving_bin / source
columns which are dropped from features.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ID_COL = "Procurement_ID"
TARGET_COL = "Final_Price_EUR"


def prepare(raw: Path, public: Path, private: Path) -> None:
    path_a = raw / "tenders_source_a.csv"
    path_b = raw / "tenders_source_b.csv"
    if not path_a.exists() or not path_b.exists():
        raise FileNotFoundError(
            f"Expected {path_a.name} and {path_b.name} under {raw}"
        )

    source_a = pd.read_csv(path_a)
    source_b = pd.read_csv(path_b)

    keep_b = [ID_COL, TARGET_COL] + (
        ["Estimated_Price_EUR"] if "Estimated_Price_EUR" in source_b.columns else []
    )
    merged = source_a.merge(source_b[keep_b], on=ID_COL, how="inner")

    drop_from_features = {
        TARGET_COL,
        "saving_bin",
        "source",
        "Estimated_Price_EUR",
    }
    feature_cols = [c for c in merged.columns if c not in drop_from_features and c != ID_COL]
    mid = len(feature_cols) // 2
    profile_cols = feature_cols[:mid]
    commercial_cols = feature_cols[mid:]

    # Keep estimate fields on the commercial extract when present
    for col in ("tender_estimatedPrice_EUR", "Estimated_Price_EUR"):
        if col in merged.columns:
            if col in profile_cols:
                profile_cols.remove(col)
            if col not in commercial_cols:
                commercial_cols.append(col)

    train_df, test_df = train_test_split(merged, test_size=0.25, random_state=0)

    assert set(train_df[ID_COL]).isdisjoint(set(test_df[ID_COL]))
    assert train_df[ID_COL].nunique() == len(train_df)
    assert test_df[ID_COL].nunique() == len(test_df)
    assert len(train_df) + len(test_df) == len(merged)

    train_profile = train_df[[ID_COL] + profile_cols + [TARGET_COL]].copy()
    train_commercial = train_df[[ID_COL] + [c for c in commercial_cols if c in train_df.columns]].copy()
    test_profile = test_df[[ID_COL] + profile_cols].copy()
    test_commercial = test_df[[ID_COL] + [c for c in commercial_cols if c in test_df.columns]].copy()

    # Row order of commercial extract is independent of profile extract
    train_commercial = train_commercial.sample(frac=1.0, random_state=1).reset_index(drop=True)
    test_commercial = test_commercial.sample(frac=1.0, random_state=2).reset_index(drop=True)

    answers = test_df[[ID_COL, TARGET_COL]].sort_values(ID_COL).reset_index(drop=True)
    sample = answers.copy()
    sample[TARGET_COL] = float(train_df[TARGET_COL].median())

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train_profile.to_csv(public / "tenders_profile.csv", index=False)
    train_commercial.to_csv(public / "tenders_commercial.csv", index=False)
    test_profile.to_csv(public / "test_profile.csv", index=False)
    test_commercial.to_csv(public / "test_commercial.csv", index=False)
    sample.to_csv(public / "sample_submission.csv", index=False)
    answers.to_csv(private / "test.csv", index=False)

    assert len(sample) == len(answers)
    assert TARGET_COL not in test_profile.columns
    assert TARGET_COL not in test_commercial.columns
    assert TARGET_COL in train_profile.columns
