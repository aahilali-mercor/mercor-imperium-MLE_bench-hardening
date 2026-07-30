#!/usr/bin/env python3
"""
Internal helper: copy Kaggle train CSVs into the neutral raw names expected by prepare.py.

  raw/tenders_source_a.csv  <- bid-2-save train
  raw/tenders_source_b.csv  <- bid-2-win train (has Final_Price_EUR)

Usage:
  HARDENING_RAW=/path/to/kaggle/unzips python stage_raw.py
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RAW_OUT = Path(os.environ.get("TASK_RAW", HERE / "procurement-tender-final-price" / "raw"))
SRC = Path(
    os.environ.get(
        "HARDENING_RAW",
        Path.home() / "hardening_demo" / "raw",
    )
)

SAVE = SRC / "bid-2-save-savings-prediction-challenge" / "unzipped" / "train.csv"
WIN = SRC / "bid-2-win-tender-outcome-prediction-challenge" / "unzipped" / "train.csv"

if not SAVE.exists() or not WIN.exists():
    raise SystemExit(f"Missing sources:\n  {SAVE}\n  {WIN}")

RAW_OUT.mkdir(parents=True, exist_ok=True)
shutil.copy(SAVE, RAW_OUT / "tenders_source_a.csv")
shutil.copy(WIN, RAW_OUT / "tenders_source_b.csv")
print(f"Wrote {RAW_OUT / 'tenders_source_a.csv'}")
print(f"Wrote {RAW_OUT / 'tenders_source_b.csv'}")
