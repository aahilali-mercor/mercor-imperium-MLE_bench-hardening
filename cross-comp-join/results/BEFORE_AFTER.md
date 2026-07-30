# Before / After — Cross-Comp Join Sample

## Problem framed the same both times
- **Target:** `Final_Price_EUR`
- **Metric:** RMSE (lower better)
- **Entities:** overlapping `Procurement_ID`s from bid-2-save × bid-2-win

## Before (easy control)
- **Surface:** single pre-joined `train.csv` / `test.csv`
- **Path:** `output/baseline_prejoined/`
- **Sklearn:** full-feature HGB ≈ **2,592,646** RMSE (oracle)
- **Agent expectation:** weak and strong models both often clear a high bar (low discrimination)

## After (hardened)
- **Surface:** `tenders_profile.csv` + `tenders_commercial.csv` (commercial rows shuffled)
- **Required skill:** join on `Procurement_ID`, then model
- **Sklearn panel:**

| Solver | RMSE |
|--------|------|
| Constant | 2,629,968 |
| One table / bad concat | ~2,614k–2,619k |
| Proper join (expert) | **2,592,646** |
| Pre-joined oracle | **2,592,646** |

## Agent panel (GPT vs Qwen / ancilla) — fill after harness run

| Variant | Qwen / ancilla | GPT-5.5 | Notes |
|---------|----------------|---------|-------|
| Before (prejoined) | _TBD_ | _TBD_ | |
| After (two-table) | _TBD_ | _TBD_ | Want weak ↓, GPT holds, spread ↑ |

## Pass criteria for this sample
1. Sklearn: expert join ≫ naive (already met).
2. Agents: after hardening, weak fail-rate up vs before; GPT within ~10% of expert gold.
3. Still feels like **one** ML asset (not dual `min()` contests).
