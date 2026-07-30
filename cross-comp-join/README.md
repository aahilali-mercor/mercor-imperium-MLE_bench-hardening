# Cross-comp join 


**[`procurement-tender-final-price/`](procurement-tender-final-price/)**

That subdirectory is a normal synth-gen-style competition package. Its `description.md`, `prepare.py`, `grade.py`, and `config.yaml` do **not** mention cross-comp join, hardening, or source competition names.

---

## Layout

```
cross-comp-join/
  README.md                          ← this file (internal)
  requirements.txt
  AGENT_PANEL.md                     ← GPT / Qwen wiring (TBD)
  scripts/stage_raw.py               ← maps Kaggle downloads → neutral raw names
  results/                           ← sklearn panel / before-after (internal)
  procurement-tender-final-price/    ← ★ upload this as the task
    config.yaml
    description.md
    prepare.py
    grade.py
    leaderboard.csv
```

Do **not** commit CSVs. `.gitignore` at `hardening/` covers `raw/`, `prepared/`, `public/`, `private/`.

---

## How the task is built (internal only)

1. Download the two related procurement comps (shared `Procurement_ID`).
2. `python scripts/stage_raw.py` → writes neutral `tenders_source_a.csv` / `tenders_source_b.csv`.
3. Run `prepare(raw, public, private)` from `procurement-tender-final-price/prepare.py`.
4. Grade with RMSE via `grade.py`.

### Reproduce prepare

```bash
cd cross-comp-join
pip install -r requirements.txt
export HARDENING_RAW=/path/to/kaggle/raw   # see scripts/stage_raw.py
python scripts/stage_raw.py

python - <<'PY'
from pathlib import Path
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader

task = Path("procurement-tender-final-price")
loader = SourceFileLoader("prepare", str(task / "prepare.py"))
mod = module_from_spec(spec_from_loader(loader.name, loader))
loader.exec_module(mod)
mod.prepare(task / "raw", task / "prepared" / "public", task / "prepared" / "private")
print("prepared OK", task / "prepared")
PY
```

---

## Metric / output (same as task package)

| | |
|--|--|
| Target | `Final_Price_EUR` |
| Metric | RMSE (lower better) |
| Submission | `Procurement_ID,Final_Price_EUR` |

Public files (agent-visible): `tenders_profile.csv`, `tenders_commercial.csv`, `test_profile.csv`, `test_commercial.csv`, `sample_submission.csv`.

---

## Sklearn panel (internal validation)

See [`results/BEFORE_AFTER.md`](results/BEFORE_AFTER.md) and [`results/panel_scores.csv`](results/panel_scores.csv).

| Solver | RMSE |
|--------|------|
| Constant | ~2,629,968 |
| One extract / bad concat | ~2,614k–2,619k |
| Proper combine + model | **2,592,646** |

GPT / Qwen before–after: pending harness ([`AGENT_PANEL.md`](AGENT_PANEL.md)).

---

## Finding more pairs

Hash-gate → overlap on entity key → one target → one metric → split across extracts → shuffle secondary extract → panel. Never put that methodology language into `description.md` / `grade.py` / `prepare.py` comments that ship with the task beyond neutral data prep.
