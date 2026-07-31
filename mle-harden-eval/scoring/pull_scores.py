#!/usr/bin/env python3
"""Pull panel scores (GPT + Qwen) from all 3 slinky boxes into a live CSV.

Wide format, one row per task+variant with per-profile run columns.
Rewrites /Users/gyannendradas/MLE_BENCH_NEW/gpt_panel_scores.csv atomically.
"""
from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

OUT = Path("/Users/gyannendradas/MLE_BENCH_NEW/gpt_panel_scores.csv")
SHEET = Path("/Users/gyannendradas/MLE_BENCH_NEW/harden_eval_tracker.csv")
ARCHIVE = Path("/Users/gyannendradas/MLE_BENCH_NEW/HARDENED_DELIVERY_20260731/"
               "hardening_tracker.csv")
PROFILES = {"gpt": "gpt5.5-high", "qwen": "qwen3.6-plus-high"}
TOTAL_RUNS = 240  # 12 tasks x 2 variants x 5 runs x 2 profiles

# user's sheet: exact row order and headers
SHEET_ORDER = [
    "viral-vision-the-you-tube-virality-predictor-challenge",
    "ufc-fight-outcome-prediction-challenge",
    "ts-forecasting",
    "telecom-churn-case-study-hackathon-c-69",
    "telecom-churn-case-study-hackathon-c-68",
    "statistella",
    "russian-car-plates-prices-prediction",
    "nanox81-lab3-sp25",
    "higgs-boson-detection-2025",
    "forest-fire-prediction-epoch-hackathon",
    "fit-5212-s-1-2025",
    "airise-image-regression-challenge-challenge",
]
SHEET_COLS = ["Task Name", "Before Qwen run Scores", "Before GPT Scores",
              "After Qwen run Scores", "After GPT Scores",
              "Tass Harden Criteria Pass"]

# metric name, lower_is_better — matches the delivery tracker
TASKS = {
    "statistella": ("F1", False),
    "nanox81-lab3-sp25": ("MSE", True),
    "ufc-fight-outcome-prediction-challenge": ("Accuracy", False),
    "fit-5212-s-1-2025": ("RMSE", True),
    "russian-car-plates-prices-prediction": ("SMAPE", True),
    "viral-vision-the-you-tube-virality-predictor-challenge": ("RMSE-type", True),
    "airise-image-regression-challenge-challenge": ("F1", False),
    "telecom-churn-case-study-hackathon-c-68": ("Accuracy", False),
    "telecom-churn-case-study-hackathon-c-69": ("Accuracy", False),
    "forest-fire-prediction-epoch-hackathon": ("log-ratio penalty", True),
    "higgs-boson-detection-2025": ("AUC", False),
    "ts-forecasting": ("Skill score", False),
}

HOSTS = ["together-gpu-slinky-0", "together-gpu-slinky-1", "together-gpu-slinky-2"]

REMOTE_SNIPPET = r"""
import json, pathlib
out = {}
art = pathlib.Path("/scratch/mle_hardening/artifacts")
for profile in ["gpt5.5-high", "qwen3.6-plus-high"]:
    for d in art.glob(f"*/{profile}-*"):
        variant, run = d.parent.name, d.name.rsplit("-", 1)[-1]
        sc = d / "scores.json"
        if sc.exists():
            try:
                p = json.loads(sc.read_text())
                v = p.get("final_score")
                out.setdefault(profile, {}).setdefault(variant, {})[run] = (
                    v if v is not None
                    else "fail:" + str((p.get("final_result") or {}).get("failure_code")))
            except Exception as e:
                out.setdefault(profile, {}).setdefault(variant, {})[run] = f"err:{e}"
        elif (d / "checkpoint.json").exists() or any(d.iterdir()):
            out.setdefault(profile, {}).setdefault(variant, {})[run] = "running"
print(json.dumps(out))
"""


def pull(host: str) -> dict:
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=20", host, "python3", "-"],
                           input=REMOTE_SNIPPET, capture_output=True, text=True, timeout=90)
        return json.loads(r.stdout.strip() or "{}")
    except Exception:
        return {}


def fmt(v) -> str:
    return f"{v:.5f}" if isinstance(v, float) else ("" if v is None else str(v))


def archive_qwen_before() -> dict[str, list[str]]:
    """Historical Qwen run scores from the delivery tracker (before-variant)."""
    out: dict[str, list[str]] = {}
    if not ARCHIVE.exists():
        return out
    with ARCHIVE.open() as fh:
        for row in csv.DictReader(fh):
            slug = row["Task Name"].removesuffix("-hardened")
            m = row.get("Before Qwen run Scores", "")
            if "runs:" in m:
                nums = m.split("runs:")[1].split("|")[0]
                out[slug] = [s.strip() for s in nums.split(";") if s.strip()]
    return out


def cell(runs: dict, fallback: list[str] | None = None) -> str:
    """Completed scores ' ; '-joined; errored/pending runs are null (omitted)."""
    vals = [f"{v:.5f}" for i in range(1, 6)
            if isinstance((v := runs.get(str(i))), (int, float))]
    if not vals and fallback:
        vals = fallback[:5]
    return " ; ".join(vals)


def write_sheet(merged: dict) -> None:
    arch = archive_qwen_before()
    rows = []
    for task in SHEET_ORDER:
        qwen, gpt = merged.get(PROFILES["qwen"], {}), merged.get(PROFILES["gpt"], {})
        rows.append({
            "Task Name": f"{task}-hardened",
            "Before Qwen run Scores": cell(qwen.get(f"{task}-before", {}),
                                           arch.get(task)),
            "Before GPT Scores": cell(gpt.get(f"{task}-before", {})),
            "After Qwen run Scores": cell(qwen.get(f"{task}-after", {})),
            "After GPT Scores": cell(gpt.get(f"{task}-after", {})),
            "Tass Harden Criteria Pass": "",
        })
    tmp = SHEET.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SHEET_COLS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(SHEET)


def main() -> None:
    merged: dict[str, dict] = {p: {} for p in PROFILES.values()}
    for host in HOSTS:
        for profile, variants in pull(host).items():
            merged.setdefault(profile, {}).update(variants)

    rows = []
    total_scored = 0
    for task, (metric, lower) in TASKS.items():
        for cond in ("before", "after"):
            variant = f"{task}-{cond}"
            row = {
                "task": task, "variant": cond,
                "metric": f"{metric} ({'lower' if lower else 'higher'} better)",
            }
            for short, profile in PROFILES.items():
                runs = merged.get(profile, {}).get(variant, {})
                vals = [runs.get(str(i)) for i in range(1, 6)]
                nums = [v for v in vals if isinstance(v, (int, float))]
                total_scored += len(nums)
                best = (min(nums) if lower else max(nums)) if nums else None
                row.update({f"{short}_run{i}": fmt(v) for i, v in enumerate(vals, 1)})
                row[f"{short}_best"] = fmt(best)
                row[f"{short}_scored"] = f"{len(nums)}/5"
            rows.append(row)

    tmp = OUT.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(OUT)
    write_sheet(merged)
    print(f"{time.strftime('%H:%M:%S')} wrote {OUT.name} + {SHEET.name}: "
          f"{total_scored}/{TOTAL_RUNS} runs scored")


if __name__ == "__main__":
    main()
