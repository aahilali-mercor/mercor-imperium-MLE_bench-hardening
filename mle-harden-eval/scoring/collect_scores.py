#!/usr/bin/env python3
"""Tabulate panel scores and compute the before/after GPT-vs-Qwen spread."""
from __future__ import annotations

import json
from pathlib import Path

ART = Path("/scratch/mle_hardening/artifacts")
TASKS = {
    "statistella": {"before": "statistella-before", "after": "statistella-after",
                    "lower_better": False, "metric": "F1"},
    "nanox81": {"before": "nanox81-lab3-sp25-before", "after": "nanox81-lab3-sp25-after",
                "lower_better": True, "metric": "MSE"},
}
PROFILES = ["qwen3.6-plus-high", "gpt5.5-high"]


def run_scores(variant: str, profile: str) -> list[tuple[str, object]]:
    out = []
    for run_dir in sorted(ART.glob(f"{variant}/{profile}-*")):
        scores = run_dir / "scores.json"
        if not scores.exists():
            state = "running" if (run_dir / "checkpoint.json").exists() else "pending"
            out.append((run_dir.name, state))
            continue
        payload = json.loads(scores.read_text())
        score = payload.get("final_score")
        out.append((run_dir.name, score if score is not None
                    else f"fail:{(payload.get('final_result') or {}).get('failure_code')}"))
    return out


def best(vals: list, lower: bool):
    nums = [v for _, v in vals if isinstance(v, (int, float))]
    if not nums:
        return None
    return min(nums) if lower else max(nums)


for name, cfg in TASKS.items():
    print(f"\n================ {name} ({cfg['metric']}, "
          f"{'lower' if cfg['lower_better'] else 'higher'} better) ================")
    summary = {}
    for cond in ("before", "after"):
        print(f"  -- {cond} ({cfg[cond]})")
        for profile in PROFILES:
            rows = run_scores(cfg[cond], profile)
            for run_name, val in rows:
                shown = f"{val:.5f}" if isinstance(val, float) else val
                print(f"     {profile:20s} {run_name.split('-')[-1]:>2s}: {shown}")
            summary[(cond, profile)] = best(rows, cfg["lower_better"])
    q_b, g_b = summary[("before", PROFILES[0])], summary[("before", PROFILES[1])]
    q_a, g_a = summary[("after", PROFILES[0])], summary[("after", PROFILES[1])]
    print(f"  best-of: qwen before={q_b} after={q_a} | gpt before={g_b} after={g_a}")
    if None not in (q_b, g_b, q_a, g_a):
        sign = -1.0 if cfg["lower_better"] else 1.0
        gap_before = sign * (g_b - q_b)
        gap_after = sign * (g_a - q_a)
        print(f"  GPT-Qwen gap (positive = GPT ahead): before={gap_before:+.5f} "
              f"after={gap_after:+.5f} | spread {'WIDENED' if gap_after > gap_before else 'did NOT widen'}")
