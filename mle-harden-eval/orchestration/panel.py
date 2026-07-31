#!/usr/bin/env python3
"""5-run GPT panel over all 24 hardening task variants.

Worker pool of 8, one worker per GPU. Skips runs that already have a
non-null final_score; failed runs (scores.json with final_score null)
are cleared and redone. Interrupted runs (artifact dir, no scores.json)
resume via --resume.

Progress is appended to /scratch/mle_hardening/logs/panel_status.log.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

BASE = Path("/scratch/mle_hardening")
HARNESS = BASE / "PIPELINE" / "harness"
LOGS = BASE / "logs"
STATUS = LOGS / os.environ.get("PANEL_STATUS", "panel_status.log")
PROFILE = os.environ.get("PANEL_PROFILE", "gpt5.5-high")
RUNS = [1, 2, 3, 4, 5]
# 12 workers/node: --cpus is a soft CFS quota and node RAM is 1.76TB, so
# 12x128GiB sandboxes oversubscribe 156 cores gracefully; GPUs shared 2-per.
WORKERS = int(os.environ.get("PANEL_WORKERS", "12"))
N_GPUS = 8

# smallest data first so early results land quickly
TASKS = [
    "forest-fire-prediction-epoch-hackathon",
    "russian-car-plates-prices-prediction",
    "viral-vision-the-you-tube-virality-predictor-challenge",
    "ufc-fight-outcome-prediction-challenge",
    "airise-image-regression-challenge-challenge",
    "fit-5212-s-1-2025",
    "telecom-churn-case-study-hackathon-c-68",
    "telecom-churn-case-study-hackathon-c-69",
    "nanox81-lab3-sp25",
    "statistella",
    "higgs-boson-detection-2025",
    "ts-forecasting",
]

_status_lock = threading.Lock()


def status(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with _status_lock:
        print(line, flush=True)
        with STATUS.open("a") as fh:
            fh.write(line + "\n")


def build_jobs() -> list[tuple[str, int]]:
    jobs: list[tuple[str, int]] = []
    for task in TASKS:
        for variant in (f"{task}-before", f"{task}-after"):
            assert (BASE / "tasks" / variant / "metadata.json").exists(), variant
            for run in RUNS:
                art = BASE / "artifacts" / variant / f"{PROFILE}-{run}"
                scores = art / "scores.json"
                if scores.exists():
                    try:
                        payload = json.loads(scores.read_text())
                    except Exception:
                        payload = {}
                    if payload.get("final_score") is not None:
                        status(f"skip scored: {variant} run{run} "
                               f"score={payload['final_score']:.5f}")
                        continue
                    status(f"redo failed: {variant} run{run}")
                    shutil.rmtree(art)
                jobs.append((variant, run))
    return jobs


def worker(slot: int, q: "queue.Queue[tuple[str, int]]") -> None:
    gpu = slot % N_GPUS
    while True:
        try:
            variant, run = q.get_nowait()
        except queue.Empty:
            return
        art = BASE / "artifacts" / variant / f"{PROFILE}-{run}"
        resume = art.is_dir()
        art.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["python3", "aide_harness.py", str(BASE / "tasks" / variant),
               "--solver-profile", PROFILE, "--run", str(run),
               "--artifact-dir", str(art), "--gpu", str(gpu)]
        if resume:
            cmd.append("--resume")
        log = LOGS / f"panel_{variant}_{PROFILE}-{run}.log"
        status(f"start: {variant} run{run} gpu={gpu}{' (resume)' if resume else ''}")
        t0 = time.time()
        with log.open("w") as fh:
            rc = subprocess.call(cmd, cwd=HARNESS, stdout=fh, stderr=fh)
        mins = (time.time() - t0) / 60
        scores = art / "scores.json"
        final = None
        if scores.exists():
            try:
                final = json.loads(scores.read_text()).get("final_score")
            except Exception:
                pass
        status(f"done: {variant} run{run} rc={rc} {mins:.1f}min score={final}")
        q.task_done()


def main() -> None:
    global TASKS
    if len(sys.argv) > 1:  # optional task-name shard, e.g. panel_gpt5.py statistella higgs-...
        unknown = set(sys.argv[1:]) - set(TASKS)
        assert not unknown, f"unknown tasks: {unknown}"
        TASKS = sys.argv[1:]
    os.environ["IMPERIUM_RUNTIME_IMAGE"] = "imperium-mlebench-runtime:fixed-20260731-pd2"
    LOGS.mkdir(exist_ok=True)
    jobs = build_jobs()
    status(f"panel start: {len(jobs)} jobs, {WORKERS} workers")
    q: "queue.Queue[tuple[str, int]]" = queue.Queue()
    for job in jobs:
        q.put(job)
    threads = [threading.Thread(target=worker, args=(g, q), daemon=True)
               for g in range(WORKERS)]
    for t in threads:
        t.start()
        time.sleep(15)  # stagger container startups
    for t in threads:
        t.join()
    status("panel complete")


if __name__ == "__main__":
    main()
