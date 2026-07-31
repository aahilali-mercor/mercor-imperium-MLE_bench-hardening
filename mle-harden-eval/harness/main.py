#!/usr/bin/env python3
"""Compatibility entrypoint for the direct three-tool evaluation harness."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

try:
    from .direct_harness import (
        MAX_PREVIEW_SUBMITS,
        ConfigurationBlocked,
        HarnessError,
        ProviderPaused,
        main,
    )
except ImportError:
    from direct_harness import (  # type: ignore[no-redef]
        MAX_PREVIEW_SUBMITS,
        ConfigurationBlocked,
        HarnessError,
        ProviderPaused,
        main,
    )


STATUS_INTERVAL_SECONDS = 150
PREVIEW_SUBMIT_LIMIT = MAX_PREVIEW_SUBMITS


def _batch_run_mode(argv: list[str]) -> str | None:
    modes = {"all", "bonus"}
    for index, arg in enumerate(argv):
        if arg.startswith("--run=") and arg.split("=", 1)[1] in modes:
            return arg.split("=", 1)[1]
        if arg == "--run" and index + 1 < len(argv) and argv[index + 1] in modes:
            return argv[index + 1]
    return None


def _all_run_requested(argv: list[str]) -> bool:
    """Compatibility predicate for either multi-run mode."""
    return _batch_run_mode(argv) is not None


def _child_args(argv: list[str], run: int) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == "--run":
            result.extend(("--run", str(run)))
            index += 2
        elif argv[index] in ("--run=all", "--run=bonus"):
            result.append(f"--run={run}")
            index += 1
        else:
            result.append(argv[index])
            index += 1
    return result


def _task_slug(argv: list[str]) -> str:
    options_with_values = {
        "--backend", "--model-name", "--api-model", "--artifact-dir", "--run",
        "--thinking", "--gpu", "--max-turns",
    }
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in options_with_values:
            index += 2
        elif arg.startswith("-"):
            index += 1
        else:
            return Path(arg).name
    raise HarnessError("a task is required")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _file_identity(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_ino, stat.st_mtime_ns, stat.st_size


def _status_line(
    run: int,
    run_dir: Path,
    process: subprocess.Popen,
    force_baseline: tuple[int, int, int] | None | object = ...,
) -> str:
    trajectory_path = run_dir / "trajectory.json"
    if force_baseline is not ... and _file_identity(trajectory_path) == force_baseline:
        step = (
            "starting"
            if process.poll() is None
            else "completed"
            if process.returncode == 0
            else f"failed(exit={process.returncode})"
        )
        return f"{run:<3} {0:>2}/{30:<2}  {step:<22.22} {'-':<14} {PREVIEW_SUBMIT_LIMIT:>2}"
    trajectory = _read_json(trajectory_path)
    scores = _read_json(run_dir / "scores.json")
    used = int(trajectory.get("preview_submits_used", 0))
    turns = int(trajectory.get("turns_used", 0))
    limit = int(trajectory.get("turn_limit", 30))
    submissions = scores.get("submissions", [])
    numeric_scores = [
        float(item["score"])
        for item in submissions
        if isinstance(item, dict) and isinstance(item.get("score"), (int, float))
    ]
    if numeric_scores:
        best = min(numeric_scores) if scores.get("is_lower_better") else max(numeric_scores)
        best_text = f"{best:.12g}"
    else:
        best_text = "-"
    events = trajectory.get("events", [])
    step = events[-1].get("type", "starting") if events else "starting"
    if process.poll() is not None:
        # A run can complete either by reaching its turn limit or by choosing
        # final-submit early.  Both are successful terminal outcomes.  Keep a
        # non-zero child exit visible instead of letting a previously written
        # score make a crashed run look successful.
        step = "completed" if process.returncode == 0 else f"failed(exit={process.returncode})"
    return (
        f"{run:<3} {turns:>2}/{limit:<2}  {step:<22.22} "
        f"{best_text:<14} {max(0, PREVIEW_SUBMIT_LIMIT - used):>2}"
    )


def _print_all_status(
    processes: dict[int, subprocess.Popen],
    artifact_root: Path,
    model_name: str,
    force_baselines: dict[int, tuple[int, int, int] | None] | None = None,
    *,
    final: bool = False,
) -> None:
    heading = "Final all-run status" if final else "All-run status"
    print(f"\n{heading} ({time.strftime('%Y-%m-%d %H:%M:%S')})", flush=True)
    print("run step   state                  best score     submits left", flush=True)
    for run, process in processes.items():
        print(
            _status_line(
                run,
                artifact_root / f"{model_name}-{run}",
                process,
                force_baselines[run] if force_baselines is not None else ...,
            ),
            flush=True,
        )


def run_all(argv: list[str]) -> int:
    if "--artifact-dir" in argv or any(arg.startswith("--artifact-dir=") for arg in argv):
        raise HarnessError("--artifact-dir cannot be combined with --run all")
    project_root = Path(__file__).resolve().parent.parent
    artifact_root = project_root / "tasks-evals" / _task_slug(argv)
    processes: dict[int, subprocess.Popen] = {}
    force_baselines: dict[int, tuple[int, int, int] | None] | None = (
        {} if "--force" in argv else None
    )
    logs = {}

    def terminate_as_interrupt(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate_as_interrupt)
    try:
        run_count = 6 if _batch_run_mode(argv) == "bonus" else 5
        model_name = _model_name(argv)
        for run in range(1, run_count + 1):
            if force_baselines is not None:
                force_baselines[run] = _file_identity(
                    artifact_root / f"{model_name}-{run}" / "trajectory.json"
                )
            log = tempfile.TemporaryFile(mode="w+")
            logs[run] = log
            processes[run] = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), *_child_args(argv, run)],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        while any(process.poll() is None for process in processes.values()):
            _print_all_status(
                processes, artifact_root, model_name, force_baselines
            )
            deadline = time.monotonic() + STATUS_INTERVAL_SECONDS
            while time.monotonic() < deadline and any(p.poll() is None for p in processes.values()):
                time.sleep(min(1, deadline - time.monotonic()))
        # Always show the settled state once after the last child exits.  This
        # makes successful early final submissions and abnormal exits obvious
        # even when they happen between periodic timer updates.
        _print_all_status(
            processes,
            artifact_root,
            model_name,
            force_baselines,
            final=True,
        )
        failed = [run for run, process in processes.items() if process.returncode != 0]
        for run in failed:
            log = logs[run]
            log.seek(0, os.SEEK_END)
            size = log.tell()
            log.seek(max(0, size - 4000))
            print(f"\nLast output from failed run {run}:\n{log.read()}", flush=True)
    except KeyboardInterrupt:
        print("\nStopping all five runs...", flush=True)
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        for process in processes.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
        return 130
    finally:
        for log in logs.values():
            log.close()
    return 0 if all(process.returncode == 0 for process in processes.values()) else 1


def _model_name(argv: list[str]) -> str:
    for index, arg in enumerate(argv):
        if arg == "--model-name" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--model-name="):
            return arg.split("=", 1)[1]
    for index, arg in enumerate(argv):
        if arg == "--backend" and index + 1 < len(argv):
            return {
                "hy3": "hy3",
                "openai": "gpt5.5-high",
                "claude": "claude-opus-4.8-high",
                "sonnet": "claude-sonnet-5-high",
                "grok": "grok-4.5-high",
                "gemini": "gemini-3.5-flash-high",
                "gemini-pro": "gemini-3.1-pro-preview-high",
                "glm": "glm-5.2-high",
            }.get(argv[index + 1], "hy3")
        if arg.startswith("--backend="):
            return {
                "hy3": "hy3",
                "openai": "gpt5.5-high",
                "claude": "claude-opus-4.8-high",
                "sonnet": "claude-sonnet-5-high",
                "grok": "grok-4.5-high",
                "gemini": "gemini-3.5-flash-high",
                "gemini-pro": "gemini-3.1-pro-preview-high",
                "glm": "glm-5.2-high",
            }.get(arg.split("=", 1)[1], "hy3")
    return "hy3"


if __name__ == "__main__":
    try:
        raise SystemExit(
            run_all(sys.argv[1:]) if _all_run_requested(sys.argv[1:]) else main()
        )
    except ConfigurationBlocked as exc:
        print(f"harness configuration: {exc}", file=sys.stderr)
        raise SystemExit(78)
    except ProviderPaused as exc:
        print(f"harness provider pause: {exc}", file=sys.stderr)
        raise SystemExit(75)
    except HarnessError as exc:
        print(f"harness: {exc}", file=sys.stderr)
        raise SystemExit(2)
