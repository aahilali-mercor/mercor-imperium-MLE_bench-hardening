#!/usr/bin/env python3
"""Run a direct-API, three-tool evaluation trajectory."""

from __future__ import annotations

import argparse
import copy
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


HARNESS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = HARNESS_DIR.parent
RUNTIME_ROOT = PROJECT_ROOT.parent / ".runtime" / "harness"
CID_ROOT = Path.home() / "snap/docker/common/hy3-direct-cids"
SANDBOX_IMAGE = os.environ.get(
    "IMPERIUM_RUNTIME_IMAGE", "imperium-mlebench-runtime:dev-20260715"
)
SANDBOX_PATH = ":".join(
    [
        "/opt/venv/bin",
        "/usr/local/cuda/bin",
        "/usr/local/sbin",
        "/usr/local/bin",
        "/usr/sbin",
        "/usr/bin",
        "/sbin",
        "/bin",
    ]
)
SANDBOX_PYTHON = "python"
PRODUCTION_TURN_LIMIT = 30
MAX_PREVIEW_SUBMITS = 4
COMMAND_TIMEOUT_SECONDS = 180
MAX_CAPTURE_BYTES = 8 * 1024 * 1024
MAX_MODEL_TOOL_RESULT_CHARS = 6_000
MAX_HISTORICAL_ARGUMENT_CHARS = 8_000
CHAT_HISTORY_CHAR_BUDGET = 36_000
MAX_PROVIDER_RETRIES = 4
CHECKPOINT_VERSION = 1
DEFAULT_HY3_MODEL = "tencent/hy3"
DEFAULT_OPENAI_MODEL = "gpt-5.5"
ATLAS_ROUTE = "atlas-cloud/fp8"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_CLAUDE_MODEL = "anthropic/claude-opus-4.8"
DEFAULT_SONNET_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_GROK_MODEL = "x-ai/grok-4.5"
DEFAULT_GEMINI_MODEL = "google/gemini-3.5-flash"
DEFAULT_GEMINI_PRO_MODEL = "google/gemini-3.1-pro-preview"
DEFAULT_GLM_MODEL = "z-ai/glm-5.2"
SERIAL_SUBMIT_LOCK = Path.home() / ".cache/hy3-harness/submit-serial.lock"
CPU_SUBMIT_QUEUE_CAPACITY = 2
CPU_SUBMIT_LOCKS = tuple(
    Path.home() / f".cache/hy3-harness/submit-cpu-{slot}.lock"
    for slot in range(1, CPU_SUBMIT_QUEUE_CAPACITY + 1)
)
CPU_SUBMIT_QUEUE_POLL_SECONDS = 0.1
RUN_LOCK_ROOT = Path.home() / ".cache/hy3-harness/run-locks"


class HarnessError(RuntimeError):
    pass


class ProviderPaused(HarnessError):
    """A retryable provider request exhausted its four-attempt cycle."""


class ConfigurationBlocked(HarnessError):
    """A missing credential or non-retryable provider response requires repair."""


def acquire_run_lock(artifact_dir: Path):
    """Hold a host-wide, nonblocking lock for one canonical run directory."""
    RUN_LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(str(artifact_dir.resolve()).encode()).hexdigest()
    handle = (RUN_LOCK_ROOT / f"{identity}.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise HarnessError(
            f"another harness process is already running {artifact_dir}"
        )
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} artifact_dir={artifact_dir}\n")
    handle.flush()
    return handle


@contextlib.contextmanager
def serial_submit_slot(
    state: "RunState", tool: str, attempt: int
):
    """Acquire the selected host-wide scoring queue across harness processes."""
    if state.serial_submits:
        queue = "gpu"
        capacity = 1
        lock_paths = (SERIAL_SUBMIT_LOCK,)
    else:
        queue = "cpu"
        capacity = CPU_SUBMIT_QUEUE_CAPACITY
        lock_paths = CPU_SUBMIT_LOCKS

    lock_paths[0].parent.mkdir(parents=True, exist_ok=True)
    state.event(
        "submit_queued",
        tool=tool,
        attempt=attempt,
        queue=queue,
        capacity=capacity,
    )
    queued_at = time.monotonic()
    lock_files = [path.open("a+") for path in lock_paths]
    acquired_slot: int | None = None
    try:
        if state.serial_submits:
            fcntl.flock(lock_files[0].fileno(), fcntl.LOCK_EX)
            acquired_slot = 0
        else:
            while acquired_slot is None:
                for slot, lock_file in enumerate(lock_files):
                    try:
                        fcntl.flock(
                            lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                        )
                    except BlockingIOError:
                        continue
                    acquired_slot = slot
                    break
                if acquired_slot is None:
                    time.sleep(CPU_SUBMIT_QUEUE_POLL_SECONDS)

        state.event(
            "submit_queue_acquired",
            tool=tool,
            attempt=attempt,
            queue=queue,
            capacity=capacity,
            slot=acquired_slot + 1,
            wait_seconds=round(time.monotonic() - queued_at, 3),
        )
        try:
            yield
        finally:
            state.event(
                "submit_queue_released",
                tool=tool,
                attempt=attempt,
                queue=queue,
                capacity=capacity,
                slot=acquired_slot + 1,
            )
    finally:
        if acquired_slot is not None:
            fcntl.flock(lock_files[acquired_slot].fileno(), fcntl.LOCK_UN)
        for lock_file in lock_files:
            lock_file.close()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(content)
    os.replace(temporary, path)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def terminal_event(event: dict[str, Any]) -> None:
    print(json.dumps(event, default=str, ensure_ascii=False), flush=True)
    for stream in ("stdout", "stderr"):
        value = event.get(stream)
        if value:
            print(f"--- {stream} begin ---", flush=True)
            print(value, flush=True)
            print(f"--- {stream} end ---", flush=True)


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(
            r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(.*)$", line
        )
        if not match:
            raise HarnessError(f"invalid environment entry at {path}:{number}")
        key, value = match.groups()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def checked_task(path: Path) -> tuple[Path, dict[str, Any]]:
    task = path.resolve()
    required = [
        task / "metadata.json",
        task / "public",
        task / "public/description.md",
        task / "private/grader.py",
    ]
    missing = [str(item) for item in required if not item.exists()]
    if not task.is_dir() or missing:
        raise HarnessError("task is missing required paths: " + ", ".join(missing))
    metadata = json.loads((task / "metadata.json").read_text())
    if metadata.get("competition_id") != task.name:
        raise HarnessError("metadata competition_id must match the task directory name")
    wall = metadata.get("wall_clock_limit_minutes")
    if not isinstance(wall, int) or isinstance(wall, bool) or wall <= 0:
        raise HarnessError("wall_clock_limit_minutes must be a positive integer")
    hidden = metadata.get("hidden_test_set")
    paths = metadata.get("hidden_test_paths")
    if not isinstance(hidden, bool) or not isinstance(paths, list):
        raise HarnessError("invalid hidden-test metadata")
    if hidden != bool(paths):
        raise HarnessError("hidden_test_set and hidden_test_paths disagree")
    for value in paths:
        relative = Path(value)
        if (
            not isinstance(value, str)
            or relative.is_absolute()
            or ".." in relative.parts
            or not value.startswith("public/")
        ):
            raise HarnessError(f"unsafe hidden_test_path: {value!r}")
        if (
            not (task / relative).exists()
            or not (task / "hidden-test" / relative).exists()
        ):
            raise HarnessError(f"hidden-test replacement is missing: {value!r}")
    return task, metadata


def system_prompt(
    metadata: dict[str, Any],
    turn_limit: int,
    turns_used: int = 0,
    submits_used: int = 0,
) -> str:
    direction = "lower" if metadata.get("is_lower_better") else "higher"
    turns_remaining = max(0, turn_limit - turns_used)
    submits_remaining = max(0, MAX_PREVIEW_SUBMITS - submits_used)
    wall = int(metadata["wall_clock_limit_minutes"])
    prompt = (PROJECT_ROOT / "prompts/evaluations/gpt-multiturn.md").read_text()
    replacements = {
        "{{evaluation_metric}}": metadata.get(
            "evaluation_metric", "the metric documented in the task"
        ),
        "{{direction}}": direction,
        "{{turns_used}}": str(turns_used),
        "{{turns_remaining}}": str(turns_remaining),
        "{{turn_limit}}": str(turn_limit),
        "{{submits_used}}": str(submits_used),
        "{{submits_remaining}}": str(submits_remaining),
        "{{submit_limit}}": str(MAX_PREVIEW_SUBMITS),
        "{{command_timeout}}": str(COMMAND_TIMEOUT_SECONDS),
        "{{wall_minutes}}": str(wall),
    }
    for key, value in replacements.items():
        prompt = prompt.replace(key, str(value))
    return prompt


def validate_prompt(prompt: str, turn_limit: int) -> None:
    required = [
        "MLE-bench-style",
        "best valid final score",
        "train.py",
        "submission/submission.csv",
        "`bash`",
        "`submit`",
        "Model turns already used",
        "Model turns remaining",
        "Hard maximum model turns",
        "Submit attempts already used",
        "Submit attempts remaining",
        f"hard {COMMAND_TIMEOUT_SECONDS}-second",
        str(turn_limit),
    ]
    missing = [phrase for phrase in required if phrase not in prompt]
    if missing:
        raise HarnessError(
            "system prompt is missing required clauses: " + repr(missing)
        )
    forbidden_strategy = [
        "use lightgbm",
        "use xgboost",
        "use a neural",
        "baseline model",
    ]
    found = [phrase for phrase in forbidden_strategy if phrase in prompt.lower()]
    if found:
        raise HarnessError("system prompt contains modeling advice: " + repr(found))


def budget_status(state: "RunState", before_next_turn: bool = False) -> str:
    remaining = max(0, state.turn_limit - state.turns_used)
    submit_remaining = max(0, MAX_PREVIEW_SUBMITS - state.preview_submits)
    lines = [
        "HARNESS BUDGET STATUS",
        f"- Model turns already used: {state.turns_used}.",
        f"- Model turns remaining: {remaining}.",
        f"- Hard maximum model turns: {state.turn_limit}.",
        f"- Submit attempts already used: {state.preview_submits}.",
        f"- Submit attempts remaining: {submit_remaining}.",
        f"- Hard maximum submit attempts: {MAX_PREVIEW_SUBMITS}.",
    ]
    if remaining == 0:
        lines.append(
            "- NO TURNS REMAIN: the best valid submitted score is final; without one this trajectory is FAIL."
        )
    elif remaining == 1:
        lines.append(
            f"- NEXT RESPONSE ORDINAL: {state.turns_used + 1}. This is the FINAL TURN. Allowed tool calls still execute while submission budget remains."
        )
    elif remaining == 2:
        lines.append(
            f"- TWO RESPONSES REMAIN. The next response ordinal is {state.turns_used + 1}; the response after that is the FINAL TURN. Stop open-ended exploration now and ensure a runnable train.py exists before the final turn begins."
        )
    return "\n".join(lines)


def safe_child_env() -> dict[str, str]:
    allowed = {
        "PATH",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CUDA_HOME",
        "CUDA_PATH",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env["PATH"] = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def read_limited(path: Path) -> tuple[str, bool]:
    size = path.stat().st_size if path.exists() else 0
    with path.open("rb") as handle:
        payload = handle.read(MAX_CAPTURE_BYTES)
    text = payload.decode("utf-8", errors="replace")
    if size > MAX_CAPTURE_BYTES:
        text += f"\n[output truncated after {MAX_CAPTURE_BYTES} bytes]"
    return text, size > MAX_CAPTURE_BYTES


def readonly_symlink_targets(workspace: Path) -> list[Path]:
    public = workspace / "public"
    candidates = [public]
    if public.is_dir() and not public.is_symlink():
        candidates.extend(public.rglob("*"))
    targets: set[Path] = set()
    root = workspace.resolve()
    for candidate in candidates:
        if candidate.is_symlink():
            target = candidate.resolve(strict=True)
            if target != root and root not in target.parents:
                targets.add(target)
    return sorted(targets, key=str)


def docker_prefix(
    workspace: Path,
    gpu: str | None,
    cidfile: Path,
    network: str = "none",
    resource_limits: dict[str, Any] | None = None,
) -> list[str]:
    docker = shutil.which("docker")
    if not docker:
        raise HarnessError("Docker is required for tool isolation")
    available = subprocess.run(
        [docker, "image", "inspect", SANDBOX_IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if available.returncode != 0:
        raise HarnessError(f"sandbox image {SANDBOX_IMAGE!r} is missing")
    CID_ROOT.mkdir(parents=True, exist_ok=True)
    cidfile.unlink(missing_ok=True)
    root = workspace.resolve()
    limits = resource_limits or {}
    cpu_cores = int(limits.get("cpuCores", 20))
    ram_bytes = int(limits.get("ramBytes", 128 * 1024**3))
    scratch_bytes = int(limits.get("scratchBytes", 32 * 1024**3))
    gpu_count = int(limits.get("gpuCount", 1 if gpu is not None else 0))
    if (
        cpu_cores < 1
        or ram_bytes < 1024**3
        or scratch_bytes < 1024**2
        or gpu_count not in {0, 1}
    ):
        raise HarnessError("task resource limits are invalid")
    if gpu_count == 0:
        gpu = None
    elif gpu is None:
        raise HarnessError("task requires one GPU but no device was assigned")
    tmpfs_bytes = min(scratch_bytes, ram_bytes)
    args = [
        docker,
        "run",
        "--rm",
        "--cidfile",
        str(cidfile),
        "--read-only",
        "--network",
        network,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--cap-drop",
        "ALL",
        "--pids-limit",
        "4096",
        "--cpus",
        str(cpu_cores),
        "--memory",
        str(ram_bytes),
        "--memory-swap",
        str(ram_bytes),
        "--tmpfs",
        f"/tmp:rw,nosuid,nodev,exec,size={tmpfs_bytes}",
        "--workdir",
        str(root),
        "--env",
        f"PATH={SANDBOX_PATH}",
        "--env",
        f"HOME={root / '.tool-home'}",
        "--env",
        f"TMPDIR={root / '.tool-tmp'}",
        "--volume",
        f"{root}:{root}:rw",
    ]
    for target in readonly_symlink_targets(workspace):
        args.extend(["--volume", f"{target}:{target}:ro"])
    for key in [
        "LANG",
        "LC_ALL",
        "TOKENIZERS_PARALLELISM",
    ]:
        args.extend(["--env", key])
    if gpu is not None:
        args.extend(
            [
                "--runtime",
                "nvidia",
                "--env",
                f"NVIDIA_VISIBLE_DEVICES={gpu}",
                "--env",
                "CUDA_VISIBLE_DEVICES=0",
                "--device",
                "/dev/nvidia-uvm",
                "--device",
                "/dev/nvidia-uvm-tools",
            ]
        )
    args.append(SANDBOX_IMAGE)
    return args


def kill_cidfile(cidfile: Path) -> None:
    if not cidfile.is_file():
        return
    container_id = cidfile.read_text().strip()
    docker = shutil.which("docker")
    if docker and container_id:
        subprocess.run(
            [docker, "kill", container_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    cidfile.unlink(missing_ok=True)


def execute_container(
    workspace: Path,
    command: str,
    timeout_seconds: int,
    gpu: str | None,
    label: str,
    resource_limits: dict[str, Any] | None = None,
) -> dict[str, Any]:
    token = uuid.uuid4().hex
    cidfile = CID_ROOT / f"hy3-{label}-{token}.cid"
    stdout_path = workspace / f".harness-{label}-{token}.stdout"
    stderr_path = workspace / f".harness-{label}-{token}.stderr"
    (workspace / ".tool-home").mkdir(exist_ok=True)
    (workspace / ".tool-tmp").mkdir(exist_ok=True)
    started = time.monotonic()
    return_code: int | None = None
    outer_timed_out = False
    with (
        stdout_path.open("wb") as stdout_handle,
        stderr_path.open("wb") as stderr_handle,
    ):
        process = subprocess.Popen(
            [
                *docker_prefix(
                    workspace,
                    gpu,
                    cidfile,
                    resource_limits=resource_limits,
                ),
                "/usr/bin/timeout",
                "--signal=TERM",
                "--kill-after=10s",
                f"{timeout_seconds}s",
                "/bin/bash",
                "-c",
                command,
            ],
            cwd=workspace,
            env=safe_child_env(),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds + 25)
        except subprocess.TimeoutExpired:
            outer_timed_out = True
            kill_cidfile(cidfile)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            return_code = process.wait()
    duration = round(time.monotonic() - started, 3)
    stdout, stdout_truncated = read_limited(stdout_path)
    stderr, stderr_truncated = read_limited(stderr_path)
    stdout_path.unlink(missing_ok=True)
    stderr_path.unlink(missing_ok=True)
    cidfile.unlink(missing_ok=True)
    # GNU timeout returns 124 when our outer limit fires, but a participant may
    # also deliberately run an inner `timeout 170 ...`. Only attribute 124 to
    # the harness limit when the elapsed time actually reached that limit.
    timed_out = outer_timed_out or (
        return_code == 124 and duration >= timeout_seconds - 1.0
    )
    return {
        "status": "fail" if timed_out or return_code != 0 else "ok",
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "duration_seconds": duration,
        "return_code": return_code,
        "timed_out": timed_out,
    }


def grade_submission(
    task: Path, submission_path: Path, active_public: Path
) -> float:
    import pandas as pd

    module_name = f"hy3_grader_{task.name.replace('-', '_')}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        module_name, task / "private/grader.py"
    )
    if spec is None or spec.loader is None:
        raise HarnessError("could not load private grader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # A grader may need the manifest that was active for this execution.  In
    # particular, final-submit grades against a temporary public tree whose
    # test inputs have been replaced with the hidden set, rather than against
    # the task package's visible public tree used when the module was loaded.
    if hasattr(module, "PUBLIC"):
        module.PUBLIC = active_public
    score = module.grade(pd.read_csv(submission_path))
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise HarnessError("grader returned a non-finite or non-numeric score")
    return float(score)


def score_train(
    task: Path,
    workspace: Path,
    timeout_seconds: int,
    gpu: str | None,
    label: str = "train",
    grade: bool = True,
    resource_limits: dict[str, Any] | None = None,
) -> dict[str, Any]:
    shutil.rmtree(workspace / "submission", ignore_errors=True)
    if not (workspace / "train.py").is_file():
        return {
            "status": "fail",
            "error": "train.py does not exist in the workspace root",
            "stdout": "",
            "stderr": "",
            "duration_seconds": 0.0,
            "return_code": None,
        }
    result = execute_container(
        workspace,
        f"{SANDBOX_PYTHON} train.py",
        timeout_seconds,
        gpu,
        label,
        resource_limits,
    )
    submission = workspace / "submission/submission.csv"
    if result.pop("timed_out", False):
        result["error"] = (
            f"train.py exceeded the {timeout_seconds}-second wall-clock limit"
        )
    elif result["return_code"] != 0:
        result["error"] = f"train.py exited with status {result['return_code']}"
    elif not submission.is_file():
        result["status"] = "fail"
        result["error"] = "train.py did not create submission/submission.csv"
    elif not grade:
        result["status"] = "ok"
        result.pop("error", None)
    else:
        try:
            result["score"] = grade_submission(
                task, submission, workspace / "public"
            )
            result["status"] = "scored"
            result.pop("error", None)
        except Exception as exc:
            result["status"] = "fail"
            result["error"] = (
                f"submission could not be scored: {type(exc).__name__}: {exc}"
            )
    return result


def make_final_public(task: Path, metadata: dict[str, Any], destination: Path) -> None:
    source = task / "public"
    replacements = {
        Path(value).relative_to("public"): task / "hidden-test" / value
        for value in metadata.get("hidden_test_paths", [])
    }

    def populate(source_dir: Path, output_dir: Path, relative_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for child in source_dir.iterdir():
            relative = relative_dir / child.name
            exact = replacements.get(relative)
            descendants = [path for path in replacements if relative in path.parents]
            target = output_dir / child.name
            if exact is not None:
                target.symlink_to(exact, target_is_directory=exact.is_dir())
            elif descendants:
                if not child.is_dir():
                    raise HarnessError(
                        f"hidden path descends through a file: public/{relative}"
                    )
                populate(child, target, relative)
            else:
                target.symlink_to(child, target_is_directory=child.is_dir())

    populate(source, destination, Path())


def format_execution_result(result: dict[str, Any], state: "RunState") -> str:
    lines = [
        f"status: {result['status']}",
        f"duration_seconds: {result.get('duration_seconds', 0)}",
        f"return_code: {result.get('return_code')}",
    ]
    if "score" in result:
        lines.append(f"score: {result['score']:.12g}")
    if result.get("error"):
        lines.append(f"error: {result['error']}")
    lines.extend(
        [
            budget_status(state),
            "stdout:",
            result.get("stdout", "") or "<empty>",
            "stderr:",
            result.get("stderr", "") or "<empty>",
        ]
    )
    return "\n".join(lines)


@dataclass
class RunState:
    task: Path
    metadata: dict[str, Any]
    scratch: Path
    artifact_dir: Path
    model_name: str
    backend: str
    run_number: int
    gpu: str | None
    turn_limit: int
    api_model: str
    serial_submits: bool = False
    started_at: str = field(default_factory=utc_now)
    turns_used: int = 0
    preview_submits: int = 0
    final_called: bool = False
    scores: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    final_result: dict[str, Any] | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def timeout_seconds(self) -> int:
        return int(self.metadata["wall_clock_limit_minutes"]) * 60

    @property
    def run_label(self) -> str:
        return f"{self.task.name}/{self.model_name}-{self.run_number}"

    @property
    def cid_label(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "-", self.run_label)

    def event(self, kind: str, **payload: Any) -> None:
        with self.lock:
            event = {"timestamp": utc_now(), "type": kind, **payload}
            self.events.append(event)
        terminal_event(event)
        persist_progress(self)

    def bash(self, command: Any) -> dict[str, Any]:
        if not isinstance(command, str) or not command.strip():
            return {"text": "command must be a non-empty string", "is_error": True}
        self.event("tool_call", tool="bash", command=command[:100_000])
        result = execute_container(
            self.scratch,
            command,
            COMMAND_TIMEOUT_SECONDS,
            self.gpu,
            f"{self.cid_label}-bash",
            self.metadata.get("resource_limits"),
        )
        if result.pop("timed_out", False):
            result["error"] = (
                f"command exceeded the {COMMAND_TIMEOUT_SECONDS}-second wall-clock limit"
            )
        elif result.get("return_code") != 0:
            result["error"] = f"command exited with status {result.get('return_code')}"
        self.event(
            "tool_result",
            tool="bash",
            status=result["status"],
            return_code=result.get("return_code"),
            duration_seconds=result.get("duration_seconds"),
            error=result.get("error"),
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
        )
        return {
            "text": format_execution_result(result, self),
            "is_error": result["status"] == "fail",
        }

    def submit(self, final: bool) -> dict[str, Any]:
        with self.lock:
            if self.final_called:
                return {
                    "text": "The trajectory is already finalized; no more tools are allowed.",
                    "is_error": True,
                }
            if final:
                return {
                    "text": "final-submit is not part of this protocol; use submit.",
                    "is_error": True,
                }
            if self.preview_submits >= MAX_PREVIEW_SUBMITS:
                return {
                    "text": "No submit attempts remain.\n"
                    + budget_status(self),
                    "is_error": True,
                }
            self.preview_submits += 1
            attempt = self.preview_submits
        tool = "submit"
        self.event("tool_call", tool=tool, attempt=attempt)
        with serial_submit_slot(self, tool, attempt):
            result = self._execute_submit(False, tool, attempt)
        if self.preview_submits >= MAX_PREVIEW_SUBMITS:
            self.final_called = True
        return result

    def _execute_submit(
        self, final: bool, tool: str, attempt: int
    ) -> dict[str, Any]:
        self.event(
            "submit_started",
            tool=tool,
            attempt=attempt,
            timeout_seconds=self.timeout_seconds,
        )
        if final:
            RUNTIME_ROOT.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=f"hy3-final-{self.task.name}-", dir=RUNTIME_ROOT
            ) as temporary:
                stage = Path(temporary)
                source = self.scratch / "train.py"
                if source.is_file():
                    shutil.copy2(source, stage / "train.py")
                make_final_public(self.task, self.metadata, stage / "public")
                result = score_train(
                    self.task,
                    stage,
                    self.timeout_seconds,
                    self.gpu,
                    f"{self.cid_label}-final",
                    resource_limits=self.metadata.get("resource_limits"),
                )
                (self.artifact_dir / "train.py").write_bytes(
                    source.read_bytes() if source.is_file() else b""
                )
            (self.artifact_dir / "stdout.txt").write_text(result.get("stdout", ""))
            result.update(kind="final", attempt=1, timestamp=utc_now())
            with self.lock:
                self.final_result = result
                self.scores.append(result)
        else:
            source = self.scratch / "train.py"
            result = score_train(
                self.task,
                self.scratch,
                self.timeout_seconds,
                self.gpu,
                f"{self.cid_label}-submit",
                resource_limits=self.metadata.get("resource_limits"),
            )
            code = source.read_text(errors="replace") if source.is_file() else ""
            attempt_dir = self.artifact_dir / "submissions"
            attempt_dir.mkdir(exist_ok=True)
            if code:
                (attempt_dir / f"attempt-{attempt}.py").write_text(code)
            submission = self.scratch / "submission/submission.csv"
            if submission.is_file():
                shutil.copy2(submission, attempt_dir / f"attempt-{attempt}.csv")
            result.update(
                kind="submit",
                attempt=attempt,
                timestamp=utc_now(),
                code=code,
                code_hash=(hashlib.sha256(code.encode()).hexdigest() if code else None),
            )
            with self.lock:
                self.scores.append(result)
        self.event(
            "tool_result",
            tool=tool,
            status=result["status"],
            score=result.get("score"),
            error=result.get("error"),
            duration_seconds=result.get("duration_seconds"),
            return_code=result.get("return_code"),
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
        )
        return {
            "text": format_execution_result(result, self),
            "is_error": result["status"] == "fail",
        }

    def finalize_missing(self, reason: str) -> None:
        if self.final_result is not None:
            return
        scored = [
            item
            for item in self.scores
            if item.get("status") == "scored"
            and isinstance(item.get("score"), (int, float))
            and math.isfinite(float(item["score"]))
        ]
        if scored:
            choose = min if self.metadata.get("is_lower_better") else max
            selected_score = choose(float(item["score"]) for item in scored)
            selected = next(
                item for item in scored if float(item["score"]) == selected_score
            )
            self.final_result = dict(selected, kind="selected_best")
            (self.artifact_dir / "train.py").write_text(selected.get("code", ""))
            (self.artifact_dir / "stdout.txt").write_text(selected.get("stdout", ""))
            self.event(
                "best_submission_selected",
                attempt=selected.get("attempt"),
                score=selected_score,
            )
            return
        source = self.scratch / "train.py"
        (self.artifact_dir / "train.py").write_bytes(
            source.read_bytes() if source.is_file() else b""
        )
        (self.artifact_dir / "stdout.txt").write_text("")
        self.final_result = {
            "kind": "terminal",
            "attempt": None,
            "timestamp": utc_now(),
            "status": "fail",
            "score": None,
            "error": reason,
            "stdout": "",
            "stderr": "",
            "duration_seconds": 0.0,
            "return_code": None,
        }
        self.event("run_without_valid_submission", error=reason)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run one non-interactive shell command in the isolated offline workspace. "
                "Only files under the current workspace persist between calls; /tmp is ephemeral. "
                "The hard wall-clock limit is 180 seconds. Returns exit status, stdout, stderr, "
                "and explicit live turn/submit counters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Complete shell command to execute.",
                    }
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Consume one submit attempt: snapshot and run train.py with no arguments, require "
                "submission/submission.csv, and return stdout, stderr, and score or error."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


def tool_names() -> set[str]:
    return {tool["function"]["name"] for tool in TOOLS}


def api_settings(backend: str, model_override: str | None) -> dict[str, Any]:
    if backend == "hy3":
        key = os.environ.get("OPENROUTER_KEY_hy3")
        if not key:
            raise ConfigurationBlocked("OPENROUTER_KEY_hy3 is required for Hy3 runs")
        return {
            "url": OPENROUTER_CHAT_URL,
            "key": key,
            "model": model_override or DEFAULT_HY3_MODEL,
        }
    if backend == "claude":
        # Deliberately do not fall back to OPENROUTER_KEY_hy3: Claude runs
        # must use the independently managed general OpenRouter account.
        key = os.environ.get("OPENROUTER_KEY")
        if not key:
            raise ConfigurationBlocked(
                "OPENROUTER_KEY is required for Claude runs; "
                "OPENROUTER_KEY_hy3 is not accepted"
            )
        return {
            "url": OPENROUTER_CHAT_URL,
            "key": key,
            "model": model_override or DEFAULT_CLAUDE_MODEL,
        }
    if backend == "sonnet":
        key = os.environ.get("OPENROUTER_KEY")
        if not key:
            raise ConfigurationBlocked(
                "OPENROUTER_KEY is required for Sonnet runs; "
                "OPENROUTER_KEY_hy3 is not accepted"
            )
        return {
            "url": OPENROUTER_CHAT_URL,
            "key": key,
            "model": model_override or DEFAULT_SONNET_MODEL,
        }
    if backend == "grok":
        # Grok uses the general OpenRouter account only.  Keeping this branch
        # explicit prevents an accidental fallback to the Hy3-specific key.
        key = os.environ.get("OPENROUTER_KEY")
        if not key:
            raise ConfigurationBlocked(
                "OPENROUTER_KEY is required for Grok runs; "
                "OPENROUTER_KEY_hy3 is not accepted"
            )
        return {
            "url": OPENROUTER_CHAT_URL,
            "key": key,
            "model": model_override or DEFAULT_GROK_MODEL,
        }
    if backend == "gemini":
        key = os.environ.get("OPENROUTER_KEY")
        if not key:
            raise ConfigurationBlocked(
                "OPENROUTER_KEY is required for Gemini runs; "
                "OPENROUTER_KEY_hy3 is not accepted"
            )
        return {
            "url": OPENROUTER_CHAT_URL,
            "key": key,
            "model": model_override or DEFAULT_GEMINI_MODEL,
        }
    if backend == "gemini-pro":
        key = os.environ.get("OPENROUTER_KEY")
        if not key:
            raise ConfigurationBlocked(
                "OPENROUTER_KEY is required for Gemini Pro runs; "
                "OPENROUTER_KEY_hy3 is not accepted"
            )
        return {
            "url": OPENROUTER_CHAT_URL,
            "key": key,
            "model": model_override or DEFAULT_GEMINI_PRO_MODEL,
        }
    if backend == "glm":
        key = os.environ.get("OPENROUTER_KEY")
        if not key:
            raise ConfigurationBlocked(
                "OPENROUTER_KEY is required for GLM runs; "
                "OPENROUTER_KEY_hy3 is not accepted"
            )
        return {
            "url": OPENROUTER_CHAT_URL,
            "key": key,
            "model": model_override or DEFAULT_GLM_MODEL,
        }
    key = os.environ.get("OPENAI_KEY")
    if not key:
        raise ConfigurationBlocked("OPENAI_KEY is required for OpenAI runs")
    return {
        "url": "https://api.openai.com/v1/responses",
        "key": key,
        "model": model_override or DEFAULT_OPENAI_MODEL,
    }


def model_request_key(state: RunState) -> str:
    identity = "|".join(
        (
            str(state.task),
            state.model_name,
            state.backend,
            str(state.run_number),
            state.started_at,
            str(state.turns_used + 1),
        )
    )
    return "hy3-" + hashlib.sha256(identity.encode()).hexdigest()


def request_model(
    state: RunState,
    settings: dict[str, Any],
    messages: list[dict[str, Any]],
    thinking: str,
) -> dict[str, Any]:
    request_tools = TOOLS
    tool_choice = "auto"
    request_messages, context_stats = bounded_chat_messages(messages)
    if (
        context_stats["omitted_groups"]
        or context_stats["request_characters"] < context_stats["original_characters"]
    ):
        state.event("context_compacted", **context_stats)
    payload: dict[str, Any] = {
        "model": settings["model"],
        "messages": request_messages,
        "tools": request_tools,
        "tool_choice": tool_choice,
    }
    if state.backend == "hy3":
        payload["provider"] = {
            "only": [ATLAS_ROUTE],
            "quantizations": ["fp8"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
        if thinking != "none":
            payload["reasoning"] = {"effort": thinking}
    elif state.backend == "glm":
        if thinking != "none":
            payload["reasoning"] = {"effort": thinking}
    elif state.backend in {"claude", "sonnet", "grok", "gemini", "gemini-pro"}:
        if thinking != "none":
            payload["reasoning"] = {"effort": thinking}
    elif thinking != "none":
        payload["reasoning_effort"] = thinking
    headers = {
        "Authorization": f"Bearer {settings['key']}",
        "Content-Type": "application/json",
        "Idempotency-Key": model_request_key(state),
    }
    last_error = "unknown provider error"
    attempts_made = 0
    for attempt in range(1, MAX_PROVIDER_RETRIES + 1):
        attempts_made = attempt
        state.event(
            "api_request",
            attempt=attempt,
            next_turn=state.turns_used + 1,
            tool_choice="auto",
        )
        try:
            response = requests.post(
                settings["url"], headers=headers, json=payload, timeout=900
            )
            body = response.json()
        except (requests.RequestException, requests.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            retryable = True
        else:
            if response.status_code == 200:
                choices = body.get("choices") or []
                first_choice = choices[0] if choices else {}
                if not choices or first_choice.get("finish_reason") == "error":
                    last_error = str(
                        first_choice.get("error")
                        or body.get("error")
                        or "provider returned an error completion"
                    )
                    retryable = True
                else:
                    if (
                        state.backend == "hy3"
                        and "atlas" not in str(body.get("provider", "")).lower()
                    ):
                        raise ConfigurationBlocked(
                            f"Hy3 unexpectedly routed through {body.get('provider')!r}, not AtlasCloud"
                        )
                    state.event(
                        "api_response",
                        provider=body.get("provider"),
                        model=body.get("model"),
                        usage=body.get("usage"),
                        retry_attempt=attempt,
                    )
                    return body
            else:
                error = body.get("error", {}) if isinstance(body, dict) else {}
                last_error = (
                    error.get("message", str(error))
                    if isinstance(error, dict)
                    else str(error)
                )
                retryable = (
                    response.status_code in {408, 409, 429}
                    or response.status_code >= 500
                )
        state.event(
            "provider_error", attempt=attempt, error=last_error, retryable=retryable
        )
        if not retryable:
            raise ConfigurationBlocked(
                f"provider rejected the request after {attempts_made} attempt(s): {last_error}"
            )
        if attempt == MAX_PROVIDER_RETRIES:
            raise ProviderPaused(
                f"provider paused after {attempts_made} attempt(s): {last_error}"
            )
        delay = random.SystemRandom().randint(60, 300)
        state.event(
            "provider_retry_scheduled",
            failed_attempt=attempt,
            next_attempt=attempt + 1,
            delay_seconds=delay,
        )
        time.sleep(delay)
    raise AssertionError("unreachable provider retry loop")


def responses_tools(final_turn: bool) -> list[dict[str, Any]]:
    selected = TOOLS
    converted = []
    for tool in selected:
        function = tool["function"]
        converted.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function["description"],
                "parameters": function["parameters"],
                "strict": True,
            }
        )
    return converted


def request_openai_model(
    state: RunState,
    settings: dict[str, Any],
    instructions: str,
    inputs: list[dict[str, Any]],
    thinking: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": settings["model"],
        "instructions": instructions,
        "input": inputs,
        "tools": responses_tools(False),
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }
    if thinking != "none":
        payload["reasoning"] = {"effort": thinking}
    headers = {
        "Authorization": f"Bearer {settings['key']}",
        "Content-Type": "application/json",
        "Idempotency-Key": model_request_key(state),
    }
    last_error = "unknown provider error"
    attempts_made = 0
    for attempt in range(1, MAX_PROVIDER_RETRIES + 1):
        attempts_made = attempt
        state.event(
            "api_request",
            attempt=attempt,
            next_turn=state.turns_used + 1,
            transport="responses",
            tool_choice="auto",
        )
        try:
            response = requests.post(
                settings["url"], headers=headers, json=payload, timeout=900
            )
            body = response.json()
        except (requests.RequestException, requests.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            retryable = True
        else:
            if response.status_code == 200 and body.get("status") == "completed":
                state.event(
                    "api_response",
                    provider="OpenAI",
                    model=body.get("model"),
                    usage=body.get("usage"),
                    retry_attempt=attempt,
                    response_id=body.get("id"),
                )
                return body
            error = body.get("error", {}) if isinstance(body, dict) else {}
            last_error = (
                error.get("message", str(error))
                if isinstance(error, dict)
                else str(error)
            )
            if not last_error or last_error == "None":
                last_error = (
                    f"response status {body.get('status')!r}: "
                    f"{body.get('incomplete_details')!r}"
                )
            retryable = (
                response.status_code in {408, 409, 429}
                or response.status_code >= 500
                or (
                    response.status_code == 200
                    and body.get("status") in {"failed", "incomplete"}
                )
            )
        state.event(
            "provider_error", attempt=attempt, error=last_error, retryable=retryable
        )
        if not retryable:
            raise ConfigurationBlocked(
                f"provider rejected the request after {attempts_made} attempt(s): {last_error}"
            )
        if attempt == MAX_PROVIDER_RETRIES:
            raise ProviderPaused(
                f"provider paused after {attempts_made} attempt(s): {last_error}"
            )
        delay = random.SystemRandom().randint(60, 300)
        state.event(
            "provider_retry_scheduled",
            failed_attempt=attempt,
            next_attempt=attempt + 1,
            delay_seconds=delay,
        )
        time.sleep(delay)
    raise AssertionError("unreachable provider retry loop")


def assistant_message_for_history(message: dict[str, Any]) -> dict[str, Any]:
    keep = {"role", "content", "tool_calls", "reasoning_details"}
    result = {key: value for key, value in message.items() if key in keep}
    result.setdefault("role", "assistant")
    result.setdefault("content", None)
    return result


def excerpt(value: str, limit: int, label: str) -> str:
    if len(value) <= limit:
        return value
    marker = f"\n...[{label}; {len(value) - limit} characters omitted]...\n"
    usable = max(0, limit - len(marker))
    before = usable // 2
    after = usable - before
    return value[:before] + marker + (value[-after:] if after else "")


def model_tool_result(value: str) -> str:
    return excerpt(
        value,
        MAX_MODEL_TOOL_RESULT_CHARS,
        "tool output excerpted for model context; full output remains in trajectory artifacts",
    )


def bounded_chat_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build a coherent, bounded request history without altering saved artifacts."""

    normalized: list[dict[str, Any]] = []
    for original in messages:
        message = copy.deepcopy(original)
        content = message.get("content")
        if message.get("role") == "tool" and isinstance(content, str):
            message["content"] = model_tool_result(content)
        elif message.get("role") == "assistant" and isinstance(content, str):
            message["content"] = excerpt(
                content, 4_000, "older assistant text excerpted"
            )
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if (
                isinstance(arguments, str)
                and len(arguments) > MAX_HISTORICAL_ARGUMENT_CHARS
            ):
                function["arguments"] = json.dumps(
                    {
                        "historical_arguments_excerpt": excerpt(
                            arguments,
                            MAX_HISTORICAL_ARGUMENT_CHARS - 200,
                            "older tool arguments excerpted",
                        ),
                        "note": "This historical call already executed; full arguments are in trajectory artifacts.",
                    }
                )
        normalized.append(message)

    base = normalized[:2]
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in normalized[2:]:
        if message.get("role") == "user" and current:
            groups.append(current)
            current = []
        current.append(message)
    if current:
        groups.append(current)

    def encoded_size(items: list[dict[str, Any]]) -> int:
        return len(json.dumps(items, ensure_ascii=False, default=str))

    total = encoded_size(base)
    selected: list[list[dict[str, Any]]] = []
    for group in reversed(groups):
        size = encoded_size(group)
        if selected and total + size > CHAT_HISTORY_CHAR_BUDGET:
            break
        selected.append(group)
        total += size
    selected.reverse()
    omitted_groups = len(groups) - len(selected)
    result = list(base)
    if omitted_groups:
        result.append(
            {
                "role": "user",
                "content": (
                    f"Context-management notice: {omitted_groups} older model/tool exchange group(s) "
                    "were omitted to stay within the provider context limit. Workspace files persist, "
                    "and full commands and outputs remain in the trajectory artifacts. Inspect the "
                    "workspace again if an omitted detail is needed."
                ),
            }
        )
    for group in selected:
        result.extend(group)
    return result, {
        "original_messages": len(messages),
        "original_characters": encoded_size(messages),
        "request_messages": len(result),
        "omitted_groups": omitted_groups,
        "request_characters": encoded_size(result),
    }


def dispatch_tool(state: RunState, name: str, arguments: Any) -> dict[str, Any]:
    if name not in tool_names():
        return {"text": f"unknown tool: {name}", "is_error": True}
    try:
        parsed = (
            json.loads(arguments or "{}") if isinstance(arguments, str) else arguments
        )
    except json.JSONDecodeError as exc:
        return {"text": f"invalid tool arguments: {exc}", "is_error": True}
    if not isinstance(parsed, dict):
        return {"text": "tool arguments must be an object", "is_error": True}
    if name == "bash":
        return state.bash(parsed.get("command"))
    return state.submit(final=False)


def mark_tool_inflight(
    state: RunState, conversation: dict[str, Any], call_index: int
) -> None:
    """Durably mark a tool call before launching its potentially costly work."""
    conversation["phase"] = "tool_inflight"
    conversation["inflight_call_index"] = call_index
    path = state.artifact_dir / "checkpoint.json"
    payload = json.loads(path.read_text())
    payload["saved_at"] = utc_now()
    payload["phase"] = "tool_inflight"
    payload["state"] = state_checkpoint_payload(state)
    payload["conversation"] = conversation
    atomic_write_text(path, json.dumps(payload, indent=2, default=str) + "\n")


def recover_interrupted_tool(
    state: RunState, conversation: dict[str, Any], transport: str
) -> None:
    """Skip an uncertain in-flight tool on resume instead of running it twice."""
    calls = conversation.get("pending_calls") or []
    index = int(conversation.get("inflight_call_index", conversation.get("next_call", 0)))
    if index >= len(calls):
        raise HarnessError("checkpoint has an invalid in-flight tool index")
    call = calls[index]
    call_id = call.get("id") or f"tool-{uuid.uuid4().hex}"
    function = call.get("function") or {}
    name = function.get("name", "")
    message = (
        "The harness was interrupted while this tool was in flight. It was not "
        "replayed, because its prior execution may already have consumed compute "
        "or produced external effects. Inspect persistent workspace files before "
        "deciding whether a new call is needed."
    )
    state.event(
        "tool_interrupted_not_retried",
        turn=state.turns_used,
        tool=name,
        call_index=index,
    )
    if transport == "responses":
        conversation["history"].append(
            {"type": "function_call_output", "call_id": call_id, "output": message}
        )
    else:
        conversation["history"].append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": message,
            }
        )
    conversation["next_call"] = index + 1
    conversation.pop("inflight_call_index", None)
    conversation["phase"] = (
        "ready_request" if conversation["next_call"] >= len(calls) else "tools_pending"
    )
    save_checkpoint(state, conversation, conversation["phase"])


def run_openai_conversation(
    state: RunState,
    settings: dict[str, Any],
    thinking: str,
    checkpoint: dict[str, Any] | None,
) -> None:
    prompt = system_prompt(state.metadata, state.turn_limit)
    validate_prompt(prompt, state.turn_limit)
    if checkpoint is None:
        conversation = {
            "transport": "responses",
            "phase": "ready_request",
            "history": [
                {
                    "role": "user",
                    "content": (state.task / "public/description.md").read_text(),
                }
            ],
            "pending_calls": [],
            "next_call": 0,
        }
        save_checkpoint(state, conversation, conversation["phase"])
    else:
        conversation = checkpoint["conversation"]
        if conversation.get("transport") != "responses":
            raise HarnessError("checkpoint transport is not OpenAI Responses")
        if conversation.get("phase") == "tool_inflight":
            recover_interrupted_tool(state, conversation, "responses")
    while state.turns_used < state.turn_limit and not state.final_called:
        phase = conversation.get("phase")
        if phase == "ready_request":
            conversation["history"].append(
                {
                    "role": "user",
                    "content": budget_status(state, before_next_turn=True),
                }
            )
            conversation["phase"] = "request_inflight"
            save_checkpoint(state, conversation, "request_inflight")
            phase = "request_inflight"
        if phase == "request_inflight":
            body = request_openai_model(
                state,
                settings,
                prompt,
                conversation["history"],
                thinking,
            )
            response_id = body.get("id")
            if not isinstance(response_id, str) or not response_id:
                raise HarnessError("OpenAI response did not contain a response id")
            output = body.get("output") or []
            content_parts: list[str] = []
            calls: list[dict[str, Any]] = []
            for item in output:
                if item.get("type") == "message":
                    for part in item.get("content") or []:
                        if part.get("type") == "output_text":
                            content_parts.append(part.get("text", ""))
                elif item.get("type") == "function_call":
                    calls.append(
                        {
                            "id": item.get("call_id") or item.get("id"),
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", "{}"),
                            },
                        }
                    )
            state.turns_used += 1
            content = "\n".join(part for part in content_parts if part) or None
            conversation["history"].extend(output)
            conversation["pending_calls"] = calls
            conversation["next_call"] = 0
            conversation["phase"] = "tools_pending" if calls else "ready_request"
            state.event(
                "assistant_response",
                turn=state.turns_used,
                content=content,
                tool_calls=calls,
                finish_reason=body.get("status"),
            )
            save_checkpoint(state, conversation, conversation["phase"])
            phase = conversation["phase"]
        if phase == "tools_pending":
            calls = conversation["pending_calls"]
            while conversation["next_call"] < len(calls) and not state.final_called:
                call = calls[conversation["next_call"]]
                call_id = call.get("id") or f"tool-{uuid.uuid4().hex}"
                function = call.get("function") or {}
                name = function.get("name", "")
                mark_tool_inflight(state, conversation, conversation["next_call"])
                result = dispatch_tool(state, name, function.get("arguments", "{}"))
                state.event(
                    "model_tool_result",
                    turn=state.turns_used,
                    tool=name,
                    is_error=result["is_error"],
                )
                conversation["history"].append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": model_tool_result(result["text"]),
                    }
                )
                conversation["next_call"] += 1
                conversation.pop("inflight_call_index", None)
                if conversation["next_call"] >= len(calls):
                    conversation["phase"] = "ready_request"
                else:
                    conversation["phase"] = "tools_pending"
                save_checkpoint(state, conversation, conversation["phase"])


def run_chat_conversation(
    state: RunState,
    settings: dict[str, Any],
    thinking: str,
    checkpoint: dict[str, Any] | None,
) -> None:
    prompt = system_prompt(state.metadata, state.turn_limit)
    validate_prompt(prompt, state.turn_limit)
    if checkpoint is None:
        conversation = {
            "transport": "chat",
            "phase": "ready_request",
            "history": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (state.task / "public/description.md").read_text(),
                },
            ],
            "pending_calls": [],
            "next_call": 0,
        }
        save_checkpoint(state, conversation, conversation["phase"])
    else:
        conversation = checkpoint["conversation"]
        if conversation.get("transport") != "chat":
            raise HarnessError("checkpoint transport is not Chat Completions")
        if conversation.get("phase") == "tool_inflight":
            recover_interrupted_tool(state, conversation, "chat")
    while state.turns_used < state.turn_limit and not state.final_called:
        phase = conversation.get("phase")
        if phase == "ready_request":
            conversation["history"].append(
                {
                    "role": "user",
                    "content": budget_status(state, before_next_turn=True),
                }
            )
            conversation["phase"] = "request_inflight"
            save_checkpoint(state, conversation, "request_inflight")
            phase = "request_inflight"
        if phase == "request_inflight":
            body = request_model(state, settings, conversation["history"], thinking)
            choices = body.get("choices") or []
            if not choices or not isinstance(choices[0].get("message"), dict):
                raise HarnessError(
                    "provider response did not contain an assistant message"
                )
            message = choices[0]["message"]
            state.turns_used += 1
            conversation["history"].append(assistant_message_for_history(message))
            calls = message.get("tool_calls") or []
            conversation["pending_calls"] = calls
            conversation["next_call"] = 0
            conversation["phase"] = "tools_pending" if calls else "ready_request"
            state.event(
                "assistant_response",
                turn=state.turns_used,
                content=message.get("content"),
                tool_calls=calls,
                finish_reason=choices[0].get("finish_reason"),
            )
            save_checkpoint(state, conversation, conversation["phase"])
            phase = conversation["phase"]
        if phase == "tools_pending":
            calls = conversation["pending_calls"]
            while conversation["next_call"] < len(calls) and not state.final_called:
                call = calls[conversation["next_call"]]
                call_id = call.get("id") or f"tool-{uuid.uuid4().hex}"
                function = call.get("function") or {}
                name = function.get("name", "")
                mark_tool_inflight(state, conversation, conversation["next_call"])
                result = dispatch_tool(state, name, function.get("arguments", "{}"))
                state.event(
                    "model_tool_result",
                    turn=state.turns_used,
                    tool=name,
                    is_error=result["is_error"],
                )
                conversation["history"].append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": model_tool_result(result["text"]),
                    }
                )
                conversation["next_call"] += 1
                conversation.pop("inflight_call_index", None)
                if conversation["next_call"] >= len(calls):
                    conversation["phase"] = "ready_request"
                else:
                    conversation["phase"] = "tools_pending"
                save_checkpoint(state, conversation, conversation["phase"])


def run_conversation(
    state: RunState,
    settings: dict[str, Any],
    thinking: str,
    checkpoint: dict[str, Any] | None,
) -> None:
    if state.backend == "openai":
        run_openai_conversation(state, settings, thinking, checkpoint)
    else:
        run_chat_conversation(state, settings, thinking, checkpoint)


def state_checkpoint_payload(state: RunState) -> dict[str, Any]:
    return {
        "started_at": state.started_at,
        "turns_used": state.turns_used,
        "preview_submits": state.preview_submits,
        "final_called": state.final_called,
        "scores": state.scores,
        "events": state.events,
        "final_result": state.final_result,
        "serial_submits": state.serial_submits,
    }


def scores_payload(state: RunState) -> dict[str, Any]:
    return {
        "task_slug": state.task.name,
        "model": state.model_name,
        "backend": state.backend,
        "provider_route": (
            ATLAS_ROUTE
            if state.backend == "hy3"
            else "openrouter"
            if state.backend in {"claude", "sonnet", "grok", "gemini", "gemini-pro"}
            else "openrouter"
            if state.backend == "glm"
            else "openai"
        ),
        "evaluation_metric": state.metadata.get("evaluation_metric"),
        "is_lower_better": state.metadata.get("is_lower_better"),
        "submits_allowed": MAX_PREVIEW_SUBMITS,
        "submissions": state.scores,
        "final_result": state.final_result,
        "final_score": (
            state.final_result.get("score") if state.final_result else None
        ),
        "final_status": (
            state.final_result.get("status") if state.final_result else "in_progress"
        ),
    }


def trajectory_payload(state: RunState) -> dict[str, Any]:
    payload = {
        "task_slug": state.task.name,
        "model": state.model_name,
        "backend": state.backend,
        "provider_route": (
            ATLAS_ROUTE
            if state.backend == "hy3"
            else "openrouter"
            if state.backend in {"claude", "sonnet", "grok", "gemini", "gemini-pro"}
            else "openrouter"
            if state.backend == "glm"
            else "openai"
        ),
        "started_at": state.started_at,
        "updated_at": utc_now(),
        "turn_limit": state.turn_limit,
        "turns_used": state.turns_used,
        "submit_limit": MAX_PREVIEW_SUBMITS,
        "submits_used": state.preview_submits,
        "terminal_boundary_reached": state.final_called,
        "events": state.events,
    }
    if state.final_result is not None:
        payload["finished_at"] = utc_now()
    return payload


def persist_progress(state: RunState) -> None:
    if not state.artifact_dir.is_dir():
        return
    atomic_write_text(
        state.artifact_dir / "scores.json",
        json.dumps(scores_payload(state), indent=2, default=str) + "\n",
    )
    atomic_write_text(
        state.artifact_dir / "trajectory.json",
        json.dumps(trajectory_payload(state), indent=2, default=str) + "\n",
    )
    source = state.scratch / "train.py"
    artifact_train = state.artifact_dir / "train.py"
    if source.is_file():
        atomic_write_bytes(artifact_train, source.read_bytes())
    elif not artifact_train.exists():
        atomic_write_bytes(artifact_train, b"")
    latest_stdout = state.scores[-1].get("stdout", "") if state.scores else ""
    atomic_write_text(state.artifact_dir / "stdout.txt", latest_stdout)


def workspace_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = {"public", ".tool-home", ".tool-tmp"}
    ignored.update(name for name in names if name.startswith(".harness-"))
    return ignored.intersection(names)


def save_checkpoint(state: RunState, conversation: dict[str, Any], phase: str) -> None:
    checkpoint_id = uuid.uuid4().hex
    snapshots = state.artifact_dir / ".checkpoints"
    snapshots.mkdir(exist_ok=True)
    snapshot = snapshots / checkpoint_id
    shutil.copytree(
        state.scratch,
        snapshot,
        symlinks=True,
        ignore=workspace_ignore,
    )
    payload = {
        "version": CHECKPOINT_VERSION,
        "checkpoint_id": checkpoint_id,
        "saved_at": utc_now(),
        "phase": phase,
        "scratch_path": str(state.scratch),
        "identity": {
            "task": str(state.task),
            "model_name": state.model_name,
            "backend": state.backend,
            "api_model": state.api_model,
            "run": state.run_number,
            "turn_limit": state.turn_limit,
        },
        "state": state_checkpoint_payload(state),
        "conversation": conversation,
    }
    atomic_write_text(
        state.artifact_dir / "checkpoint.json",
        json.dumps(payload, indent=2, default=str) + "\n",
    )
    for child in snapshots.iterdir():
        if child.name != checkpoint_id:
            shutil.rmtree(child, ignore_errors=True)
    persist_progress(state)
    terminal_event(
        {
            "timestamp": utc_now(),
            "type": "checkpoint_saved",
            "phase": phase,
            "checkpoint_id": checkpoint_id,
        },
    )


def refresh_checkpoint_state(state: RunState) -> None:
    path = state.artifact_dir / "checkpoint.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text())
    payload["saved_at"] = utc_now()
    payload["state"] = state_checkpoint_payload(state)
    atomic_write_text(path, json.dumps(payload, indent=2, default=str) + "\n")


def record_checkpoint_interruption(state: RunState, reason: str) -> None:
    path = state.artifact_dir / "checkpoint.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text())
    payload.setdefault("interruptions", []).append(
        {"timestamp": utc_now(), "type": "checkpoint_interruption", "reason": reason}
    )
    atomic_write_text(path, json.dumps(payload, indent=2, default=str) + "\n")


def load_checkpoint(state: RunState) -> dict[str, Any]:
    path = state.artifact_dir / "checkpoint.json"
    if not path.is_file():
        raise HarnessError(f"no resumable checkpoint exists in {state.artifact_dir}")
    payload = json.loads(path.read_text())
    if payload.get("version") != CHECKPOINT_VERSION:
        raise HarnessError("unsupported checkpoint version")
    expected = {
        "task": str(state.task),
        "model_name": state.model_name,
        "backend": state.backend,
        "api_model": state.api_model,
        "run": state.run_number,
        "turn_limit": state.turn_limit,
    }
    anonymized_expected = {
        **expected,
        "task": f"/workspace/tasks/{state.task.name}",
    }
    checkpoint_identity = payload.get("identity")
    if checkpoint_identity not in (expected, anonymized_expected):
        raise HarnessError(
            "checkpoint identity mismatch: expected "
            f"{expected!r} (or anonymized {anonymized_expected!r}), "
            f"got {checkpoint_identity!r}"
        )
    checkpoint_id = payload.get("checkpoint_id", "")
    if not re.fullmatch(r"[0-9a-f]{32}", checkpoint_id):
        raise HarnessError("checkpoint contains an invalid workspace id")
    snapshot = state.artifact_dir / ".checkpoints" / checkpoint_id
    if not snapshot.is_dir():
        raise HarnessError("checkpoint workspace snapshot is missing")
    shutil.rmtree(state.scratch)
    shutil.copytree(snapshot, state.scratch, symlinks=True)
    (state.scratch / "public").symlink_to(
        state.task / "public", target_is_directory=True
    )
    saved = payload["state"]
    state.started_at = saved["started_at"]
    state.turns_used = saved["turns_used"]
    state.preview_submits = saved["preview_submits"]
    state.final_called = saved["final_called"]
    state.scores = saved["scores"]
    state.events = saved["events"]
    state.events.extend(payload.get("interruptions", []))
    state.final_result = saved["final_result"]
    # Preserve GPU-vs-CPU queue selection across an exact resume. The fallback
    # keeps checkpoints written before --serial was introduced compatible.
    state.serial_submits = saved.get("serial_submits", state.serial_submits)
    old_scratch = Path(payload.get("scratch_path", ""))
    if (
        old_scratch != state.scratch
        and old_scratch.parent == RUNTIME_ROOT
        and old_scratch.name.startswith(
            f"hy3-direct-{state.task.name}-{state.model_name}-{state.run_number}-"
        )
    ):
        shutil.rmtree(old_scratch, ignore_errors=True)
    persist_progress(state)
    terminal_event(
        {
            "timestamp": utc_now(),
            "type": "checkpoint_loaded",
            "phase": payload.get("phase"),
            "checkpoint_id": checkpoint_id,
        },
    )
    return payload


def clear_checkpoint(state: RunState) -> None:
    (state.artifact_dir / "checkpoint.json").unlink(missing_ok=True)
    shutil.rmtree(state.artifact_dir / ".checkpoints", ignore_errors=True)


def kill_run_containers(state: RunState) -> None:
    if not CID_ROOT.is_dir():
        return
    for cidfile in CID_ROOT.glob(f"hy3-{state.cid_label}-*.cid"):
        kill_cidfile(cidfile)


def persist(state: RunState) -> None:
    if state.final_result is None:
        if state.turns_used >= state.turn_limit:
            reason = f"trajectory reached the {state.turn_limit}-turn limit without a valid score"
        elif state.preview_submits >= MAX_PREVIEW_SUBMITS:
            reason = "trajectory consumed all submit attempts without a valid score"
        else:
            reason = "trajectory ended without a valid score"
        state.finalize_missing(reason)
    persist_progress(state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    parser.add_argument(
        "--backend",
        choices=(
            "hy3",
            "openai",
            "claude",
            "sonnet",
            "grok",
            "gemini",
            "gemini-pro",
            "glm",
        ),
        required=True,
    )
    parser.add_argument("--model-name")
    parser.add_argument("--api-model")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="exact directory in which to save this trajectory's artifacts",
    )
    parser.add_argument(
        "--run",
        type=int,
        choices=range(1, 7),
        default=1,
        metavar="{1,2,3,4,5,6,all,bonus}",
        help=(
            "run number; harness/main.py also accepts 'all' for runs 1-5 "
            "or 'bonus' for runs 1-6"
        ),
    )
    parser.add_argument(
        "--thinking", choices=("none", "low", "medium", "high"), default="high"
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--max-turns",
        type=int,
        choices=range(1, PRODUCTION_TURN_LIMIT + 1),
        default=PRODUCTION_TURN_LIMIT,
        help="hard model-turn limit (default: 30; use 5 for a short debug run)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--serial",
        action="store_true",
        help=(
            "place preview and final scoring in the capacity-one host-wide GPU "
            "queue instead of the default capacity-two CPU queue"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the exact saved conversation/workspace step for this run",
    )
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--print-system-prompt", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.force and args.resume:
        raise HarnessError("--force and --resume cannot be used together")
    load_dotenv(PROJECT_ROOT.parent / ".env")
    task_arg = args.task
    if not task_arg.exists() and len(task_arg.parts) == 1:
        task_arg = PROJECT_ROOT / "tasks" / task_arg
    task, metadata = checked_task(task_arg)
    prompt = system_prompt(metadata, args.max_turns)
    validate_prompt(prompt, args.max_turns)
    if args.print_system_prompt:
        print(prompt)
        return 0
    settings = api_settings(args.backend, args.api_model)
    default_model_names = {
        "hy3": "hy3",
        "openai": "gpt5.5-high",
        "claude": "claude-opus-4.8-high",
        "sonnet": "claude-sonnet-5-high",
        "grok": "grok-4.5-high",
        "gemini": "gemini-3.5-flash-high",
        "gemini-pro": "gemini-3.1-pro-preview-high",
        "glm": "glm-5.2-high",
    }
    model_name = args.model_name or default_model_names[args.backend]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model_name):
        raise HarnessError("model-name contains unsafe characters")
    expected_run_name = f"{model_name}-{args.run}"
    artifact_dir = (
        args.artifact_dir.resolve()
        if args.artifact_dir is not None
        else PROJECT_ROOT / "tasks-evals" / task.name / expected_run_name
    )
    if artifact_dir.name != expected_run_name:
        raise HarnessError(
            f"artifact-dir must end in {expected_run_name!r}: {artifact_dir}"
        )
    # Keep the handle alive for the lifetime of main(). This prevents two
    # supervisors or terminals from executing the same logical trajectory.
    run_lock = acquire_run_lock(artifact_dir)
    if args.resume:
        if not artifact_dir.is_dir():
            raise HarnessError(f"run does not exist for --resume: {artifact_dir}")
    else:
        if artifact_dir.exists():
            if not args.force:
                raise HarnessError(
                    f"run already exists: {artifact_dir} (pass --force or --resume)"
                )
            shutil.rmtree(artifact_dir)
        artifact_dir.parent.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir()
    RUNTIME_ROOT.mkdir(exist_ok=True)
    scratch = Path(
        tempfile.mkdtemp(
            prefix=f"hy3-direct-{task.name}-{model_name}-{args.run}-",
            dir=RUNTIME_ROOT,
        )
    )
    (scratch / "public").symlink_to(task / "public", target_is_directory=True)
    limits = metadata.get("resource_limits")
    selected_gpu = (
        args.gpu
        if not isinstance(limits, dict) or int(limits.get("gpuCount", 0)) > 0
        else None
    )
    state = RunState(
        task=task,
        metadata=metadata,
        scratch=scratch,
        artifact_dir=artifact_dir,
        model_name=model_name,
        backend=args.backend,
        run_number=args.run,
        gpu=selected_gpu,
        turn_limit=args.max_turns,
        api_model=settings["model"],
        serial_submits=args.serial,
    )
    checkpoint: dict[str, Any] | None = None
    if args.resume:
        kill_run_containers(state)
        checkpoint = load_checkpoint(state)
        state.event(
            "run_resumed",
            phase=checkpoint.get("phase"),
            checkpoint_id=checkpoint.get("checkpoint_id"),
        )
    else:
        persist_progress(state)
        state.event(
            "run_started",
            backend=state.backend,
            api_model=state.api_model,
            turn_limit=state.turn_limit,
        )
    paused_exit_code = 0

    def terminate_as_interrupt(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate_as_interrupt)
    try:
        run_conversation(state, settings, args.thinking, checkpoint)
    except KeyboardInterrupt as exc:
        paused_exit_code = 76
        reason = f"{type(exc).__name__}: interrupted"
        state.event(
            "run_paused",
            reason=reason,
            resumable=(artifact_dir / "checkpoint.json").is_file(),
        )
        record_checkpoint_interruption(state, reason)
    except ConfigurationBlocked as exc:
        paused_exit_code = 78
        reason = f"{type(exc).__name__}: {exc}"
        state.event(
            "run_configuration_blocked",
            reason=reason,
            resumable=(artifact_dir / "checkpoint.json").is_file(),
        )
        record_checkpoint_interruption(state, reason)
    except ProviderPaused as exc:
        paused_exit_code = 75
        reason = f"{type(exc).__name__}: {exc}"
        state.event(
            "run_provider_paused",
            reason=reason,
            resumable=(artifact_dir / "checkpoint.json").is_file(),
        )
        record_checkpoint_interruption(state, reason)
    except Exception as exc:
        paused_exit_code = 76
        reason = f"{type(exc).__name__}: {exc}"
        state.event(
            "run_paused",
            reason=reason,
            resumable=(artifact_dir / "checkpoint.json").is_file(),
        )
        record_checkpoint_interruption(state, reason)
    finally:
        kill_run_containers(state)
        if args.keep_scratch:
            print(f"scratch preserved at {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)
    if paused_exit_code:
        print(
            json.dumps(
                {
                    "status": "paused",
                    "resumable": (artifact_dir / "checkpoint.json").is_file(),
                    "turns_used": state.turns_used,
                },
                indent=2,
            ),
            flush=True,
        )
        return paused_exit_code
    persist(state)
    state.event(
        "run_completed",
        status=state.final_result.get("status") if state.final_result else None,
        score=state.final_result.get("score") if state.final_result else None,
        turns_used=state.turns_used,
    )
    clear_checkpoint(state)
    persist_progress(state)
    final = state.final_result or {}
    print(
        json.dumps(
            {
                "status": final.get("status"),
                "score": final.get("score"),
                "turns_used": state.turns_used,
            },
            indent=2,
        )
    )
    run_lock.close()
    return 0 if final.get("status") == "scored" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigurationBlocked as exc:
        print(f"harness configuration: {exc}", file=sys.stderr)
        raise SystemExit(78)
    except ProviderPaused as exc:
        print(f"harness provider pause: {exc}", file=sys.stderr)
        raise SystemExit(75)
    except HarnessError as exc:
        print(f"harness: {exc}", file=sys.stderr)
        raise SystemExit(2)
