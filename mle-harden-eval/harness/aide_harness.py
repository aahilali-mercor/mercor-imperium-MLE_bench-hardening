#!/usr/bin/env python3
"""One-solver-response AIDE-compatible evaluation harness."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import requests

try:
    from .direct_harness import (
        CPU_SUBMIT_LOCKS,
        OPENROUTER_CHAT_URL,
        SERIAL_SUBMIT_LOCK,
        HarnessError,
        atomic_write_text,
        checked_task,
        excerpt,
        load_dotenv,
        score_train,
        utc_now,
    )
except ImportError:
    from direct_harness import (  # type: ignore[no-redef]
        CPU_SUBMIT_LOCKS,
        OPENROUTER_CHAT_URL,
        SERIAL_SUBMIT_LOCK,
        HarnessError,
        atomic_write_text,
        checked_task,
        excerpt,
        load_dotenv,
        score_train,
        utc_now,
    )


HARNESS_ROOT = Path(__file__).resolve().parent
PIPELINE_ROOT = HARNESS_ROOT.parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent
# Vercel AI Gateway: OpenAI-compatible chat completions. Used for models whose
# OpenRouter route is blocked by the account's ZDR-only data policy (qwen).
VERCEL_CHAT_URL = "https://ai-gateway.vercel.sh/v1/chat/completions"
RUNTIME_ROOT = WORKSPACE_ROOT / ".runtime" / "aide"
PROMPT_ROOT = PIPELINE_ROOT / "prompts" / "evaluations"
CHECKPOINT_VERSION = 1
MAX_PROVIDER_ATTEMPTS = 4
PROVIDER_TIMEOUT_SECONDS = 900
REVIEW_OUTPUT_LIMIT = 240_000


@dataclass(frozen=True)
class ModelProfile:
    logical_name: str
    provider: str
    model_id: str
    reasoning: str
    credential: str


SOLVER_PROFILES = {
    # Same logical name as the old OpenRouter route so artifact layouts and
    # score collection stay uniform; only the transport changed.
    "qwen3.6-plus-high": ModelProfile(
        "qwen3.6-plus-high",
        "vercel",
        os.environ.get("QWEN_VERCEL_MODEL", "alibaba/qwen3.6-plus"),
        "high",
        "AI_GATEWAY_API_KEY",
    ),
    "gpt5.5-high": ModelProfile(
        "gpt5.5-high", "openai", "gpt-5.5", "high", "OPENAI_KEY"
    ),
    "claude-opus-4.8-high": ModelProfile(
        "claude-opus-4.8-high",
        "openrouter",
        "anthropic/claude-opus-4.8",
        "high",
        "OPENROUTER_KEY",
    ),
}

REVIEWER_PROFILE = ModelProfile(
    "gpt5.6-luna-medium",
    "openai",
    "gpt-5.6-luna",
    "medium",
    "OPENAI_KEY",
)


class ProviderPaused(HarnessError):
    """A transient provider request exhausted its retry cycle."""


class ConfigurationBlocked(HarnessError):
    """A non-transient provider or harness configuration failed."""


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _profile_key(profile: ModelProfile) -> str:
    key = os.environ.get(profile.credential)
    if not key:
        raise ConfigurationBlocked(f"{profile.credential} is required for {profile.logical_name}")
    return key


def _replace_tokens(template: str, replacements: dict[str, object]) -> str:
    result = template
    for name, value in replacements.items():
        result = result.replace("{{" + name + "}}", str(value))
    unresolved = sorted(set(re.findall(r"\{\{[A-Za-z0-9_]+\}\}", result)))
    if unresolved:
        raise ConfigurationBlocked(f"Unresolved prompt tokens: {unresolved}")
    return result


def render_solver_prompt(task: Path, metadata: dict[str, Any]) -> str:
    wall_minutes = int(metadata["wall_clock_limit_minutes"])
    hours, minutes = divmod(wall_minutes, 60)
    time_text = f"{hours}hrs {minutes}mins" if hours else f"{minutes}mins"
    template = (PROMPT_ROOT / "aide-solver.md").read_text(encoding="utf-8")
    return _replace_tokens(
        template,
        {
            "compute_description": metadata.get(
                "compute_description",
                "20 CPU cores, 128 GiB RAM, and one NVIDIA RTX 6000 Ada GPU.",
            ),
            "wall_minutes": wall_minutes,
            "wall_hours_minutes": time_text,
            "task_description": (task / "public" / "description.md").read_text(
                encoding="utf-8"
            ),
            "data_overview": metadata.get("data_overview", "See the files under ./input/.")
        },
    )


def render_reviewer_prompt(
    task: Path,
    metadata: dict[str, Any],
    code: str,
    execution: dict[str, Any],
) -> str:
    output = "\n".join(
        [
            f"status: {execution.get('status')}",
            f"duration_seconds: {execution.get('duration_seconds')}",
            f"return_code: {execution.get('return_code')}",
            f"error: {execution.get('error') or '<none>'}",
            "stdout:",
            str(execution.get("stdout") or "<empty>"),
            "stderr:",
            str(execution.get("stderr") or "<empty>"),
            "submission_present: " + str(bool(execution.get("submission_present"))),
        ]
    )
    output = excerpt(
        output,
        REVIEW_OUTPUT_LIMIT,
        "execution output truncated for reviewer context; full output is retained",
    )
    template = (PROMPT_ROOT / "aide-reviewer.md").read_text(encoding="utf-8")
    return _replace_tokens(
        template,
        {
            "wall_minutes": metadata["wall_clock_limit_minutes"],
            "task_description": (task / "public" / "description.md").read_text(
                encoding="utf-8"
            ),
            "implementation": code.replace("```", "` ` `"),
            "execution_output": output.replace("```", "` ` `"),
        },
    )


def parse_solver_response(response: str) -> dict[str, str]:
    pattern = re.compile(
        r"\A\s*(?P<plan>[\s\S]*?)\s*\n```(?:python|py)?\s*\n(?P<code>[\s\S]*?)\n```\s*\Z",
        re.IGNORECASE,
    )
    match = pattern.fullmatch(response)
    if not match:
        raise ValueError("response must contain plan text followed by exactly one Python code block")
    plan = match.group("plan").strip()
    code = match.group("code").strip() + "\n"
    if not plan or not code.strip():
        raise ValueError("plan and Python implementation must both be non-empty")
    if "```" in plan or "```" in code:
        raise ValueError("response contains more than one code fence")
    # AIDE asks for a three-to-five-sentence sketch, but its response parser does
    # not turn stylistic noncompliance into a missing solution.  Qualification
    # must therefore execute any unambiguous plan-plus-code response and let the
    # authoritative execution determine whether the solver succeeded.
    return {"plan": plan, "code": code}


def _review_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "is_bug": {"type": "boolean"},
            "has_csv_submission": {"type": "boolean"},
            "summary": {"type": "string"},
            "metric": {"type": ["number", "null"]},
            "lower_is_better": {"type": "boolean"},
        },
        "required": [
            "is_bug",
            "has_csv_submission",
            "summary",
            "metric",
            "lower_is_better",
        ],
        "additionalProperties": False,
    }


def validate_review(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(_review_schema()["required"]):
        raise HarnessError("reviewer response does not match the required field set")
    if not isinstance(value["is_bug"], bool) or not isinstance(
        value["has_csv_submission"], bool
    ):
        raise HarnessError("reviewer boolean fields are invalid")
    if not isinstance(value["summary"], str) or not value["summary"].strip():
        raise HarnessError("reviewer summary is invalid")
    metric = value["metric"]
    if metric is not None and (
        isinstance(metric, bool)
        or not isinstance(metric, (int, float))
        or not math.isfinite(float(metric))
    ):
        raise HarnessError("reviewer metric must be finite or null")
    if not isinstance(value["lower_is_better"], bool):
        raise HarnessError("reviewer metric direction is invalid")
    return dict(value)


class Run:
    def __init__(
        self,
        task: Path,
        metadata: dict[str, Any],
        artifact_dir: Path,
        run_number: int,
        solver_profile: ModelProfile,
        gpu: str | None,
    ) -> None:
        self.task = task
        self.metadata = metadata
        self.artifact_dir = artifact_dir
        self.run_number = run_number
        self.solver_profile = solver_profile
        self.gpu = gpu
        self.events: list[dict[str, Any]] = []
        self.provider_attempts = 0

    @property
    def checkpoint_path(self) -> Path:
        return self.artifact_dir / "checkpoint.json"

    def event(self, kind: str, **fields: Any) -> None:
        event = {"timestamp": utc_now(), "type": kind, **fields}
        self.events.append(event)
        print(json.dumps(event, ensure_ascii=False, default=str), flush=True)
        self.persist_trajectory()

    def identity(self) -> dict[str, Any]:
        return {
            "task": str(self.task),
            "run": self.run_number,
            "solverProfile": self.solver_profile.logical_name,
            "solverModel": self.solver_profile.model_id,
            "reviewerProfile": REVIEWER_PROFILE.logical_name,
            "reviewerModel": REVIEWER_PROFILE.model_id,
        }

    def save(self, phase: str, **fields: Any) -> dict[str, Any]:
        old: dict[str, Any] = {}
        if self.checkpoint_path.is_file():
            old = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        payload = {
            **old,
            "version": CHECKPOINT_VERSION,
            "identity": self.identity(),
            "phase": phase,
            "updatedAt": utc_now(),
            "providerAttempts": self.provider_attempts,
            "events": self.events,
            **fields,
        }
        if phase not in {"provider_paused", "configuration_blocked"} and not phase.endswith(
            "_retry_wait"
        ):
            for transient in (
                "blockedStage",
                "pauseReason",
                "retryAt",
                "retryDelaySeconds",
            ):
                payload.pop(transient, None)
        atomic_write_text(self.checkpoint_path, json.dumps(payload, indent=2) + "\n")
        return payload

    def load(self) -> dict[str, Any]:
        payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        if payload.get("version") != CHECKPOINT_VERSION:
            raise ConfigurationBlocked("unsupported AIDE checkpoint version")
        if payload.get("identity") != self.identity():
            raise ConfigurationBlocked("AIDE checkpoint identity mismatch")
        self.events = list(payload.get("events") or [])
        self.provider_attempts = int(payload.get("providerAttempts") or 0)
        return payload

    def persist_trajectory(self, terminal: bool = False) -> None:
        if not self.artifact_dir.is_dir():
            return
        payload = {
            "task_slug": self.task.name,
            "run_number": self.run_number,
            "solver_profile": self.solver_profile.logical_name,
            "solver_model": self.solver_profile.model_id,
            "reviewer_profile": REVIEWER_PROFILE.logical_name,
            "reviewer_model": REVIEWER_PROFILE.model_id,
            "provider_request_attempts": self.provider_attempts,
            "model_turns_used": 1 if any(e["type"] == "solver_response" for e in self.events) else 0,
            "events": self.events,
            "updated_at": utc_now(),
            "terminal": terminal,
        }
        atomic_write_text(
            self.artifact_dir / "trajectory.json", json.dumps(payload, indent=2) + "\n"
        )


def _request_key(run: Run, stage: str) -> str:
    value = json.dumps({**run.identity(), "stage": stage}, sort_keys=True).encode()
    return "imperium-aide-" + hashlib.sha256(value).hexdigest()


def _retryable(status: int) -> bool:
    return status in {408, 409, 429} or status >= 500


def _provider_request(
    run: Run,
    stage: str,
    profile: ModelProfile,
    request: Callable[[dict[str, str]], requests.Response],
) -> dict[str, Any]:
    key = _profile_key(profile)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Idempotency-Key": _request_key(run, stage),
    }
    last_error = "unknown provider error"
    for cycle_attempt in range(1, MAX_PROVIDER_ATTEMPTS + 1):
        run.provider_attempts += 1
        run.event(
            "provider_request",
            stage=stage,
            profile=profile.logical_name,
            cycle_attempt=cycle_attempt,
            total_attempt=run.provider_attempts,
        )
        try:
            response = request(headers)
            body = response.json()
        except (requests.RequestException, requests.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            can_retry = True
        else:
            if response.status_code == 200:
                failed_status = body.get("status") in {"failed", "incomplete"}
                failed_choice = any(
                    choice.get("finish_reason") == "error"
                    for choice in (body.get("choices") or [])
                    if isinstance(choice, dict)
                )
                if not failed_status and not failed_choice:
                    run.event(
                        "provider_response",
                        stage=stage,
                        profile=profile.logical_name,
                        provider=body.get("provider", profile.provider),
                        model=body.get("model"),
                        response_id=body.get("id"),
                        usage=body.get("usage"),
                    )
                    return body
                last_error = str(
                    body.get("error")
                    or body.get("incomplete_details")
                    or "provider returned an error completion"
                )
                can_retry = True
            else:
                error = body.get("error") if isinstance(body, dict) else body
                if isinstance(error, dict):
                    last_error = str(error.get("message") or error)
                else:
                    last_error = str(error)
                can_retry = _retryable(response.status_code)
        run.event(
            "provider_error",
            stage=stage,
            cycle_attempt=cycle_attempt,
            retryable=can_retry,
            error=last_error,
        )
        if not can_retry:
            run.save(
                "configuration_blocked",
                blockedStage=stage,
                pauseReason=last_error,
            )
            raise ConfigurationBlocked(f"{stage}: {last_error}")
        if cycle_attempt == MAX_PROVIDER_ATTEMPTS:
            run.save("provider_paused", blockedStage=stage, pauseReason=last_error)
            raise ProviderPaused(f"{stage} provider paused after four attempts: {last_error}")
        delay = random.SystemRandom().randint(60, 300)
        retry_at = int(time.time() * 1000) + delay * 1000
        run.save(
            stage + "_retry_wait",
            blockedStage=stage,
            retryAt=retry_at,
            retryDelaySeconds=delay,
            pauseReason=last_error,
        )
        run.event(
            "provider_retry_scheduled",
            stage=stage,
            failed_attempt=cycle_attempt,
            next_attempt=cycle_attempt + 1,
            delay_seconds=delay,
            retry_at=retry_at,
        )
        time.sleep(delay)
    raise AssertionError("unreachable provider retry loop")


def _openrouter_solver(run: Run, prompt: str) -> tuple[str, dict[str, Any]]:
    profile = run.solver_profile
    payload = {
        "model": profile.model_id,
        "messages": [{"role": "system", "content": prompt}],
        "reasoning": {"effort": profile.reasoning},
    }
    body = _provider_request(
        run,
        "solver_request",
        profile,
        lambda headers: requests.post(
            OPENROUTER_CHAT_URL,
            headers=headers,
            json=payload,
            timeout=PROVIDER_TIMEOUT_SECONDS,
        ),
    )
    choices = body.get("choices") or []
    content = choices[0].get("message", {}).get("content") if choices else None
    if not isinstance(content, str) or not content.strip():
        raise HarnessError("solver response did not contain text")
    return content, body


def _vercel_solver(run: Run, prompt: str) -> tuple[str, dict[str, Any]]:
    profile = run.solver_profile
    payload = {
        "model": profile.model_id,
        "messages": [{"role": "system", "content": prompt}],
        # OpenAI-compat field; the gateway translates per provider.
        "reasoning_effort": profile.reasoning,
    }
    body = _provider_request(
        run,
        "solver_request",
        profile,
        lambda headers: requests.post(
            VERCEL_CHAT_URL,
            headers=headers,
            json=payload,
            timeout=PROVIDER_TIMEOUT_SECONDS,
        ),
    )
    choices = body.get("choices") or []
    content = choices[0].get("message", {}).get("content") if choices else None
    if not isinstance(content, str) or not content.strip():
        raise HarnessError("solver response did not contain text")
    return content, body


def _responses_text(body: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    if not chunks:
        raise HarnessError("OpenAI response did not contain output text")
    return "\n".join(chunks)


def _openai_solver(run: Run, prompt: str) -> tuple[str, dict[str, Any]]:
    profile = run.solver_profile
    payload = {
        "model": profile.model_id,
        "instructions": prompt,
        "input": "Produce the required one-step solution response now.",
        "reasoning": {"effort": profile.reasoning},
        "store": False,
    }
    body = _provider_request(
        run,
        "solver_request",
        profile,
        lambda headers: requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=payload,
            timeout=PROVIDER_TIMEOUT_SECONDS,
        ),
    )
    return _responses_text(body), body


def request_solver(run: Run, prompt: str) -> tuple[str, dict[str, Any]]:
    if run.solver_profile.provider == "openrouter":
        return _openrouter_solver(run, prompt)
    if run.solver_profile.provider == "vercel":
        return _vercel_solver(run, prompt)
    return _openai_solver(run, prompt)


def request_review(run: Run, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = REVIEWER_PROFILE
    tool = {
        "type": "function",
        "name": "submit_review",
        "description": "Submit the structured diagnostic execution review.",
        "parameters": _review_schema(),
        "strict": True,
    }
    payload = {
        "model": profile.model_id,
        "instructions": prompt,
        "input": "Review the supplied execution and call submit_review exactly once.",
        "reasoning": {"effort": profile.reasoning},
        "tools": [tool],
        "tool_choice": {"type": "function", "name": "submit_review"},
        "parallel_tool_calls": False,
        "store": False,
    }
    body = _provider_request(
        run,
        "reviewer_request",
        profile,
        lambda headers: requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=payload,
            timeout=PROVIDER_TIMEOUT_SECONDS,
        ),
    )
    calls = [
        item
        for item in body.get("output") or []
        if item.get("type") == "function_call" and item.get("name") == "submit_review"
    ]
    if len(calls) != 1:
        raise HarnessError(f"reviewer returned {len(calls)} submit_review calls")
    try:
        value = json.loads(calls[0].get("arguments") or "{}")
    except json.JSONDecodeError as exc:
        raise HarnessError(f"reviewer returned invalid JSON arguments: {exc}") from exc
    return validate_review(value), body


@contextmanager
def execution_slot(gpu: str | None) -> Iterator[None]:
    paths = (SERIAL_SUBMIT_LOCK,) if gpu is not None else CPU_SUBMIT_LOCKS
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    handles = [path.open("a+") for path in paths]
    acquired: int | None = None
    try:
        while acquired is None:
            for index, handle in enumerate(handles):
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                acquired = index
                break
            if acquired is None:
                time.sleep(0.1)
        yield
    finally:
        if acquired is not None:
            fcntl.flock(handles[acquired].fileno(), fcntl.LOCK_UN)
        for handle in handles:
            handle.close()


def execute_solution(run: Run, code: str) -> dict[str, Any]:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"aide-{run.task.name}-{run.run_number}-", dir=RUNTIME_ROOT
    ) as temporary:
        workspace = Path(temporary)
        (workspace / "input").symlink_to(run.task / "public", target_is_directory=True)
        (workspace / "public").symlink_to(run.task / "public", target_is_directory=True)
        (workspace / "working").mkdir()
        (workspace / "train.py").write_text(code, encoding="utf-8")
        with execution_slot(run.gpu):
            result = score_train(
                run.task,
                workspace,
                int(run.metadata["wall_clock_limit_minutes"]) * 60,
                run.gpu,
                f"aide-{run.task.name}-{run.run_number}",
                resource_limits=run.metadata.get("resource_limits"),
            )
        submission = workspace / "submission" / "submission.csv"
        result["submission_present"] = submission.is_file()
        if submission.is_file():
            shutil.copy2(submission, run.artifact_dir / "submission.csv")
        return result


def _failure_code(result: dict[str, Any]) -> str:
    error = str(result.get("error") or "").lower()
    if "exceeded" in error and "wall-clock" in error:
        return "solver_timeout"
    if "did not create" in error:
        return "missing_submission"
    if "could not be scored" in error:
        return "invalid_submission"
    if "does not exist" in error:
        return "no_usable_train_py"
    return "solver_runtime_error"


def finalize(
    run: Run,
    code: str,
    execution: dict[str, Any] | None,
    review: dict[str, Any],
    parser_error: str | None = None,
) -> int:
    submissions: list[dict[str, Any]] = []
    if execution is not None:
        item = {
            "kind": "submit",
            "attempt": 1,
            "status": execution["status"],
            "score": execution.get("score"),
            "error": execution.get("error"),
            "duration_seconds": execution.get("duration_seconds", 0.0),
            "return_code": execution.get("return_code"),
            "stdout": execution.get("stdout", ""),
            "stderr": execution.get("stderr", ""),
            "code_hash": _sha256(code.encode()),
        }
        submissions.append(item)
    scored = bool(
        execution
        and execution.get("status") == "scored"
        and isinstance(execution.get("score"), (int, float))
        and math.isfinite(float(execution["score"]))
    )
    error = parser_error or (str(execution.get("error") or "") if execution else "")
    final_result = {
        "status": "scored" if scored else "fail",
        "score": float(execution["score"]) if scored and execution else None,
        "error": error,
        "failure_code": None if scored else ("solver_parser_failure" if parser_error else _failure_code(execution or {})),
    }
    payload = {
        "task_slug": run.task.name,
        "model": run.solver_profile.logical_name,
        "backend": run.solver_profile.provider,
        "reviewer": REVIEWER_PROFILE.logical_name,
        "evaluation_metric": run.metadata.get("evaluation_metric"),
        "is_lower_better": run.metadata.get("is_lower_better"),
        "submits_allowed": 1,
        "submissions": submissions,
        "review": review,
        "parser_error": parser_error,
        "final_result": final_result,
        "final_score": final_result["score"],
        "final_status": final_result["status"],
    }
    atomic_write_text(run.artifact_dir / "scores.json", json.dumps(payload, indent=2) + "\n")
    run.save("terminal", finalResult=final_result, review=review)
    run.event("run_terminal", **final_result)
    run.persist_trajectory(terminal=True)
    return 0 if scored else 1


def run_harness(run: Run, resume: bool) -> int:
    prompt = render_solver_prompt(run.task, run.metadata)
    atomic_write_text(run.artifact_dir / "solver-prompt.md", prompt)
    if resume:
        checkpoint = run.load()
        phase = str(checkpoint["phase"])
        if phase == "terminal":
            scores = json.loads((run.artifact_dir / "scores.json").read_text())
            return 0 if scores.get("final_status") == "scored" else 1
        if phase in {"provider_paused", "configuration_blocked"}:
            phase = str(checkpoint.get("blockedStage"))
            run.event("run_resumed", resumed_stage=phase)
        elif phase.endswith("_retry_wait"):
            phase = str(checkpoint.get("blockedStage"))
        elif phase == "execution_inflight":
            phase = "execute"
        checkpoint = run.save(phase)
    else:
        checkpoint = run.save("solver_request")
        run.event("run_started")
    phase = str(checkpoint["phase"])

    if phase == "solver_request":
        response, raw = request_solver(run, prompt)
        atomic_write_text(run.artifact_dir / "solver-response.txt", response)
        client_raw = {key: value for key, value in raw.items() if key not in {"authorization"}}
        atomic_write_text(
            run.artifact_dir / "solver-provider-response.json",
            json.dumps(client_raw, indent=2, default=str) + "\n",
        )
        run.event("solver_response", response_characters=len(response))
        try:
            parsed = parse_solver_response(response)
        except ValueError as exc:
            error = str(exc)
            review = {
                "status": "review_not_applicable_parser_failure",
                "reason": error,
            }
            run.event("solver_parser_failure", error=error)
            return finalize(run, "", None, review, parser_error=error)
        atomic_write_text(run.artifact_dir / "plan.txt", parsed["plan"] + "\n")
        atomic_write_text(run.artifact_dir / "train.py", parsed["code"])
        checkpoint = run.save(
            "execute",
            plan=parsed["plan"],
            codeHash=_sha256(parsed["code"].encode()),
        )
        phase = "execute"

    code = (run.artifact_dir / "train.py").read_text(encoding="utf-8")
    if phase == "execute":
        run.save("execution_inflight")
        run.event("execution_started", wall_minutes=run.metadata["wall_clock_limit_minutes"])
        execution = execute_solution(run, code)
        atomic_write_text(
            run.artifact_dir / "execution.json",
            json.dumps(execution, indent=2, default=str) + "\n",
        )
        run.event(
            "execution_complete",
            status=execution.get("status"),
            score=execution.get("score"),
            duration_seconds=execution.get("duration_seconds"),
            error=execution.get("error"),
        )
        reviewer_prompt = render_reviewer_prompt(run.task, run.metadata, code, execution)
        atomic_write_text(run.artifact_dir / "reviewer-prompt.md", reviewer_prompt)
        run.save("reviewer_request")
        phase = "reviewer_request"

    execution = json.loads((run.artifact_dir / "execution.json").read_text())
    if phase == "reviewer_request":
        reviewer_prompt = (run.artifact_dir / "reviewer-prompt.md").read_text()
        review, raw_review = request_review(run, reviewer_prompt)
        atomic_write_text(
            run.artifact_dir / "review.json", json.dumps(review, indent=2) + "\n"
        )
        atomic_write_text(
            run.artifact_dir / "reviewer-provider-response.json",
            json.dumps(raw_review, indent=2, default=str) + "\n",
        )
        run.event("review_complete", review=review)
        return finalize(run, code, execution, review)
    raise ConfigurationBlocked(f"unsupported AIDE phase: {phase}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    parser.add_argument("--solver-profile", choices=sorted(SOLVER_PROFILES), required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--gpu")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(WORKSPACE_ROOT / ".env")
    task, metadata = checked_task(args.task)
    if args.run < 1:
        raise ConfigurationBlocked("run number must be positive")
    expected_name = f"{args.solver_profile}-{args.run}"
    artifact_dir = args.artifact_dir.resolve()
    if artifact_dir.name != expected_name:
        raise ConfigurationBlocked(f"artifact directory must end in {expected_name}")
    if args.resume:
        if not artifact_dir.is_dir():
            raise ConfigurationBlocked("resume artifact directory does not exist")
    else:
        if artifact_dir.exists():
            raise ConfigurationBlocked("artifact directory already exists; use --resume")
        artifact_dir.mkdir(parents=True)
    run = Run(
        task,
        metadata,
        artifact_dir,
        args.run,
        SOLVER_PROFILES[args.solver_profile],
        args.gpu,
    )
    try:
        return run_harness(run, args.resume)
    except ProviderPaused as exc:
        run.event("run_paused", reason=str(exc), owner="provider")
        return 75
    except ConfigurationBlocked as exc:
        run.event("run_paused", reason=str(exc), owner="configuration")
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
