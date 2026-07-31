from __future__ import annotations

import json
from pathlib import Path

import pytest

import aide_harness as aide


def _task(tmp_path: Path) -> tuple[Path, dict]:
    task = tmp_path / "fixture-task"
    public = task / "public"
    public.mkdir(parents=True)
    (public / "description.md").write_text(
        "## Task\nPredict y.\n\n## Metric\nRMSE; lower is better.\n"
    )
    metadata = {
        "competition_id": task.name,
        "wall_clock_limit_minutes": 15,
        "evaluation_metric": "RMSE",
        "is_lower_better": True,
        "compute_description": "2 CPU cores.",
        "data_overview": "input/train.csv (10 rows, 2 columns: x, y)",
    }
    return task, metadata


def test_solver_parser_accepts_historical_aide_shape() -> None:
    response = """I'll build a grouped gradient-boosting model. I'll validate it with the task metric and derive stable aggregate features. I'll refit on all rows and verify the exact submission schema.

```python
from pathlib import Path
Path("submission").mkdir(exist_ok=True)
```
"""
    parsed = aide.parse_solver_response(response)
    assert parsed["plan"].count(".") == 3
    assert "Path(\"submission\")" in parsed["code"]


@pytest.mark.parametrize(
    "plan",
    [
        "One compact sentence that still supplies executable code.",
        "First compound sentence with several modelling clauses. A second sentence.",
        "# Model plan\nUse a robust tabular estimator.",
    ],
)
def test_solver_parser_does_not_promote_style_violations_to_solver_failures(
    plan: str,
) -> None:
    parsed = aide.parse_solver_response(f'{plan}\n\n```python\nprint("ok")\n```\n')
    assert parsed == {"plan": plan, "code": 'print("ok")\n'}


@pytest.mark.parametrize(
    "response",
    [
        "One. Two. Three.\n```python\nprint(1)\n```\nextra",
        "One. Two. Three.\n```python\nprint(1)\n```\n```python\nprint(2)\n```",
        "```python\nprint(1)\n```",
    ],
)
def test_solver_parser_rejects_contract_violations(response: str) -> None:
    with pytest.raises(ValueError):
        aide.parse_solver_response(response)


def test_prompt_is_fully_rendered(tmp_path: Path) -> None:
    task, metadata = _task(tmp_path)
    prompt = aide.render_solver_prompt(task, metadata)
    assert "{{" not in prompt
    assert "15 minutes" in prompt
    assert "input/train.csv" in prompt
    assert "Predict y" in prompt


def test_profiles_cover_required_solver_adapters() -> None:
    assert aide.SOLVER_PROFILES["qwen3.6-plus-high"].model_id == "qwen/qwen3.6-plus"
    assert aide.SOLVER_PROFILES["gpt5.5-high"].provider == "openai"
    assert aide.SOLVER_PROFILES["claude-opus-4.8-high"].provider == "openrouter"
    assert aide.REVIEWER_PROFILE.model_id == "gpt-5.6-luna"
    assert aide.REVIEWER_PROFILE.reasoning == "medium"


def test_four_retryable_failures_create_resumable_provider_pause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, metadata = _task(tmp_path)
    artifact = tmp_path / "qwen3.6-plus-high-1"
    artifact.mkdir()
    run = aide.Run(
        task,
        metadata,
        artifact,
        1,
        aide.SOLVER_PROFILES["qwen3.6-plus-high"],
        None,
    )
    monkeypatch.setenv("OPENROUTER_KEY", "test-only-key")
    monkeypatch.setattr(aide.time, "sleep", lambda _: None)
    monkeypatch.setattr(aide.random.SystemRandom, "randint", lambda _self, _a, _b: 60)

    class Response:
        status_code = 429

        @staticmethod
        def json() -> dict:
            return {"error": {"message": "overused key"}}

    with pytest.raises(aide.ProviderPaused):
        aide._provider_request(run, "solver_request", run.solver_profile, lambda _h: Response())
    checkpoint = json.loads(run.checkpoint_path.read_text())
    assert run.provider_attempts == 4
    assert checkpoint["phase"] == "provider_paused"
    assert checkpoint["blockedStage"] == "solver_request"
    assert [event["delay_seconds"] for event in run.events if event["type"] == "provider_retry_scheduled"] == [60, 60, 60]


def test_review_validation_does_not_accept_nonfinite_metric() -> None:
    with pytest.raises(aide.HarnessError):
        aide.validate_review(
            {
                "is_bug": False,
                "has_csv_submission": True,
                "summary": "Execution succeeded and produced a file.",
                "metric": float("nan"),
                "lower_is_better": True,
            }
        )


def test_resuming_a_pause_clears_transient_blocker_fields(tmp_path: Path) -> None:
    task, metadata = _task(tmp_path)
    artifact = tmp_path / "qwen3.6-plus-high-1"
    artifact.mkdir()
    run = aide.Run(
        task,
        metadata,
        artifact,
        1,
        aide.SOLVER_PROFILES["qwen3.6-plus-high"],
        None,
    )
    run.save(
        "configuration_blocked",
        blockedStage="solver_request",
        pauseReason="invalid credential",
    )
    resumed = run.save("solver_request")
    assert "blockedStage" not in resumed
    assert "pauseReason" not in resumed
