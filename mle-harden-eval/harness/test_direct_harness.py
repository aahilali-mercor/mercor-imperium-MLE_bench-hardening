from pathlib import Path
from types import SimpleNamespace

import direct_harness as harness
import main as compatibility_main
import pytest


def test_compatibility_status_uses_canonical_four_submit_budget() -> None:
    assert compatibility_main.PREVIEW_SUBMIT_LIMIT == harness.MAX_PREVIEW_SUBMITS == 4


class _RequestState:
    def __init__(self, task: Path) -> None:
        self.task = task
        self.model_name = "gpt5.5-high"
        self.backend = "openai"
        self.run_number = 1
        self.started_at = "fixture-start"
        self.turns_used = 0
        self.events: list[dict[str, object]] = []

    def event(self, kind: str, **values: object) -> None:
        self.events.append({"type": kind, **values})


def test_missing_openai_key_is_configuration_blocked(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_KEY", raising=False)
    with pytest.raises(harness.ConfigurationBlocked):
        harness.api_settings("openai", "gpt-5.5")


def test_score_train_uses_the_container_interpreter(monkeypatch, tmp_path: Path) -> None:
    task = tmp_path / "task"
    workspace = tmp_path / "workspace"
    task.mkdir()
    workspace.mkdir()
    (workspace / "train.py").write_text("print('ok')\n", encoding="utf-8")
    seen: dict[str, str] = {}

    def execute_container(
        active_workspace: Path,
        command: str,
        timeout_seconds: int,
        gpu: str | None,
        label: str,
        resource_limits: dict[str, object] | None = None,
    ) -> dict[str, object]:
        seen["command"] = command
        submission = active_workspace / "submission"
        submission.mkdir()
        (submission / "submission.csv").write_text("id,prediction\n1,0\n", encoding="utf-8")
        return {
            "status": "ok",
            "stdout": "",
            "stderr": "",
            "duration_seconds": 0.1,
            "return_code": 0,
            "timed_out": False,
        }

    monkeypatch.setattr(harness, "execute_container", execute_container)
    result = harness.score_train(task, workspace, 60, None, grade=False)

    assert result["status"] == "ok"
    assert seen["command"] == "python train.py"


def test_docker_prefix_does_not_forward_host_python_or_library_paths(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(harness.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    for name in (
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
    ):
        monkeypatch.setenv(name, f"/host-only/{name.lower()}")

    args = harness.docker_prefix(tmp_path, None, tmp_path / "container.cid")
    rendered = " ".join(args)

    assert "/host-only/" not in rendered
    assert "CONDA_PREFIX" not in args
    assert "PYTHONPATH" not in args
    assert "LD_LIBRARY_PATH" not in args


def test_docker_prefix_enforces_contract_resources_and_gpu_assignment(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(harness.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    cpu_limits = {
        "cpuCores": 4,
        "ramBytes": 8 * 1024**3,
        "scratchBytes": 2 * 1024**3,
        "gpuCount": 0,
    }
    cpu_args = harness.docker_prefix(
        tmp_path,
        "1",
        tmp_path / "cpu.cid",
        resource_limits=cpu_limits,
    )
    rendered_cpu = " ".join(cpu_args)
    assert "--cpus 4" in rendered_cpu
    assert f"--memory {8 * 1024**3}" in rendered_cpu
    assert f"size={2 * 1024**3}" in rendered_cpu
    assert "NVIDIA_VISIBLE_DEVICES" not in rendered_cpu

    gpu_args = harness.docker_prefix(
        tmp_path,
        "1",
        tmp_path / "gpu.cid",
        resource_limits={**cpu_limits, "gpuCount": 1},
    )
    rendered_gpu = " ".join(gpu_args)
    assert "NVIDIA_VISIBLE_DEVICES=1" in rendered_gpu
    assert "CUDA_VISIBLE_DEVICES=0" in rendered_gpu


def test_nonretryable_provider_response_is_configuration_blocked(
    monkeypatch, tmp_path: Path
) -> None:
    state = _RequestState(tmp_path / "task")

    class Response:
        status_code = 401

        @staticmethod
        def json() -> dict:
            return {"error": {"message": "invalid credential"}}

    monkeypatch.setattr(harness.requests, "post", lambda *_a, **_k: Response())
    with pytest.raises(harness.ConfigurationBlocked):
        harness.request_openai_model(
            state,
            {"url": "https://example.invalid", "key": "test", "model": "fixture"},
            "instructions",
            [],
            "high",
        )
    assert len([event for event in state.events if event["type"] == "api_request"]) == 1


def test_four_retryable_provider_responses_create_provider_pause(
    monkeypatch, tmp_path: Path
) -> None:
    state = _RequestState(tmp_path / "task")

    class Response:
        status_code = 429

        @staticmethod
        def json() -> dict:
            return {"error": {"message": "overused credential"}}

    monkeypatch.setattr(harness.requests, "post", lambda *_a, **_k: Response())
    monkeypatch.setattr(harness.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(
        harness.random.SystemRandom,
        "randint",
        lambda _self, _minimum, _maximum: 60,
    )
    with pytest.raises(harness.ProviderPaused):
        harness.request_openai_model(
            state,
            {"url": "https://example.invalid", "key": "test", "model": "fixture"},
            "instructions",
            [],
            "high",
        )
    assert len([event for event in state.events if event["type"] == "api_request"]) == 4
    assert [
        event["delay_seconds"]
        for event in state.events
        if event["type"] == "provider_retry_scheduled"
    ] == [60, 60, 60]
