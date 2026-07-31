#!/usr/bin/env python3
"""Validate the bundled/default MLEbench-style task-package contract.

Repository-local contracts are authoritative. Adapt this helper when a repository
intentionally uses a different package shape or metadata vocabulary.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any


REQUIRED_METADATA = {
    "competition_id",
    "dataset",
    "category",
    "short_description",
    "is_lower_better",
    "evaluation_metric",
    "gold_solution_score",
    "score_tiers",
    "hidden_test_set",
    "hidden_test_paths",
    "wall_clock_limit_minutes",
    "task_difficulty",
}
FORBIDDEN_METADATA = {
    "baseline_score",
    "minimum_score",
    "score_tier_rule",
    "environment_constraints",
    "system_prompt_addendum",
}
SCORE_FIELDS = ("gold_solution_score",)
TIER_NAMES = ("baseline", "bronze", "silver", "gold")
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
DEFAULT_GRADER_TIMEOUT_SECONDS = 60.0
GRADER_SMOKE_CODE = r"""
import importlib.util
import math
import numbers
import sys
from pathlib import Path

import pandas as pd

grader_path = Path(sys.argv[1])
sample_path = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("_task_grader_smoke_test", grader_path)
if spec is None or spec.loader is None:
    raise RuntimeError("could not construct grader import spec")
module = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(grader_path.parent))
try:
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)
grade_fn = getattr(module, "grade", None)
if not callable(grade_fn):
    raise RuntimeError("grade is not callable")
score = grade_fn(pd.read_csv(sample_path))
if isinstance(score, bool) or not isinstance(score, numbers.Real) or not math.isfinite(float(score)):
    raise RuntimeError("grader returned a non-finite or non-numeric score")
"""


class DuplicateJsonKey(ValueError):
    pass


def unique_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("task_dir", type=Path)
    parser.add_argument(
        "--final",
        action="store_true",
        help="Require gold_solution_score and all tier thresholds to be numeric",
    )
    parser.add_argument(
        "--trust-grader-code",
        action="store_true",
        help=(
            "Execute private/grader.py in a child process and grade the sample submission. "
            "Use only for code you trust: process separation and the timeout are not sandboxing."
        ),
    )
    parser.add_argument(
        "--grader-timeout-seconds",
        type=positive_seconds,
        default=DEFAULT_GRADER_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=(
            "Timeout for --trust-grader-code "
            f"(default: {DEFAULT_GRADER_TIMEOUT_SECONDS:g})"
        ),
    )
    parser.add_argument(
        "--key-columns",
        help="Comma-separated sample-submission key columns to check for duplicates",
    )
    return parser.parse_args()


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_task_slug(task_dir: Path, errors: list[str]) -> None:
    slug = task_dir.name
    if not SLUG_RE.fullmatch(slug):
        errors.append(
            "task directory name must be a stable lowercase kebab-case slug; "
            f"got {slug!r}"
        )


def validate_no_symlinks(task_dir: Path, errors: list[str]) -> None:
    try:
        symlinks = sorted(
            str(path.relative_to(task_dir))
            for path in task_dir.rglob("*")
            if path.is_symlink()
        )
    except OSError:
        errors.append("could not scan the task tree for symlinks")
        return
    if symlinks:
        preview = symlinks[:10]
        suffix = f" (and {len(symlinks) - len(preview)} more)" if len(symlinks) > len(preview) else ""
        errors.append(f"task package must not contain symlinks: {preview}{suffix}")


def validate_metadata(task_dir: Path, errors: list[str], warnings: list[str], final: bool) -> dict[str, Any] | None:
    path = task_dir / "metadata.json"
    if not path.is_file():
        errors.append("missing metadata.json")
        return None
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object_pairs)
    except (json.JSONDecodeError, DuplicateJsonKey) as exc:
        errors.append(f"invalid metadata.json: {exc}")
        return None
    if not isinstance(metadata, dict):
        errors.append("metadata.json must contain an object")
        return None

    missing = sorted(REQUIRED_METADATA - set(metadata))
    if missing:
        errors.append(f"metadata missing required fields: {missing}")
    forbidden = sorted(FORBIDDEN_METADATA & set(metadata))
    if forbidden:
        errors.append(f"metadata contains obsolete/forbidden fields: {forbidden}")
    extra = sorted(set(metadata) - REQUIRED_METADATA)
    if extra:
        warnings.append(f"metadata contains repository-specific extra fields: {extra}")

    slug = task_dir.resolve().name
    if metadata.get("competition_id") != slug:
        errors.append(f"competition_id must exactly match directory slug {slug!r}")
    if not isinstance(metadata.get("category"), str) or not metadata["category"].strip():
        errors.append("category must be nonempty text")
    if not isinstance(metadata.get("is_lower_better"), bool):
        errors.append("is_lower_better must be a boolean")
    if not isinstance(metadata.get("dataset"), str) or not metadata.get("dataset", "").strip():
        errors.append("dataset must be a nonempty string")
    if not isinstance(metadata.get("evaluation_metric"), str) or not metadata.get("evaluation_metric", "").strip():
        errors.append("evaluation_metric must be a nonempty string")

    short = metadata.get("short_description")
    if not isinstance(short, str) or not short.strip():
        errors.append("short_description must be a nonempty one- or two-sentence string")
    else:
        sentence_count = len(re.findall(r"[.!?](?:\s|$)", short.strip()))
        if sentence_count not in (1, 2):
            warnings.append("short_description does not appear to contain exactly one or two sentences")
        if len(short) > 600:
            warnings.append("short_description is unusually long")

    wall = metadata.get("wall_clock_limit_minutes")
    if not isinstance(wall, int) or isinstance(wall, bool) or wall <= 0:
        errors.append("wall_clock_limit_minutes must be a positive integer")
    difficulty = metadata.get("task_difficulty")
    if not isinstance(difficulty, int) or isinstance(difficulty, bool) or not 1 <= difficulty <= 5:
        errors.append("task_difficulty must be an integer from 1 to 5")

    for field in SCORE_FIELDS:
        value = metadata.get(field)
        if value is not None and not is_number(value):
            errors.append(f"{field} must be finite numeric or null")
        if final and value is None:
            errors.append(f"{field} must be numeric for a final task")

    tiers = metadata.get("score_tiers")
    if not isinstance(tiers, dict):
        errors.append("score_tiers must be an object")
        return metadata
    if set(tiers) != set(TIER_NAMES):
        errors.append(f"score_tiers must contain exactly {list(TIER_NAMES)}")
    for name in TIER_NAMES:
        value = tiers.get(name)
        if value is not None and not is_number(value):
            errors.append(f"score_tiers.{name} must be finite numeric or null")
        if final and value is None:
            errors.append(f"score_tiers.{name} must be numeric for a final task")

    if all(is_number(tiers.get(name)) for name in TIER_NAMES) and isinstance(metadata.get("is_lower_better"), bool):
        values = [float(tiers[name]) for name in TIER_NAMES]
        ordered = all(a >= b for a, b in zip(values, values[1:])) if metadata["is_lower_better"] else all(
            a <= b for a, b in zip(values, values[1:])
        )
        if not ordered:
            direction = "nonincreasing" if metadata["is_lower_better"] else "nondecreasing"
            errors.append(f"tier thresholds must be {direction} in baseline→gold order")
        gold_score = metadata.get("gold_solution_score")
        if is_number(gold_score):
            clears = float(gold_score) <= values[-1] if metadata["is_lower_better"] else float(gold_score) >= values[-1]
            if not clears:
                errors.append("gold_solution_score does not clear the gold tier threshold")
            margin = abs(float(gold_score) - values[-1])
            scale = max(1.0, abs(values[-1]))
            if clears and margin / scale < 1e-4:
                warnings.append("gold solution clears the threshold by a very small relative margin")
    return metadata


def validate_hidden_test_contract(
    task_dir: Path,
    metadata: dict[str, Any],
    errors: list[str],
) -> None:
    enabled = metadata.get("hidden_test_set")
    raw_paths = metadata.get("hidden_test_paths")
    if not isinstance(enabled, bool):
        errors.append("hidden_test_set must be a boolean")
        return
    if not isinstance(raw_paths, list):
        errors.append("hidden_test_paths must be a list")
        return

    parsed_paths: list[PurePosixPath] = []
    for index, value in enumerate(raw_paths):
        if not isinstance(value, str) or not value:
            errors.append(f"hidden_test_paths[{index}] must be a nonempty string")
            continue
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or path.as_posix() != value
            or "\\" in value
            or any(part in {"", ".", ".."} for part in path.parts)
            or len(path.parts) < 2
            or path.parts[0] != "public"
        ):
            errors.append(
                f"hidden_test_paths[{index}] must be a normalized task-root-relative path below public/: {value!r}"
            )
            continue
        parsed_paths.append(path)

    if len(set(parsed_paths)) != len(parsed_paths):
        errors.append("hidden_test_paths must not contain duplicates")
    for index, first in enumerate(parsed_paths):
        for second in parsed_paths[index + 1 :]:
            if first in second.parents or second in first.parents:
                errors.append(
                    "hidden_test_paths must not contain overlapping parent/child entries: "
                    f"{first.as_posix()!r}, {second.as_posix()!r}"
                )

    hidden_root = task_dir / "hidden-test"
    if not enabled:
        if raw_paths:
            errors.append("hidden_test_paths must be empty when hidden_test_set is false")
        if hidden_root.exists():
            errors.append("hidden-test/ must not exist when hidden_test_set is false")
        return

    if not parsed_paths:
        errors.append("hidden_test_paths must be nonempty when hidden_test_set is true")
    if not hidden_root.is_dir():
        errors.append("hidden-test/ must exist as a directory when hidden_test_set is true")
        return

    for relative in parsed_paths:
        live = task_dir.joinpath(*relative.parts)
        replacement = hidden_root.joinpath(*relative.parts)
        if not live.exists():
            errors.append(f"hidden-test live path does not exist: {relative.as_posix()}")
            continue
        if not replacement.exists():
            errors.append(
                "hidden-test replacement does not exist: "
                f"hidden-test/{relative.as_posix()}"
            )
            continue
        if live.is_file() != replacement.is_file() or live.is_dir() != replacement.is_dir():
            errors.append(
                "hidden-test live/replacement kinds differ for "
                f"{relative.as_posix()}"
            )

    unlisted_payloads: list[str] = []
    for payload in hidden_root.rglob("*"):
        if not payload.is_file():
            continue
        relative = PurePosixPath(payload.relative_to(hidden_root).as_posix())
        if not any(relative == declared or declared in relative.parents for declared in parsed_paths):
            unlisted_payloads.append(relative.as_posix())
    if unlisted_payloads:
        preview = sorted(unlisted_payloads)[:10]
        suffix = f" (and {len(unlisted_payloads) - len(preview)} more)" if len(unlisted_payloads) > len(preview) else ""
        errors.append(f"hidden-test/ contains payloads outside hidden_test_paths: {preview}{suffix}")


def attribute_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def validate_solution_artifacts(
    task_dir: Path,
    errors: list[str],
    warnings: list[str],
    final: bool,
) -> None:
    solution_dir = task_dir / "solutions"
    if not solution_dir.is_dir():
        if final:
            errors.append("final task must contain solutions/train.py and solutions/README.md")
        return

    relative_entries = sorted(
        path.relative_to(solution_dir).as_posix()
        for path in solution_dir.rglob("*")
    )
    expected = ["README.md", "train.py"]
    if relative_entries != expected:
        message = f"solutions/ must contain exactly {expected}; found {relative_entries}"
        if final:
            errors.append(message)
        else:
            warnings.append(message)

    train_path = solution_dir / "train.py"
    readme_path = solution_dir / "README.md"
    if not train_path.is_file() or not readme_path.is_file():
        return

    source = train_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(train_path))
    except SyntaxError as exc:
        errors.append(f"solutions/train.py has a syntax error: {exc}")
        return

    module_docstring = ast.get_docstring(tree, clean=False)
    if not module_docstring:
        errors.append("solutions/train.py must have one concise module docstring")
    elif len(module_docstring.strip()) > 600:
        errors.append("solutions/train.py module docstring must be concise (600 characters or fewer)")

    nested_docstrings = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and ast.get_docstring(node, clean=False) is not None
    ]
    if nested_docstrings:
        errors.append(
            "solutions/train.py must not contain function/class docstrings: "
            f"{sorted(nested_docstrings)}"
        )

    forbidden_import_roots = {"argparse", "click", "fire", "subprocess", "typer"}
    forbidden_imports: set[str] = set()
    relative_imports = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            forbidden_imports.update(
                alias.name for alias in node.names if alias.name.split(".", 1)[0] in forbidden_import_roots
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative_imports = True
            if node.module and node.module.split(".", 1)[0] in forbidden_import_roots:
                forbidden_imports.add(node.module)
            if node.module == "os" and any(alias.name in {"popen", "system"} for alias in node.names):
                forbidden_imports.add("os shell function")
            if node.module == "sys" and any(alias.name in {"argv", "path"} for alias in node.names):
                forbidden_imports.add("sys CLI/import state")
    if forbidden_imports:
        errors.append(
            "solutions/train.py must not import CLI/subprocess frameworks: "
            f"{sorted(forbidden_imports)}"
        )
    if relative_imports:
        errors.append("solutions/train.py must not use relative imports")

    forbidden_attributes = {
        "os.popen",
        "os.system",
        "sys.argv",
        "sys.path",
    }
    used_forbidden_attributes = sorted(
        {
            name
            for node in ast.walk(tree)
            if (name := attribute_name(node)) in forbidden_attributes
        }
    )
    if used_forbidden_attributes:
        errors.append(
            "solutions/train.py must not expose a CLI, invoke shell commands, or mutate imports: "
            f"{used_forbidden_attributes}"
        )

    strings = [
        node.value.replace("\\", "/")
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    if "submission/submission.csv" not in strings:
        errors.append("solutions/train.py must name the exact output path submission/submission.csv")
    lower_source = source.lower()
    banned_mentions = [
        term for term in ("readme", "competition", "prior code", "previous implementation") if term in lower_source
    ]
    if banned_mentions:
        errors.append(
            "solutions/train.py contains prohibited historical/documentation mentions: "
            f"{banned_mentions}"
        )

    readme = readme_path.read_text(encoding="utf-8").strip()
    if not readme:
        errors.append("solutions/README.md must contain the concise approach/score history")
    elif len(readme) > 12000:
        warnings.append("solutions/README.md is unusually long for a concise score history")


def validate_description(task_dir: Path, errors: list[str]) -> None:
    path = task_dir / "public" / "description.md"
    if not path.is_file():
        errors.append("missing public/description.md")
        return
    text = path.read_text(encoding="utf-8")
    headings = [match.group(1).strip().lower() for match in re.finditer(r"^##\s+(.+?)\s*$", text, re.MULTILINE)]
    required = ["task", "metric", "submission format", "dataset"]
    positions: list[int] = []
    for heading in required:
        if heading not in headings:
            errors.append(f"description.md missing level-2 heading {heading!r}")
        else:
            positions.append(headings.index(heading))
    if len(positions) == len(required) and positions != sorted(positions):
        errors.append("description.md headings must appear as Task, Metric, Submission Format, Dataset")


def validate_grader_source(task_dir: Path, errors: list[str]) -> None:
    path = task_dir / "private" / "grader.py"
    if not path.is_file():
        errors.append("missing private/grader.py")
        return
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        errors.append(f"private/grader.py has a syntax error: {exc}")
        return
    async_grade_functions = [
        node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "grade"
    ]
    grade_functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "grade"
    ]
    if async_grade_functions:
        errors.append("private/grader.py grade() must be synchronous, not async")
    if len(grade_functions) != 1:
        errors.append("private/grader.py must define exactly one synchronous top-level grade(submission)")
        return

    arguments = grade_functions[0].args
    positional = [*arguments.posonlyargs, *arguments.args]
    if (
        len(positional) != 1
        or arguments.defaults
        or arguments.vararg is not None
        or arguments.kwonlyargs
        or arguments.kwarg is not None
    ):
        errors.append("grade() must accept exactly one required positional argument and no other arguments")


def validate_sample(
    task_dir: Path,
    errors: list[str],
    warnings: list[str],
    key_columns_arg: str | None,
    final: bool,
) -> None:
    path = task_dir / "public" / "sample_submission.csv"
    if not path.is_file():
        errors.append("missing public/sample_submission.csv")
        return

    key_columns = [name.strip() for name in key_columns_arg.split(",") if name.strip()] if key_columns_arg else []
    if final and not key_columns:
        errors.append("--key-columns is required with --final")
    if len(set(key_columns)) != len(key_columns):
        errors.append("--key-columns must not contain duplicate column names")
        key_columns = []

    row_count = 0
    malformed_row_count = 0
    blank_key_count = 0
    duplicate_key = False
    seen_keys: set[tuple[str, ...]] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            fieldnames = reader.fieldnames
            if not fieldnames or any(not name or not name.strip() for name in fieldnames):
                errors.append("sample_submission.csv must have nonempty column names")
                return
            if len(set(fieldnames)) != len(fieldnames):
                errors.append("sample_submission.csv has duplicate column names")

            missing = sorted(set(key_columns) - set(fieldnames))
            if missing:
                errors.append(f"requested key columns absent from sample submission: {missing}")
                key_columns = []

            for row in reader:
                row_count += 1
                if None in row or any(value is None for value in row.values()):
                    malformed_row_count += 1
                    continue
                if key_columns:
                    key = tuple(row[name] for name in key_columns)
                    if any(not value.strip() for value in key):
                        blank_key_count += 1
                        continue
                    if key in seen_keys:
                        duplicate_key = True
                    else:
                        seen_keys.add(key)
    except (OSError, UnicodeError, csv.Error) as exc:
        errors.append(f"cannot parse sample_submission.csv: {exc}")
        return
    if row_count == 0:
        errors.append("sample_submission.csv must contain prediction rows, not only a header")
        return

    if malformed_row_count:
        errors.append(
            "sample_submission.csv has "
            f"{malformed_row_count} malformed row(s) with missing or extra fields"
        )
    if blank_key_count:
        errors.append(f"sample submission has {blank_key_count} row(s) with blank key values")
    if duplicate_key:
        errors.append(f"sample submission has duplicate keys for {key_columns}")
    if not key_columns_arg and not final:
        warnings.append("sample key uniqueness not checked; pass --key-columns for task-specific coverage")


def validate_public_assets(task_dir: Path, errors: list[str], warnings: list[str]) -> None:
    public = task_dir / "public"
    private = task_dir / "private"
    if not public.is_dir():
        errors.append("missing public/ directory")
        return
    if not private.is_dir():
        errors.append("missing private/ directory")
    data_assets = [path for path in public.rglob("*") if path.is_file() and path.name not in {"description.md", "sample_submission.csv"}]
    if not data_assets:
        errors.append("public/ contains no training or test data assets")
    suspicious = [
        str(path.relative_to(public))
        for path in data_assets
        if re.search(r"(^|[_\-.])(answer|answers|ground[_-]?truth|private[_-]?label)([_\-.]|$)", path.name, re.IGNORECASE)
    ]
    if suspicious:
        warnings.append(f"participant-visible filenames look answer-bearing and require review: {suspicious}")


def grader_environment() -> dict[str, str]:
    allowed = (
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "TEMP",
        "TMP",
    )
    return {name: os.environ[name] for name in allowed if name in os.environ}


def run_trusted_grader(task_dir: Path, errors: list[str], timeout_seconds: float) -> None:
    grader_path = task_dir / "private" / "grader.py"
    sample_path = task_dir / "public" / "sample_submission.csv"
    if not grader_path.is_file() or not sample_path.is_file():
        return
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                GRADER_SMOKE_CODE,
                str(grader_path),
                str(sample_path),
            ],
            cwd=task_dir,
            env=grader_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        errors.append(
            "trusted grader smoke test timed out after "
            f"{timeout_seconds:g} seconds; child output was suppressed"
        )
        return
    except OSError:
        errors.append("trusted grader smoke test could not start; child output was suppressed")
        return

    if completed.returncode != 0:
        errors.append(
            "trusted grader smoke test failed in the child process "
            f"(exit code {completed.returncode}); child output was suppressed"
        )


def main() -> int:
    args = parse_args()
    requested_task_dir = args.task_dir.expanduser()
    if requested_task_dir.is_symlink():
        print("ERROR: task directory itself must not be a symlink", file=sys.stderr)
        return 1
    task_dir = requested_task_dir.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    if not task_dir.is_dir():
        print(f"ERROR: task directory does not exist: {task_dir}", file=sys.stderr)
        return 2

    validate_task_slug(task_dir, errors)
    validate_no_symlinks(task_dir, errors)
    metadata = validate_metadata(task_dir, errors, warnings, args.final)
    if metadata is not None:
        validate_hidden_test_contract(task_dir, metadata, errors)
    validate_solution_artifacts(task_dir, errors, warnings, args.final)
    validate_description(task_dir, errors)
    validate_grader_source(task_dir, errors)
    validate_sample(task_dir, errors, warnings, args.key_columns, args.final)
    validate_public_assets(task_dir, errors, warnings)
    if args.trust_grader_code and not errors:
        run_trusted_grader(task_dir, errors, args.grader_timeout_seconds)

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print(f"FAILED with {len(errors)} error(s) and {len(warnings)} warning(s)")
        return 1
    print(f"PASS with {len(warnings)} warning(s): {task_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
