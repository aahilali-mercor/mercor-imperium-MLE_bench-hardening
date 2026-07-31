#!/usr/bin/env python3
"""Replace selected CSV/Parquet identifier columns with compact task-local IDs.

Unique values are ordered by a secret-keyed HMAC, not by source value or first
appearance. The script never writes a raw-value mapping. Use the same secret,
namespace, and complete input collection to reproduce a mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class IdSpec:
    column: str
    prefix: str


def parse_id_spec(value: str) -> IdSpec:
    if ":" not in value:
        raise argparse.ArgumentTypeError("ID specs must use COLUMN:PREFIX")
    column, prefix = value.split(":", 1)
    if not column or "\0" in column or not PREFIX_RE.fullmatch(prefix):
        raise argparse.ArgumentTypeError(f"Invalid ID spec: {value!r}")
    return IdSpec(column=column, prefix=prefix)


def parse_task_namespace(value: str) -> str:
    if not value or value != value.strip() or "\0" in value:
        raise argparse.ArgumentTypeError(
            "Task namespace must be nonempty, have no surrounding whitespace, and contain no NUL"
        )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="CSV or Parquet files sharing ID semantics")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--task-namespace",
        required=True,
        type=parse_task_namespace,
        help="Stable unique task/version namespace included in every ID HMAC",
    )
    parser.add_argument(
        "--id",
        dest="id_specs",
        action="append",
        type=parse_id_spec,
        required=True,
        help="Column and output prefix as COLUMN:PREFIX; repeatable",
    )
    parser.add_argument("--drop", action="append", default=[], help="Column to omit; repeatable")
    parser.add_argument(
        "--key-env",
        default="TABULAR_TASK_ID_KEY",
        help=(
            "Environment variable containing a high-entropy secret of at least 32 UTF-8 bytes; "
            "generate it randomly and keep it out of shell history"
        ),
    )
    parser.add_argument(
        "--blank-id-policy",
        choices=("preserve", "null", "error"),
        default="preserve",
        help="How to handle empty or whitespace-only IDs (default: preserve exactly)",
    )
    parser.add_argument("--manifest", type=Path, help="Write a no-raw-values JSON manifest")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace existing output files and manifest",
    )
    return parser.parse_args()


def require_polars() -> Any:
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise SystemExit("anonymize_ids.py requires polars") from exc
    return pl


def scan(path: Path, pl: Any) -> tuple[Any, str]:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pl.scan_parquet(path), "parquet"
    if suffixes.endswith(".csv"):
        return (
            pl.scan_csv(
                path,
                infer_schema_length=10000,
                ignore_errors=False,
                missing_utf8_is_empty_string=True,
            ),
            "csv",
        )
    raise SystemExit(f"Unsupported input format for {path}; use .csv or .parquet")


def collect_streaming(lazy_frame: Any) -> Any:
    try:
        return lazy_frame.collect(engine="streaming")
    except TypeError:  # Compatibility with older Polars.
        return lazy_frame.collect(streaming=True)


def keyed_digest(secret: bytes, task_namespace: str, column_namespace: str, value: str) -> bytes:
    payload = (
        task_namespace.encode("utf-8")
        + b"\0"
        + column_namespace.encode("utf-8")
        + b"\0"
        + value.encode("utf-8")
    )
    return hmac.new(secret, payload, hashlib.sha256).digest()


def build_mapping(
    values: set[str], spec: IdSpec, secret: bytes, task_namespace: str
) -> dict[str, str]:
    ordered = sorted(
        values,
        key=lambda value: (
            keyed_digest(secret, task_namespace, spec.column, value),
            value,
        ),
    )
    width = max(4, len(str(len(ordered))))
    return {value: f"{spec.prefix}{index:0{width}d}" for index, value in enumerate(ordered, start=1)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paths_refer_to_same_file(first: Path, second: Path) -> bool:
    if first == second:
        return True
    if not first.exists() or not second.exists():
        return False
    try:
        return first.samefile(second)
    except OSError:
        return False


def reject_path_overlaps(paths: list[tuple[str, Path]]) -> None:
    for index, (first_role, first_path) in enumerate(paths):
        for second_role, second_path in paths[index + 1 :]:
            if paths_refer_to_same_file(first_path, second_path):
                raise SystemExit(
                    "Resolved input, output, and manifest paths must be distinct; "
                    f"{first_role} overlaps {second_role}: {first_path}"
                )


def preflight_paths(args: argparse.Namespace) -> tuple[list[Path], Path, list[Path], Path | None]:
    input_paths = [path.expanduser().resolve() for path in args.inputs]
    if len(set(input_paths)) != len(input_paths):
        raise SystemExit("The same resolved input may be specified only once")
    if len({path.name for path in input_paths}) != len(input_paths):
        raise SystemExit("Input basenames collide; use uniquely named files")
    for path in input_paths:
        if not path.is_file():
            raise SystemExit(f"Input does not exist or is not a file: {path}")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise SystemExit(f"Output directory path is not a directory: {output_dir}")
    unresolved_outputs = [output_dir / path.name for path in input_paths]
    symlink_outputs = [path for path in unresolved_outputs if path.is_symlink()]
    if symlink_outputs:
        raise SystemExit(f"Refusing symlink output paths: {symlink_outputs}")
    output_paths = [path.resolve() for path in unresolved_outputs]

    manifest_path: Path | None = None
    if args.manifest is not None:
        unresolved_manifest = args.manifest.expanduser()
        if unresolved_manifest.is_symlink():
            raise SystemExit(f"Refusing symlink manifest path: {unresolved_manifest}")
        manifest_path = unresolved_manifest.resolve()
        if manifest_path.parent.exists() and not manifest_path.parent.is_dir():
            raise SystemExit(f"Manifest parent is not a directory: {manifest_path.parent}")

    labeled_paths = [(f"input[{index}]", path) for index, path in enumerate(input_paths)]
    labeled_paths.extend((f"output[{index}]", path) for index, path in enumerate(output_paths))
    if manifest_path is not None:
        labeled_paths.append(("manifest", manifest_path))
    reject_path_overlaps(labeled_paths)

    invalid_outputs = [path for path in output_paths if path.exists() and not path.is_file()]
    if invalid_outputs:
        raise SystemExit(f"Output path exists but is not a regular file: {invalid_outputs}")
    collisions = [path for path in output_paths if path.exists() and not args.force]
    if collisions:
        raise SystemExit(f"Output exists; use --force to replace: {collisions}")
    if manifest_path is not None and manifest_path.exists():
        if not manifest_path.is_file():
            raise SystemExit(f"Manifest exists but is not a regular file: {manifest_path}")
        if not args.force:
            raise SystemExit(f"Manifest exists; use --force to replace: {manifest_path}")

    return input_paths, output_dir, output_paths, manifest_path


def stage_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            json.dump(manifest, temporary, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def main() -> int:
    args = parse_args()
    secret_text = os.environ.get(args.key_env)
    if not secret_text:
        raise SystemExit(
            f"Set {args.key_env} to a private stable secret; do not pass secrets on the command line"
        )
    secret = secret_text.encode("utf-8")
    if len(secret) < 32:
        raise SystemExit(
            f"{args.key_env} must contain at least 32 UTF-8 bytes from a high-entropy random "
            "source; length alone does not make a predictable secret safe"
        )

    id_columns = [spec.column for spec in args.id_specs]
    if len(set(id_columns)) != len(id_columns):
        raise SystemExit("Each --id column may be specified only once")
    prefixes = [spec.prefix for spec in args.id_specs]
    if len(set(prefixes)) != len(prefixes):
        raise SystemExit("Each --id prefix may be specified only once")
    overlap = sorted(set(id_columns) & set(args.drop))
    if overlap:
        raise SystemExit(f"Columns cannot be both anonymized and dropped: {overlap}")

    input_paths, output_dir, output_paths, manifest_path = preflight_paths(args)
    pl = require_polars()

    scans: dict[Path, tuple[Any, str, set[str]]] = {}
    for path in input_paths:
        lazy, file_format = scan(path, pl)
        names = set(lazy.collect_schema().names())
        scans[path] = (lazy, file_format, names)

    required_columns = {spec.column for spec in args.id_specs}
    present_somewhere = set().union(*(names for _, _, names in scans.values()))
    absent_everywhere = sorted(required_columns - present_somewhere)
    if absent_everywhere:
        raise SystemExit(f"ID columns absent from every input: {absent_everywhere}")

    mappings: dict[str, dict[str, str]] = {}
    blank_values: dict[str, set[str]] = {}
    for spec in args.id_specs:
        values: set[str] = set()
        blanks: set[str] = set()
        for lazy, _, names in scans.values():
            if spec.column not in names:
                continue
            frame = collect_streaming(
                lazy.select(pl.col(spec.column).cast(pl.String, strict=False).alias(spec.column)).unique()
            )
            for value in frame.get_column(spec.column).to_list():
                if value is None:
                    continue
                if value.strip() == "":
                    blanks.add(value)
                else:
                    values.add(value)
        blank_values[spec.column] = blanks
        mappings[spec.column] = build_mapping(values, spec, secret, args.task_namespace)

    columns_with_blanks = sorted(column for column, values in blank_values.items() if values)
    if args.blank_id_policy == "error" and columns_with_blanks:
        raise SystemExit(
            "Blank or whitespace-only IDs found in columns: "
            f"{columns_with_blanks}; choose --blank-id-policy preserve or null"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_temporary_path: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix=".anonymize_ids-", dir=output_dir) as staging:
            staging_dir = Path(staging)
            staged_outputs: list[tuple[Path, Path]] = []
            outputs: list[dict[str, Any]] = []
            for input_path, output_path in zip(input_paths, output_paths, strict=True):
                lazy, file_format, names = scans[input_path]
                expressions = []
                for spec in args.id_specs:
                    if spec.column not in names:
                        continue
                    replacements: dict[str, str | None] = dict(mappings[spec.column])
                    if args.blank_id_policy == "preserve":
                        replacements.update({value: value for value in blank_values[spec.column]})
                    elif args.blank_id_policy == "null":
                        replacements.update({value: None for value in blank_values[spec.column]})
                    expressions.append(
                        pl.col(spec.column)
                        .cast(pl.String, strict=False)
                        .replace_strict(replacements, return_dtype=pl.String)
                        .alias(spec.column)
                    )
                transformed = lazy.with_columns(expressions)
                to_drop = [column for column in args.drop if column in names]
                if to_drop:
                    transformed = transformed.drop(to_drop)

                staged_path = staging_dir / output_path.name
                if file_format == "parquet":
                    transformed.sink_parquet(staged_path, compression="zstd")
                else:
                    transformed.sink_csv(staged_path)
                staged_outputs.append((staged_path, output_path))
                outputs.append(
                    {
                        "input_basename": input_path.name,
                        "output_basename": output_path.name,
                        "output_sha256": sha256_file(staged_path),
                    }
                )

            manifest = {
                "task_namespace": args.task_namespace,
                "key_environment_variable": args.key_env,
                "blank_id_policy": args.blank_id_policy,
                "id_columns": [
                    {
                        "column": spec.column,
                        "prefix": spec.prefix,
                        "unique_nonblank_values": len(mappings[spec.column]),
                        "column_namespace": spec.column,
                    }
                    for spec in args.id_specs
                ],
                "dropped_columns": sorted(set(args.drop)),
                "outputs": outputs,
                "raw_value_mapping_written": False,
            }
            if manifest_path is not None:
                manifest_temporary_path = stage_manifest(manifest_path, manifest)

            # Every output and the optional manifest are complete before the first commit.
            for staged_path, output_path in staged_outputs:
                os.replace(staged_path, output_path)
            if manifest_path is not None and manifest_temporary_path is not None:
                os.replace(manifest_temporary_path, manifest_path)
                manifest_temporary_path = None
            else:
                print(json.dumps(manifest, indent=2))
    finally:
        if manifest_temporary_path is not None:
            manifest_temporary_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
