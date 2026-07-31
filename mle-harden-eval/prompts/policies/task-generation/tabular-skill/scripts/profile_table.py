#!/usr/bin/env python3
"""Create a privacy-conscious structural profile of one CSV or Parquet table.

The report contains schema, sampled missingness/cardinality, and name-based risk
flags. It intentionally emits no example values or frequency-table values.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


CAMEL_ACRONYM_BOUNDARY_RE = re.compile(r"([A-Z]+)([A-Z][a-z])")
CAMEL_WORD_BOUNDARY_RE = re.compile(r"([a-z0-9])([A-Z])")
NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]+")

SENSITIVE_NAME_TOKENS = {
    "account",
    "address",
    "cookie",
    "device",
    "email",
    "employee",
    "gps",
    "guid",
    "ip",
    "latitude",
    "longitude",
    "name",
    "order",
    "password",
    "patient",
    "payment",
    "phone",
    "player",
    "receipt",
    "referrer",
    "secret",
    "session",
    "signature",
    "token",
    "transaction",
    "uri",
    "url",
    "user",
    "uuid",
}
ID_NAME_TOKENS = {"guid", "hash", "id", "identifier", "key", "uuid"}
TIME_NAME_TOKENS = {
    "created",
    "creation",
    "date",
    "datetime",
    "day",
    "epoch",
    "eventtime",
    "hour",
    "millis",
    "millisecond",
    "modified",
    "month",
    "time",
    "timestamp",
    "updated",
    "year",
}
OUTCOME_NAME_TOKENS = {
    "after",
    "click",
    "conversion",
    "convert",
    "cost",
    "future",
    "label",
    "next",
    "outcome",
    "post",
    "response",
    "result",
    "reward",
    "spend",
    "target",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path, help="CSV/TSV/Parquet file or Parquet directory")
    parser.add_argument("--output", type=Path, help="Write JSON here; stdout when omitted")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output file; input/output overlap is always rejected",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=15000,
        help="Maximum rows profiled (default: 15000)",
    )
    parser.add_argument(
        "--exact-count",
        action="store_true",
        help="Scan CSV input for an exact row count; Parquet counts are exact by default",
    )
    parser.add_argument(
        "--null-token",
        action="append",
        default=[],
        help="Additional exact string treated as missing in the sample; repeatable",
    )
    parser.add_argument(
        "--blank-is-missing",
        action="store_true",
        help="Treat empty and whitespace-only strings as missing; otherwise only count them",
    )
    parser.add_argument(
        "--permissive-csv",
        action="store_true",
        help="Permit CSV value parse errors by coercing them to null; strict parsing is the default",
    )
    parser.add_argument(
        "--columns",
        help="Optional comma-separated column subset; all columns by default",
    )
    parser.add_argument(
        "--separator",
        help="CSV separator override; defaults to tab for .tsv and comma otherwise",
    )
    return parser.parse_args()


def name_tokens(name: str) -> set[str]:
    """Return separator/camel-case tokens plus conservative inflection variants."""
    separated = CAMEL_ACRONYM_BOUNDARY_RE.sub(r"\1 \2", name)
    separated = CAMEL_WORD_BOUNDARY_RE.sub(r"\1 \2", separated)
    tokens = {part.lower() for part in NON_ALNUM_RE.split(separated) if part}
    variants = set(tokens)
    for token in tokens:
        if len(token) > 3 and token.endswith("ies"):
            variants.add(token[:-3] + "y")
        if len(token) > 3 and token.endswith("es"):
            variants.add(token[:-2])
        if len(token) > 2 and token.endswith("s") and not token.endswith("ss"):
            variants.add(token[:-1])
        if len(token) > 4 and token.endswith("ed"):
            variants.add(token[:-2])
        if len(token) > 5 and token.endswith("ing"):
            variants.add(token[:-3])
        if len(token) > 3 and token.endswith("ids"):
            variants.update({"id", token[:-3]})
        elif len(token) > 2 and token.endswith("id"):
            variants.update({"id", token[:-2]})
    return variants


def name_flags(name: str) -> list[str]:
    tokens = name_tokens(name)
    flags: list[str] = []
    if tokens & ID_NAME_TOKENS:
        flags.append("identifier_like_name")
    if tokens & SENSITIVE_NAME_TOKENS:
        flags.append("sensitive_name_review")
    if tokens & TIME_NAME_TOKENS:
        flags.append("time_candidate")
    if tokens & OUTCOME_NAME_TOKENS:
        flags.append("outcome_or_post_state_name_review")
    return flags


def require_polars() -> Any:
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise SystemExit("profile_table.py requires polars") from exc
    return pl


def parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"No .parquet files found below {path}")
    return files


def parquet_row_count(files: list[Path]) -> int:
    try:
        import pyarrow.parquet as pq

        return sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    except Exception:
        pl = require_polars()
        sources = [str(path) for path in files]
        return int(pl.scan_parquet(sources).select(pl.len()).collect().item())


def collect_streaming(lazy_frame: Any) -> Any:
    try:
        return lazy_frame.collect(engine="streaming")
    except TypeError:  # Compatibility with older Polars.
        return lazy_frame.collect(streaming=True)


def load_lazy(args: argparse.Namespace, pl: Any) -> tuple[Any, str, int | None]:
    path = args.table
    suffixes = "".join(path.suffixes).lower()
    if path.is_dir() or suffixes.endswith(".parquet"):
        files = parquet_files(path)
        lazy = pl.scan_parquet([str(file) for file in files])
        return lazy, "parquet", parquet_row_count(files)
    if suffixes.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz")):
        separator = args.separator or ("\t" if ".tsv" in suffixes else ",")
        lazy = pl.scan_csv(
            path,
            separator=separator,
            infer_schema_length=10000,
            ignore_errors=args.permissive_csv,
            null_values=None,
        )
        row_count = int(lazy.select(pl.len()).collect().item()) if args.exact_count else None
        return lazy, "csv", row_count
    raise SystemExit(f"Unsupported table format: {path}")


def select_columns(lazy: Any, requested: str | None) -> tuple[Any, list[str]]:
    names = lazy.collect_schema().names()
    if not requested:
        return lazy, names
    chosen = [name.strip() for name in requested.split(",") if name.strip()]
    missing = sorted(set(chosen) - set(names))
    if missing:
        raise SystemExit(f"Unknown requested columns: {missing}")
    return lazy.select(chosen), chosen


def take_sample(lazy: Any, row_count: int | None, sample_rows: int, pl: Any) -> tuple[Any, str]:
    if sample_rows <= 0:
        raise SystemExit("--sample-rows must be positive")
    if row_count is None or row_count <= sample_rows:
        return collect_streaming(lazy.head(sample_rows)), "head"

    third = max(1, sample_rows // 3)
    middle_start = max(0, (row_count // 2) - (third // 2))
    tail_start = max(0, row_count - third)
    sampled = pl.concat(
        [lazy.head(third), lazy.slice(middle_start, third), lazy.slice(tail_start, third)],
        how="vertical",
    )
    return collect_streaming(sampled), "head-middle-tail"


def scalar_int(value: Any) -> int:
    return int(value) if value is not None else 0


def profile_column(
    series: Any,
    null_tokens: set[str],
    blank_is_missing: bool,
    pl: Any,
) -> dict[str, Any]:
    sample_n = len(series)
    null_mask = series.is_null()
    physical_null_count = scalar_int(null_mask.sum())
    string_like = str(series.dtype).lower().startswith(("string", "utf8", "categorical", "enum"))
    if string_like:
        as_text = series.cast(pl.String, strict=False)
        blank_mask = as_text.str.strip_chars().eq("").fill_null(False)
        token_mask = (
            as_text.is_in(sorted(null_tokens)).fill_null(False)
            if null_tokens
            else pl.Series([False] * sample_n)
        )
    else:
        blank_mask = pl.Series([False] * sample_n)
        token_mask = pl.Series([False] * sample_n)
        if null_tokens and not series.dtype.is_nested():
            try:
                as_text = series.cast(pl.String, strict=False)
                token_mask = as_text.is_in(sorted(null_tokens)).fill_null(False)
            except Exception:
                # Binary/object/extension scalar types may not support a string
                # cast. They remain profiled structurally without sentinel matching.
                pass
    # Blank/whitespace values have their own policy even if a configured
    # token happens to be blank, keeping the three reported classes disjoint.
    token_mask = token_mask & ~blank_mask
    missing_mask = null_mask | token_mask
    if blank_is_missing:
        missing_mask = missing_mask | blank_mask
    blank_count = scalar_int(blank_mask.sum())
    token_count = scalar_int(token_mask.sum())

    valid = series.filter(~missing_mask)
    valid_n = len(valid)
    unique_n = scalar_int(valid.n_unique()) if valid_n else 0
    unique_ratio = (unique_n / valid_n) if valid_n else None
    name = series.name
    missing_n = sample_n - valid_n

    flags: list[str] = []
    if valid_n and unique_n <= 1:
        flags.append("constant_in_sample")
    if sample_n and missing_n / sample_n >= 0.95:
        flags.append("at_least_95_percent_missing_in_sample")
    if valid_n >= 100 and unique_ratio is not None and unique_ratio >= 0.98:
        flags.append("near_unique_in_sample")
    flags.extend(name_flags(name))

    return {
        "name": name,
        "dtype": str(series.dtype),
        "sample_nonmissing": valid_n,
        "sample_missing": missing_n,
        "sample_missing_fraction": (missing_n / sample_n) if sample_n else None,
        "sample_physical_nulls": physical_null_count,
        "sample_blank_strings": blank_count,
        "sample_configured_null_tokens": token_count,
        "sample_unique_nonmissing": unique_n,
        "sample_unique_ratio": unique_ratio,
        "flags": flags,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    pl = require_polars()
    lazy, table_format, row_count = load_lazy(args, pl)
    lazy, selected_names = select_columns(lazy, args.columns)
    schema = {name: str(dtype) for name, dtype in lazy.collect_schema().items()}
    sample, strategy = take_sample(lazy, row_count, args.sample_rows, pl)
    null_tokens = set(args.null_token)
    columns = [
        profile_column(sample.get_column(name), null_tokens, args.blank_is_missing, pl)
        for name in selected_names
    ]

    flag_counts: dict[str, int] = {}
    for column in columns:
        for flag in column["flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    paired_prefixes: list[dict[str, Any]] = []
    for left, right in (("pre", "post"), ("before", "after"), ("pb", "pa")):
        left_names = {name.split(".", 1)[1] for name in selected_names if name.startswith(left + ".")}
        right_names = {name.split(".", 1)[1] for name in selected_names if name.startswith(right + ".")}
        overlap = left_names & right_names
        if overlap:
            paired_prefixes.append({"left": left, "right": right, "paired_suffix_count": len(overlap)})

    return {
        "source": args.table.name,
        "format": table_format,
        "permissive_csv": bool(args.permissive_csv),
        "exact_row_count": row_count,
        "column_count": len(schema),
        "profiled_column_count": len(columns),
        "sampled_row_count": sample.height,
        "sample_strategy": strategy,
        "blank_is_missing": bool(args.blank_is_missing),
        "configured_null_token_count": len(null_tokens),
        "schema": schema,
        "flag_counts": dict(sorted(flag_counts.items())),
        "paired_prefix_hints": paired_prefixes,
        "columns": columns,
        "limitations": [
            "Cardinality, constants, and missingness are sample estimates unless the full table fit in the sample.",
            "Name-based flags are review prompts, not proof of sensitivity or leakage.",
            "Blank strings are counted separately and are missing only when blank_is_missing is true.",
            "Configured null-token matches exclude blank strings and are never emitted literally.",
            "No example cell values or top-value frequencies are included by design.",
        ],
    }


def validate_output_target(table: Path, output: Path | None, force: bool) -> None:
    if output is None:
        return

    table_resolved = table.resolve()
    output_resolved = output.resolve()
    overlaps = table_resolved == output_resolved
    if table.is_dir() and table_resolved in output_resolved.parents:
        overlaps = True
    if os.path.lexists(output):
        try:
            overlaps = overlaps or os.path.samefile(table, output)
        except (FileNotFoundError, OSError):
            pass
    if overlaps:
        raise SystemExit("--output must not overlap the input table or Parquet directory")
    if os.path.lexists(output) and not force:
        raise SystemExit(f"Output already exists (pass --force to replace it): {output}")
    if output.is_dir():
        raise SystemExit(f"Output path is a directory: {output}")


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:  # pragma: no cover - not supported by every platform/filesystem
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(path: Path, text: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        if force:
            os.replace(temporary_path, path)
        else:
            try:
                # A same-directory hard link publishes the complete file atomically
                # and, unlike replace(), cannot clobber an output created in a race.
                os.link(temporary_path, path)
            except FileExistsError as exc:
                raise SystemExit(f"Output already exists (pass --force to replace it): {path}") from exc
            temporary_path.unlink()
        fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    validate_output_target(args.table, args.output, args.force)
    report = build_report(args)
    rendered = json.dumps(report, indent=2, sort_keys=False) + "\n"
    if args.output:
        atomic_write_text(args.output, rendered, args.force)
        print(
            f"Wrote privacy-conscious profile for {report['sampled_row_count']} sampled rows "
            f"and {report['profiled_column_count']} columns to {args.output}",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
