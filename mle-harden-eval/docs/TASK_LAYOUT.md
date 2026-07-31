# Task and delivery formats

## Harness task layout (what `aide_harness.py` consumes)

```
tasks/<competition_id>/
├── metadata.json        # competition_id MUST equal the directory name
├── public/              # everything the solver sees
│   ├── description.md
│   ├── train.csv / test.csv / sample_submission.csv / ...
└── private/
    ├── answers.csv      # ground truth
    └── grader.py        # exposes: grade(submission_df: pd.DataFrame) -> float
```

`metadata.json` keys used by the harness: `competition_id`,
`wall_clock_limit_minutes`, `hidden_test_set` / paths, `data_overview`,
`compute_description`, optional `resource_limits`
(`cpuCores`/`ramBytes`/`scratchBytes`/`gpuCount`).

`grader.py` wraps the original mlebench-style `grade(submission, answers)`:
vendor `InvalidSubmissionError`, load `answers.csv`, and expose the
single-argument `grade(submission_df)` the harness calls. Invalid submissions
should raise, not return 0 — the harness records them as failures.

## Before/after pairs

A hardening sample is two sibling task dirs, `<slug>-before` (the delivered
task, converted 1:1) and `<slug>-after` (the hardened variant). Metric and
submission contract must be identical; only data/split/description change,
and every change is disclosed in the description.

## Delivery zip format (synth-gen-pipeline)

```
<slug>-hardened.zip
└── <slug>-hardened/
    ├── raw/                      # source data sufficient to rebuild everything
    └── task/
        ├── prepare.py            # prepare(raw, public, private) — base prep + hardening step
        ├── grade.py              # grade(submission, answers) — unchanged from source task
        ├── description.md        # hardened description (scrubbed of removed columns)
        ├── config.yaml
        ├── HARDENING.md          # what changed, why, evidence
        └── leaderboard.csv
```

Verification rule before shipping: re-run `task/prepare.py` from the zip's own
`raw/` and diff the outputs against the audited build (byte-equality for CSVs,
value-equality fallback for reformatted floats, decompressed-content hash for
gz). A package that cannot regenerate itself from raw/ does not ship.
