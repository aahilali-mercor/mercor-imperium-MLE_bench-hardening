# Package, benchmark, and validation contract

Use repository-local task-shape and difficulty documents when available. This reference captures this skill's bundled/default contract and the validation discipline needed for a robust MLEbench-style task; adapt hardcoded validators when an authoritative local contract differs.

## 1. Directory boundaries

The participant package should be self-contained:

```text
<task-slug>/
├── metadata.json
├── public/
│   ├── description.md
│   ├── sample_submission.csv
│   └── <documented training and test assets>
├── hidden-test/                    # only when hidden_test_set is true
│   └── public/
│       └── <replacement paths mirroring the task root>
├── private/
│   ├── grader.py
│   └── <hidden evaluation assets>
└── solutions/
    ├── train.py
    └── README.md
```

Author-side materials belong outside this boundary:

```text
<author-workdir>/
├── task_plans.md
├── build_task.py
├── config.*
├── source_fingerprints.json
├── source_to_task_ledger/       # protected; omit if unnecessary
├── tests/
├── benchmark_results.json
└── audit_results/
```

The task keeps exactly one author solution under `solutions/`. It is not a participant input during solving. Development experiments and benchmark runners remain author-side and are removed from the final task.

## 2. Metadata contract

Default `metadata.json` fields:

```json
{
  "competition_id": "<task-slug>",
  "dataset": "<human-readable source name>",
  "category": "Tabular",
  "short_description": "One or two sentences describing the prediction task and its main challenge.",
  "is_lower_better": true,
  "evaluation_metric": "<exact human-readable metric name>",
  "gold_solution_score": null,
  "score_tiers": {
    "baseline": null,
    "bronze": null,
    "silver": null,
    "gold": null
  },
  "hidden_test_set": false,
  "hidden_test_paths": [],
  "wall_clock_limit_minutes": 120,
  "task_difficulty": 1
}
```

Semantics:

- `competition_id`: exact directory slug.
- `dataset`: canonical human-readable source name.
- `category`: concise non-empty task category. Prefer an existing repository label, but preserve a useful new label for later metadata review rather than rejecting an otherwise valid task.
- `short_description`: one or two sentences naming the objective and main technical challenge.
- `is_lower_better`: native metric direction.
- `evaluation_metric`: exact metric implemented by the grader, including important macro dimensions or constraints.
- `gold_solution_score`: best reproducible author solution using packaged public inputs only.
- `score_tiers`: native-metric thresholds for a competitive non-medal baseline and progressively stronger bronze, silver, and gold performance.
- `hidden_test_set`: whether the harness swaps development test inputs before the final code run.
- `hidden_test_paths`: normalized task-root-relative public files/directories replaced from identical relative paths below `hidden-test/`; empty when the boolean is false.
- `wall_clock_limit_minutes`: positive integer for the complete participant run.
- `task_difficulty`: repository rubric value, normally 1–5.

`gold_solution_score` and tier thresholds may be null during construction and must be numeric after final benchmarking. Do not add the obsolete `baseline_score` or `minimum_score` fields; the score tiers carry those thresholds directly. Do not add `score_tier_rule`, `environment_constraints`, or `system_prompt_addendum` when the harness derives direction and prompt constraints from existing metadata.

Threshold ordering:

- lower is better: `baseline >= bronze >= silver >= gold`;
- higher is better: `baseline <= bronze <= silver <= gold`.

The reproducible gold score must clear the gold threshold with enough margin for reruns and library differences. Avoid a threshold separated from the measured score only by floating-point dust.

The `baseline` tier is not an educated guess or a reward for producing valid output. Set it from a reproducible competitive solution that uses the correct task-specific approach and required meaningful validation, iteration, tuning, or calibration. A generic one-shot model should normally remain below it. Bronze begins the medal-competitive range; silver and gold require progressively stronger execution.

## 3. Public description

Write these top-level sections in order:

1. **Task** — prediction unit, target, cutoff/horizon/candidate set, and intended population.
2. **Metric** — exact formula, direction, range if bounded, averaging cells, weights/scales, labels, constraints, and tie behavior.
3. **Submission Format** — exact columns, key semantics, allowed values, row count/mapping, constraints, and a short example.
4. **Dataset** — every public asset, schema, units, targets, missing semantics, train/test differences, split, and privacy-relevant transformations.

The description is an executable specification. A solver should not need task plans, builder code, grader inspection, or private knowledge to interpret the data or score.

If input corruption, candidate sampling, cold-start construction, normalization, or hierarchy constraints affect strategy, disclose them at the level needed to solve the task without exposing labels.

## 4. Hidden-test replacement

When `hidden_test_set` is false, require an empty `hidden_test_paths` and no `hidden-test/` directory.

When true:

- list every replaced live path under `public/`, using normalized task-root-relative paths;
- reject absolute paths, `..`, duplicates, and overlapping parent/child entries;
- require both the live path and `hidden-test/<same path>` to exist and have the same file/directory kind;
- keep `hidden-test/` completely unavailable during development and participant solution runs;
- have the harness replace each listed live path immediately before the final run;
- preserve the documented role and compatible format/schema while permitting different IDs, row counts, filenames, and distributions;
- make participant code discover final inputs dynamically rather than rely on preview IDs or sizes;
- store replacement inputs only in `hidden-test/`; answer keys remain private.

Document the switch in the public task description. The hidden tree mirrors task-root paths: for example, `public/test.csv` comes from `hidden-test/public/test.csv` and `public/test_images` comes from `hidden-test/public/test_images`.

## 5. Grader requirements

Expose:

```python
def grade(submission: pandas.DataFrame) -> float:
    ...
```

The grader should:

- require exact column names and reject extras;
- coerce only where the public format explicitly allows it;
- reject missing, extra, or duplicate prediction keys;
- compare the exact key set and merge one-to-one;
- reject unknown labels/categories and malformed list/vector lengths;
- reject nonnumeric, NaN, infinity, and out-of-domain values;
- enforce probability, monotonicity, hierarchy, conservation, or ordering constraints;
- compute the exact documented metric;
- reindex macro cells against a fixed expected set and fail if cells disappear;
- ignore submission row order;
- load private assets relative to `__file__`;
- return one finite native-scale numeric score without leaking row-level truth.

Do not silently clip, normalize, deduplicate, reconcile, or fill predictions unless that behavior is explicitly part of the public metric.

Treat graders from external or unknown task packages as untrusted code: inspect them statically and do not import or execute them. A smoke test is appropriate only for grader code authored or reviewed in the current task. Even a timeout-bounded subprocess is not a security sandbox and still has the caller's filesystem/network authority.

## 6. Task-specific grader tests

Start from a known-valid submission and mutate one property at a time:

- shuffled rows: same score;
- missing row/key: reject;
- extra row/key: reject;
- duplicate key: reject;
- extra/missing column: reject;
- string in numeric field: reject;
- NaN and positive/negative infinity: reject;
- negative or out-of-domain value: reject when prohibited;
- unknown class or step: reject;
- probabilities not summing to one: reject when required;
- crossed quantiles or cumulative values: reject;
- hierarchy/conservation violation: reject;
- exact boundary values: accept or reject as documented;
- hand-calculated tiny fixture: exact expected score;
- sample submission: structurally valid and produces a finite score.

Test error paths without printing answer values.

## 7. Public package validation

Assert at minimum:

- every documented asset exists and every asset is documented;
- training contains the intended target; test does not;
- test prediction IDs are stable and unique;
- sample submission has every required key exactly once;
- no package symlink escapes the task root or exposes source/private assets to participants;
- hidden-test metadata, path mirroring, file/directory kinds, and folder presence agree exactly;
- participant solutions cannot access `hidden-test/` before the harness switch;
- `solutions/` contains exactly `train.py` and `README.md` for a final task;
- train/test schemas and dtypes match the description;
- declared arrays have fixed/valid lengths;
- scenario quotas and score cells are populated;
- missingness encodings match the prose;
- no public path, cache, metadata, or statistic reveals private labels;
- package size and load time fit the wall-clock budget.

Run `scripts/validate_task.py` for generic checks, then add schema-aware assertions in the task builder/tests.

## 8. Deterministic construction

Freeze:

- source file SHA-256 hashes and row counts;
- builder/config/code revision;
- all seeds and random generators;
- time cutoffs and purge widths;
- eligibility and sampling order;
- stable sort and tie-breaking rules;
- ID namespaces and protected keyed-ordering secret identity;
- library versions that affect serialization or models;
- compression and output ordering.
- development and hidden replacement inputs, with both artifact sets fingerprinted.

Build twice into fresh directories. Compare relative file lists and SHA-256 hashes. When a format has unavoidable byte-level metadata differences, compare canonical content and document the exception.

Make caches content-addressed by source fingerprint plus config, or clear them. A fixed cache filename can silently preserve an obsolete split or feature definition.

## 9. Solution isolation

Run every participant solution from a clean working directory with only `public/` readable; withhold both `private/` and `hidden-test/`. Check for:

- absolute or parent-relative private paths;
- imports from builder/ledger modules;
- stale local predictions or model caches;
- target-bearing training artifacts unintentionally reused for test;
- internet or external-data dependencies not allowed by the harness;
- reliance on current working directory rather than located public assets.

Copy the generated submission into the grading environment only after the solution exits.

### Final solution artifact

After development, keep only the strongest eligible approach:

- `solutions/train.py` is self-contained and imports no task-specific helper modules;
- it accepts no CLI arguments, calls no shell/subprocess commands, and selects its own paths;
- it reads participant-visible inputs only and remains valid after hidden-test replacement;
- it always creates or replaces task-root-relative `submission/submission.csv`;
- it has one concise module docstring explaining the method and important choices;
- it has no other docstrings or routine comments; comments are reserved for intentional non-standard behavior that could look mistaken;
- it contains no references to prior implementations, competitions, benchmark history, or documentation files;
- its cold score is the recorded `gold_solution_score`.

`solutions/README.md` is a terse approach/score table. Give each materially different prior approach one concise explanation, state metric direction once, and identify the method used by `train.py`. Do not keep prior approach code in the task.

## 10. Benchmark ladder

Benchmark in separate cold processes:

| Run | Purpose |
| --- | --- |
| Constant/zero | Metric and difficulty sanity check |
| Domain heuristic | Lowest credible domain-aware diagnostic; does not define a metadata tier |
| Fast generic frontier | Strong one-shot/generic diagnostic; does not automatically define a metadata tier |
| Competitive baseline | Correct task-specific approach plus meaningful validation, iteration, tuning, or calibration; sets the baseline tier |
| Bronze | Clearly stronger competitive refinement entering medal range |
| Reference/silver | Task-specific features/modeling and robust validation; fair strong comparison point |
| Gold | Best reproducible packaged-input solution |
| Oracle diagnostic | Ceiling/leakage diagnosis; never participant-eligible |

This ladder is a development process, not a final file set. Preserve the measurements in `solutions/README.md`, consolidate the best public-input approach into `solutions/train.py`, and remove the executable diagnostics and alternatives.

For every run record:

- task package hash and solution code hash;
- cold/warm state and hardware;
- seed and library versions;
- end-to-end wall time, active tuning time, and peak memory;
- aggregate score and declared slice/cell scores;
- validation score used for selection;
- private-development feedback count, if any;
- exact submission hash.

Do not compare warm-cache gold timing to cold baseline timing. The wall-clock limit includes data loading, preprocessing, fitting, inference, and output writing.

## 11. Tier setting

Set tiers after the benchmark ladder is stable:

1. Baseline requires a reproducible competitive solution using the correct task-specific approach plus meaningful effort and calibration. It must sit above heuristics, generic one-shot attempts, and mere correctness.
2. Bronze marks entry into the genuinely medal-competitive range through a clear improvement over the competitive baseline.
3. Silver corresponds to the credible strong reference solution or equivalent modeling competence.
4. Gold is beatable by the packaged gold solution with a reproducibility buffer.
5. Adjacent thresholds exceed ordinary seed/platform variance.
6. Slice performance confirms that a threshold is not reached through one dominant easy regime.

For small, clustered, or repeated-view test sets, estimate uncertainty by bootstrapping the true independence unit. Tier gaps should exceed both rerun/seed variance and evaluation uncertainty; merge unstable metric cells rather than presenting tie-sensitive decimals as capability differences.

Keep measured solution scores distinct from thresholds when useful, but reconcile their semantics in metadata and benchmark documentation.

## 12. Difficulty validation

Difficulty measures expected *active human time* for a competent domain expert using frontier AI. It excludes author task construction and unattended compute.

Under the common rubric:

| Difficulty | Active expert time |
| --- | ---: |
| 1 | 1–60 minutes |
| 2 | 61–120 minutes |
| 3 | 121–180 minutes |
| 4 | 181–240 minutes |
| 5 | 241–300 minutes |

A theoretical 301+ minute task should be simplified rather than labeled 6 when the repository only accepts 1–5.

Before clean trials, freeze the success threshold/tier, stopwatch rules, allowed tools, frontier model/version, prompt/harness version, and how failed or timed-out trials are counted. Then:

- give only the participant package and normal harness prompt;
- prevent access to task plans, source analysis, gold code, private labels, and earlier conversation;
- capture active inspection, prompting, review, coding, debugging, validation, and packaging time;
- do not count unattended training/download time;
- record achieved tier and failure modes;
- run at least three independent trials, preferably five;
- report success rate and the median active time among successful trials, not the fastest outlier;
- with fewer than three successes, mark difficulty unresolved or redesign rather than inferring a band from one survivor.

Difficulty 4 is defensible only when clean solvers generally need 181–240 active minutes to reach the required standard. A complex author build, a failed generic booster, or a narrow gold margin is not enough.

Use the repository difficulty label in the task slug and title: Difficulty 2 is `easy`, Difficulty 3 is `medium`, and Difficulty 4 is `hard`.

Active solving time and participant wall clock measure different things. Active time can include dataset inspection, prompting, debugging, and multiple experimental runs; every final end-to-end solution execution must still fit `wall_clock_limit_minutes`, including load, preprocessing, training, inference, and output.

## 13. Redesign triggers

Redesign rather than adjust a number when:

- the generic fast solution reaches gold;
- a simple heuristic or generic one-shot solution reaches the competitive baseline tier;
- gold wins only by seed noise;
- private feedback drives most of the gold gain;
- scored classes/regimes are unlearnable from training;
- one leak-prone column dominates;
- test size creates unstable ties or leaderboard luck;
- I/O or format confusion consumes most active time;
- the task exceeds its own wall-clock budget;
- privacy requires removing the features that make the target valid.

After redesign, regenerate private answers, rerun every solution, and reset thresholds from the new evidence.

## 14. Final audit record

Store a machine-readable or concise Markdown record containing:

- source/build/package hashes;
- split and leakage test results;
- privacy scan result;
- grader mutation-test result;
- deterministic rebuild comparison;
- hidden-test path/mirroring and pre-switch isolation checks when enabled;
- isolated solution benchmark table;
- final `train.py` output-path/style/isolation audit and README score history;
- score tier rationale and variance margin;
- clean-agent difficulty evidence;
- remaining limitations and sign-off date.

The task is finished only when the package, grader, solutions, metadata, and evidence all refer to the same frozen build.
