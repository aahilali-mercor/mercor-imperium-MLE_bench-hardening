---
name: task-generation-tabular
description: Design, harden, build, privacy-sanitize, benchmark, and validate self-contained MLEbench-style tasks from arbitrary raw tabular datasets or dataset-analysis reports. Use when turning CSV, Parquet, event logs, panels, snapshots, transactions, or relational tables into a nontrivial prediction competition; redesigning an easy first pass so boilerplate boosting no longer reaches the competitive range; choosing a defensible target and split; preventing target leakage and identifier disclosure; adding fair controlled shifts or missingness; implementing public/private/hidden-test packaging and the grader; creating diagnostic, competitive-baseline, medal, and gold solutions; setting score tiers; or auditing claimed task difficulty.
---

# Generate a hard tabular ML task

Turn raw tables into a learnable, privacy-safe competition whose difficulty comes from meaningful modeling work. Treat the source as design material, not as a schema that must be published unchanged.

## Operating contract

Match the requested scope:

- If asked to assess ideas, inspect and benchmark enough to recommend or reject them, but do not build the package.
- For design-only or read-only assessment, use existing analyses and safe read-only probes. If a transformed pilot would be required to measure hardness, stop with a provisional design and an explicit experiment plan rather than creating files against scope.
- If asked to create or implement a task, continue through construction, grader tests, reference solutions, benchmarking, metadata, and validation. Do not stop at a proposal.
- If asked to increase difficulty, change the task only after measuring why the current solution is easy. Rebuild and rebenchmark after each material change.
- When a requested difficulty is missed, do not mistake one easy formulation for the source dataset's ceiling. Run a structured hardening pass while valid, learnable redesigns remain; stop only when the target difficulty is supported or measured evidence rules out the reasonable hardening paths.
- If the source cannot support an honest task, say so and identify the missing observation, label, negative, censoring rule, or privacy clearance. Never fabricate an unsupported real-world claim.

Keep source data, build code, source-to-task mappings, task plans, and benchmark diagnostics outside the participant-visible package. Work from public inputs when implementing participant solutions.

## Load the right guidance

Before selecting a task, read [references/design-playbook.md](references/design-playbook.md). Always read [references/privacy-leakage.md](references/privacy-leakage.md) before opening raw rows, source analyses, or schema-frozen build code: sensitivity may be hidden, and its universal leakage rules apply even to public data.

Before writing the task package or score thresholds, read [references/package-benchmark-validation.md](references/package-benchmark-validation.md). Before finalizing the author solution, read repository-local `docs/solution.md` when present. If the source ontology is ambiguous, read [references/contrasting-examples.md](references/contrasting-examples.md) for the panel-forecasting and transaction-event counterexamples.

When a generic tabular model saturates a pilot, the requested difficulty is missed, or the user asks to make a task harder, read [references/hardening-pass.md](references/hardening-pass.md) and follow its diagnose-redesign-falsify loop before packaging the easy version.

Repository-local specifications are authoritative. Read files such as `docs/task_shape.md` and `docs/difficulty.md` when present; use the bundled references only as the fallback and design guide.

In commands below, set `SKILL_DIR` to the directory containing this `SKILL.md`. The bundled profile/anonymization helpers require Polars; trusted-grader smoke testing additionally requires pandas.

## End-to-end workflow

### 1. Establish the source and workspace

Locate the raw assets, any `dataset_analysis.md`, data dictionary, provenance, license/contract terms, existing task plans, and repository task contract. Record source file fingerprints, schema, row counts, time range, missing-value encodings, and analysis assumptions. Treat analyses as potentially sensitive because they may contain raw exemplars.

Record the requested difficulty band, participant compute/runtime budget, and release context: design exercise, internal prototype, or redistributable benchmark. If difficulty is unspecified, do not assume Difficulty 4; build the best defensible task and rate it from evidence. Follow the repository slug labels—Difficulty 2 is `easy`, Difficulty 3 is `medium`, and Difficulty 4 is `hard`. Assessment and internal prototyping may continue with unresolved release rights, but public packaging may not.

Prefer an existing trustworthy analysis, but verify claims that determine the label, split, or leakage boundary. When no adequate profile exists, run:

```bash
python "$SKILL_DIR/scripts/profile_table.py" <table.csv-or-parquet> --output <author-workdir>/profile.json
```

The profiler deliberately reports no example or frequency-table cell values. Inspect raw examples only when necessary, keep them out of public notes, and avoid printing secrets into tool logs.

For implementation work, create or update an author-side `task_plans.md`. Track hypotheses, candidate tasks, rejected ideas, the availability contract, split design, privacy decisions, benchmark results, redesigns, and unresolved risks. For assessment-only work, keep this record in the response unless the user authorized documentation changes.

### 2. Infer the data ontology before choosing a target

Identify:

- the physical row and the real-world observation unit;
- whether the data is a cross-section, regular panel, irregular event log, transaction-only log, pre/post snapshot pair, censored process, relational collection, or a mixture;
- which events or entities are absent from the table;
- time, entity, hierarchy, session, and repeated-measure structure;
- candidate labels and when they become observable;
- the exact prediction moment and every feature actually observable to the prediction service then, not merely a state that logically existed earlier;
- natural deployment shifts and the intended generalization unit.

Write a one-sentence prediction contract in this form:

> For each **prediction unit**, using only information available **strictly by this time or state**, predict **this target** over **this horizon or candidate set**, and evaluate on **this population**.

If that sentence is ambiguous, do not model yet.

### 3. Generate and rank multiple task candidates

Produce at least three ontology-compatible candidates unless only one label is defensible. For each, state:

- target, prediction unit, permitted inputs, split, metric, and expected solver work;
- the intended participant bottleneck, why a flat generic model should miss it, and at least one legitimate specialized path;
- what real claim the score supports;
- invalid interpretations the source does not support;
- likely leakage, privacy, triviality, and compute risks;
- the weakest baseline that could expose the idea as easy.

Rank candidates on label validity, prediction-time realism, sample size, split support, privacy safety, metric quality, headroom above simple baselines, and room for progressively stronger solutions. Reject a candidate rather than hiding a fundamental flaw with noise.

### 4. Pass the privacy and leakage gate

Build a whitelist of public columns and derivations. For every feature family, document its source, timestamp, delivery/latency semantics, transformation fit set, and actual prediction-time observability. Remove post-outcome fields, target proxies, current-transaction descriptors, future-fitted aggregates, duplicate target representations, and label-derived selection gates.

Drop unique alignment keys when they add no signal. When linkable IDs are necessary, replace them with compact task-local identifiers whose assignment does not reveal source values or lexical order. Set a random high-entropy per-task key outside shell history/logs, and pass every file that shares an ID universe in one anonymizer invocation:

```bash
python "$SKILL_DIR/scripts/anonymize_ids.py" train.parquet test.parquet \
  --output-dir <derived-dir> --task-namespace <task-slug>-v1 \
  --id accountId:a --id campaignId:c --drop row_id --drop _id
```

Running linked files separately can create inconsistent compact IDs. Keep the key and any mapping outside the task, document its protected lifecycle, and scan every emitted artifact for raw identifiers and embedded tokens.

Fit vocabularies, imputers, encoders, scalers, candidate generators, and aggregate statistics on the permitted training partition only. Define same-timestamp ordering explicitly. Run single-column and feature-family leakage probes before accepting the schema.

Do not proceed while a private-data disclosure or target reconstruction path remains plausible. Before public release, also verify provenance, redistribution/data-use rights, required attribution, and any domain-specific review; public licensing does not automatically remove location, trajectory, or other sensitivity.

### 5. Falsify easy versions quickly

Before investing in the full build, create a representative pilot split and run, as applicable:

1. constant, majority, prevalence, or global-popularity prediction;
2. an educated domain heuristic;
3. a fast generic boosted-tree or linear model;
4. an obvious task-specialized model;
5. suspicious single-column and feature-family probes;
6. a private oracle diagnostic that is never shipped as a participant solution.

Measure aggregate and hard-slice scores. Tiny extreme-only test sets, target ties, majority dominance, deterministic accounting identities, current/post-state columns, or generic boosting near the apparent ceiling are redesign signals.

If the fast generic model reaches the intended competitive-baseline range, or the specialized model adds no stable post-generic gain, do not merely tighten score thresholds. Enter the hardening pass and change the representation, prediction unit, output structure, or learnable scenario design.

### 6. Freeze a canonical construction spec

Create the clean canonical prediction pool before splitting, sampling, or corrupting inputs. Freeze:

- source fingerprint and seed/config;
- eligibility rules based on prediction-time covariates;
- label derivation and prediction unit;
- time/group/entity split, purge, and overlap rules;
- scenario quotas and sampling order;
- feature availability and as-of joins;
- anonymization namespaces;
- public/private schemas;
- metric cells, constraints, and tie behavior.

Apply splits before fitting transformations. Prevent target-window, entity, trajectory, duplicate, and aggregate leakage. When a fixed label vocabulary is necessary, define it from training-visible data or a declared pre-cutoff catalog, map unsupported future labels to a documented `OTHER`/unknown class, and do not quietly drop final-test rows based on hidden labels. Sample with stable sorting and tie-breaking so rebuilds are byte-stable where formats permit.

### 7. Add difficulty through independent, learnable competencies

For an easy first pass or a missed requested difficulty, execute [the hardening pass](references/hardening-pass.md). Diagnose what the package has pre-solved for participants, generate at least three structural redesigns, and pilot the smallest promising change. Prefer restoring meaningful sequence, set, relational, hierarchy, as-of, multi-target, or calibration work over adding nuisance. A hardening pass is successful only when a specialized public-input approach beats reasonable generic families by more than seed and evaluation uncertainty while the task remains learnable.

Use the smallest combination needed to reach the target difficulty:

- leakage-safe temporal or relational feature engineering;
- natural time, entity, catalog, or environment shift;
- cold-start groups alongside warm groups;
- long-tail or rare-regime performance;
- multiple horizons, targets, quantiles, or candidate ranking;
- hierarchy, conservation, monotonicity, or probability constraints;
- calibrated uncertainty and asymmetric costs;
- deterministic input-side missingness, truncation, coarsening, or sensor bias represented in training;
- macro-averaging across declared scenarios so easy volume cannot dominate.

Each lever must create a useful modeling decision, be inferable from public data, and have an analogous learnable mechanism in training or documented domain structure. A cold entity need not literally occur in training, but training must teach the solver how no-history or unseen-identity cases are represented. Do not use random target noise, label-dependent corruption, hidden test-only tricks, arbitrary feature removal, giant raw files, or confusing formats as substitutes for difficulty.

Re-run the pilot ladder after every redesign. Prefer a task with several viable solution paths and measurable gains from better validation, features, modeling, calibration, or ensembling.

Do not declare the dataset exhausted after one candidate matrix. Record which hardening branches were piloted, which generic shortcut each addressed, the specialized hypothesis, the measured outcome, and why remaining branches are infeasible or invalid.

### 8. Build the self-contained task

Derive compact public data rather than publishing raw sensitive or unwieldy tables. Include enough labeled data and metadata to learn every scored regime. Publish target-free test inputs and an exact sample submission; keep answers and the grader private.

Decide whether the final harness will replace development test inputs. If not, set `hidden_test_set` to false, use an empty `hidden_test_paths`, and do not create `hidden-test/`. If so, set it to true, list every replaced public file or folder in `hidden_test_paths`, and store each final replacement at the same task-root-relative path below `hidden-test/`. Keep that entire folder unavailable to participant solutions and ensure code can handle changed IDs, counts, and directory contents dynamically.

Write the participant description in the repository-required order and make the metric executable from the prose. Implement `grade(submission: pandas.DataFrame) -> float` with exact key coverage, type/domain checks, finite-value checks, task constraints, fixed scoring cells, and row-order invariance.

Run the static package audit during construction:

```bash
python "$SKILL_DIR/scripts/validate_task.py" <task-directory>
python "$SKILL_DIR/scripts/validate_task.py" <task-directory> --final \
  --key-columns <comma-separated-prediction-key-columns>
```

The generic audit enforces this skill's bundled/default repository contract; adapt it when authoritative local rules differ. It is necessary but not sufficient, so add task-specific split, leakage, schema, grader-mutation, and privacy tests. Only for grader code authored or reviewed in the current task, add `--trust-grader-code`; this runs a bounded subprocess but is not a security sandbox. Never execute an untrusted grader during an audit.

### 9. Benchmark a ladder, then keep one solution

During development, measure these approaches using public assets only:

- zero/constant diagnostic;
- simple domain heuristic diagnostic;
- standard fast generic frontier-LLM diagnostic;
- competitive baseline solution using the correct task-specific approach plus real validation, iteration, tuning, or calibration;
- bronze-competitive solution;
- credible reference or silver solution;
- tuned gold solution;
- private oracle diagnostic, kept author-only.

Benchmark each in a cold process under the published wall-clock limit, including loading, preprocessing, training, prediction, and writing. Record package hash, code hash, seed, library versions, score, slice scores, elapsed time, and peak memory.

The final task keeps only `solutions/train.py` and `solutions/README.md`. Consolidate the strongest eligible public-input approach into the self-contained `train.py`; remove all baseline, reference, benchmark, helper, notebook, and superseded solution code. The script takes no arguments, invokes no CLI or shell commands, discovers inputs relative to itself, reads neither `private/` nor `hidden-test/`, and always creates or replaces `submission/submission.csv` at the task/project root.

Use one concise module docstring to say what the script does and why its important choices were made. Add no other docstrings or routine comments. Comment only an intentional non-standard choice that would otherwise look erroneous. Do not mention competitions, prior code, benchmark history, or documentation from executable code.

Write a terse README table listing each materially different prior approach, a one-line explanation, and its measured score. Identify the approach now implemented by `train.py`; keep all narrative and historical comparison out of the script.

Tune on public-style validation and an author development holdout. Do not repeatedly optimize against the final private labels. If private feedback was explicitly authorized for pipeline debugging, log every use, then regenerate and freeze a fresh untouched final holdout before claiming an unbiased gold score.

Choose baseline, bronze, silver, and gold thresholds only after measured runs. The `baseline` tier is the floor for serious competitive performance, not an educated guess, one-shot implementation, generic model, or merely valid submission. It should require the right approach plus enough effort and calibration to be credible while remaining below the medal-competitive bronze range. Respect metric direction, leave reproducibility margin around gold, and ensure each tier represents a meaningful capability jump rather than decimal decoration.

### 10. Validate difficulty with clean solver trials

Difficulty is active expert human time with frontier AI, not author construction effort or unattended compute. Before trials, freeze the success tier, stopwatch protocol, allowed tools, model/version, prompt/harness version, and failure treatment. Run at least three independent clean trials, preferably five, with no task-design notes or private feedback. Record active time, failure/censoring, final score, and tier. Use the median active time among successful trials and report the success rate; with fewer than three successes, treat the rating as unresolved or the task as too hard rather than selectively timing one success.

For a claimed Difficulty 4, evidence should support a typical successful 181–240 minutes of active work under the repository rubric. Active solving time may include inspection, iteration, and several compliant experimental runs; the final end-to-end solution run must still fit `wall_clock_limit_minutes`. If only the author solution is hard to write, or fresh agents solve it quickly, lower the rating or redesign. Mark untested ratings provisional.

### 11. Complete the validation gates

Before handoff:

- rebuild twice from clean output directories and compare artifact hashes;
- verify solution isolation from all private and `hidden-test/` files;
- confirm `solutions/` contains exactly `train.py` and `README.md`, and that `train.py` has no CLI interface or task-specific imports;
- run `train.py` without arguments from a cold process and verify it writes only the required `submission/submission.csv` output;
- test the grader with valid, shuffled, missing, extra, duplicate, NaN, infinity, and constraint-violating submissions;
- audit temporal/group overlap, fitted-transform scope, target absence, and scenario quotas;
- verify source provenance, release/data-use rights, attribution, and domain-specific clearance;
- scan public and private artifacts for source identifiers, tokens, URLs, emails, and forbidden columns;
- confirm the sample submission grades and covers every prediction key once;
- reconcile metadata scores with fresh benchmark results;
- when modifying this skill itself, run the `quick_validate.py` supplied by the `skill-creator` skill.

Do not claim completion with failing gates. Document any residual limitation that cannot be eliminated without changing the task claim.

## Completion deliverables

A completed task-generation run should leave:

- an author-side analysis/task plan and frozen build configuration;
- a deterministic builder and task-specific validation tests;
- the self-contained public/private task package and, when enabled, harness-only `hidden-test/` replacements;
- one self-contained best-score `solutions/train.py` and a concise `solutions/README.md` score history;
- cold benchmark results with slice metrics and resource use;
- a privacy/leakage/determinism audit;
- evidence-backed metadata, tier thresholds, wall-clock limit, and difficulty rating.
- when the requested difficulty was initially missed, a hardening record covering rejected redesigns and the generic-versus-specialized evidence for the retained form.

Lead the user with the resulting task, measured scores, and remaining risks—not a diary of implementation steps.
