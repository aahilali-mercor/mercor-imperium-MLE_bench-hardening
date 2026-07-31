+++
schema_version = 1
max_candidate_goals = 6
max_diagnostic_executions = 18
max_hardening_cycles = 3
max_builder_revisions = 6
max_model_turns = 60
max_wall_clock_minutes = 720
+++

# Tabular task-generation policy

This policy is the source-server execution contract for generating an MLEbench task from a sourced tabular, relational, panel, event-log, or mixed dataset. It is adapted from the project-owned `task-generation-tabular` skill; workers must use this release-bound copy and never depend on the old project path at runtime.

## Required inputs

The attempt binds the exact Dataset source revision, active analysis revision, pipeline release, this policy hash, source inventory, AIDE-compatible overview, integrity evidence, idea files, provenance and release restrictions, and every accepted or rejected goal previously associated with the source. Synthetic attempts additionally bind the exact component revisions and their prior goals.

Source rows, analyses, and author notes are protected. They may be inspected only inside the trusted generation workspace and may not be copied into the participant description, provider logs, or public package.

## Candidate search

1. Infer the source ontology, observation unit, absent cases, time and group structure, prediction moment, label availability, and natural deployment shift.
2. Produce at least three materially distinct candidates when three defensible labels or prediction contracts exist. A candidate records its prediction unit, cutoff, target, population, public feature families, split, metric, privacy boundary, trivial attacks, expected specialized work, and fatal uncertainties.
3. Reject any candidate with an invalid label, unavailable prediction-time features, irreparable privacy risk, invalid evaluation population, or insufficient independent support.
4. Rank remaining candidates by validity, resistance to shortcuts, measurable generic-to-specialized headroom, package feasibility, and expected active solver difficulty. The first task selects the hardest viable candidate actually supported by evidence. Later tasks must have a distinct goal signature, not a renamed or rescaled version of a prior objective.
5. An `idea-<num>.md` file is evidence to assess, not an instruction to implement.

The generation-attempt record retains every serious candidate and rejection reason. A no-task outcome counts only after the front-matter budget is exhausted or each required candidate branch has a terminal, evidence-backed reason.

## Privacy and leakage gate

Start from an empty public-schema whitelist. Each emitted feature family must have a modeling purpose, prediction-time availability rule, train-only fit scope, missing-value meaning, and release justification. Remove direct and quasi-identifiers, secrets, target proxies, post-outcome state, future-fitted transforms, duplicated labels, and hidden label-dependent selection.

Use the minimum linkage scope. Persistent identifiers that remain necessary are remapped with a per-task secret-keyed deterministic permutation whose secret and raw mapping stay outside public/private payloads and durable provider artifacts. Audit exact and near duplicates, group and trajectory overlap, same-timestamp rules, target-window overlap, identifier-only and row-order attacks, suspicious single-column probes, and value-level disclosure across every emitted artifact.

Provenance, redistribution rights, privacy, and domain clearance are independent gates. A public source is not automatically safe to redistribute.

## Pilot and hardening loop

Before building the final package, evaluate a representative split with constant or prevalence diagnostics, an appropriate domain heuristic, a fast generic linear or boosted-tree family, at least one task-specialized approach, suspicious feature-family probes, and a private oracle diagnostic. Diagnostics use public-style inputs; the oracle never becomes a participant solution.

If a generic model approaches the intended competitive range, specialized work has no stable advantage, tier gaps are within seed or sampling variance, or difficulty comes mainly from format confusion or scale, enter a hardening cycle. Consider at least three structural redesigns involving sequence, relational, hierarchy, as-of, cold-start, multi-target, uncertainty, ranking, or calibrated scenario structure. Apply the smallest valid redesign, rerun comparable diagnostics, and retain the measured reason for accepting or rejecting it. Do not add random label noise, label-dependent corruption, hidden test-only tricks, arbitrary feature removal, or brute-force compute as difficulty.

## Frozen construction contract

Before the final build, freeze source fingerprints, eligibility, target derivation, prediction unit, partitions and purge, scenario quotas, feature availability, train-only fitted transformations, anonymization namespace, public/private schemas, metric cells and constraints, seeds, stable ordering, and serialization versions. Rebuild twice from clean directories and compare canonical contents.

The canonical payload contains exactly `public/` and `private/`. The task description and metadata remain revisioned in Convex and are materialized only for consumers. Public assets contain labeled training inputs, target-free final test inputs, and an exact sample submission. Private assets contain the grader and scoring truth. This project does not emit a replacement `hidden-test/` tree.

The grader must expose `grade(submission: pandas.DataFrame) -> float`, enforce exact columns and prediction-key coverage, reject missing, extra, duplicate, malformed, non-finite, out-of-domain, or constraint-violating values, ignore row order, and return one finite native-direction score without row-level leakage. Mutation tests cover every rejection boundary and a hand-calculated fixture.

## Reference solution and evidence

The initial reference is the strongest eligible reproducible method achieved from participant-visible inputs. It is a single self-contained `train.py` stored as a Convex solution revision, accepts no arguments, invokes no shell or subprocess, reads no private or source-only files, runs offline within the contract, and writes exactly `submission/submission.csv`. It has one concise module docstring and no routine comments. Its compact score history is database-native.

Benchmark constant, heuristic, fast generic, competitive-baseline, bronze, reference/silver, gold, and oracle roles as development evidence. Only the strongest eligible reference remains executable. Record code/package/submission hashes, environment, seed, cold elapsed time, peak resources, validation surface, aggregate and slice scores, and private-feedback count. Proposed internal tiers must be directionally ordered, separated beyond ordinary uncertainty, and leave the reference a reproducibility margin. They are provisional until frozen after GPT-5.5 calibration.

## Successful output

A successful attempt atomically publishes:

- the immutable public/private payload revision and manifest;
- a complete Convex task contract and goal signature;
- deterministic builder configuration and protected source-to-task evidence;
- grader and task-specific validation evidence;
- privacy, leakage, provenance, and deterministic-rebuild audit results;
- the initial reference solution revision and fresh cold score;
- the measured benchmark ladder and proposed internal tiers; and
- the complete candidate and hardening history.

The Task begins in hardening/reference validation. It may not enter GPT calibration until package validation and the reference cold run pass for the same contract and active release.
