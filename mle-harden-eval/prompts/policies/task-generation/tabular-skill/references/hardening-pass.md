# Hardening an easy tabular task

Use this pass when a valid first formulation is too easy, a generic booster reaches the competitive range, or the requested difficulty is not supported. The goal is to make meaningful structure necessary, not to make the data obscure or the score noisy.

## 1. Diagnose what made the first pass easy

Do not start by adding models or tightening thresholds. Write a short saturation diagnosis from measured evidence:

| Question | Evidence to record |
| --- | --- |
| What does the best generic model exploit? | Important feature families, single-feature probes, row representation, slice scores |
| What work has the builder already done? | Precomputed histories, aggregates, joins, candidate features, labels, reconciled outputs |
| What structure was discarded? | Sequences, sets, hierarchy, relations, shared origins, repeated targets, censoring, uncertainty |
| Where is remaining error concentrated? | Time, group, cold-start, tail, horizon, target family, missingness regime |
| Is the apparent ceiling real? | Oracle, stronger specialist, label ambiguity, repeated-seed and bootstrap uncertainty |
| Why is solving fast? | One flat fit, ordinary cross-validation, dominant target, weak constraints, no representation work |

Common saturation causes are a single flat row per prediction, participant-ready aggregates, independent scalar targets, an IID split, a metric dominated by easy volume, or a target whose natural structure was binned away.

## 2. Search hardening paths in a sensible order

Generate at least three materially different redesigns. Search from the source ontology outward; use later options only when earlier ones are invalid or ineffective.

### A. Restore structure hidden by the easy package

- Publish sanitized long-form observations and require query-specific as-of reconstruction.
- Represent histories as variable-length event sequences rather than only fixed lag/count columns.
- Represent inventories, memberships, documents, or components as sets or relational child tables rather than only aggregate counts.
- Group related predictions at a shared origin so information can be pooled across siblings, targets, or horizons.

This is the preferred first move when the builder converted a rich source into one modeling-ready row. Keep participant data compact enough to process within the wall-clock limit; file size and parsing pain do not count as difficulty.

### B. Strengthen the prediction contract

- Predict several related horizons, target families, quantiles, or candidate outcomes from one origin.
- Use a sampled conditional ranking contract with declared, label-independent decoy generation when a full opportunity log does not exist.
- Predict a joint or conditional outcome whose components have meaningful dependence instead of scoring several unrelated easy labels.
- Retain natural granularity when coarse bands turn the problem into majority-class classification.

Every output must remain observable after the declared origin. Macro-average components so easy targets cannot subsidize hard ones.

### C. Add natural structure and constraints

- Preserve a real hierarchy, conservation law, probability simplex, quantile order, monotone path, or nested candidate structure.
- Require participants to produce coherent outputs; reject invalid submissions rather than silently repairing them.
- Combine warm, cold, tail, temporal, environment, or missingness regimes that correspond to plausible deployment conditions.

Constraints should create modeling and reconciliation decisions. Do not add arbitrary algebra that has no source interpretation.

### D. Make representation learning useful

Expose a participant-visible auxiliary corpus when the source contains repeated entities, events, fields, items, or components that can teach reusable representations. Valid options include:

- supervised pretraining on earlier labels or related outcomes;
- masked-field or masked-event prediction under a disclosed label-independent mask;
- item/entity embeddings learned from co-occurrence, transition, or train-period effect signatures;
- sequence, set, graph, or multi-task encoders fine-tuned on the scored prediction units.

Fit auxiliary labels, vocabularies, graphs, and signatures on permitted training-era data only. Keep evaluation groups and times out of auxiliary targets. Run a same-capacity from-scratch ablation and a simpler aggregate-feature model. Pretraining counts as hardening only when its gain is stable, comes from reusable public structure, and fits the participant runtime budget.

Do not force one named architecture. A good design permits at least two plausible specialist routes, such as learned embeddings versus carefully engineered relational statistics.

### E. Add coherent input-side observation effects

Use disclosed, target-independent truncation, delayed reporting, missing blocks, coarsening, or sensor views only when training teaches the same mechanism. Make the mask or observation state visible. Prefer multiple views of the same underlying training unit when denoising or robustness is the intended competence.

Input corruption is a last-mile lever, not a substitute for a substantive target. Random label noise, hidden test-only corruption, and removal of the only necessary signal are invalid.

### F. Use an explicit simulator only as a last resort

When the natural labels cannot support the requested difficulty but the public covariates support a useful synthetic problem, define the task as synthetic. Generate labels from a documented family of mechanisms, use fresh hidden seeds/configurations, and audit for shortcuts. The simulator must reward transferable modeling choices rather than recovery of a secret constant or brute-force search.

## 3. Write a competence ledger before building

For each redesign, state:

| Field | Required content |
| --- | --- |
| Generic shortcut being blocked | The measured reason the current task saturates |
| New participant work | Concrete as-of, relational, sequence, representation, calibration, or reconciliation steps |
| Learnable public mechanism | Where training data teaches every scored regime |
| Generic diagnostic | At least one reasonable boosting/linear pipeline and one simple structural heuristic |
| Specialized hypothesis | A public-input model expected to gain and why |
| Ablations | Features, pretraining, constraints, calibration, or scenario components to remove |
| Failure condition | Result that would reject this hardening branch |
| Active-time effect | Additional hands-on work expected from a frontier-assisted solver |

Count competencies, not files or targets. One relational reconstruction plus a joint calibrated model can be harder than ten independent prediction columns.

## 4. Pilot with a learnability sandwich

Use a representative split and compare:

1. simple heuristic;
2. strong generic flat model with reasonable encoding and tuning;
3. generic neural model when representation learning is relevant;
4. intended specialized approach;
5. specialist ablations;
6. private oracle or clean upper-bound diagnostic.

Measure every important slice and use the true independence unit for uncertainty. Run enough seeds to distinguish an improvement from optimizer variation.

Accept the hardening branch only when all of these hold:

- The generic flat solution remains below the intended competitive-baseline tier.
- The specialized approach improves by more than seed variation and evaluation uncertainty, preferably with a paired or clustered interval excluding zero.
- At least one ablation attributes the gain to the intended competence rather than extra compute or accidental leakage.
- All difficult regimes are represented or structurally learnable in public training data.
- The oracle or strongest specialist shows useful remaining headroom; the task is not merely noisier.
- The final solution fits the runtime limit and the expected active solver work moves toward the requested difficulty band.

Reject or redesign when a standard booster still wins, the specialized gain is unstable, only one brittle trick works, or hard slices are irreducibly unlearnable.

## 5. Iterate without over-hardening

After every material redesign, rerun the complete ladder. Change one structural hypothesis at a time when practical so the source of headroom remains measurable.

Stop when:

- clean solver trials support the requested active-time band;
- further valid changes only add compute, format friction, or irreducible error;
- privacy, ontology, sample support, or runtime makes the remaining branches invalid;
- the source cannot teach the competence needed by the harder evaluation.

Do not stop merely because the first three objectives were easy. Conversely, do not keep adding mechanisms after the requested difficulty is supported.

## 6. Parallel hardening branches

When parallel work is authorized, share only the source audit, canonical-pool rules, release gates, and comparison rubric. Give each branch an isolated work directory, distinct task slug, frozen development split, and explicit hardening hypothesis.

Compare branches on:

- prediction-contract validity and privacy;
- generic-to-specialized gain relative to variance;
- ablation support for the intended competence;
- hard-slice behavior and oracle headroom;
- participant runtime and estimated active work;
- distinctness from other retained objectives.

Do not compare raw metric values across different objectives. Retain multiple branches only when each passes independently and their prediction contracts or required solver competencies are materially different.

## 7. Record the hardening outcome

Add a hardening section to `task_plans.md` containing:

1. the first-pass saturation diagnosis;
2. the competence ledger for every branch;
3. pilot schemas and split fingerprints;
4. generic, specialist, ablation, slice, runtime, and uncertainty results;
5. rejected branches and exact rejection reasons;
6. the retained branch and why its difficulty is participant work rather than author-side construction;
7. clean solver-trial evidence or a provisional rating.

An easy retained prototype may remain useful, but label it at its measured difficulty. Do not let its existence prevent a separate hardening branch from using a richer representation or prediction contract.
