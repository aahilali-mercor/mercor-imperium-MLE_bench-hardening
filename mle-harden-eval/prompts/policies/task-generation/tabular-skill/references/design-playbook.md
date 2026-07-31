# Tabular task design playbook

Use this reference before selecting a target. The same physical table can support several very different claims; its observation process determines which ones are legitimate.

## 1. Start with the observation process

Answer these questions from documentation, invariants, and small audits:

1. What caused a row to exist?
2. What real-world cases are missing because nothing was recorded?
3. Is the row an entity, state, event, transaction, opportunity, interval, pair, or aggregate?
4. Which columns describe the world before the outcome, the event itself, and the world after it?
5. Are repeated rows linked by an entity, session, hierarchy, or time grid?
6. When would a prediction have been requested in deployment?
7. Which fields would be known at that instant?
8. When is the label observable, and is it censored?
9. What population should the score generalize to?
10. What natural shifts occur across time, entities, sites, versions, catalogs, or regimes?

Do not infer negatives from absence unless the logging process records the complete set of opportunities. A purchase-only table supports modeling *which item is chosen conditional on a purchase*; it does not, by itself, support purchase propensity, conversion, uplift, or causal recommendation.

## 2. Route by dataset ontology

| Ontology | Defining evidence | Usually defensible tasks | Typical split | Common invalid claim |
| --- | --- | --- | --- | --- |
| Cross-sectional entities | One row per independent entity at a declared snapshot | Classification, regression, ordinal prediction | Group/site/time holdout when available | Random split when duplicates or household/site clusters cross folds |
| Regular panel | Entity × fixed time grid with explicit zero/missing semantics | Forecasting, quantiles, anomaly labels if verified | Forward time with purge; cold-entity slices | Treating missing rows as zero without proving panel completeness |
| Irregular event log | Rows appear when events occur | Next event/type/time, sequence prediction, conditional ranking | Forward time and trajectory-safe group split | Event occurrence probability without non-event exposure windows |
| Transaction-only log | Only completed positive transactions | Conditional item/category/value prediction, next-transaction modeling | Temporal catalog/player shift | Conversion, propensity, uplift, churn, or catalog-wide recommendation claims |
| Pre/post snapshot pair | Same event has state before and after | Outcome from pre-state, transition class, validated delta prediction | Time/entity split using pre-state only | Publishing post-state or current-event descriptors as predictors |
| Survival-capable process | Entry time, at-risk intervals, event time, censoring are all defined | Time-to-event, hazard, competing risks | Time-based origin with censoring-safe evaluation | Calling inactivity churn without observation-window/censoring rules |
| Opportunity/impression log | Every offered candidate/opportunity and response is recorded | Conversion, CTR, ranking, propensity; causal work only with assumptions | Time/user/campaign shift | Treating logged actions as unbiased counterfactual labels |
| Relational collection | Several tables with stable keys and timestamps | Entity prediction with as-of joins, link/ranking tasks | Group/time split across the prediction entity | Full-corpus aggregates or joins that include post-cutoff facts |
| Repeated measurements / sites | Samples nested in person, machine, batch, hospital, geography, etc. | Group-generalization, calibrated regression/classification | Hold out the deployment-level group | Row-random validation that memorizes group signatures |
| Partially completed records | Some fields are present for some records, with an unknown missingness mechanism | Explicitly simulated field completion on observed labels; conditional imputation | Split original records/groups before masking | Treating organic null as a negative or claiming performance on naturally missing labels without ground truth |

Mixed datasets need an explicit routing decision for each table. Create a small entity-relationship and time-availability diagram when more than two tables participate.

## 3. Candidate worksheet

Generate at least three candidates and fill this table before choosing:

| Field | Required answer |
| --- | --- |
| Prediction contract | Unit, cutoff/state, target, horizon/candidate set, population |
| Scientific claim | What a lower/higher score actually means |
| Invalid claims | What cannot be concluded from this source |
| Public inputs | Whitelisted feature families and as-of rules |
| Label | Exact derivation, availability, censoring, exclusions |
| Split | Time/group boundaries, purge, cold-start definition |
| Metric | Formula, direction, aggregation cells, constraints |
| Simple attacks | Majority, identity, accounting relation, current/post-state, duplicates |
| Hardness path | Concrete improvements from baseline to gold |
| Privacy | Direct identifiers, tokens, quasi-identifiers, trajectory risks |
| Size/runtime | Rows, width, submission size, expected end-to-end limit |
| Fatal uncertainty | Evidence still needed before implementation |

Score candidates from 0–2 on each of the following. A zero in label validity, prediction-time validity, privacy releasability, or evaluation validity is fatal rather than something to average away.

- Label validity
- Prediction-time feature validity
- Split realism
- Sample and class support
- Privacy releasability
- Metric alignment
- Resistance to trivial shortcuts
- Multiple legitimate modeling paths
- Feasible package and submission size
- Plausible baseline-to-gold headroom

Prefer the simplest candidate that can support the desired difficulty. Complexity in the builder does not count as participant difficulty.

Pre-register what “adequate support” means for the candidate. There is no universal row minimum: use effective independent units, class/cell counts, seed variance, and clustered bootstrap confidence intervals. Merge cells or reject a design when ordinary resampling uncertainty could reverse leaderboard or tier ordering.

## 4. Define availability, not merely columns

For each feature family, record:

| Feature family | Source time | Available at prediction? | Fit population | Missing meaning | Public representation |
| --- | --- | --- | --- | --- | --- |

Important distinctions:

- A historical post-state may be legal for a later prediction; the target event's post-state is not. A logically pre-event state is legal only if it was actually delivered or observable before the prediction decision.
- A row timestamp does not define ordering among same-timestamp events. Group them, exclude ambiguous cases, or document a stable causal ordering.
- A globally computed category frequency, vocabulary, target encoding, candidate slate, scaler, or imputer can leak test/future information even when the source column is historical.
- Empty string, absent row, null, zero, and not-applicable are separate states until proven equivalent.
- Publishing later observations for test entities can reveal hidden targets indirectly. Pre-extract test histories when the long-form table cannot be safely truncated.

## 5. Build a canonical prediction pool

Derive one clean table of eligible prediction units before scenario sampling or artificial corruption. Each row should have:

- internal source locator, never participant-visible;
- prediction cutoff/origin;
- target window or candidate set;
- group/entity/hierarchy keys;
- label and label availability flag;
- eligibility fields based only on permitted covariates;
- natural slice attributes;
- a stable, non-sensitive task-local prediction key assigned after selection.

Freeze the order of operations:

1. parse and normalize source semantics;
2. derive prediction-time covariates and label separately;
3. apply declared eligibility rules, label-independent by default and using only the narrow population-definition exception below;
4. assign train/development/final partitions;
5. fit train-only transforms;
6. sample scenarios with stable ordering and fixed seeds;
7. apply declared input-side transformations;
8. anonymize and emit public/private artifacts.

If label support is needed to ensure an evaluable test set, make that a transparent class-support rule at task-design time, audit its effect, and do not expose the hidden target distribution through scenario selection.

For closed-set classification, prefer a vocabulary fixed from training-visible data or a catalog known before the cutoff. Map later unsupported labels to one documented unknown/`OTHER` class when that preserves the claim. Do not discard final cases after inspecting their hidden labels. A label-dependent eligibility exception is permissible only when it defines the target population itself, is frozen before model tuning, is disclosed, and does not select hard or extreme cases within that population.

### Masked-field completion

Do not equate organic nulls with a known class or with random missingness. A clean record-completion task normally starts from records where the field is observed, splits original records/groups, and only then applies a disclosed label-independent masking mechanism. The score supports performance under that simulated masking distribution, not necessarily imputation of naturally missing values, which may be MNAR. Keep all masked views of one original record in one partition and bootstrap by the original independence unit.

## 6. Split design

Choose the split that mirrors deployment and blocks memorization:

- **Forward time:** train before validation before test; add a purge at least as long as overlapping context/target windows require.
- **Group-disjoint:** keep people, devices, sites, documents, products, or source objects wholly within one partition when generalizing to unseen groups.
- **Cold-start:** withhold complete entities or categories from labeled training while retaining only the context genuinely available at prediction.
- **Hybrid:** time first, then stratify scored scenarios into warm/cold/rare regimes without moving future data backward.
- **Spatial/site:** hold out deployment regions/sites and prevent near-duplicate or household leakage.

Audit:

- exact and fuzzy duplicates;
- entity, parent, trajectory, session, and source-object overlap;
- overlapping target windows;
- train statistics fitted on validation/test;
- labels embedded in later public observations;
- target-derived eligibility or sampling;
- time zones, late arrivals, and event-time versus ingestion-time confusion.

## 7. Select a metric that resists easy dominance

| Task family | Strong default | Add when needed |
| --- | --- | --- |
| Balanced classification | Log loss or macro F1 | Per-group/class macro averaging, calibration |
| Long-tail multiclass | Log loss, macro recall, Recall@K | Head/mid/tail or seen/unseen macro cells |
| Regression | MAE/RMSE appropriate to costs | Per-group scaling, asymmetric loss, interval coverage |
| Count forecasting | Poisson/deviance, scaled MAE, pinball | Horizon/scenario/hierarchy macro averaging |
| Probabilistic forecasting | Pinball or proper scoring rule | Quantile order, calibration, hierarchy coherence |
| Ranking | NDCG/MRR/Recall@K on a declared candidate set | Query-group macro average, sampled-ranking disclosure |
| Survival | C-index or time-dependent proper score | Censoring rule and horizon-specific reporting |

Macro-average over a small, fixed Cartesian set of important dimensions when volume would otherwise hide hard regimes. Every cell must have adequate support and be reconstructible by the grader. Expose normalizers and scenario labels unless doing so directly reveals the target.

For heterogeneous targets, horizons, or units, derive normalizers from training-visible data with a declared formula and robust floor/cap. Verify through ablations that no component dominates merely because of units, variance, row count, or an arbitrary weight. Report per-component scores and use clustered bootstrap or repeated resampling to confirm tier gaps exceed evaluation noise.

Structural constraints are valuable only when natural to the task: probability simplex, nonnegative quantities, ordered quantiles, hierarchical sums, conservation, monotone cumulative incidence, or valid ranking vocabularies. Reject invalid submissions; do not silently repair them unless repair is part of the documented metric.

## 8. Difficulty levers

Combine independent solver competencies rather than stacking arbitrary nuisances.

### High-value levers

- As-of reconstruction from event or observation tables
- Entity/time/catalog/domain shift
- Warm and cold-start scenarios
- Long-tail labels or rare but supported regimes
- Multiple horizons or related targets
- Probabilistic calibration
- Relational or hierarchical aggregation
- Meaningful output constraints and reconciliation
- Variable-length histories, sessions, or candidate sets
- Scenario-specific model selection or ensembling
- Train/test missingness patterns that have declared semantics

### Controlled artificial transformations

Artificial transformations are acceptable when they represent a coherent observation process and remain learnable:

- deterministic history masking, truncation, coarsening, delayed reporting, quantization, sensor dropout, or bounded feature noise;
- withholding complete entities/categories to create a declared cold-start regime;
- scenario-balanced sampling based on public covariates;
- task-local opaque remapping;
- reproducible positive-included candidate slates whose decoy proposal, count, ordering, and deduplication are independent of which item is positive; guaranteed positive insertion is the only allowed label-dependent step and must be disclosed as sampled conditional ranking;
- synthetic labels from a documented simulator when the task is explicitly synthetic, the simulator is audited for shortcuts, and a fresh hidden seed/config is used for final evaluation.

Requirements:

1. Transformations are independent of hidden target values unless target dependence is the explicitly modeled, disclosed mechanism.
2. Every scored transformation has an analogous learnable mechanism in labeled training or is inferable from declared structure; genuinely unseen entity identities need not occur in training.
3. Scenario identity or missingness is observable when a participant would observe it.
4. The transformation adds a modeling decision, not only irreducible error.
5. A clean oracle and a learnability pilot establish a useful ceiling.

Avoid random label flips, arbitrary target noise, hidden test-only bias, target-dependent row selection, undisclosed adversarial corruption, removing the one scientifically necessary feature, or requiring brute-force compute. These lower scores without reliably increasing active expert work.

## 9. Triviality and leakage probes

Run cheap probes on a representative pilot:

- constant/mean/median/majority/global popularity;
- group-, month-, version-, or site-conditioned popularity;
- last value, seasonal naive, moving average, exponential smoothing;
- previous class, Markov transition, nearest neighbor, entity memorization;
- regularized linear model;
- generic CatBoost/LightGBM/XGBoost with conservative runtime;
- each suspicious feature alone and each feature family alone;
- exact mapping rates from columns/tuples to labels;
- public-key or row-order attacks;
- candidate-only and candidate-position attacks for ranking tasks;
- a private oracle using forbidden/post-target information to confirm the theoretical ceiling.

Record overall and slice scores. Redesign if:

- a constant or popularity baseline nearly reaches gold;
- a single suspicious column recovers much of the label;
- a generic model saturates the score with little feature work;
- the test set is tiny or dominated by ties/extremes;
- performance differences are mostly random-seed noise;
- no legitimate developed model can beat the simple diagnostic heuristics;
- a hard slice has unseen/unlearnable labels;
- the strongest improvement comes from violating the prediction contract.

## 10. Design the solution ladder with the task

Before the full build, write down the intended capability jump:

| Level | Expected capability |
| --- | --- |
| Diagnostic | Constant or invalidly simple shortcut; sanity check only |
| Domain heuristic | Correct schema plus one or two domain-aware statistics; diagnostic only |
| Fast generic frontier solution | Generic model and reasonable validation within a short implementation cycle; diagnostic only |
| Competitive baseline | Correct task-specific approach plus meaningful validation, iteration, tuning, or calibration; establishes the non-medal baseline tier |
| Bronze | A clearly stronger, robust competitive solution entering the medal range |
| Reference / silver | Correct as-of features, slice-aware validation, task-specific modeling, calibration, or ensembling |
| Gold | Tuned features/models, ensembling, reconciliation/calibration, robust validation |
| Oracle | Private diagnostic using unavailable truth; never a participant solution |

If there is no credible progression from baseline to gold, the task is probably fragile, trivial, or luck-driven.

These are development experiments, not separate final solution files. Record their concise descriptions and scores in `solutions/README.md`, consolidate the strongest eligible approach into a self-contained `solutions/train.py`, and remove every other executable solution artifact.

For small or clustered test sets, accompany every ladder score with uncertainty from resampling the true independence unit, not individual duplicated or masked rows. Tier gaps and claimed model improvements should comfortably exceed both seed variance and evaluation uncertainty.

## 11. Author-side task plan template

Use these sections in `task_plans.md`:

1. Source facts and unresolved semantics
2. Privacy/security classification
3. Dataset ontology and prediction-time contract
4. Candidate task matrix and rejected claims
5. Pilot probes and initial results
6. Selected task and why
7. Canonical pool, split, purge, and scenario specification
8. Public/private schemas and anonymization
9. Metric and grader constraints
10. Builder config, fingerprints, and determinism
11. Diagnostic, competitive-baseline, bronze, silver, and gold designs
12. Benchmark table and slice metrics
13. Difficulty trial evidence
14. Known limitations and final freeze record

Keep rejected ideas. They prevent a later maintainer from reintroducing a known leak or invalid task claim.
