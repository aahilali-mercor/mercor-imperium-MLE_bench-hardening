# Contrasting examples: route by source semantics

Use these examples when a raw dataset could tempt a mechanical task design. They illustrate reusable reasoning, not templates to copy verbatim.

## Example A: regular advertising-reporting panel

### Source shape

- Tens of millions of timestamped reporting rows
- Nested account/campaign/creative-like entities
- Exposure and funnel counts such as recommendations, impressions, clicks, conversions, and cost
- Repeated observations that can be aggregated to a stable hourly grid

### Rejected first idea

Predict hidden outcomes for only the ten highest-cost or highest-exposure rows.

Why it fails:

- Ten rows make the metric unstable and tie-sensitive.
- Selecting by an extreme can reveal strong constraints on the hidden values.
- Funnel/accounting relationships may make the targets nearly deterministic.
- A leaderboard can be dominated by luck rather than transferable modeling.

### Defensible harder design

Forecast click quantiles over several future intervals at leaf, campaign, and account levels from a leakage-safe history and known future exposure covariates. Evaluate warm, cold-entity, and reporting-outage scenarios with macro-averaged scaled pinball loss. Require ordered quantiles and hierarchy coherence.

Useful difficulty comes from:

- reconstructing training histories strictly as of each forecast origin;
- time and entity shift;
- missing historical blocks represented in training;
- probabilistic calibration across horizons;
- reconciling nested forecasts;
- validating per scenario rather than optimizing a single dominant slice.

Privacy handling:

- omit unique source row keys;
- replace required hierarchy values with compact task-local identifiers;
- ensure code assignment does not preserve source lexical order;
- keep source mappings and the builder outside the package;
- pre-extract test histories if later long-form observations would reveal targets.

Empirical lesson: generic boosting must be tested, but a task-specific exposure-rate/calibration method can outperform it. Difficulty is supported by the observed solution ladder and clean-agent time, not by data volume or the author's engineering effort.

## Example B: sparse purchase/state-transition event log

### Source shape

- Millions of all-string transaction rows and over a thousand sparse columns
- Current item/transaction descriptors
- Pre-state and post-state feature families
- Flattened positional inventories and embedded JSON-like structures
- Many one-event players, an imbalanced catalog, temporal catalog drift, and a collapsed late collection tail
- Only observed purchases; no logged nonpurchase opportunities

### Invalid mechanical ports

- A panel forecast copied from the advertising example: the log has no complete regular grid.
- Purchase propensity or conversion: there are no negative opportunities.
- Uplift or causal recommendation: exposure/assignment and counterfactual assumptions are absent.
- Ordinary churn: at-risk windows and censoring are unresolved.
- Using post-state/current item name/cost fields: they reveal the target event.
- Treating 1,395 raw sparse string columns as the main challenge: that is mostly I/O and schema annoyance.

### Defensible candidate

Predict the current or next item conditional on a transaction using structured pre-state and strictly prior history. Split forward in time, quarantine the abnormal collection tail, and macro-evaluate seen/unseen players and head/mid/tail catalog regimes. A candidate-slate ranking version is valid when only positive insertion depends on the label, decoy proposal/order/deduplication do not, candidate-only attacks fail, and the task is explicitly described as positive-included sampled conditional ranking.

Useful difficulty comes from:

- reconstructing order-invariant summaries from positional state;
- strictly prior sequence/history features;
- severe cold-player and temporal catalog shift;
- long-tail classification or ranking;
- feature-family leakage ablations;
- validation that mirrors the final scenario mixture.

Privacy handling:

- whitelist derived fields instead of blacklisting raw columns;
- remove receipt, token, device, transaction, product, and embedded commerce identifiers;
- avoid globally consistent player pseudonyms when trajectory linkage is unnecessary;
- use case-local sequence keys or precomputed prior-history features;
- coarsen exact time/quantities if rare trajectories remain linkable;
- treat empty string as missing, not zero.

Empirical lesson: the observation process constrains the claim. Artificially inventing nonpurchases would create a synthetic task, not evidence about real purchase propensity. If synthetic negatives are used for candidate ranking, disclose that the score is sampled ranking conditional on a transaction.

## General deduction

The reusable pattern is not “add cold start, outages, and quantiles.” It is:

1. infer why rows exist and what is absent;
2. define a legal prediction moment;
3. select a claim the observation process supports;
4. derive a compact safe representation;
5. expose natural shifts and structure through an exact metric;
6. falsify shortcuts with probes;
7. add only learnable input-side transformations;
8. measure the full baseline-to-gold ladder and clean-solver difficulty.

The two sources may both be tabular, yet the valid targets, splits, privacy choices, and hard modeling work are fundamentally different.
