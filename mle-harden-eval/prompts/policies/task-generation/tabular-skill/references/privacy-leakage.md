# Privacy, security, and leakage gate

Apply the leakage portions of this gate to every dataset and the privacy controls proportionally to the release threat model. Anonymization is not just renaming obvious ID columns, and leakage is not limited to a literal target column.

## 1. Establish a release threat model

Classify source fields and derived outputs:

- direct identity: name, email, phone, address, account, device, player, patient, employee;
- authentication/commerce secrets: tokens, receipts, order IDs, transaction IDs, payment payloads, signatures, session keys;
- location/network: IP, GPS, precise site, URL, referrer, host, route;
- free text and embedded structures: JSON, query strings, metadata blobs, arrays, logs;
- persistent pseudonyms: hashes, UUIDs, advertising IDs, source row keys;
- quasi-identifiers: exact timestamps, rare category combinations, exact balances, unique trajectories;
- proprietary identifiers and catalog names that may reveal the source organization;
- sensitive outcomes or attributes;
- post-outcome state and operational internals.

Publicly licensed data can still expose sensitive precise locations, endangered species, vulnerable facilities, rare medical/behavioral trajectories, or people named in institutional metadata. License, privacy, security, and ethical/domain release clearance are separate questions.

Decide which entity linkages the prediction task actually needs. Every unnecessary persistent key increases memorization and re-identification risk.

Do not copy raw examples from a source analysis into public descriptions. Dataset-analysis documents themselves may contain sensitive values; sanitize them before sharing and treat them as protected source artifacts by default.

## 2. Prefer a public-schema whitelist

Start with an empty public schema. Add a field only when all are true:

1. It has a documented modeling purpose.
2. It is available at prediction time.
3. It is safe to release at the chosen precision and linkage scope.
4. Its missing-value semantics are defined.
5. Its derivation is fitted only on permitted training data.

Whitelisting is safer than dropping a remembered blacklist from a very wide table. Parse free text, JSON, arrays, and flattened slots into purpose-built aggregates; never pass through opaque payloads by default.

## 3. Identifier strategy

Choose the minimum linkage scope:

- **No linkage needed:** drop the ID and assign a random task-local case key after splitting.
- **Within-case linkage only:** use a case-local sequence/node key that changes across cases.
- **Train/test entity linkage needed:** use a consistent task-local compact ID.
- **History needed but identity memorization is not:** precompute strictly prior aggregates or sequences, then remove the persistent source entity key.

For consistent compact IDs, do not enumerate source values in lexical, numeric, timestamp, or first-seen order. Those orderings can leak source information. Use a secret-keyed deterministic permutation:

1. Normalize type without changing identity semantics.
2. Compute `HMAC-SHA256(secret, task_namespace || NUL || column_namespace || NUL || raw_value)`.
3. Sort unique values by the keyed digest.
4. Assign namespace-specific codes such as `a0001`, `c0042`, or `u01937`.
5. Keep the secret and any raw mapping outside public/private task directories.

Use different HMAC namespaces for semantically different ID columns even if raw strings overlap. Use a unique task/version namespace and a random per-task secret with at least 256 bits of entropy. Store the secret in a secrets manager or protected author-only file outside the repository and shell history; record only a nonsecret key identifier in the build record, restrict access, back it up for reproducible rebuilds, and rotate/regenerate the public IDs if it is exposed. Do not reuse the key or mapping across unrelated tasks. Pass every linked train/test/history file in one invocation so the value universe and compact mapping are consistent. The included `scripts/anonymize_ids.py` implements this pattern for CSV/Parquet files.

Unique `row_id`, `_id`, event IDs, ingestion IDs, and source offsets usually provide alignment rather than signal; omit them entirely and create a new non-reversible prediction key.

## 4. Trajectory and quasi-identifier risk

Replacing an account with `u0042` does not anonymize a unique exact trajectory. Audit combinations of:

- precise timestamp sequences;
- rare location/site/version/device combinations;
- exact balances, prices, currencies, or item sequences;
- long histories and unique hierarchy paths;
- low-frequency categories or exact text lengths;
- public external facts that could link a pseudonym back to a person or organization.

Mitigations include coarsening time, bucketing quantities, truncating sequences, suppressing rare groups, using case-local IDs, publishing aggregates, or removing linkage altogether. Measure the effect on task validity before applying them.

Do not claim formal anonymity unless a qualified review and a defined privacy standard support that claim. Pre-register task-specific suppression/coarsening thresholds and the relevant independence unit; no universal k or count is safe for every threat model. “Task-local identifiers and screened derived features” is usually the accurate statement.

## 5. Leakage taxonomy

### Direct target leakage

- target or alternate target encoding;
- post-event/post-state fields;
- current transaction name, cost, reward, quantity, or status that deterministically identifies the target;
- labels embedded inside arrays, JSON, filenames, paths, or IDs;
- exact accounting identities or target-derived aggregates.

### Temporal leakage

- rows at or after the prediction cutoff;
- future-fitted frequency tables, encoders, scalers, imputers, vocabularies, and candidate sets;
- backward joins without strict timestamp conditions;
- later snapshots that encode the target event;
- overlapping label windows between train and validation/test;
- ingestion time mistaken for event time;
- same-timestamp events treated as ordered without justification.

### Group and duplicate leakage

- the same person/device/site/source object in nominally disjoint partitions;
- near-duplicate records, resampled rows, or parent/child copies;
- multiple target views of the same event;
- persistent identifiers that enable lookup or memorization;
- random row folds on repeated-measure data.

### Construction leakage

- selecting test cases using hidden label magnitude, rarity, or difficulty;
- sampling decoys after observing the positive label without a declared ranking protocol;
- computing scenario quotas or scales from hidden outcomes;
- stale caches built from the full corpus;
- deriving public metadata from private labels;
- tuning repeatedly on final private feedback.

### Evaluation leakage

- answer files accessible to solutions;
- grader error messages exposing labels or per-row losses;
- sample submission populated with non-placeholder truth-like values;
- predictable task IDs or row order tied to target ranking;
- private files recoverable through relative paths, symlinks, caches, or build leftovers.

## 6. Availability ledger

Maintain an author-side table for every public feature family:

| Family | Raw source | Event/state time | Legal cutoff rule | Transform fit set | Linkage scope | Privacy action | Probe result |
| --- | --- | --- | --- | --- | --- | --- | --- |

For time-indexed data, specify `< cutoff` versus `<= cutoff`. For current-row state, explain why it is pre-outcome. For historical post-state, ensure it belongs to a strictly earlier event. For asynchronous systems, account for reporting delay.

## 7. Required leakage probes

Run these before optimizing the intended model:

- target determinism from every individual column;
- determinism from suspicious tuples such as cost/currency/quantity;
- feature-family-only models for current event, post-state, IDs, time, and history;
- exact duplicate and near-duplicate overlap;
- train/test entity, parent, session, source-object, and trajectory overlap;
- target recovery from public later rows or differenced snapshots;
- performance with IDs alone and row order alone;
- fit-scope audit for all learned transforms;
- private oracle with known forbidden features to understand the ceiling.

High probe performance is not automatically leakage, but it requires a causal availability explanation and an ablation-based decision.

## 8. Output scanning

Perform both schema and value-level scans:

1. Assert forbidden source columns are absent.
2. Extract source direct-ID/token values in a protected process and check their intersection with all participant-visible strings.
3. Scan text, CSV, JSON, Parquet string columns, filenames, and metadata for URLs, emails, IPs, UUIDs, receipt/token patterns, and organization-specific domains.
4. Check compact IDs against allowed namespace regexes and referential consistency.
5. Check that raw ID counts, lexical ordering, or first-seen ordering cannot be inferred from the new codes.
6. Inspect embedded JSON/arrays after serialization, not only source columns.
7. Scan task plans and logs before accidentally copying them into the public package.

Pattern scans are only a supplement. Comparing against actual source values catches tokens whose format was not anticipated.

## 9. Private-test hygiene

Use at least three partitions when tuning task difficulty:

- public-style training/validation for participant development;
- author development holdout for builder and gold-solution iteration;
- untouched final private holdout frozen after design.

If a user explicitly authorizes private-label optimization, record queries and treat that private set as author development data. Regenerate or reserve a new final holdout before reporting a clean score. Otherwise the score may still debug the pipeline, but it is not unbiased evidence of generalization.

Ensure participant solutions run with the private directory unavailable. The grader alone should load answers, and grader output should be a single aggregate score unless the harness explicitly requires more.

## 10. Release gate

Do not release until all are true:

- public fields are whitelisted and prediction-time valid;
- source provenance, redistribution/data-use rights, attribution, and domain-specific release obligations are verified;
- identifiers use the minimum required linkage scope;
- direct identifiers, secrets, commerce tokens, and raw persistent IDs are absent;
- quasi-identifier and trajectory risk has been reviewed;
- all learned transforms respect split boundaries;
- target, temporal, duplicate, construction, and evaluation leakage probes pass;
- public/private/source directories are isolated;
- output scans find no forbidden source values;
- privacy and leakage limitations are accurately documented.

When data covers demographic or otherwise protected groups, report measurable coverage and performance disparities only where group definitions and sample support are valid. Do not infer demographic fairness, causal discrimination, or representativeness from unverified proxy columns.
