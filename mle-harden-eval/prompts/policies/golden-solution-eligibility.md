# Golden-solution eligibility policy

A reference candidate is eligible only when all conditions below are proven against the exact active task contract and pipeline release.

## Artifact contract

- The candidate is one self-contained Python file with one concise module docstring, no function or class docstrings, and only comments needed to explain an intentional non-standard choice.
- It accepts no arguments, exposes no CLI, invokes no shell or subprocess, imports no task-specific author module, and contains no prior-model, competition, benchmark-history, or documentation provenance.
- It locates participant inputs from its materialized task workspace, discovers supplied rows and files dynamically, reads only participant-visible inputs, and never reads `private/`, source datasets, hidden author work, prior attempts, credentials, or provider artifacts.
- It runs offline in the exact release environment and creates or replaces exactly `submission/submission.csv`.

## Validation contract

- Static isolation, forbidden-import, path, token, and output scans pass.
- A clean cold run from a new workspace finishes inside the task's wall-clock and resource limits.
- The submission passes the exact grader schema, key-coverage, finite-value, domain, and task-specific constraints.
- The actual unrounded cold-run score is finite and reproducible within the task contract's nonnegative absolute tolerance.
- Promotion comparisons use raw binary64 scores in the native metric direction. Reproducibility tolerance never converts a tie or regression into an improvement.

## Promotion contract

For a GPT-triggered consolidation, the cold-run score must match or improve upon the triggering GPT score and be strictly better than the incumbent reference. If either comparison fails, the incumbent remains current and the candidate is retained only as failed promotion evidence.

A successful promotion atomically records the code and documentation revision, code and output hashes, actual score, environment and release, elapsed time and resources, eligibility checks, and this policy path and hash. Shipped executable code and rendered documentation must not mention GPT provenance. Superseded candidates remain immutable internal history; only one solution revision is current.
