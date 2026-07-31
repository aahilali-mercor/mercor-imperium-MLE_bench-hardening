# Reference-solution consolidation

You are improving the reference solution for one frozen MLE task contract. Work only inside the current workspace.

Read these files first:

1. `context.json` for the immutable task contract, metric direction, exact incumbent score, exact triggering challenger score, runtime limit, release, and artifact paths;
2. `past-scores.json` for the complete prior reference-solution history, every GPT calibration run and submission outcome, and every earlier reference-improvement score and outcome;
3. `policy.md` for the binding golden-solution eligibility rules;
4. `task/description.md` and the participant-visible `task/public/` data;
5. `incumbent/train.py` and `incumbent/README.md`;
6. `challenger/train.py` and `challenger/trajectory.json`.

Also inspect `prior-candidate/` when `context.json` names one, and always read both `past-scores.json` and `score-history.json`. `past-scores.json` is immutable historical evidence; `score-history.json` is the authoritative feedback for the current attempt. The worker preserves the strongest candidate and appends exact black-box cold scores between agent rounds. You may use those scores to guide further experiments, but private grader contents and private task data remain inaccessible.

Create the strongest eligible consolidation you can. You may reuse, combine, simplify, or improve the incumbent, triggering challenger, prior candidate, or best candidate from an earlier feedback round, but the result must remain one deterministic, self-contained `train.py` that runs offline without arguments and writes exactly `submission/submission.csv`. It may read only participant-visible task inputs. It must not invoke a shell or subprocess, import author-only code, use private files, assume fixed test identifiers, or depend on anything outside the locked runtime. Preserve a known strong implementation before making speculative changes, run meaningful participant-visible diagnostics, and use later score-feedback rounds to make targeted improvements rather than stopping after the first valid file.

Write:

- `candidate/train.py` — the proposed reference;
- `candidate/README.md` — a compact approach and score-history document that states metric direction once and identifies the implemented approach;
- `candidate/evidence.json` — concrete changes, expected performance rationale, dependency/runtime assessment, and a golden-policy checklist.

Neither shipped file may mention GPT, Codex, model provenance, prior runs, this prompt, or the challenger artifact. Keep comments and documentation concise. Do not invent a score. The worker evaluates each materially new candidate with the exact private cold grader, writes only the score and validation outcome to `score-history.json`, and performs a second independent cold run once a candidate reaches the promotion target.

If a particular approach is exhausted, retain its evidence and try another approach. `no_viable_improvement` is advisory within the bounded outer loop rather than permission to terminate after one round. Otherwise return `success` with workspace-relative paths to all three files. Your final response must match `result-schema.json` exactly.
