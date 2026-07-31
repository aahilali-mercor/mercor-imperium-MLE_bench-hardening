# Source-server task-generation worker

You are constructing one complete, hard-but-honest MLEbench task from the exact source revision mounted at `source/`. Work only inside this attempt directory. The source mount is read-only by policy: never edit, rename, delete, or generate caches inside it.

Read these files completely before inspecting raw source rows or making task decisions:

1. `policy.md`
2. `skill/SKILL.md`
3. `skill/references/privacy-leakage.md`
4. `skill/references/design-playbook.md`
5. `skill/references/package-benchmark-validation.md`
6. `context.json`

Read `skill/references/hardening-pass.md` whenever a generic pilot is too strong or the first viable formulation misses the intended difficulty. Read `skill/references/contrasting-examples.md` when the observation process or negative population is ambiguous.

The context contains the immutable analysis, source inventory summary, idea-file names, previous accepted/rejected goals, active release, and finite policy limits. Inspect source content only as needed and keep raw examples, identifiers, source mappings, privacy secrets, and private diagnostics out of your final response and out of `candidate/public/`.

Continue through the full skill workflow. Do not stop at an idea or package sketch. Within the finite limits, produce candidate comparison, privacy/leakage review, pilot evidence, hardening evidence when triggered, deterministic construction, public/private package, grader mutation tests, isolated reference solution, benchmark ladder, and audit evidence. Difficulty comes from meaningful modeling work, not noise, hidden tricks, confusing formats, or excessive I/O.

## Attempt layout

Keep author-only material under `author/`. A successful attempt must create:

```text
candidate/
├── public/
│   ├── sample_submission.csv
│   └── task-specific participant inputs
├── private/
│   ├── grader.py
│   └── private scoring assets
├── description.md
├── train.py
└── solution-README.md
author/
├── task_plans.md
├── build_task.py
├── config.json
├── generation-evidence.json
├── validation-evidence.json
└── benchmark-evidence.json
```

The canonical payload has no `metadata.json`, `description.md`, `solutions/`, or hidden-test tree inside public/private. Description, metadata, and solution revisions are database-native after publication. The builder and author evidence stay outside the payload.

`description.md` must contain `## Task`, `## Metric`, `## Submission Format`, and `## Dataset` in that order. `private/grader.py` exposes `grade(submission: pandas.DataFrame) -> float`. `train.py` follows the golden-solution policy exactly, runs without arguments and offline, reads only participant-visible inputs, and writes exactly `submission/submission.csv` relative to its execution workspace.

Use one lowercase kebab-case slug that describes the goal and does not encode a provisional difficulty rating. Return a concise, non-empty category label. Existing repository labels are preferred, but a genuinely useful new category is accepted and can be normalized during later metadata review. CPU tasks must choose 600–900 seconds. GPU tasks must choose 600–3600 seconds and justify the value in evidence. Resource values must fit this source server.

Return proposed baseline, bronze, silver, and gold thresholds from the measured benchmark ladder. They must be finite, directionally ordered, separated beyond ordinary rerun/evaluation uncertainty, and leave the cold reference score a reproducibility margin. These are proposal evidence for GPT-calibrated freezing; do not create the post-Qwen client threshold.

If no candidate survives all validity, privacy, leakage, learnability, package, and finite-search gates, create `author/generation-evidence.json` with the complete terminal search evidence and return `no_viable_task`. Do not fabricate a task to avoid a failed generation attempt.

Your final response must contain only the structured object required by `result-schema.json`. All paths are relative to this attempt directory. Do not include raw source values in the response.
