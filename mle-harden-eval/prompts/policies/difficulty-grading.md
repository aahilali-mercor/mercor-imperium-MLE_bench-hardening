# Post-qualification difficulty grading policy

Estimate the active human time for one competent domain expert using frontier AI to reach the
client threshold on the supplied task. Count comprehension, prompting, review, implementation,
debugging, validation, and submission preparation. Do not count dataset sourcing, authoring,
unattended training, downloads, or other machine-only waiting.

Treat the recorded five GPT-5.5 and ten Qwen outcomes as primary empirical evidence. Strong model
performance reduces the estimate even when the task sounds sophisticated. A familiar recipe is not
hard merely because it uses images, forecasting, geospatial data, or a custom model family; higher
ratings require task-specific wrangling, metric, leakage, validation, modeling, or submission work.

Use these bands:

- 0: essentially immediate/toy work below a meaningful one-minute workflow.
- 1: 1–60 active minutes.
- 2: 61–120 active minutes.
- 3: 121–180 active minutes.
- 4: 181–240 active minutes.
- 5: 241–300 active minutes.

If the realistic estimate exceeds 300 active minutes, return rating 5, retain the actual estimated
range, and include `above_nominal_range` in warnings. Choose the lower defensible band when evidence
is ambiguous. Difficulty is descriptive and must not alter qualification.
