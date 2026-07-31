# mle-harden-eval

Toolkit for triggering GPT-vs-Qwen evaluation runs on hardened MLE-Bench
(Imperium) task pairs — the one-turn "Tencent-style" qualification harness plus
the multi-node panel orchestration and live score collection used for the
hardening sprint (July 2026).

```
harness/         one-turn AIDE-emulating harness (solver → sandbox → grade → review)
prompts/         solver / reviewer prompt templates the harness loads
environment/     Dockerfile for the sandbox image (pandas pinned — see Gotchas)
orchestration/   panel launcher (N runs × before/after × task shards) + chaining
scoring/         live score pullers → CSV trackers
docs/            task-layout spec, delivery packaging example
```

## How a run works

One run = one solver call + one sandboxed execution + one grade:

1. `harness/aide_harness.py` renders the task (description + file listing) into
   the solver prompt and calls the solver model **once** (one-turn).
2. The returned script runs inside the sandbox image (`docker run --network
   none --cpus 20 --memory 128g`, one GPU via CDI), wall-clocked per
   `metadata.json` (60 min for all hardening tasks).
3. `private/grader.py::grade(submission_df) -> float` scores the submission;
   `scores.json` records `final_score` (null + `failure_code` if the run
   produced no submission).
4. A reviewer model (`gpt-5.6-luna`) writes a structured diagnostic of the
   execution.

Solver profiles (in `aide_harness.py::SOLVER_PROFILES`):

| profile | provider | transport |
|---|---|---|
| `gpt5.5-high` | OpenAI | `api.openai.com/v1/responses` |
| `qwen3.6-plus-high` | Vercel AI Gateway | `ai-gateway.vercel.sh/v1/chat/completions` |

Qwen routes through **Vercel AI Gateway** (model `alibaba/qwen3.6-plus`)
because the shared OpenRouter account is ZDR-locked and qwen3.6-plus is served
only by Alibaba's non-ZDR endpoint.

## Box setup (per node)

Assumes the working root `/scratch/mle_hardening/` (all scripts hardcode it):

```bash
/scratch/mle_hardening/
├── PIPELINE/{harness,prompts,environment}   # this repo's harness/, prompts/, environment/
├── tasks/<slug>-{before,after}/             # metadata.json + public/ + private/grader.py
├── artifacts/                               # run outputs (one dir per task/profile-run)
├── logs/
└── .env                                     # copy .env.example, fill keys, chmod 600
```

1. Build the sandbox image: `docker build --network=host -t
   imperium-mlebench-runtime:fixed-20260731-pd2 environment/` (or `docker save
   | ssh <node> docker load` from a node that has it).
2. Register all GPUs with CDI (fresh nodes only know GPU 0):
   `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`
3. Single run:

```bash
cd PIPELINE/harness
export IMPERIUM_RUNTIME_IMAGE=imperium-mlebench-runtime:fixed-20260731-pd2
python3 aide_harness.py /scratch/mle_hardening/tasks/<task> \
  --solver-profile gpt5.5-high --run 1 \
  --artifact-dir /scratch/mle_hardening/artifacts/<task>/gpt5.5-high-1 --gpu 0
```

## Panel orchestration

`orchestration/panel.py` runs 5 runs × before/after over a task shard with a
worker pool (default 12 workers/node — see Gotchas on oversubscription):

```bash
cd /scratch/mle_hardening
# GPT panel over this node's shard:
nohup python3 panel.py statistella forest-fire-prediction-epoch-hackathon ... \
  > logs/panel_driver.log 2>&1 &
# Qwen panel, separate status file:
PANEL_PROFILE=qwen3.6-plus-high PANEL_STATUS=panel_status_qwen.log \
  nohup python3 panel.py <same shard> > logs/panel_driver_qwen.log 2>&1 &
```

- **Skip/redo/resume**: runs with a non-null score are skipped, failed runs are
  cleared and redone, interrupted runs resume — so re-launching is idempotent.
- **Sharding**: split the task list across nodes (we used 4 tasks/node × 3
  nodes); shard membership is just the argv list.
- **Chaining**: `orchestration/qwen_chain.sh` waits for the running GPT driver
  to exit and for a `/scratch/mle_hardening/QWEN_SMOKE_OK` flag, then launches
  the Qwen panel — used to queue the second panel without oversubscribing RAM.
- **Smoke first**: `orchestration/qwen_smoke.sh` pings the gateway and does one
  real harness run on the smallest task before you commit to a full panel.

## Score collection

`scoring/pull_scores.py` (runs on a workstation with ssh access to the nodes)
scans every `artifacts/*/{profile}-{run}/scores.json` across nodes and rewrites
two CSVs atomically — run it in a loop for a live tracker:

- `gpt_panel_scores.csv` — wide diagnostic view (per-run cells, `running`,
  `fail:<code>`)
- `harden_eval_tracker.csv` — the team-sheet format (5-run score lists per
  task × variant × model; errored runs are null)

Edit the `HOSTS`/`SHARDS`/`TASKS` tables at the top to match your deployment.
`scoring/collect_scores.py` is the older single-node best-of/spread tabulator.

## Reading results

Best-of-5 per variant per model, then compare spreads:
`(GPT_after − Qwen_after)` vs `(GPT_before − Qwen_before)` (sign-adjusted for
lower-is-better metrics). A hardened task passes when the spread widens — Qwen
drops while GPT holds — not merely when the task gets harder for everyone.
A score of `0.0` is a *valid* score (e.g. zero skill vs baseline), distinct
from a failed run (null + failure code).

## Gotchas (all cost us real hours)

- **pandas 3.x breaks sandbox code**: `str` dtype makes
  `select_dtypes("object")` return nothing → silent all-zero submissions. The
  image pins pandas 2.2.3 (`fixed-20260731-pd2`).
- **CDI registry**: fresh nodes resolve only `nvidia.com/gpu=0`; regenerate
  (setup step 2) or every run on GPU 1–7 dies with exit 125.
- **`bash -lc` in the sandbox resets PATH** (login shell sources
  `/etc/profile`) → `python: command not found`. The harness uses `bash -c`.
- **`--resume` returns the recorded result** if `scores.json` exists — delete
  the artifact dir to force a rerun.
- **Worker count**: `--cpus` is a soft CFS quota, so 12 sandboxes/node on 156
  cores timeshare fine, but `--memory 128g` × workers must stay under node RAM
  (1.76 TB ⇒ 12 max). Never run two full-width panels at once — chain them.
- **OpenRouter ZDR lock**: request-level `provider.data_collection` overrides
  do NOT bypass an account-level ZDR restriction; use the Vercel route.
- **Flaky ssh to the nodes**: wrap long transfers in retry loops
  (`rsync --partial --timeout=60`), and `nohup` anything long-running on the
  node itself.

## Task layout + delivery packaging

See `docs/TASK_LAYOUT.md` for the harness task format and the
`<slug>-hardened.zip` delivery format (raw/ + task/), and
`docs/build_team_delivery.py.example` for a real packaging + verification
script (regenerates every artifact from raw/ and diffs against the audited
build before zipping).
