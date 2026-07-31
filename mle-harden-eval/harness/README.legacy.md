# Direct evaluation harness

The harness talks directly to the provider API. It does not launch Cline or Codex and does not use MCP. The model receives exactly three tools: `bash`, `submit`, and `final-submit`.

```bash
# Production run (30-turn default)
python harness/main.py aviation-attention-easy --backend hy3 --model-name hy3 --run 1 --force

# Run all five trajectories concurrently, with a status summary every 2.5 minutes
python harness/main.py aviation-attention-easy --backend hy3 --model-name hy3 --run all --force

# Run six trajectories, including one spare replacement trajectory
python harness/main.py aviation-attention-easy --backend hy3 --model-name hy3 --run bonus --force

# Use the capacity-one GPU queue for all preview/final scoring
python harness/main.py aviation-attention-easy --backend hy3 --run bonus --serial --force

# Short five-turn protocol/debug run
python harness/main.py aviation-attention-easy --backend hy3 --model-name hy3 --run 1 --max-turns 5 --force

# Resume an interrupted trajectory from its exact saved step
python harness/main.py aviation-attention-easy --backend hy3 --model-name hy3 --run 1 --resume

# Supervise all five GPT-5.5-high and five Hy3 zas-medium runs, five at a time
python harness/run_zas_batch.py --max-parallel 5

# Direct OpenAI API backend
python harness/main.py aviation-attention-easy --backend openai --model-name gpt5.5-high --run 1

# Claude Opus 4.8 with high reasoning through the general OpenRouter account
python harness/main.py aviation-attention-easy --backend claude --run 1

# Claude Sonnet 5 with high reasoning through the general OpenRouter account
python harness/main.py aviation-attention-easy --backend sonnet --run 1

# Grok 4.5 with high reasoning through the general OpenRouter account
python harness/main.py aviation-attention-easy --backend grok --run 1

# Gemini 3.5 Flash with high reasoning through the general OpenRouter account
python harness/main.py aviation-attention-easy --backend gemini --run 1

# Gemini 3.1 Pro Preview with high reasoning through the general OpenRouter account
python harness/main.py aviation-attention-easy --backend gemini-pro --run 1

# GLM 5.2 with high reasoning through OpenRouter
python harness/main.py aviation-attention-easy --backend glm --run 1
```

`--force` replaces only the selected `<model-name>-<run>` directory. Run numbers are restricted to 1–6. `--api-model` changes the provider model ID without changing the artifact directory name. `--print-system-prompt` prints the exact generated system prompt without starting a run.

`--run all` starts runs 1–5 concurrently; `--run bonus` starts runs 1–6 so one spare trajectory is available as a replacement. Both modes print an immediate status table, refresh it every 2.5 minutes, and print a clearly labeled final table when all children exit. A zero-exit run is shown as `completed` whether it used every turn or called `final-submit` early; abnormal exits remain visible as `failed(exit=N)`. Each row also includes the model turn, best score seen so far, and remaining preview-submit attempts. Ctrl-C is forwarded to every child so resumable checkpoints are preserved.

By default, model requests and ordinary `bash` work remain concurrent while every `submit` and `final-submit` scoring operation enters a host-wide CPU queue that admits at most two scoring jobs at once. Passing `--serial` selects the separate host-wide GPU queue, which admits one scoring job at a time. Commands launched from different terminals share the corresponding queue. Queue slots are held until the complete scoring operation exits and are automatically released by the operating system if a waiting or active harness process dies. Queue wait, acquisition, and release events—including the queue name, capacity, and acquired slot—are recorded in the trajectory.

The default and production ceiling are 30 turns. A smaller value such as `--max-turns 5` is useful for protocol debugging.

Credentials are always required for a run. Hy3 exclusively uses `OPENROUTER_KEY_hy3`; Claude, Sonnet, Grok, Gemini, Gemini Pro, and GLM exclusively use the separate `OPENROUTER_KEY`; the OpenAI backend uses `OPENAI_KEY`. General OpenRouter backends never fall back to the Hy3 key. Keys are held by the parent process and are never exposed to model tools, prompts, scratch files, or artifacts. Hy3 requests explicitly pin `atlas-cloud/fp8`, require FP8/tool parameters, and disable fallbacks. Claude defaults to `anthropic/claude-opus-4.8` with high reasoning and the artifact name `claude-opus-4.8-high`. Sonnet defaults to `anthropic/claude-sonnet-5` with high reasoning and the artifact name `claude-sonnet-5-high`. Grok defaults to `x-ai/grok-4.5` with high reasoning and the artifact name `grok-4.5-high`. Gemini defaults to `google/gemini-3.5-flash` with high reasoning and the artifact name `gemini-3.5-flash-high`. Gemini Pro defaults to `google/gemini-3.1-pro-preview` with high reasoning and the artifact name `gemini-3.1-pro-preview-high`. GLM defaults to `z-ai/glm-5.2` with high reasoning and the artifact name `glm-5.2-high`; normal turns use OpenRouter's default provider selection. The mandatory final tool-call turn uses the provider-agnostic `:exacto` route for tool-call compliance.

The parent process retains the complete conversation history and retries transient provider errors without consuming model turns. Hy3 uses OpenRouter Chat Completions. GPT-5.5 uses OpenAI's Responses API with stateless, Zero-Data-Retention-compatible history replay (including encrypted reasoning continuity). On the final allowed response, the API exposes only `final-submit` and requires a tool call; the prompt tells the model to have `train.py` ready before then.

Every harness event is emitted immediately as a JSON terminal line. Tool stdout and stderr are printed in labeled blocks. API requests, provider errors and scheduled retries, assistant responses, tool calls, submit starts, submit results, checkpoint saves, pauses, resumes, and completion are all visible live.

While a trajectory is active, all four canonical artifacts are atomically refreshed after every event. `checkpoint.json` and `.checkpoints/<id>/` additionally store the exact provider conversation, counters, pending tool-call index, and a versioned workspace snapshot at every safe boundary. An interrupted run exits as resumable instead of being recorded as a false final failure. `--resume` restores the referenced snapshot and continues from `ready_request`, `request_inflight`, or the next pending tool call. A tool marked in-flight at interruption is reported back as interrupted and is never replayed automatically, avoiding duplicate command or scoring compute. Checkpoint internals are removed after terminal completion, leaving exactly the four canonical artifacts.

The zas batch supervisor writes line-flushed per-run logs and its own durable state under `logs/zas-medium/`. A paused or crashed child with a checkpoint is automatically relaunched with `--resume`, up to 20 resumptions. It runs no more than five new trajectories concurrently.

Every `bash`, preview, and final execution runs in a fresh network-disabled, read-only-root Docker container. The writable scratch workspace contains a read-only `public` symlink; task `private`, `hidden-test`, `solutions`, credentials, and other runs are not mounted. `bash` receives 180 seconds. Each `submit` runs `train.py` once on visible inputs and each `final-submit` runs it once on hidden-final inputs; both receive the task metadata's full wall-clock limit. One selected GPU is exposed (`--gpu 0` by default). A host-wide per-run lock rejects concurrent harness processes targeting the same artifact directory.

Permanent output is:

```text
tasks-evals/<slug>/<model-name>-<run>/
├── trajectory.json
├── stdout.txt
├── scores.json
└── train.py
```

The system prompt describes only the MLE-bench-style objective and operational contract: score direction, exact file names, tool behavior, used/remaining/maximum budgets, compute limits, hidden-final behavior, and terminal failure conditions. It deliberately gives no modeling-strategy advice.

Completed `trajectory.json` and `scores.json` artifacts do not embed the run number or launch directory, and terminal event logs do not prefix events with it. The containing `<model-name>-<run>` directory is the sole permanent run-number label, so a completed spare directory can be renamed as a replacement without stale identity metadata. Resumable checkpoints retain private run identity until completion to prevent restoring a checkpoint into the wrong active run.
