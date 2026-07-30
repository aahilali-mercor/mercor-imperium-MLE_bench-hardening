# Synthetic Perturbation Hardening — Playbook

**Status:** method only (no locked sample yet)  
**Goal:** take an *existing* easy Kaggle / Studio task and **search** over random data perturbations until the score **spread** between a weak model and a strong model is large enough to ship as Difficulty 2–3.

This is **not** cross-comp join.  
This is **not** “staple two contests with `min()`.”  
This is: **optimize the dataset surface** (columns + noise + missingness + decoys) under an explicit agent panel objective.

---

## 1. Why this exists

Many delivered tasks saturate: **Qwen / weak (“ancilla”) and GPT both score near the ceiling** → little ranking signal.

Cross-comp join fixes that when a real pair exists. When it doesn’t, we still need a lever that:
1. Makes the **weak model fail more often**
2. Lets **GPT (frontier) still do well** (near expert gold, within ~10%)
3. Keeps a **normal** single target + single metric (same problem statement)

We treat hardness as something we **search for**, not guess once.

---

## 2. Calibration rules (non-negotiable)

Use the same Imperium rules while searching:

| Observation | Action |
|-------------|--------|
| Qwen / ancilla **beats** gold | Bad — update gold or ease perturbation |
| Qwen clears accept band too often (≥2/5 runs) | Bad — harden further or fix gold/band |
| GPT / Opus within **~10%** of gold | OK |
| GPT / Opus **>10%** better than gold | Bad — update gold |
| Weak and strong both near ceiling | Not hardened — keep searching |
| Weak and strong both collapse | Too destructive (noise killed signal) — roll back |

**North star:** weak fails ↑ · GPT holds · expert gold still leads · spread widens vs unperturbed baseline.

---

## 3. Inputs you need

For **one** base task (already in Studio / local package):

| Artifact | Purpose |
|----------|---------|
| Public train / test (or rebuildable from prepare) | Surface to perturb |
| Private answers + `grade.py` / metric | Score every candidate |
| Expert / gold recipe (or strong HGB baseline) | Calibration anchor |
| **Weak runner** — Qwen / ancilla agent (fixed prompt, fixed budget) | Fail-rate signal |
| **Strong runner** — GPT-5.5 (same budget/tooling) | Hold-rate signal |
| Optional: constant + sklearn AutoML | Cheap smoke panel before burning LLM $ |

**Do not** put API keys in the repo. Use env vars only:

```bash
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
```

---

## 4. What you are allowed to perturb

Build a **search space** of reversible transforms. Prefer transforms that change *reasoning / feature use*, not just Bayes noise floor.

### 4.1 Column selection / masking (primary)
- Randomly **drop** a subset of columns from public train+test (same mask).
- Randomly **keep only K** columns (K small).
- Move a few mildly predictive columns to a second file (light multi-source) — optional, keep story coherent.
- Rename columns to opaque names (`f_017`) so agents can’t rely on semantic shortcuts — optional.

### 4.2 Noise / missingness (use carefully)
- Feature noise: Gaussian / Laplace on numeric cols (σ as search param).
- Label noise: **avoid or keep tiny** — destroys gold and shrinks gaps.
- Missingness: MCAR or column-wise NaN rates.
- Junk columns: random or weakly correlated decoys.

### 4.3 Row / split tricks (secondary)
- Stratified subsample of train (harder estimation).
- Hard-negative test slice (only if honest to the original task story).

### 4.4 Explicitly secondary / last resort
Uniform heavy noise that hurts GPT as much as Qwen → **reject** (same reason Option 1 was rejected as primary).

---

## 5. Objective to optimize

For each candidate perturbation `θ`, run the panel and compute a **spread score**.

Example objective (maximize):

```text
spread(θ) =
    w1 * (score_gpt  − score_qwen)          # or metric-oriented gap
  + w2 * (score_gold − score_qwen)
  − w3 * max(0, score_gpt − score_gold − 0.10 * |score_gold|)   # penalty if GPT >> gold
  − w4 * collapse_penalty                   # if gpt also tanks
```

Adapt signs for “lower better” metrics (RMSE): use negated scores or rank gaps.

**Accept θ only if all hold:**
1. `score_qwen` clearly worse than baseline_qwen (or fail rate ↑).
2. `score_gpt` still near gold (within ~10%).
3. Gold still beats / matches GPT appropriately.
4. Task description remains truthful (document what changed).

If many random θ fail, **that base task may not be hardenable this way** — park it and try another.

---

## 6. Search procedure (anyone can follow)

### Step 0 — Freeze baseline
1. Run constant, sklearn, Qwen, GPT, gold on the **unperturbed** task.  
2. Record `reward_mean/std`, pass rate, trajectories.  
3. If already discriminating, **don’t perturb** — ship as-is or use cross-comp instead.

### Step 1 — Define θ space
Example knobs (start small):

| Knob | Example range |
|------|----------------|
| `n_drop_cols` | 0 … 40% of features |
| `n_junk_cols` | 0 … 20 |
| `numeric_noise_σ` | 0 … 0.5 × column std |
| `missing_rate` | 0 … 0.3 |
| `opaque_rename` | on/off |
| seed | many |

### Step 2 — Random search (or coarse Bayesian opt)
For `t = 1…N` (start N=20–50 with sklearn-only; then N=10–20 with LLM panel):

1. Sample `θ_t`.
2. Apply transforms → write a **candidate bundle** (`public/`, same `private/` unless labels untouched).
3. **Smoke panel (cheap):** constant + HistGB / AutoML.  
   - If even strong sklearn collapses → discard θ (too harsh).  
   - If nothing changes vs baseline → discard θ (too weak).
4. **LLM panel (expensive):** Qwen/ancilla × ≥3–5 seeds, GPT × ≥3–5 seeds, same budgets.
5. Score `spread(θ_t)`; keep Pareto: high spread, GPT near gold, Qwen down.

### Step 3 — Lock winner
1. Re-run winner with fresh seeds (confirm not fluke).  
2. Freeze: `perturbation.json` (exact θ + seed + code version).  
3. Rebuild gold on the **perturbed public** data (gold must not see private answers).  
4. Package like any Imperium task: description, public, private, grade, gold.

### Step 4 — Report before/after
Always publish a one-pager row:

| | Baseline | Perturbed winner |
|--|----------|------------------|
| Qwen mean / pass | | |
| GPT mean / pass | | |
| Gold | | |
| Spread / rank order | | |
| θ summary | — | drops, noise, junk, seed |

---

## 7. Recommended workflow diagram

```text
base task
   │
   ├─ baseline panel (Qwen, GPT, gold)
   │
   ├─ sample θ ──apply──► candidate public data
   │                         │
   │                         ├─ sklearn smoke (reject trash)
   │                         └─ LLM panel (Qwen vs GPT)
   │
   ├─ keep if spread↑ and GPT holds and Qwen fails more
   │
   └─ freeze θ + rebuild gold + ship
```

---

## 8. Implementation sketch (for engineers)

Keep transforms pure and logged:

```python
# pseudocode
def apply_theta(train, test, theta, rng):
    cols = list(train.columns)
    # 1) drop
    drop = rng.sample(feature_cols, k=theta.n_drop)
    # 2) junk
    for j in range(theta.n_junk):
        train[f"junk_{j}"] = rng.normal(size=len(train))
        test[f"junk_{j}"] = rng.normal(size=len(test))
    # 3) noise on numeric keep-set
    ...
    # 4) missingness
    ...
    return train, test, audit_dict
```

Persist `audit_dict` next to the bundle so the perturbation is reproducible.

**Agent runners:** plug whatever harness you use for Imperium (OpenAI + OpenRouter). Same tools, same step budget, same temperature policy across θ — otherwise the search is invalid.

---

## 9. Pitfalls

| Pitfall | Why it hurts | Fix |
|---------|--------------|-----|
| Label noise first | Shrinks everyone’s ceiling | Prefer feature/column ops |
| One lucky seed | Fake win | ≥3–5 seeds per model |
| Gold trained on old public | Unfair / broken calibration | Rebuild gold on perturbed public |
| Description lies about data | Invalid task | Document drops/noise honestly or keep opaque but consistent |
| Optimizing only sklearn | Misses agent failure modes | Always confirm with Qwen vs GPT |
| Same θ for all tasks | Overfit method | Per-task search; many tasks won’t qualify |

---

## 10. When to use this vs cross-comp join

| Situation | Prefer |
|-----------|--------|
| Two comps share a real entity key (hash-gate pass) | **Cross-comp join** |
| Single easy flat table, no join pair | **Synthetic perturbation search** |
| Need client story “real multi-source” | Cross-comp / relational |
| Need volume of harder tasks from Studio inventory | Perturbation search on many bases |

Both can coexist. Join is structural; perturbation is **optimization for spread**.

---

## 11. Deliverable checklist (per hardened task)

- [ ] `perturbation.json` (θ, seeds, code hash)
- [ ] Before panel stats (Qwen, GPT, gold)
- [ ] After panel stats + ≥1 failing Qwen trajectory
- [ ] Gold rebuilt on perturbed public
- [ ] Description + grade unchanged in *contract* (metric/target) unless intentionally changed
- [ ] No secrets in git; no raw dumps beyond task package policy

---

## 12. Status / next

1. Pick 2–3 easy Studio tabular bases with **poor current spread**.  
2. Implement transform library + search loop.  
3. Wire Qwen (OpenRouter) + GPT-5.5 (OpenAI) with shared budgets.  
4. Accept first θ that meets calibration; package as sample.  
5. Only then scale the search across more Studio tasks.

**Owner note:** API keys and harness trigger commands are provided out-of-band — never commit them here.
