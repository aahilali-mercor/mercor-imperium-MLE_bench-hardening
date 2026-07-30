# Agent panel — placeholder

Wire GPT-5.5 (OpenAI) and Qwen / ancilla (OpenRouter) here once the Imperium trigger path is shared.

Expected interface (sketch):

```bash
# never commit real keys
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...

# before
python run_agent_panel.py --variant baseline_prejoined --models gpt,qwen

# after
python run_agent_panel.py --variant hardened_two_table --models gpt,qwen
```

Results should update `results/BEFORE_AFTER.md` and `results/readout.json` → `gpt_qwen_panel`.
