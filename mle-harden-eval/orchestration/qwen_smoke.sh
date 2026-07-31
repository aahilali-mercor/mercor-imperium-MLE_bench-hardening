#!/bin/bash
# Smoke-test the Vercel AI Gateway qwen route, then one harness run.
# Usage: AI_GATEWAY_API_KEY must be in /scratch/mle_hardening/.env first.
set -u
cd /scratch/mle_hardening
export $(grep -v '^#' .env | xargs)

if [ -z "${AI_GATEWAY_API_KEY:-}" ]; then
  echo "FATAL: AI_GATEWAY_API_KEY not in .env"; exit 1
fi

echo "== 1. raw gateway ping (both candidate slugs) =="
for slug in alibaba/qwen3.6-plus qwen/qwen3.6-plus; do
  echo "--- $slug"
  curl -sS -m 120 https://ai-gateway.vercel.sh/v1/chat/completions \
    -H "Authorization: Bearer $AI_GATEWAY_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$slug\", \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly: pong\"}], \"max_tokens\": 2000}" \
    | python3 -c 'import json,sys
b = json.load(sys.stdin)
if "error" in b: print("ERROR:", b["error"])
else:
    c = b["choices"][0]["message"]["content"]
    print("OK model=", b.get("model"), "| content=", (c or "")[:80])'
done

echo "== 2. harness smoke run (forest-fire-before, qwen run 1) =="
cd PIPELINE/harness
export IMPERIUM_RUNTIME_IMAGE=imperium-mlebench-runtime:fixed-20260731-pd2
art=/scratch/mle_hardening/artifacts/forest-fire-prediction-epoch-hackathon-before/qwen3.6-plus-high-1
rm -rf "$art"
python3 aide_harness.py /scratch/mle_hardening/tasks/forest-fire-prediction-epoch-hackathon-before \
  --solver-profile qwen3.6-plus-high --run 1 --artifact-dir "$art" --gpu 7 \
  2>&1 | tail -5
echo "== scores =="
cat "$art/scores.json" 2>/dev/null | python3 -m json.tool | head -12 || echo "no scores.json"
