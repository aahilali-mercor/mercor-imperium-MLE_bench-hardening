#!/bin/bash
# Launch the Qwen panel once the GPT driver exits and the smoke flag exists.
while pgrep -f 'python3 panel_gpt5.py' >/dev/null; do sleep 120; done
end=$((SECONDS + 14400))
while [ ! -f /scratch/mle_hardening/QWEN_SMOKE_OK ]; do
  if [ $SECONDS -gt $end ]; then echo "$(date -u) no smoke flag after 4h; aborted" >> /scratch/mle_hardening/logs/qwen_chain.log; exit 1; fi
  sleep 120
done
cd /scratch/mle_hardening
PANEL_PROFILE=qwen3.6-plus-high PANEL_STATUS=panel_status_qwen.log nohup python3 panel_gpt5.py statistella forest-fire-prediction-epoch-hackathon russian-car-plates-prices-prediction viral-vision-the-you-tube-virality-predictor-challenge > logs/panel_driver_qwen.log 2>&1 &
echo "$(date -u) qwen panel launched pid=$!" >> /scratch/mle_hardening/logs/qwen_chain.log
