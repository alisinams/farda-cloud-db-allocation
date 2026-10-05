#!/usr/bin/env bash
# Runs after the main plan: regimes, weights, overhead (28 workers each), then the
# solver scaling study on a quiet machine (4 workers), then the analysis.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
while ! grep -q "finished" logs/plan_main.out 2>/dev/null; do sleep 60; done
python -m farda.experiments regimes  --processes 28 > logs/plan_regimes.out  2>&1
python -m farda.experiments weights  --processes 28 > logs/plan_weights.out  2>&1
python -m farda.experiments overhead --processes 28 > logs/plan_overhead.out 2>&1
python -m farda.experiments scale    --processes 4  > logs/plan_scale.out    2>&1
python -m farda.analysis > logs/analysis.out 2>&1
echo "ALL PLANS DONE" >> logs/plan_scale.out
