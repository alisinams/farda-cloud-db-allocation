#!/usr/bin/env bash
# Full experiment chain after the forecasting refits: margin selection, main matrix, regimes,
# weight sweep, commit-delay sweep, solver scaling (quiet machine), isolated solve times, analysis.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
while [ ! -f results/forecasts/20110403_lstm64_val.npz ] || [ "$(ls results/forecasts/*_val.npz 2>/dev/null | wc -l)" -lt 4 ]; do sleep 30; done
echo "refits complete $(date)"
python scripts/tune_margin.py > logs/tune_margin.log 2>&1
echo "margin selected: $(tail -1 logs/tune_margin.log)"
python -m farda.experiments main     --processes 28 > logs/plan_main.out     2>&1
python -m farda.experiments regimes  --processes 28 > logs/plan_regimes.out  2>&1
python -m farda.experiments weights  --processes 28 > logs/plan_weights.out  2>&1
python -m farda.experiments overhead --processes 28 > logs/plan_overhead.out 2>&1
python -m farda.experiments scale    --processes 4  > logs/plan_scale.out    2>&1
python scripts/solve_time_isolated.py > logs/solve_time_isolated.log 2>&1
python -m farda.analysis > logs/analysis.out 2>&1
python scripts/make_figures.py > logs/figures.out 2>&1
echo "ALL PLANS DONE" >> logs/plan_scale.out
echo "ALL PLANS DONE $(date)"
