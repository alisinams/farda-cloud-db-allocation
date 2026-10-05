#!/usr/bin/env bash
# Resume the experiment chain. Every plan skips runs whose result file already exists, so this
# script can be re-run after an interruption. The solver timing studies run first so that they
# are measured on a quiet machine; the simulation plans follow.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
echo "resume started $(date)"
python -m farda.experiments scale    --processes 4  > logs/plan_scale.out    2>&1
echo "scale done $(date)"
python scripts/solve_time_isolated.py > logs/solve_time_isolated.log 2>&1
echo "isolated done $(date)"
python -m farda.experiments main     --processes 28 > logs/plan_main.out     2>&1
echo "main done $(date)"
python -m farda.experiments overhead --processes 28 > logs/plan_overhead.out 2>&1
echo "overhead done $(date)"
python -m farda.experiments regimes  --processes 28 > logs/plan_regimes.out  2>&1
echo "regimes done $(date)"
python -m farda.experiments weights  --processes 28 > logs/plan_weights.out  2>&1
echo "weights done $(date)"
python -m farda.analysis > logs/analysis.out 2>&1
python scripts/make_figures.py > logs/figures.out 2>&1
echo "ALL PLANS DONE $(date)"
