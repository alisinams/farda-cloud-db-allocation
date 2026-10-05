#!/usr/bin/env bash
# After the resume chain: refine the weight sweep from step 0.2 to step 0.1, then re-run analysis and figures.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
while ! grep -q "ALL PLANS DONE" logs/run_pipeline_resume.out 2>/dev/null; do sleep 60; done
echo "fine sweep started $(date)"
python -m farda.experiments weights_fine --processes 28 > logs/plan_weights_fine.out 2>&1
python -m farda.analysis > logs/analysis.out 2>&1
python scripts/make_figures.py > logs/figures.out 2>&1
echo "FINE DONE $(date)"
