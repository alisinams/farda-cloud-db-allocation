#!/usr/bin/env bash
# After the resume chain: re-run the adversarial runs made before the tampered-latency double-count fix
# (2026-09-28 10:15), then refine the weight sweep from step 0.2 to step 0.1, re-running the analysis and
# figures after each stage. Runs whose result files exist are skipped, so the script can be re-run.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
while ! grep -q "ALL PLANS DONE" logs/run_pipeline_resume.out 2>/dev/null; do sleep 60; done
echo "adversarial rerun started $(date)"
if [ ! -f logs/adversarial_fix.done ]; then
    rm -f results/runs/adversarial_*_overhead.json results/runs/adversarial_*_regimes.json
    touch logs/adversarial_fix.done
fi
python -m farda.experiments regimes  --processes 28 > logs/plan_regimes_rerun.out  2>&1
python -m farda.experiments overhead --processes 28 > logs/plan_overhead_rerun.out 2>&1
python -m farda.analysis > logs/analysis.out 2>&1
python scripts/make_figures.py > logs/figures.out 2>&1
echo "ADV RERUN DONE $(date)"
echo "fine sweep started $(date)"
python -m farda.experiments weights_fine --processes 28 > logs/plan_weights_fine.out 2>&1
python -m farda.analysis > logs/analysis.out 2>&1
python scripts/make_figures.py > logs/figures.out 2>&1
echo "FINE DONE $(date)"
