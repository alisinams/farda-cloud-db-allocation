# FARDA: forecast-driven mixed-integer request allocation for cloud database replicas

Code, configuration and results of the trace-driven evaluation reported in the manuscript
*Forecast-driven mixed-integer request allocation for multi-tenant cloud database replicas evaluated on PlanetLab traces* (submitted for publication).

FARDA plans request assignments once per 300 s control interval with a mixed-integer linear programme
(HiGHS, 0.1% gap, 5 s limit). The programme is fed by an LSTM forecast of each virtual machine's background
processor utilisation plus a calibrated margin. Requests are admitted against the plan in constant time, and the
accountability (ledger) layer is modelled as a commit delay. The evaluation replays the public PlanetLab
CPU-utilisation traces in a discrete-event simulator and compares FARDA with tuned particle swarm, genetic and ant
colony allocators and a no-optimiser proportional baseline, with 30 paired seeds at 10, 30 and 50 hosts.

## Layout

| Path | Content |
|---|---|
| `farda/data.py` | PlanetLab trace loading (ten days, one file per VM, 288 samples at 300 s) |
| `farda/forecasting.py` | Rolling-origin forecasting study: LSTM (16 to 256 units), last value, seasonal naive, LightGBM |
| `farda/allocators.py` | Exact MILP on HiGHS and the PSO, GA and ACO comparators on the same objective |
| `farda/simulator.py` | Discrete-event simulator: hosts, VMs bound to traces, processor-sharing queues (Numba), plan-based admission, commit-delay model, energy |
| `farda/tuning.py` | Equal-budget random-search tuning of the comparators (30 configurations × 24 instances × 2 restarts) |
| `farda/experiments.py` | Experiment plans: `main` (forecaster × allocator cells), `regimes`, `weights`, `weights_fine`, `overhead`, `scale` |
| `farda/analysis.py` | Descriptive statistics, paired Wilcoxon tests with Holm correction, bootstrap intervals, Welch ANOVA, Games-Howell, Friedman, Cliff's delta, TOST, Pareto fronts |
| `scripts/fetch_data.py` | Downloads the PlanetLab traces to `data/` |
| `scripts/run_forecasting.py`, `scripts/refit_cpu.py` | Forecasting study (24 jobs) and single-fold refits |
| `scripts/tune_margin.py` | Selects the margin quantile on the validation day 20110403 (`results/margin_alpha.json`) |
| `scripts/solve_time_isolated.py` | Uncontended solver timing at 10, 30 and 50 hosts |
| `scripts/run_*.sh` | The experiment chains as they were run (resumable: existing run files are skipped) |
| `figures_svis/` | Scripts for Figures 1 to 4 of the manuscript and the exported figures (`out/`) |
| `scripts/make_figures.py`, `figures/` | Quick-look plots used during the study (not the manuscript figures) |
| `results/runs/` | 1907 run files, one per simulated day: configuration, seed, every metric, per-interval solver log |
| `results/summary/` | CSV tables and `report.json`, the single source of every number in the manuscript |
| `results/forecasts/` | One-step LSTM forecasts consumed by the simulator, per held-out day and hidden size |
| `results/*.json` | Forecasting metrics per fold, tuning results, margin selection, solver timing studies |

## Setup

```
pip install -r requirements.txt
python scripts/fetch_data.py
```

Python 3.13 was used. A CUDA GPU shortens the forecasting study (about 35 minutes on an RTX 3090) but is not
required. The traces can be placed elsewhere and located with the `PLANETLAB_ROOT` environment variable.

## Reproducing the results

```
python scripts/run_forecasting.py        # forecasting study: results/forecast_*.json and results/forecasts/
python -m farda.tuning 24                # comparator tuning: results/tuning.json
python scripts/tune_margin.py            # margin level on the validation day: results/margin_alpha.json
python -m farda.experiments main --processes 28
python -m farda.experiments regimes --processes 28
python -m farda.experiments weights --processes 28
python -m farda.experiments overhead --processes 28
python -m farda.experiments scale --processes 4
python scripts/solve_time_isolated.py
python -m farda.analysis                 # results/summary/
python figures_svis/build_all.py         # figures_svis/out/
```

Every run is deterministic given its seed: seed `s` fixes the held-out day (`s mod 4`), the host draw, the trace
sample, the arrival stream and the metaheuristic random streams. The 30 seeds therefore share four trace days, and
effects are also reported per day. Runs already present in `results/runs/` are skipped, so a plan can be resumed.

## Notes on the reported results

* **Excluded cell.** The 30 runs of the exact allocator on the LSTM point forecast without margin at 50 hosts read
  LSTM forecast files written before the final forecasts. `farda/analysis.py` excludes this cell (`EXCLUDED_CELLS`,
  recorded in `report.json` under `excluded_cells`); the run files are kept for transparency.
* **Solver timing.** Solve times in the main plans were measured with up to 28 single-threaded solver processes
  sharing one 32-thread workstation, so they are conservative; `results/solve_time_isolated.json` holds the
  uncontended reference. Solve time is not charged to plan activation in the simulator.
* **Measured versus modelled.** Forecasting errors are measured on the real traces. Latency, energy, service-level
  violations, exposure and solve time are simulated with real trace input. Request arrivals, request classes,
  sensitivity labels and shared-tier flags are synthetic. The accountability layer is a declared commit delay, not a
  deployed ledger, and the commit-cost index is a property of that delay model.
* **Not evaluated.** An adversary's co-residency success probability, a deployed ledger, and memory or network
  contention.

## Licence

The code is released under the MIT licence (`LICENSE`). The PlanetLab traces belong to their original publishers,
are not redistributed here and are not covered by this licence.

## Citation

Please cite the accompanying article (details will be added on publication) and the archived release of this
repository (DOI to be added).
