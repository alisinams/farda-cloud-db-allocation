"""FARDA reproducibility package.

Forecast-driven mixed-integer resource allocation with an auditable decision
record, evaluated on the public PlanetLab CPU-utilisation traces.

Modules
-------
data          PlanetLab trace loading and characterisation
forecasting   rolling-origin LSTM study with naive, seasonal-naive and LightGBM baselines
allocators    exact MILP (HiGHS) and the PSO, GA and ACO comparators
simulator     discrete-event simulation of the control loop, admission path and hosts
tuning        equal-budget random-search tuning of the comparators
experiments   experiment matrix and sweeps (30 seeds, regimes, weights, commit delay)
analysis      statistics, tables and figures reported in the manuscript
"""
__version__ = "1.0.0"
