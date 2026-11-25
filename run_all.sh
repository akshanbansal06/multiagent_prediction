#!/bin/bash
# Runs the whole project in order and saves the output to results/run_all.txt
set -e
mkdir -p results
{
  echo "=== Tests ===";                  python -m pytest -q
  echo; echo "=== Agents vs bookmaker ==="; python -m experiments.run_dc
  echo; echo "=== Fault damage ===";       python -m experiments.fault_demo
  echo; echo "=== Equal vs Hedge ===";     python -m experiments.run_trust
  echo; echo "=== Hedge tuning ===";       python -m experiments.tune_trust
  echo; echo "=== Trust comparison ===";   python -m experiments.run_compare
} 2>&1 | tee results/run_all.txt
