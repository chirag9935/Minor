#!/bin/bash
# Reproduces every figure (plots/) and result (results/) in this project
# from a clean checkout, per the Phase 2 spec section 10 ("Tests and
# quality gates"). Run from ris_project/ or anywhere (the script cd's to
# its own directory first).
#
# Usage:
#   ./run_all.sh          full run (what actually regenerates trustworthy
#                          numbers -- this takes HOURS, not minutes: the
#                          DRL stages (M7) and the SDR sweep (M6b) alone
#                          took most of a day on this machine, mainly due
#                          to unrelated background processes repeatedly
#                          triggering OS-level low-memory kills, not the
#                          compute itself -- see docs/ASSUMPTIONS.md)
#   ./run_all.sh --fast   smoke test: every script runs at a trivial
#                          Monte-Carlo scale (RIS_FAST=1, honoured via
#                          experiments._common.mc_draws()/FAST) -- confirms
#                          every script still runs end-to-end without
#                          error in a few minutes. Its numbers are NOT
#                          meant to be trusted or committed.
#
# Both modes run pytest first and stop immediately if it fails (a broken
# unit test means every downstream number is suspect).
set -e
cd "$(dirname "$0")"

FAST=0
if [ "$1" == "--fast" ]; then
    FAST=1
    export RIS_FAST=1
    echo "=== run_all.sh --fast: smoke-test mode (RIS_FAST=1) ==="
fi

PY="./.venv/Scripts/python.exe"
if [ ! -f "$PY" ]; then
    PY="python"
fi

echo ""
echo "=== pytest ==="
"$PY" -m pytest -q

run() {
    echo ""
    echo "=== $1 ==="
    "$PY" -m "experiments.$1"
}

# Phase 1
run plot_geometry
run exp1_rate_vs_power
run exp2_rate_vs_M
run exp2b_crossover
run exp3_convergence
run exp4_ee_vs_se
run exp5_distance
run exp6_frequency
run exp7_drl_demo

# exp1's "realistic" (lower-gain) antenna preset -- an explicit, committed
# sensitivity point alongside the default preset, not a replacement for it
# (see Config.realistic()'s docstring). Skipped in --fast mode: it is a
# second full sweep of the same scale as the default-preset run just
# above, adding smoke-test time without exercising any new code path.
if [ "$FAST" -eq 0 ]; then
    echo ""
    echo "=== exp1_rate_vs_power (realistic preset) ==="
    "$PY" -c "from experiments.exp1_rate_vs_power import main; main('realistic')"
fi

# Phase 2
run exp_blockage_sweep        # Milestone 0
run exp_objB_power_vs_sinr    # Milestone 1
run exp_objB_power_vs_M       # Milestone 1
run exp_objB_outage           # Milestone 1
run exp_objB_convergence      # Milestone 1
run exp_objC_ee_optimal       # Milestone 2
run exp_absorption_itu        # Milestone 3
run exp_hw_quantization       # Milestone 4
run exp_csi_error             # Milestone 5
run exp_rate_vs_K             # Milestone 6(a)
run exp_sdr_vs_manifold       # Milestone 6(b)
run exp_drl_stageA            # Milestone 7, Stage A
run exp_drl_stageB            # Milestone 7, Stage B

run make_summary

echo ""
echo "run_all.sh done ($([ "$FAST" -eq 1 ] && echo "fast/smoke-test" || echo "full") mode)."
echo "See plots/ and results/ (results/summary.csv for the headline numbers)."
