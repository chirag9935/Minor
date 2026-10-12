"""
Phase 2, Milestone 7, Stage B: K=1, M in {32, 64} DRL study (phase-only
agent), same algorithms/settings as Stage A (see exp_drl_stageA.py and
docs/ASSUMPTIONS.md for the environment design and the CPU-time budget
fix this reuses). Run per the spec only because Stage A succeeded (the
agent clearly learned, beating random phase, in a reproducible way across
seeds) -- per the spec's own staging ("Stage B ... same settings").

Run: python -m experiments.exp_drl_stageB
"""
from config import Config
from experiments.exp_drl_stageA import run_stage


def main():
    run_stage(Config(K=1, Mx=8, My=4), "exp_drl_stageB_M32")  # M=32
    run_stage(Config(K=1, Mx=8, My=8), "exp_drl_stageB_M64")  # M=64


if __name__ == "__main__":
    main()
