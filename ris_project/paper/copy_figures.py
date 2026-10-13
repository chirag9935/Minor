"""
Copies the curated subset of plots/*.png actually referenced by
paper/main.tex into paper/figures/, per the Phase 2 spec ("figures/
copied from plots/ at the end of the run via a script"). Run this after
run_all.sh (or at least after the experiments that produce the figures
below) so paper/figures/ reflects the current results.

Run: python -m paper.copy_figures   (from ris_project/)
  or: python paper/copy_figures.py
"""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(ROOT, "plots")
FIGURES_DIR = os.path.join(ROOT, "paper", "figures")

# Every figure cited in main.tex, in the order they appear there.
FIGURES = [
    "geometry.png",
    "exp1_rate_vs_power_K1.png",
    "exp2_rate_vs_M_K1.png",
    "exp2b_crossover_K1.png",
    "exp3_convergence_K1.png",
    "exp_blockage_sweep_M64.png",
    "exp_objB_power_vs_sinr.png",
    "exp_objC_ee_se_tradeoff.png",
    "exp6_rate_vs_frequency.png",
    "exp_absorption_vs_frequency.png",
    "exp_hw_quantization_M64.png",
    "exp_csi_error_vs_epsilon_K1.png",
    "exp_csi_error_vs_epsilon_K4.png",
    "exp_rate_vs_K_sum.png",
    "exp_sdr_vs_manifold_sinr.png",
    "exp_drl_stageA_learning_curve.png",
    "exp_drl_stageB_M64_learning_curve.png",
]


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    missing = []
    for name in FIGURES:
        src = os.path.join(PLOTS_DIR, name)
        dst = os.path.join(FIGURES_DIR, name)
        if not os.path.exists(src):
            missing.append(name)
            continue
        shutil.copyfile(src, dst)
        print(f"  copied {name}")
    if missing:
        print(f"\nWARNING: {len(missing)} figure(s) referenced by main.tex were not found "
              f"in plots/ -- run the corresponding experiment script(s) first: {missing}")
    print(f"\n{len(FIGURES) - len(missing)}/{len(FIGURES)} figures copied to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
