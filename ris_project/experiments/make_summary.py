"""
Builds results/summary.csv: headline numbers pulled from the already-saved
results/*.npz files -- run run_all.sh (or at least the exp1-exp6 and the
Phase 2 milestone scripts) first.
Run: python -m experiments.make_summary
"""
import csv
import os

import numpy as np

from experiments._common import RESULTS_DIR


def load(name):
    return np.load(os.path.join(RESULTS_DIR, f"{name}.npz"))


def main():
    rows = []

    d = load("exp1_rate_vs_power_K1")
    i30 = int(np.argmin(np.abs(d["ptx_dbm"] - 30)))
    rows.append(("AO-RIS rate @ Ptx=30dBm, K=1 [bit/s/Hz]", d["AO-RIS"][i30]))
    rows.append(("Random-phase rate @ Ptx=30dBm, K=1 [bit/s/Hz]", d["Random-phase"][i30]))
    rows.append(("No-RIS rate @ Ptx=30dBm, K=1 [bit/s/Hz]", d["No-RIS"][i30]))
    rows.append(("AF relay rate @ Ptx=30dBm, K=1 [bit/s/Hz]", d["AF relay"][i30]))
    rows.append(("AO gain over random-phase @ Ptx=30dBm, K=1 [x]",
                  d["AO-RIS"][i30] / d["Random-phase"][i30]))
    rows.append(("AO gain over no-RIS @ Ptx=30dBm, K=1 [x]",
                  d["AO-RIS"][i30] / d["No-RIS"][i30]))

    d4 = load("exp1_rate_vs_power_K4")
    i30_4 = int(np.argmin(np.abs(d4["ptx_dbm"] - 30)))
    rows.append(("AO-RZF rate @ Ptx=30dBm, K=4 [bit/s/Hz]", d4["AO-RZF"][i30_4]))
    rows.append(("AO-ZF rate @ Ptx=30dBm, K=4 [bit/s/Hz]", d4["AO-ZF"][i30_4]))
    rows.append(("Random-phase rate @ Ptx=30dBm, K=4 [bit/s/Hz]", d4["Random-phase"][i30_4]))
    rows.append(("No-RIS rate @ Ptx=30dBm, K=4 [bit/s/Hz]", d4["No-RIS"][i30_4]))
    rows.append(("AO-RZF gain over random-phase @ Ptx=30dBm, K=4 [x]",
                  d4["AO-RZF"][i30_4] / d4["Random-phase"][i30_4]))

    e2 = load("exp2_rate_vs_M_K1")
    i256 = int(np.argmin(np.abs(e2["M"] - 256)))
    i64 = int(np.argmin(np.abs(e2["M"] - 64)))
    rows.append(("AO-RIS rate @ M=64, Ptx=30dBm, K=1 [bit/s/Hz]", e2["AO-RIS"][i64]))
    rows.append(("AO-RIS rate @ M=256, Ptx=30dBm, K=1 [bit/s/Hz]", e2["AO-RIS"][i256]))
    rows.append(("AO-RIS array-gain scaling M=64->256 [x, expect ~(256/64)^2=16]",
                  (2 ** e2["AO-RIS"][i256]) / (2 ** e2["AO-RIS"][i64])))

    e2_4 = load("exp2_rate_vs_M_K4")
    rows.append(("AO-RZF rate @ M=64, Ptx=30dBm, K=4 [bit/s/Hz]", e2_4["AO-RZF"][i64]))
    rows.append(("AO-RZF rate @ M=256, Ptx=30dBm, K=4 [bit/s/Hz]", e2_4["AO-RZF"][i256]))
    rows.append(("No-RIS rate @ Ptx=30dBm, K=4 (M-independent) [bit/s/Hz]", e2_4["No-RIS"][i64]))

    e3 = load("exp3_convergence_K1")
    rows.append(("AO (K=1) iterations to converge (mean curve)", e3["curves"].shape[1]))
    e3b = load("exp3_convergence_K4")
    rows.append(("AO (K=4) iterations to converge (mean curve)", e3b["curves"].shape[1]))

    e6 = load("exp6_rate_vs_frequency")
    i0 = 0
    rows.append(("AO-RIS rate @ 28 GHz, shortest distance [bit/s/Hz]", e6["28 GHz"][i0]))
    rows.append(("AO-RIS rate @ 140 GHz, shortest distance [bit/s/Hz]", e6["140 GHz"][i0]))
    rows.append(("AO-RIS rate @ 300 GHz, shortest distance [bit/s/Hz]", e6["300 GHz"][i0]))

    # ---- Phase 2 ----------------------------------------------------
    # Each block is independently try/except'd: under RIS_FAST=1 some
    # grids shrink (e.g. SDR drops M=64, DRL drops 2 of 3 seeds) and a
    # result file may be from a partial/fast run rather than the full
    # one -- a missing/reshaped metric here should not take down every
    # other row. Index lookups use argmin-to-nearest rather than a fixed
    # position so they degrade gracefully to whatever grid is present.
    try:  # Milestone 0: blockage realism
        bs = load("exp_blockage_sweep_M64")
        i_cross = int(np.argmin(np.abs(bs["AO-RIS"] - bs["No-RIS"])))
        rows.append(("Blockage AO-RIS/No-RIS crossover [dB] (M=64, nearest grid point)",
                      bs["blockage_db"][i_cross]))
    except Exception as e:
        print(f"  (skipped Milestone 0 row: {e})")

    try:  # Milestone 1: Objective B, min-power
        ob_sinr = load("exp_objB_power_vs_sinr")
        i15 = int(np.argmin(np.abs(ob_sinr["sinr_db"] - 15)))
        rows.append(("Min Tx power for AO-RIS @ target SINR=15dB, K=1 [W]", ob_sinr["AO-RIS_power_w"][i15]))
        rows.append(("Min Tx power for No-RIS @ target SINR=15dB, K=1 [W]", ob_sinr["No-RIS_power_w"][i15]))
        ob_M = load("exp_objB_power_vs_M")
        i64b, i256b = int(np.argmin(np.abs(ob_M["M"] - 64))), int(np.argmin(np.abs(ob_M["M"] - 256)))
        rows.append(("Min Tx power for AO-RIS, M=64->256 ratio [x, power should fall]",
                      ob_M["AO-RIS_power_w"][i256b] / ob_M["AO-RIS_power_w"][i64b]))
    except Exception as e:
        print(f"  (skipped Milestone 1 rows: {e})")

    try:  # Milestone 2: Objective C, energy efficiency
        oc = load("exp_objC_ee_optimal")
        i64c = int(np.argmin(np.abs(oc["M"] - 64)))
        rows.append(("EE-optimal Ptx @ M=64 [dBm]", oc["ee_optimal_ptx_dbm"][i64c]))
        rows.append(("Max EE @ M=64 [bit/Joule]", oc["ee_max"][i64c]))
    except Exception as e:
        print(f"  (skipped Milestone 2 rows: {e})")

    try:  # Milestone 3: ITU-R P.676 molecular absorption
        ab = load("exp_absorption_300ghz_distance")
        i_mid = int(np.argmin(np.abs(ab["distance"] - 20)))
        rows.append(("AO-RIS rate @ 300 GHz, d=20m, ITU model [bit/s/Hz]", ab["AO-RIS_itu"][i_mid]))
        rows.append(("AO-RIS rate @ 300 GHz, d=20m, placeholder model [bit/s/Hz]", ab["AO-RIS_placeholder"][i_mid]))
    except Exception as e:
        print(f"  (skipped Milestone 3 rows: {e})")

    try:  # Milestone 4: discrete RIS phase shifts
        hw = load("exp_hw_quantization")
        rows.append(("Continuous-phase rate @ M=64 [bit/s/Hz]", np.mean(hw["M64_rate_cont"])))
        rows.append(("1-bit refined-quantised rate @ M=64 [bit/s/Hz]", np.mean(hw["M64_rate_refined_1bit"])))
        rows.append(("4-bit refined-quantised rate @ M=64 [bit/s/Hz]", np.mean(hw["M64_rate_refined_4bit"])))
    except Exception as e:
        print(f"  (skipped Milestone 4 rows: {e})")

    try:  # Milestone 5: imperfect CSI
        csi1 = load("exp_csi_error_vs_epsilon_K1")
        rows.append(("AO-RIS rate, perfect CSI (eps=0), K=1 [bit/s/Hz]", csi1["AO-RIS"][0]))
        rows.append(("AO-RIS rate, eps=0.5, K=1 [bit/s/Hz]", csi1["AO-RIS"][-1]))
        csi4 = load("exp_csi_error_vs_epsilon_K4")
        rows.append(("AO-RZF rate, perfect CSI (eps=0), K=4 [bit/s/Hz]", csi4["AO-RZF (plain)"][0]))
        rows.append(("AO-RZF rate, eps=0.5, K=4 [bit/s/Hz]", csi4["AO-RZF (plain)"][-1]))
    except Exception as e:
        print(f"  (skipped Milestone 5 rows: {e})")

    try:  # Milestone 6(a): rate vs K
        rk = load("exp_rate_vs_K")
        i_peak = int(np.argmax(rk["sum_AO-RIS"]))
        rows.append((f"Peak AO-RIS sum rate over K (at K={int(rk['K'][i_peak])}) [bit/s/Hz]",
                      rk["sum_AO-RIS"][i_peak]))
    except Exception as e:
        print(f"  (skipped Milestone 6a row: {e})")

    try:  # Milestone 6(b): SDR vs manifold
        sdr = load("exp_sdr_vs_manifold")
        i_last = -1  # largest M actually run (M=64 normally, M=32 under --fast)
        rows.append((f"Manifold max-min SINR @ M={int(sdr['M'][i_last])}, K=4", sdr["sinr_manifold"][i_last]))
        rows.append((f"SDR+randomization max-min SINR @ M={int(sdr['M'][i_last])}, K=4", sdr["sinr_sdr"][i_last]))
    except Exception as e:
        print(f"  (skipped Milestone 6b rows: {e})")

    try:  # Milestone 7: DRL (Stage A, M=16)
        drl = load("exp_drl_stageA")
        rows.append(("AO oracle rate, K=1 M=16 (DRL comparison) [bit/s/Hz]", drl["ao_rate"]))
        rows.append(("PPO final rate, K=1 M=16 [bit/s/Hz]", drl["PPO_final_mean"]))
        rows.append(("SAC final rate, K=1 M=16 [bit/s/Hz]", drl["SAC_final_mean"]))
    except Exception as e:
        print(f"  (skipped Milestone 7 rows: {e})")

    out_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for name, val in rows:
            writer.writerow([name, float(val)])
    print(f"saved {out_path}")
    for name, val in rows:
        print(f"  {name}: {val:.4f}")


if __name__ == "__main__":
    main()
