"""
Builds results/summary.csv: headline numbers pulled from the already-saved
results/*.npz files (run the exp1-exp6 scripts first).
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
