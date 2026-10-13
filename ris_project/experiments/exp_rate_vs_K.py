"""
Phase 2, Milestone 6(a): rate vs number of users K.

K in {1, 2, 4, 6} (Nt=8, M=64, Ptx=30 dBm): sum rate and mean per-user
rate for AO-ZF, AO-RZF (K=1 uses the MRT/closed-form anchor, labelled
"AO-RIS"), random-phase RIS, and no-RIS.
Run: python -m experiments.exp_rate_vs_K
"""
import numpy as np
from tqdm import tqdm

from config import Config
from channel import get_ue_positions, sv_channel_H, sv_channel_g, direct_channel
from ao import run_ao
from benchmarks import no_ris_rate, random_phase_rate
from metrics import sinr_per_user, effective_channel
from experiments._common import new_fig, style_and_save, save_results, COLORS, mc_draws

N_DRAWS = mc_draws(200)
K_VALUES = [1, 2, 4, 6]


def sweep_K():
    names = ["AO-RIS", "Random-phase", "No-RIS"]
    sum_rate_res = {k: np.zeros(len(K_VALUES)) for k in names}
    per_user_rate_res = {k: np.zeros(len(K_VALUES)) for k in names}

    for ki, K in enumerate(tqdm(K_VALUES, desc="exp_rate_vs_K")):
        mode = "mrt" if K == 1 else "rzf"
        acc_sum = {k: [] for k in names}
        acc_peruser = {k: [] for k in names}
        for i in range(N_DRAWS):
            cfg = Config(K=K)
            rng = cfg.rng(i)
            ue_pos = get_ue_positions(cfg, rng)
            H = sv_channel_H(cfg, rng)
            G = sv_channel_g(cfg, rng, ue_pos)
            d = direct_channel(cfg, rng, ue_pos)
            P, sigma2 = cfg.Pmax_w, cfg.noise_power_w

            max_iter = 15 if K == 1 else 10
            manifold_iters = 50 if K == 1 else 15
            theta, W, hist = run_ao(H, G, P, sigma2, mode=mode, max_iter=max_iter,
                                     manifold_iters=manifold_iters)
            H_eff = effective_channel(G, H, theta)
            sinr = sinr_per_user(H_eff, W, sigma2)
            per_user_rate = np.mean(np.log2(1 + sinr))
            acc_sum["AO-RIS"].append(hist[-1])
            acc_peruser["AO-RIS"].append(per_user_rate)

            rand_rate = random_phase_rate(H, G, P, sigma2, mode, cfg.rng(i + 10000))
            acc_sum["Random-phase"].append(rand_rate)
            acc_peruser["Random-phase"].append(rand_rate / K)

            noris_rate = no_ris_rate(d, P, sigma2)
            acc_sum["No-RIS"].append(noris_rate)
            acc_peruser["No-RIS"].append(noris_rate / K)

        for k in names:
            sum_rate_res[k][ki] = np.mean(acc_sum[k])
            per_user_rate_res[k][ki] = np.mean(acc_peruser[k])

    return sum_rate_res, per_user_rate_res


def main():
    sum_rate_res, per_user_rate_res = sweep_K()
    names = ["AO-RIS", "Random-phase", "No-RIS"]

    fig, ax = new_fig()
    for name in names:
        ax.plot(K_VALUES, sum_rate_res[name], marker="o", color=COLORS.get(name, None), label=name)
    style_and_save(fig, ax, "exp_rate_vs_K_sum", "Number of users K", "Sum rate [bit/s/Hz]",
                   title="Sum rate vs K (Nt=8, M=64, Ptx=30 dBm)")

    fig, ax = new_fig()
    for name in names:
        ax.plot(K_VALUES, per_user_rate_res[name], marker="o", color=COLORS.get(name, None), label=name)
    style_and_save(fig, ax, "exp_rate_vs_K_peruser", "Number of users K", "Mean per-user rate [bit/s/Hz]",
                   title="Per-user rate vs K (Nt=8, M=64, Ptx=30 dBm)")

    save_results("exp_rate_vs_K", K=np.array(K_VALUES),
                 **{f"sum_{k}": v for k, v in sum_rate_res.items()},
                 **{f"peruser_{k}": v for k, v in per_user_rate_res.items()})
    print("exp_rate_vs_K done.")


if __name__ == "__main__":
    main()
