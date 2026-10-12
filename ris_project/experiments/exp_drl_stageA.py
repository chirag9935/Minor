"""
Phase 2, Milestone 7, Stage A: K=1, M=16 DRL study (phase-only agent),
per RIS_Phase2_Prompt.md section 8. Trains SAC and PPO (3 seeds each) via
stable-baselines3, evaluates periodically on a FIXED 500-channel held-out
set, and reports learning curves against the AO and random-phase
baselines, a final-rate table, and inference-time-per-decision.

Time-boxing (explicit in the spec: "cap at about 20 minutes per run"):
each of the 6 runs (2 algorithms x 3 seeds) is capped at both
TOTAL_TIMESTEPS and TIME_BUDGET_S, whichever comes first, via
_TimeBudgetCallback. Nothing here claims a "tuned" result -- per the
spec's own framing ("a fair, honest DRL study, not a guaranteed win"),
we report whatever Stage A actually shows, including an honest limitations
paragraph.

Run: python -m experiments.exp_drl_stageA
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from config import Config
from drl_env import RISPhaseEnv, random_phase_rate
from channel import get_ue_positions, sv_channel_H, sv_channel_g
from experiments._common import new_fig, style_and_save, save_results, RESULTS_DIR

# Checkpointing: each of the 6 (algo, seed) runs takes ~15-20 minutes, and
# this machine has repeatedly had unrelated background processes (Chrome,
# etc.) trigger OS-level low-memory kills of long-running jobs during
# Phase 2 M6 (see docs/ASSUMPTIONS.md) -- not from this script's own
# footprint, but worth protecting against regardless. Every completed
# (algo, seed) run's results are checkpointed to disk immediately; re-
# running this script resumes by skipping already-completed runs.
def _checkpoint_path(name: str) -> str:
    return os.path.join(RESULTS_DIR, f"{name}_checkpoint.pkl")


def _load_checkpoint(name: str) -> dict:
    path = _checkpoint_path(name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def _save_checkpoint(name: str, all_results: dict):
    with open(_checkpoint_path(name), "wb") as f:
        pickle.dump(all_results, f)

TOTAL_TIMESTEPS = 200_000
TIME_BUDGET_S = 20 * 60
SEEDS = [0, 1, 2]
ALGOS = {"SAC": SAC, "PPO": PPO}
EVAL_FREQ = 5_000
N_EVAL_CHANNELS = 500
# Disjoint from every training stream: training envs use seed*1_000_000 + i
# for seed in {0,1,2,...}; eval uses a separate high range so no training
# draw can ever coincide with a held-out evaluation draw.
EVAL_SEEDS = list(range(50_000_000, 50_000_000 + N_EVAL_CHANNELS))
NET_ARCH = [256, 256]


class _TimeBudgetCallback(BaseCallback):
    """Stops training once TIME_BUDGET_S of this PROCESS's own CPU time
    (time.process_time(), not wall-clock) is exceeded, regardless of how
    many of TOTAL_TIMESTEPS have been reached -- the spec's "cap at about
    20 minutes per run" is meant as a compute-effort cap. Using wall-clock
    time.time() instead was tried first and is NOT safe on a machine that
    can sleep: a run that happened to straddle the machine going idle for
    hours (confirmed here -- see docs/ASSUMPTIONS.md) would get starved
    down to a few hundred steps while a luckier seed got tens of
    thousands, making any seed-to-seed comparison meaningless. CPU time is
    immune to both sleep and (to first order) general system contention
    from unrelated processes, so every run gets a genuinely comparable
    budget regardless of what else happens on the machine meanwhile."""

    def __init__(self, budget_s: float):
        super().__init__()
        self.budget_s = budget_s
        self._t0 = None

    def _on_training_start(self):
        self._t0 = time.process_time()

    def _on_step(self) -> bool:
        return (time.process_time() - self._t0) < self.budget_s


class _PeriodicEvalCallback(BaseCallback):
    """Every eval_freq steps, runs the CURRENT policy (deterministic) on
    the fixed held-out set and records (timesteps, mean raw rate, mean
    normalised reward). Bypasses SB3's own EvalCallback because that only
    tracks reward, not the raw (bit/s/Hz) rate we need for the required
    learning-curve plot."""

    def __init__(self, cfg: Config, eval_freq: int, record: list):
        super().__init__()
        self.cfg = cfg
        self.eval_freq = eval_freq
        self.record = record
        self._eval_env = RISPhaseEnv(cfg, seed=999_999, eval_seeds=EVAL_SEEDS)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            raw_rates, norm_rewards = [], []
            for _ in range(len(EVAL_SEEDS)):
                obs, _ = self._eval_env.reset()
                action, _ = self.model.predict(obs, deterministic=True)
                _, reward, _, _, info = self._eval_env.step(action)
                raw_rates.append(info["raw_rate"])
                norm_rewards.append(reward)
            self.record.append((self.num_timesteps, float(np.mean(raw_rates)), float(np.mean(norm_rewards))))
        return True


def _ao_and_random_baselines(cfg: Config):
    """AO and random-phase mean rate over the SAME fixed held-out set (AO
    is deterministic given (H,G), so this is a single pass, not per-seed)."""
    from ao import run_ao
    from metrics import effective_channel, sum_rate

    ao_rates, random_rates, ao_times = [], [], []
    for s in EVAL_SEEDS:
        rng = cfg.rng(s)
        ue_pos = get_ue_positions(cfg, rng)
        H = sv_channel_H(cfg, rng)
        G = sv_channel_g(cfg, rng, ue_pos)
        sigma2 = cfg.noise_power_w

        t0 = time.time()
        theta_ao, W_ao, _ = run_ao(H, G, cfg.Pmax_w, sigma2, mode="mrt")
        ao_times.append(time.time() - t0)
        ao_rates.append(sum_rate(effective_channel(G, H, theta_ao), W_ao, sigma2))

        random_rates.append(random_phase_rate(cfg, H, G, sigma2, rng))

    return float(np.mean(ao_rates)), float(np.mean(random_rates)), float(np.mean(ao_times))


def _inference_time_per_decision(model, cfg: Config, n: int = 200) -> float:
    env = RISPhaseEnv(cfg, seed=888_888)
    obs, _ = env.reset()
    t0 = time.time()
    for _ in range(n):
        model.predict(obs, deterministic=True)
    return (time.time() - t0) / n


def train_one(algo_name: str, seed: int, cfg: Config):
    AlgoCls = ALGOS[algo_name]
    env = Monitor(RISPhaseEnv(cfg, seed=seed))
    policy_kwargs = dict(net_arch=NET_ARCH)
    if algo_name == "SAC":
        model = AlgoCls("MlpPolicy", env, verbose=0, seed=seed, learning_starts=1000,
                         buffer_size=100_000, policy_kwargs=policy_kwargs)
    else:  # PPO
        model = AlgoCls("MlpPolicy", env, verbose=0, seed=seed, n_steps=2048,
                         policy_kwargs=policy_kwargs)

    record = []
    callbacks = CallbackList([_TimeBudgetCallback(TIME_BUDGET_S), _PeriodicEvalCallback(cfg, EVAL_FREQ, record)])

    t0_wall, t0_cpu = time.time(), time.process_time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
    elapsed_wall = time.time() - t0_wall
    elapsed_cpu = time.process_time() - t0_cpu
    steps_reached = model.num_timesteps

    infer_time = _inference_time_per_decision(model, cfg)
    print(f"  {algo_name} seed={seed}: {steps_reached} steps, {elapsed_cpu:.1f}s CPU "
          f"({elapsed_wall:.1f}s wall-clock), inference {infer_time*1e3:.3f} ms/decision", flush=True)
    return record, steps_reached, elapsed_cpu, infer_time


def run_stage(cfg: Config, name: str):
    """Shared training/eval/report driver for a Stage A/B run: SAC+PPO x
    len(SEEDS) seeds at the given cfg, checkpointed and named by `name`
    (e.g. "exp_drl_stageA", "exp_drl_stageB_M32").

    Skips entirely if name's final results .npz already exists: the
    checkpoint is deleted on success (it's scratch, not a deliverable),
    which means a later re-run of a multi-stage caller (e.g.
    exp_drl_stageB.py re-invoking run_stage for an already-finished M)
    would otherwise have no way to tell "already done" from "never
    started," and would silently retrain everything from scratch --
    confirmed to actually happen (an M=32 run that had already completed
    successfully got ~an hour of its results discarded and retrained from
    zero by a resume after a later stage was killed) before this check
    was added."""
    results_path = os.path.join(RESULTS_DIR, f"{name}.npz")
    if os.path.exists(results_path) and not os.path.exists(_checkpoint_path(name)):
        print(f"{name}: results already exist at {results_path} and no checkpoint is pending -- skipping.", flush=True)
        return

    print(f"{name}: K={cfg.K}, M={cfg.M}. {len(ALGOS)} algos x {len(SEEDS)} seeds, "
          f"cap {TOTAL_TIMESTEPS} steps / {TIME_BUDGET_S/60:.0f} min per run.", flush=True)

    ao_rate, random_rate, ao_time = _ao_and_random_baselines(cfg)
    print(f"AO rate (held-out mean): {ao_rate:.4f} bit/s/Hz  "
          f"(run time {ao_time*1e3:.3f} ms/instance)")
    print(f"Random-phase rate (held-out mean): {random_rate:.4f} bit/s/Hz")

    all_results = _load_checkpoint(name)
    if all_results:
        done = {(a, i) for a, runs in all_results.items() for i in range(len(runs))}
        print(f"Resuming from checkpoint: {len(done)} of {len(ALGOS)*len(SEEDS)} runs already done.", flush=True)

    for algo_name in ALGOS:
        algo_results = all_results.setdefault(algo_name, [])
        if len(algo_results) >= len(SEEDS):
            print(f"\n{algo_name}: all {len(SEEDS)} seeds already done (checkpoint).", flush=True)
            continue
        print(f"\nTraining {algo_name} ...", flush=True)
        for seed in SEEDS[len(algo_results):]:
            record, steps, elapsed, infer_time = train_one(algo_name, seed, cfg)
            algo_results.append({"record": record, "steps": steps, "elapsed": elapsed, "infer_time": infer_time})
            _save_checkpoint(name, all_results)

    # ---- learning curves: mean +/- std across seeds, common x-grid ----
    fig, ax = new_fig()
    colors = {"SAC": "#1f77b4", "PPO": "#d62728"}
    save_dict = {"ao_rate": ao_rate, "random_rate": random_rate, "ao_time": ao_time}

    final_table = []
    for algo_name, algo_results in all_results.items():
        # align on the shortest run's set of eval checkpoints (in case of early time-budget stops)
        min_len = min(len(r["record"]) for r in algo_results)
        if min_len == 0:
            print(f"{algo_name}: no evaluation checkpoints recorded (ran out of time budget too early).")
            continue
        steps_grid = np.array([algo_results[0]["record"][i][0] for i in range(min_len)])
        rates = np.array([[r["record"][i][1] for i in range(min_len)] for r in algo_results])  # (n_seeds, n_checkpoints)
        mean_rate = rates.mean(axis=0)
        std_rate = rates.std(axis=0)

        ax.plot(steps_grid, mean_rate, color=colors.get(algo_name, None), label=f"{algo_name} (mean of {len(algo_results)} seeds)")
        ax.fill_between(steps_grid, mean_rate - std_rate, mean_rate + std_rate, color=colors.get(algo_name, None), alpha=0.2)

        save_dict[f"{algo_name}_steps_grid"] = steps_grid
        save_dict[f"{algo_name}_rates"] = rates
        save_dict[f"{algo_name}_final_mean"] = mean_rate[-1]
        save_dict[f"{algo_name}_final_std"] = std_rate[-1]
        save_dict[f"{algo_name}_elapsed"] = np.array([r["elapsed"] for r in algo_results])
        save_dict[f"{algo_name}_infer_time"] = np.array([r["infer_time"] for r in algo_results])

        final_table.append((algo_name, mean_rate[-1], std_rate[-1],
                             np.mean([r["infer_time"] for r in algo_results]),
                             np.mean([r["elapsed"] for r in algo_results])))

    ax.axhline(ao_rate, color="#2ca02c", linestyle="--", label="AO (closed-form)")
    ax.axhline(random_rate, color="#ff7f0e", linestyle="--", label="Random phase")
    style_and_save(fig, ax, f"{name}_learning_curve", "Training steps", "Held-out mean rate [bit/s/Hz]",
                   title=f"DRL learning curves vs AO/random (K={cfg.K}, M={cfg.M})")

    save_results(name, **save_dict)

    print(f"\n{'Algo':>6} {'Final rate':>12} {'+/- std':>10} {'Infer ms/dec':>14} {'Train time[s]':>15}")
    print(f"{'AO':>6} {ao_rate:12.4f} {'':>10} {ao_time*1e3:14.4f} {'(closed-form)':>15}")
    print(f"{'Random':>6} {random_rate:12.4f}")
    for algo_name, final_mean, final_std, infer_t, elapsed in final_table:
        print(f"{algo_name:>6} {final_mean:12.4f} {final_std:10.4f} {infer_t*1e3:14.4f} {elapsed:15.1f}")

    if os.path.exists(_checkpoint_path(name)):
        os.remove(_checkpoint_path(name))

    print(f"\n{name} done. See docs/ASSUMPTIONS.md for the honest write-up of what this does and doesn't show.")


def main():
    run_stage(Config(K=1, Mx=4, My=4), "exp_drl_stageA")  # M=16


if __name__ == "__main__":
    main()
