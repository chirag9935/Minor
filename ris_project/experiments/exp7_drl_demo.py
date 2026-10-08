"""
Experiment 7 (OPTIONAL, proof of concept): single-user DRL demo.

A Gymnasium environment for K=1, M=16: each episode draws a fresh channel
and a *fixed* reference BS beamformer w_ref = MRT(H, theta_random) (i.e.
the agent only controls the RIS phases, not the beamformer -- per the
master prompt's milestone 5 spec). Observation: real/imag parts of the
cascaded vector c_m = conj(g_m)*(H w_ref). Action: M phase values in
[-1,1], scaled to [-pi,pi]. Reward: the resulting achievable rate. One
step per episode. Trained with stable-baselines3 SAC on CPU.

This is a proof of concept, not a tuned result (see master prompt section
8 / the master prompt's own framing of milestone 5): we do not tune
further, and report plainly how close it gets to the single-step AO
(closed-form) oracle and to a random-phase baseline.
Run: python -m experiments.exp7_drl_demo
"""
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import Config
from channel import get_ue_positions, sv_channel_H, sv_channel_g
from beamforming import mrt
from ris_opt import phase_update_single_user
from metrics import effective_channel, sum_rate
from experiments._common import new_fig, style_and_save, save_results

TOTAL_TIMESTEPS = 40000
TIME_BUDGET_S = 15 * 60


class RISPhaseEnv(gym.Env):
    """One-step-per-episode env: choose RIS phases to maximise rate, BS
    beamformer fixed (per episode) to MRT for a random reference theta."""

    def __init__(self, cfg: Config, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.M = cfg.M
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.M,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 * self.M,), dtype=np.float32)
        self._episode_count = seed * 1_000_000  # keep per-worker seed streams disjoint
        self.H = self.G = self.w_ref = self.c = None

    def _draw_episode(self):
        rng = self.cfg.rng(self._episode_count)
        self._episode_count += 1
        ue_pos = get_ue_positions(self.cfg, rng)
        H = sv_channel_H(self.cfg, rng)
        G = sv_channel_g(self.cfg, rng, ue_pos)
        g = G[:, 0]
        theta_rand = np.exp(1j * rng.uniform(0, 2 * np.pi, self.M))
        h_eff_rand = effective_channel(G, H, theta_rand)
        w_ref = mrt(h_eff_rand[0], self.cfg.Pmax_w)
        a = H @ w_ref.reshape(-1)
        c = np.conj(g) * a
        return H, G, w_ref, c

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.H, self.G, self.w_ref, self.c = self._draw_episode()
        obs = np.concatenate([self.c.real, self.c.imag]).astype(np.float32)
        return obs, {}

    def step(self, action):
        phases = np.asarray(action, dtype=np.float64) * np.pi
        theta = np.exp(1j * phases)
        h_eff = effective_channel(self.G, self.H, theta)
        reward = float(sum_rate(h_eff, self.w_ref, self.cfg.noise_power_w))
        obs = np.concatenate([self.c.real, self.c.imag]).astype(np.float32)
        return obs, reward, True, False, {}


def evaluate_baselines(cfg: Config, n_episodes: int = 300):
    """AO-oracle (closed-form theta for the episode's fixed w_ref) and
    random-phase baselines, evaluated on freshly drawn episodes."""
    ao_rates, rand_rates = [], []
    for i in range(n_episodes):
        rng = cfg.rng(5_000_000 + i)
        ue_pos = get_ue_positions(cfg, rng)
        H = sv_channel_H(cfg, rng)
        G = sv_channel_g(cfg, rng, ue_pos)
        g = G[:, 0]
        theta_rand0 = np.exp(1j * rng.uniform(0, 2 * np.pi, cfg.M))
        h_eff0 = effective_channel(G, H, theta_rand0)
        w_ref = mrt(h_eff0[0], cfg.Pmax_w)

        theta_ao = phase_update_single_user(g, H, w_ref.reshape(-1))
        ao_rates.append(sum_rate(effective_channel(G, H, theta_ao), w_ref, cfg.noise_power_w))

        theta_r = np.exp(1j * rng.uniform(0, 2 * np.pi, cfg.M))
        rand_rates.append(sum_rate(effective_channel(G, H, theta_r), w_ref, cfg.noise_power_w))
    return float(np.mean(ao_rates)), float(np.mean(rand_rates))


def main():
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    cfg = Config(K=1, Mx=4, My=4)  # M=16
    print(f"exp7 DRL demo: K=1, M={cfg.M}. Training SAC for up to {TOTAL_TIMESTEPS} steps "
          f"(time-boxed at {TIME_BUDGET_S/60:.0f} min) ...")

    env = Monitor(RISPhaseEnv(cfg, seed=0))
    model = SAC("MlpPolicy", env, verbose=0, seed=0, learning_starts=200)

    t0 = time.time()
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
    except Exception as e:
        print(f"exp7 FAILED during training: {e}")
        print("Skipping the DRL demo honestly, as allowed by the master prompt (time-boxed, optional).")
        return
    elapsed = time.time() - t0
    print(f"Training took {elapsed:.1f} s for {TOTAL_TIMESTEPS} steps.")

    rewards = np.array(env.get_episode_rewards())
    if len(rewards) < 50:
        print(f"exp7: too few completed episodes ({len(rewards)}) to plot a learning curve. Skipping.")
        return

    window = 200
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    ao_rate, rand_rate = evaluate_baselines(cfg)
    print(f"AO (closed-form) oracle mean rate: {ao_rate:.3f} bit/s/Hz")
    print(f"Random-phase mean rate:            {rand_rate:.3f} bit/s/Hz")
    print(f"SAC final smoothed rate:            {smoothed[-1]:.3f} bit/s/Hz")
    gap_pct = 100 * (ao_rate - smoothed[-1]) / ao_rate
    print(f"Gap to AO oracle: {gap_pct:.1f}%")

    fig, ax = new_fig()
    ax.plot(np.arange(len(smoothed)), smoothed, color="#1f77b4", label=f"SAC (smoothed, window={window})")
    ax.axhline(ao_rate, color="#2ca02c", linestyle="--", label="AO (closed-form) oracle")
    ax.axhline(rand_rate, color="#ff7f0e", linestyle="--", label="Random phase")
    style_and_save(fig, ax, "exp7_drl_learning_curve", "Training episode", "Rate [bit/s/Hz]",
                   title=f"SAC learning curve vs AO oracle (K=1, M={cfg.M})")
    save_results("exp7_drl_learning_curve", rewards=rewards, smoothed=smoothed,
                 ao_rate=ao_rate, rand_rate=rand_rate)
    print("exp7 done. DRL demo is a proof of concept -- not tuned further, see docs/ASSUMPTIONS.md.")


if __name__ == "__main__":
    main()
