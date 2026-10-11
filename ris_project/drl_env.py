"""
Phase 2, Milestone 7: Gymnasium environments for the DRL approach (Approach
B) to RIS configuration, per RIS_Phase2_Prompt.md section 8 (citing Huang,
Mo & Yuen, JSAC 2020 -- see docs/REFERENCES.md).

Episode structure: each episode draws ONE fresh channel realisation and is
exactly one step long -- the agent picks an action for that instance,
receives a reward, and the episode ends. This is the standard "learn a
per-instance configuration policy under i.i.d. channel draws" formulation
used in the cited reference (a contextual-bandit-style framing of RL, not
a multi-step sequential-decision episode); it is NOT a simplification we
introduced -- the spec's own episode definition ("each episode draws a new
channel realisation") only makes sense this way for a problem with no
state that carries across time.

Two variants:
  RISPhaseEnv (Stages A/B): action = M RIS phases in [-pi, pi]. The BS
    beamformer is NOT part of the action -- it is recomputed from the
    CURRENT action's phases via a fixed rule (MRT for K=1, RZF for K>1),
    exactly mirroring AO's "fix theta, update W" half-step. This differs
    from (and supersedes, for M7's purposes) exp7_drl_demo.py's earlier
    Phase 1 proof-of-concept, which fixed w_ref from a RANDOM phase
    independent of the action; here the reward faithfully reflects the
    agent's own phase choice, per the M7 spec's explicit design.
  RISJointEnv (Stage C): action = M RIS phases + 2*Nt*K real BS-beamformer
    coefficients (real/imag of W, flattened), projected to ||W||_F^2=Pmax.

State: real/imag parts of the cascaded channel feature(s) c_k = conj(g_k) *
(H @ w_ref), stacked over users (K=1: length-M complex -> 2M real; K>1:
length K*M complex -> 2*K*M real). w_ref is a FIXED, non-learned reference
precoder (all-antennas-equal-phase, power Pmax/Nt) used ONLY to extract a
fixed-size, phase-relevant feature of the channel for the agent to observe
-- it is NOT the beamformer used to compute the reward (that one tracks
the action, as above). Observations are normalised to zero mean / unit
variance using a fixed (not running) empirical mean/std, calibrated once
at construction from a few hundred channel-only draws (no AO, so this is
fast) -- a deliberate simplification vs. SB3's VecNormalize (which needs
extra eval/train normalisation-syncing machinery we chose not to spend
this time-boxed milestone's budget on); it still satisfies the spec's
"observation normalisation" requirement.

Reward: achievable rate for the chosen action, normalised by the AO rate
for the SAME channel instance (closed-form AO for K=1 via ao.run_ao mode=
"mrt"; full iterative AO for K>1 via mode="rzf"), so reward is in [0, ~1]
during training, as the spec requires -- both raw and AO-normalised rate
are also returned in `info` for logging. Episodes are seeded via
Config.rng(draw_seed), matching the project-wide reproducibility
convention; pass `eval_seeds` (a fixed list of draw indices) to get a
deterministic, cycling held-out evaluation stream instead of an
ever-advancing training stream.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import Config
from channel import get_ue_positions, sv_channel_H, sv_channel_g
from beamforming import mrt, rzf
from metrics import effective_channel, sum_rate
from ao import run_ao

_CALIBRATION_DRAWS = 300


class _RISEnvBase(gym.Env):
    """Shared episode-drawing, reference-feature and AO-baseline logic for
    RISPhaseEnv and RISJointEnv."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: Config, seed: int = 0, eval_seeds: list | None = None,
                 manifold_iters: int = 50):
        super().__init__()
        self.cfg = cfg
        self.M = cfg.M
        self.K = cfg.K
        self.Nt = cfg.Nt
        self._ao_mode = "mrt" if self.K == 1 else "rzf"
        self._ao_manifold_iters = manifold_iters
        # Disjoint seed streams per parallel worker/instance (seed) so
        # vectorised training envs never draw the same channel sequence.
        self._episode_count = seed * 1_000_000
        self._eval_seeds = eval_seeds
        self._eval_idx = 0

        self.H = self.G = self.sigma2 = None
        self._raw_c = None
        self.ao_rate = None

        obs_dim = 2 * self.K * self.M
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self._obs_mean, self._obs_std = self._calibrate_obs_stats()

    # -- reference feature / observation --------------------------------
    def _calibrate_obs_stats(self):
        """Fixed (not running) per-Config empirical mean/std of the raw
        [c.real, c.imag] feature, from channel-only draws (no AO -- fast
        even at the largest M used in Phase 2). See module docstring."""
        rng = np.random.default_rng(12345)  # independent of cfg.seed: this is a feature-scale
        # calibration constant, not a training/eval draw -- must not collide with either stream.
        feats = np.zeros((_CALIBRATION_DRAWS, 2 * self.K * self.M))
        for i in range(_CALIBRATION_DRAWS):
            draw_seed = int(rng.integers(0, 10_000_000))
            H, G = self._draw_channel(draw_seed)
            c = self._reference_feature(H, G)
            feats[i] = np.concatenate([c.real, c.imag])
        mean = feats.mean(axis=0)
        std = feats.std(axis=0)
        std[std < 1e-12] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)

    def _draw_channel(self, draw_seed):
        rng = self.cfg.rng(draw_seed)
        ue_pos = get_ue_positions(self.cfg, rng)
        H = sv_channel_H(self.cfg, rng)
        G = sv_channel_g(self.cfg, rng, ue_pos)
        return H, G

    def _reference_feature(self, H, G):
        """c_k = conj(g_k) * (H @ w_ref), stacked over k; w_ref is a fixed,
        equal-phase, power-Pmax/Nt vector (not the reward's beamformer)."""
        w_ref = np.full(self.Nt, np.sqrt(self.cfg.Pmax_w / self.Nt), dtype=complex)
        a = H @ w_ref  # (M,)
        c = np.conj(G) * a[:, None]  # (M, K)
        return c.reshape(-1, order="F")  # [c_user0; c_user1; ...]

    def _obs(self):
        raw = np.concatenate([self._raw_c.real, self._raw_c.imag]).astype(np.float32)
        return ((raw - self._obs_mean) / self._obs_std).astype(np.float32)

    def _next_draw_seed(self):
        if self._eval_seeds is not None:
            s = self._eval_seeds[self._eval_idx % len(self._eval_seeds)]
            self._eval_idx += 1
            return s
        s = self._episode_count
        self._episode_count += 1
        return s

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        draw_seed = self._next_draw_seed()
        self.H, self.G = self._draw_channel(draw_seed)
        self.sigma2 = self.cfg.noise_power_w
        self._raw_c = self._reference_feature(self.H, self.G)

        theta_ao, W_ao, _ = run_ao(self.H, self.G, self.cfg.Pmax_w, self.sigma2,
                                    mode=self._ao_mode, manifold_iters=self._ao_manifold_iters)
        H_eff_ao = effective_channel(self.G, self.H, theta_ao)
        self.ao_rate = max(float(sum_rate(H_eff_ao, W_ao, self.sigma2)), 1e-6)

        return self._obs(), {"ao_rate": self.ao_rate, "draw_seed": draw_seed}


class RISPhaseEnv(_RISEnvBase):
    """Phase-only agent (Stages A/B): action = M RIS phases in [-pi, pi].
    The BS beamformer is recomputed from the CURRENT phases (MRT for K=1,
    RZF for K>1), not learned."""

    def __init__(self, cfg: Config, seed: int = 0, eval_seeds: list | None = None):
        super().__init__(cfg, seed=seed, eval_seeds=eval_seeds)
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(self.M,), dtype=np.float32)

    def _W_for_theta(self, theta):
        H_eff = effective_channel(self.G, self.H, theta)
        if self.K == 1:
            W = mrt(H_eff[0], self.cfg.Pmax_w)
        else:
            W = rzf(H_eff, self.cfg.Pmax_w, self.sigma2)
        return W, H_eff

    def step(self, action):
        theta = np.exp(1j * np.asarray(action, dtype=np.float64))
        W, H_eff = self._W_for_theta(theta)
        rate = float(sum_rate(H_eff, W, self.sigma2))
        reward = rate / self.ao_rate
        info = {"raw_rate": rate, "ao_rate": self.ao_rate, "normalized_rate": reward}
        return self._obs(), reward, True, False, info


class RISJointEnv(_RISEnvBase):
    """Joint agent (Stage C): action = M RIS phases + 2*Nt*K real BS-
    beamformer coefficients, projected to ||W||_F^2 = Pmax."""

    def __init__(self, cfg: Config, seed: int = 0, eval_seeds: list | None = None):
        super().__init__(cfg, seed=seed, eval_seeds=eval_seeds, manifold_iters=15)
        n_w = 2 * self.Nt * self.K
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.M + n_w,), dtype=np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        theta = np.exp(1j * action[: self.M] * np.pi)
        w_raw = action[self.M:]
        half = self.Nt * self.K
        W = w_raw[:half].reshape(self.Nt, self.K) + 1j * w_raw[half:].reshape(self.Nt, self.K)
        norm = np.linalg.norm(W, "fro")
        if norm < 1e-12:
            W, norm = np.ones((self.Nt, self.K), dtype=complex), np.sqrt(self.Nt * self.K)
        W = W * (np.sqrt(self.cfg.Pmax_w) / norm)

        H_eff = effective_channel(self.G, self.H, theta)
        rate = float(sum_rate(H_eff, W, self.sigma2))
        reward = rate / self.ao_rate
        info = {"raw_rate": rate, "ao_rate": self.ao_rate, "normalized_rate": reward}
        return self._obs(), reward, True, False, info


def random_phase_rate(cfg: Config, H, G, sigma2, rng, mode: str | None = None):
    """Random-phase reference rate for the SAME (H, G), with the matching
    fixed-W rule (MRT/RZF for K=1/K>1) -- used as a reporting baseline
    alongside the AO rate, never as a training signal."""
    mode = ("mrt" if cfg.K == 1 else "rzf") if mode is None else mode
    theta = np.exp(1j * rng.uniform(0, 2 * np.pi, cfg.M))
    H_eff = effective_channel(G, H, theta)
    if mode == "mrt":
        W = mrt(H_eff[0], cfg.Pmax_w)
    else:
        W = rzf(H_eff, cfg.Pmax_w, sigma2)
    return float(sum_rate(H_eff, W, sigma2))
