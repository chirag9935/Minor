# MASTER PROMPT FOR CLAUDE CODE — RIS-Assisted mmWave/THz Communication

Paste everything below this line into Claude Code.

---

You are building a complete, student-explainable simulation project in Python: **RIS-assisted mmWave/THz downlink communication with a blocked direct link**, solved with **Alternating Optimization (AO)**. Work in a new folder `ris_project/`. Build in the milestones below, in order. After each milestone run its tests and fix failures before moving on. Keep code simple, commented, and with a fixed random seed (default 42). Do not over-engineer.

## Reference files provided

- `RIS_Communication_Project_Workflow.md` is the student's **original assignment** (Steps 1–6). Read it fully before writing any code. It is the source of truth for the project's topic, system model, constraints, benchmarks, sensitivity ranges and required plots.
- This master prompt is the implementation plan. It deliberately covers only a subset of the assignment (see Scope below). If anything in this prompt conflicts with the assignment file, stop and tell the student before proceeding. Do not silently change the scope.
- Ignore the "APPENDIX: Updated Multiplexer Implementation Table" at the end of the assignment file. It is unrelated to this project.
- Keep the same names and notation as the assignment (BS, RIS, UE, H, g, Φ, θ_m, W, Nt, M, Ptx, Pmax, Lmol) in code, plots and docs. In the README, add a short table mapping each assignment step (1–6) to the files that implement it, marking each step as done, partial or deferred.

## 0. Scope

IN SCOPE:
- Blocked-LoS scenario, sparse Saleh-Valenzuela channels, THz molecular absorption
- Objective: maximize sum achievable rate (Option A). Energy efficiency is *evaluated* from the same solutions (Option C metric), not separately optimized.
- AO: BS beamforming (MRT for K=1; ZF and regularized-ZF/MMSE for K>1) alternating with RIS phase optimization (closed form for K=1, **Riemannian gradient ascent on the unit-modulus manifold** for K>1)
- Benchmarks: no-RIS, random-phase RIS, half-duplex AF relay
- Sensitivity: M, Ptx, distance, frequency (28 GHz vs 140 GHz)
- Plots 1–6, a results summary, and README
- OPTIONAL last milestone: small single-user DRL (SAC) demo

OUT OF SCOPE (do NOT implement): SDR with Gaussian randomization, Objective B (min-power), realistic channel estimation, hardware impairments, multi-user DRL, DRL hyperparameter tuning to beat AO, paper writing. Mention them in README under "Future work".

## 1. Files

```
ris_project/
├── README.md
├── requirements.txt          # numpy scipy matplotlib torch (CPU) stable-baselines3 gymnasium tqdm
├── config.py
├── channel.py
├── beamforming.py
├── ris_opt.py
├── ao.py
├── benchmarks.py
├── metrics.py                # sum rate, EE
├── experiments/
│   ├── exp1_rate_vs_power.py
│   ├── exp2_rate_vs_M.py
│   ├── exp3_convergence.py
│   ├── exp4_ee_vs_se.py
│   ├── exp5_distance.py
│   ├── exp6_frequency.py
│   └── exp7_drl_demo.py      # optional
├── plots/
├── results/                  # .npz / .csv of every curve
└── tests/test_core.py
```

## 2. Milestone 1 — config.py and channel.py

**config.py** (one dataclass; every parameter overridable):
- fc = 140e9 (also support 28e9), c = 3e8, BW = 100e6, noise PSD = −174 dBm/Hz, NF = 8 dB
- Nt = 8 (BS ULA, half-wavelength spacing), RIS UPA with M = Mx·My (default 8×8 = 64; support 16, 32, 64, 128, 256), K = 1 or 4 single-antenna UEs
- Pmax_dBm = 30
- Geometry (metres): BS at (0, 0, 10), RIS at (20, 10, 10), UEs placed within 3–8 m of the RIS, in the area behind a blocker. Draw the geometry as a figure (`plots/geometry.png`) with a blocker rectangle between BS and UEs.
- Direct BS→UE link: blocked. Implement as the same path-loss model plus a configurable `blockage_loss_db = 40` penetration loss (so the "no RIS" curve is finite and plottable). State this assumption in README.
- BS and UE antenna gains `bs_gain_dbi`, `ue_gain_dbi`: choose defaults so that for K=1, M=64, Ptx=30 dBm at 140 GHz the mean SNR with AO is roughly 15–25 dB. **Print the SNR you obtain and the gains you chose, and record them in the README as assumptions.**
- Molecular absorption coefficient (dB/km, approximate placeholder assumptions, label them as such): 28 GHz → 0.1, 140 GHz → 2, 300 GHz → 8. Loss factor L_mol = 10^(−k·d/10000) with d in metres and k in dB/km.
- Power model for EE: P_total = Ptx/η + P_BS_static + M·P_RIS + K·P_UE with η = 0.4, P_BS_static = 1 W, P_RIS = 5 mW per element, P_UE = 0.1 W (assumptions).

**channel.py**
- `ula_steering(N, angle)`, `upa_steering(Mx, My, az, el)`
- `sv_channel_H(...)`: BS→RIS, shape (M, Nt). L = 3 paths (1 LoS with K-factor 10 dB + 2 NLoS), complex gains CN(0,1) scaled, angles of LoS from geometry, NLoS angles random. Multiply by sqrt(path loss × L_mol(fc, d)). Free-space path loss: (λ/4πd)².
- `sv_channel_g(...)`: RIS→UE, shape (M, K), same model per user.
- `direct_channel(...)`: BS→UE, shape (K, Nt), Rayleigh with path loss plus blockage loss.
- Normalise carefully and document the exact formula in the docstring so the student can explain it.

Tests: shapes; average channel power matches the path-loss formula within tolerance; absorption makes the 140 GHz channel weaker than 28 GHz at equal distance.

## 3. Milestone 2 — beamforming.py, ris_opt.py, ao.py

System: y_k = (g_kᴴ Φ H + d_kᴴ) W x + n_k, Φ = diag(θ), |θ_m| = 1. Effective channel row for user k: h_kᴴ = g_kᴴ Φ H (+ direct term in the benchmark that uses it). SINR_k = |h_kᴴ w_k|² / (Σ_{j≠k} |h_kᴴ w_j|² + σ²). Sum rate = Σ log2(1 + SINR_k) (report in bit/s/Hz).

**beamforming.py**
- `mrt(h_eff, P)` for K=1
- `zf(H_eff, P)`: W = H_effᴴ (H_eff H_effᴴ)⁻¹, scaled so ‖W‖_F² = P
- `rzf(H_eff, P, sigma2)`: W = H_effᴴ (H_eff H_effᴴ + (K σ²/P) I)⁻¹, scaled so ‖W‖_F² = P (this is the MMSE-type precoder)
- Every function must return a W that satisfies ‖W‖_F² = P (assert in tests).

**ris_opt.py**
- `phase_update_single_user(g, H, w)`: a = H w; θ_m = exp(−j·angle(conj(g_m)·a_m)). Closed form, exactly unit modulus.
- `phase_update_manifold(G, H, W, theta0, sigma2, iters=50)`: maximise the sum rate over θ on the complex-circle manifold with **Riemannian gradient ascent**: compute the Euclidean gradient with PyTorch autograd (CPU, complex64/128), project it onto the tangent space (grad − Re(conj(θ)⊙grad)⊙θ), take a step with Armijo backtracking line search, retract by θ ← θ/|θ|. Return the new θ and the objective trace. The objective must never decrease (assert in tests).

**ao.py**
- `run_ao(H, G, P, sigma2, mode, theta0=None, max_iter=20, tol=1e-4)`, mode ∈ {"mrt" (K=1), "zf", "rzf"}.
- Loop: fix Φ → update W; fix W → update θ; record the sum rate after every half-step; stop when the improvement < tol.
- Returns θ, W, rate history.

Tests: |θ_m| = 1 within 1e-9; ‖W‖² = P; rate history non-decreasing (tolerance 1e-9); AO rate ≥ random-phase rate on at least 95% of 100 channel draws; K=1 closed-form AO matches the manifold solver within 1% on the same instance.

## 4. Milestone 3 — benchmarks.py and metrics.py

- `no_ris_rate(...)`: direct (blocked) link only, with MRT (K=1) or ZF (K>1).
- `random_phase_rate(...)`: average over 200 random θ draws.
- `af_relay_rate(...)` (K=1 only): half-duplex AF relay placed at the RIS location with one antenna, relay transmit power = Pmax, pre-log factor 1/2, end-to-end SNR γ = γ1·γ2 / (γ1 + γ2 + 1), where γ1 is the BS→relay SNR using MRT and γ2 is the relay→UE SNR. Same noise, same path-loss and absorption model. Document the assumptions in the docstring.
- `energy_efficiency(rate, BW, Ptx_W)`: EE = (BW · rate) / P_total [bit/Joule], and SE = rate [bit/s/Hz].

## 5. Milestone 4 — experiments (Monte-Carlo 200 realisations per point, mean values; save curves to results/ as .npz and PNG to plots/ at dpi 200, labelled axes, legend, grid)

1. **exp1 Rate vs Ptx** (Ptx = 0, 5, …, 30 dBm). Run the K=1 setup with four curves: AO-RIS, random-phase RIS, no-RIS, AF relay. Then a second figure for K=4 with AO-ZF, AO-RZF, random-phase, no-RIS.
2. **exp2 Rate vs M** (M = 16, 32, 64, 128, 256; for 128 use 8×16). Same curves as above at Ptx = 30 dBm.
3. **exp3 Convergence**: the rate against AO iteration for K=1 (MRT) and K=4 (RZF + manifold). Show 5 random initialisations in light lines plus their mean in bold.
4. **exp4 EE vs SE**: sweep Ptx from 0 to 30 dBm and for several M (16, 64, 256) plot EE (bit/Joule) against SE (bit/s/Hz), one curve per M, with a marker at each Ptx.
5. **exp5 Distance sensitivity**: move the UE away from the RIS (2 to 30 m) and plot the rate for AO, random, no-RIS, and relay. Also vary the RIS-to-BS distance (10 to 40 m) for a second panel.
6. **exp6 Frequency**: compare 28 GHz and 140 GHz (and 300 GHz if SNR allows) for AO-RIS vs distance, showing the effect of the absorption loss.

Also write `results/summary.csv` with the headline numbers (e.g., AO gain over random at M=64, Ptx=30 dBm).

## 6. Milestone 5 (OPTIONAL — time-box it; if it does not work in a reasonable attempt, skip it and say so honestly)

`exp7_drl_demo.py`: a Gymnasium environment for K=1, M=16, with the BS beamformer fixed to MRT given the RIS phases. Each episode is one step: a new channel is drawn, the observation is the real and imaginary parts of the cascaded vector c_m = conj(g_m)·(H w_ref) (w_ref is MRT for random phases, so it is fixed per episode), the action is M phase values in [−π, π] (scaled from [−1, 1]), and the reward is the achievable rate. Train stable-baselines3 SAC on CPU for at most ~50k steps (≤ 15 minutes). Plot the learning curve against the AO optimum (a horizontal line) and random phase (another line). Report plainly how close it gets; do not tune further. Write "DRL demo is a proof of concept" in the README.

## 7. Final deliverables

1. All code, tests passing (`pytest -q`).
2. All plots in `plots/` and data in `results/`.
3. `README.md` with: project overview; the system model equations (H, g, Φ, SINR, rate, EE); algorithm description (AO steps in numbered form); how to run each experiment; the table of assumptions and parameter values; a results section with the plots and 1–2 sentences of physical interpretation each; honest limitations; the future work list from section 0.
4. At the end, print a short summary for the student: what was built, headline numbers, anything that failed or was skipped.

## 8. Skills and working practices you must apply

1. **Wireless/signal-processing literacy.** Before coding each module, write its equation in the docstring (channel model, SINR, rate, EE). Check units: dB vs linear, dBm vs W, Hz vs bit/s/Hz. Unit bugs are the most likely error here.
2. **Complex-valued numerics.** Use `complex128` in NumPy, conjugate transposes (`.conj().T`) carefully, and a seeded `np.random.default_rng`.
3. **Optimization on manifolds.** The unit-modulus set is the complex circle manifold. Use tangent-space projection, retraction by normalisation, and Armijo backtracking (see refs [4], [9]). If torch autograd for complex tensors misbehaves, derive the Euclidean gradient analytically and verify it against finite differences in a test.
4. **Verification before trust.** Add sanity checks: AO rate must not decrease; with M→large the RIS gain should grow roughly as M²; a perfect-alignment upper bound (θ_m aligned for K=1) must be ≥ any other scheme; rate must increase with Ptx. If a curve violates these, treat it as a bug.
5. **Reproducibility.** One seed in config, all experiments runnable by one command each, and every plot reproducible from `results/*.npz`.
6. **Plot quality.** One consistent style, labelled axes with units, legend, grid, dpi 200, same colour for the same scheme in every figure.
7. **Honesty.** Separate "assumption" from "result". Never hide a failed run. Keep a `docs/ASSUMPTIONS.md`.
8. **Student-explainable code.** Short functions, no cleverness, comments that explain the *why*. Add `docs/EXPLAIN.md`: a one-page walkthrough of the pipeline (channel → beamformer → RIS phase → rate → plots) that the student can use to prepare for an oral evaluation, plus 10 likely viva questions with short answers.

## 9. References (all verified to exist; use them for the README bibliography and to justify design choices; cite only what you actually use; do not invent other citations)

| # | Reference | Use it for |
|---|---|---|
| [1] | Q. Wu and R. Zhang, "Intelligent Reflecting Surface Enhanced Wireless Network via Joint Active and Passive Beamforming," IEEE Trans. Wireless Commun., 2019. arXiv:1810.03961 | Core AO structure: alternate active (BS) and passive (RIS) beamforming; unit-modulus constraint; rate-vs-M scaling |
| [2] | Q. Wu and R. Zhang, "Towards Smart and Reconfigurable Environment: Intelligent Reflecting Surface Aided Wireless Network," IEEE Commun. Mag., 2020. arXiv:1905.00152 | Overview and motivation, blocked-link use case |
| [3] | M. Di Renzo et al., "Smart Radio Environments Empowered by Reconfigurable Intelligent Surfaces: How it Works, State of Research, and Road Ahead," IEEE JSAC, 2020. arXiv:2004.09352 | Background, RIS vs relay comparison (supports Benchmark 3) |
| [4] | X. Yu, D. Xu, R. Schober, "MISO Wireless Communication Systems via Intelligent Reflecting Surfaces," IEEE ICCC, 2019. arXiv:1904.12199 | Manifold optimization (and SDR) for RIS phases, MISO setting |
| [5] | C. Huang, A. Zappone, G. C. Alexandropoulos, M. Debbah, C. Yuen, "Reconfigurable Intelligent Surfaces for Energy Efficiency in Wireless Communication," IEEE Trans. Wireless Commun., 2019. arXiv:1810.06934 | Power-consumption model and energy-efficiency metric |
| [6] | C. Huang, R. Mo, C. Yuen, "Reconfigurable Intelligent Surface Assisted Multiuser MISO Systems Exploiting Deep Reinforcement Learning," IEEE JSAC, 2020. arXiv:2002.10072 | State/action/reward design for the optional DRL demo and for future work |
| [7] | O. El Ayach et al., "Spatially Sparse Precoding in Millimeter Wave MIMO Systems," IEEE Trans. Wireless Commun., 2014. arXiv:1305.2460 | Sparse Saleh-Valenzuela-style mmWave channel and array-response model |
| [8] | W. Tang et al., "Wireless Communications with Reconfigurable Intelligent Surface: Path Loss Modeling and Experimental Measurement," IEEE Trans. Wireless Commun., 2021. arXiv:1911.05326 | Realistic RIS path-loss modelling; list as a limitation of the simplified cascaded model |
| [9] | N. Boumal, B. Mishra, P.-A. Absil, R. Sepulchre, "Manopt, a Matlab Toolbox for Optimization on Manifolds," JMLR, 2014 (and the Pymanopt Python port) | Riemannian optimization background |

Notes for the README: [1] and [4] justify the algorithm, [7] the channel model, [5] the EE model, [3] the relay comparison, [6] the future DRL extension. THz absorption values in config.py are approximate placeholders; say that an ITU-R P.676 / HITRAN-based model is the proper source and is left as future work. Put the table above in `docs/REFERENCES.md` and cite by number in the README.

## 10. Extra deliverables

`docs/ASSUMPTIONS.md`, `docs/EXPLAIN.md`, `docs/REFERENCES.md`.

Rules: never fake results; if a test or experiment fails, show the failure and fix the cause. Do not change the scope. Run `python -m experiments.exp1_rate_vs_power` etc. once at the end to prove everything runs from a clean state.
