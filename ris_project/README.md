# RIS-Assisted mmWave/THz Downlink Communication (AO)

A student-explainable simulation of a Reconfigurable Intelligent Surface
(RIS) assisting a downlink mmWave/THz link whose direct line-of-sight is
blocked, solved with Alternating Optimization (AO) between BS active
beamforming and RIS passive phase shifts. Built per `RIS_Master_Prompt.md`,
implementing a defined subset of the full assignment in
`RIS_Communication_Project_Workflow.md` (see the mapping table below).

## Assignment-step -> file mapping

| Step | Assignment topic | Implemented in | Status |
|---|---|---|---|
| 1 | System model & scenario definition | `config.py`, `channel.py` | Done |
| 2 | Optimization problem formulation | `metrics.py` (SINR/rate), `beamforming.py`/`ris_opt.py` (constraints) | Done (Option A only; Option B min-power out of scope) |
| 3 | Algorithm design | `ao.py`, `ris_opt.py`, `beamforming.py` | Done (Approach A / AO with ZF-MMSE + manifold optimization; SDR and Approach B/DRL for the *joint* problem out of scope; see exp7 for an optional single-user DRL proof of concept) |
| 4 | Implementation & simulation | all of the above + `benchmarks.py` | Done |
| 5 | Result analysis & plots | `experiments/exp1`-`exp6`, `results/summary.csv` | Done |
| 6 | Documentation & paper writing | this README, `docs/` | Partial (results + assumptions documented; no full paper manuscript written) |

## Scope

See `RIS_Master_Prompt.md` section 0 for the authoritative scope. In short:
blocked-LoS, sparse Saleh-Valenzuela + THz molecular absorption channels,
sum-rate maximization (Option A) with energy efficiency evaluated (not
separately optimized), AO (MRT/ZF/RZF + closed-form/manifold RIS phases),
three benchmarks, sensitivity sweeps over M/Ptx/distance/frequency, and an
optional single-user DRL demo.

**Out of scope** (per the master prompt, listed here as future work):
SDR with Gaussian randomization, Objective B (min-power), realistic channel
estimation, hardware impairments, multi-user DRL, DRL hyperparameter tuning
to beat AO, full paper writing.

## System model

- BS: `Nt`-antenna ULA (half-wavelength spacing). RIS: `M = Mx*My`-element
  UPA (half-wavelength spacing). UEs: single antenna each, `K` of them.
- Channels (see `channel.py` docstring for exact formulas): `H` (BS->RIS,
  M x Nt), `g` (RIS->UE, M x K), `d` (BS->UE direct/blocked, K x Nt). `H`
  and `g` are sparse Saleh-Valenzuela (SV) models (1 LoS + NLoS paths,
  Rician K-factor), each path an array-steering outer product scaled by
  `sqrt(pathloss * Lmol(fc,d) * antenna_gain)`. `d` is Rayleigh, scaled by
  pathloss and an added `blockage_loss_db` penalty.
- RIS reflection: `Phi = diag(theta)`, `|theta_m| = 1`.
- Effective channel (user k): `h_k^H = g_k^H * Phi * H` (`+ d_k^H` for the
  no-RIS benchmark).
- Received signal: `y_k = h_k^H W x + n_k`, `n_k ~ CN(0, sigma^2)`.
- `SINR_k = |h_k^H w_k|^2 / (sum_{j!=k} |h_k^H w_j|^2 + sigma^2)`.
- Sum rate: `R = sum_k log2(1 + SINR_k)` [bit/s/Hz].
- Energy efficiency: `EE = BW*R / P_total`, `P_total = Ptx/eta +
  P_BS_static + M*P_RIS + K*P_UE` [bit/Joule].
- Constraints: `||W||_F^2 <= Pmax`, `|theta_m| = 1` for all m.

## Algorithm: Alternating Optimization (AO)

Repeat until the sum rate stops improving (or `max_iter` is reached):

1. **Fix Phi, update W** (active beamforming): `mrt(h_eff, P)` for K=1;
   `zf(H_eff, P)` or `rzf(H_eff, P, sigma2)` for K>1.
2. **Fix W, update theta** (passive beamforming): closed-form phase
   alignment for K=1 (`phase_update_single_user`, globally optimal for a
   fixed W); Riemannian gradient ascent on the unit-modulus manifold for
   K>1 (`phase_update_manifold`): Euclidean gradient via PyTorch autograd
   -> tangent-space projection -> normalise -> Armijo backtracking line
   search -> retraction to the unit circle.
3. Each half-step is only accepted if it does not decrease the sum rate
   (safeguarded AO), so the rate history returned by `run_ao` is
   non-decreasing by construction.

See `docs/EXPLAIN.md` for a walkthrough with worked justifications, and
`docs/REFERENCES.md` for the papers behind each design choice.

## How to run

```bash
cd ris_project
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt

pytest -q                                   # all tests

python -m experiments.plot_geometry         # plots/geometry.png
python -m experiments.exp1_rate_vs_power    # plots/exp1_*.png
python -m experiments.exp2_rate_vs_M        # plots/exp2_*.png
python -m experiments.exp2b_crossover       # bonus, not required: plots/exp2b_crossover_K1.png
python -m experiments.exp3_convergence      # plots/exp3_*.png
python -m experiments.exp4_ee_vs_se         # plots/exp4_*.png
python -m experiments.exp5_distance         # plots/exp5_*.png
python -m experiments.exp6_frequency        # plots/exp6_*.png
python -m experiments.exp7_drl_demo         # optional, plots/exp7_*.png
```

Every experiment saves its raw curves to `results/*.npz` (so plots can be
regenerated without re-running the Monte Carlo sweeps) and its figures to
`plots/*.png` at 200 dpi.

## Assumptions and parameters

See `docs/ASSUMPTIONS.md` for the full list with reasoning. Headline
values:

| Parameter | Value | Note |
|---|---|---|
| `fc` | 140 GHz (28 / 300 GHz also tested) | |
| `Nt`, default `M`, `K` | 8, 64 (8x8), 1 or 4 | |
| `Pmax_dBm` | 30 dBm | swept 0-30 dBm in exp1/exp4 |
| `blockage_loss_db` | 40 dB | direct-link penetration loss assumption |
| `bs_gain_dbi` / `ue_gain_dbi` | 45 / 25 dBi | calibrated for ~15-25 dB mean AO SNR at K=1,M=64,Ptx=30dBm,140GHz (measured ≈20.2 dB); deliberately high -- see docs/ASSUMPTIONS.md |
| Absorption k(28/140/300 GHz) | 0.1 / 2 / 8 dB/km | placeholders, see [8] |
| EE power model | eta=0.4, P_BS=1W, P_RIS=5mW/elem, P_UE=0.1W | see [5] |

## Results

*(Figures are generated by the experiments above; this section summarises
the physical interpretation of each. Headline numbers are in
`results/summary.csv`.)*

1. **Rate vs Ptx** (`exp1_rate_vs_power_K1/K4.png`): all schemes' rate
   increases monotonically with Ptx, as expected, and AO-RIS clearly beats
   random-phase RIS at every power level (both K=1 and K=4) -- that
   ordering among the RIS-phase schemes is the required comparison and it
   always holds. Against the un-optimized blocked direct link (no-RIS),
   however, AO-RIS does *not* win at the assignment's default M=64 in this
   geometry: no-RIS is clearly ahead across the whole Ptx range, for both
   K=1 and K=4. This is the RIS "multiplicative path loss" effect
   (a cascaded two-hop link's path loss is a *product*, not a sum, of the
   two legs' losses) discussed in detail in `docs/ASSUMPTIONS.md` -- it is
   a real, literature-consistent finding (see [3],[8]), not a tuning
   failure, and exp2 (below) is exactly the experiment designed to show
   where it stops holding.
2. **Rate vs M** (`exp2_rate_vs_M_K1/K4.png`): AO-RIS rate grows steeply
   with M (coherent M-element combining, ~M^2 in SNR -- see the sanity
   check in `docs/ASSUMPTIONS.md`); the random-phase curve grows much more
   slowly (incoherent, ~M-fold combining only). AO-RIS (K=1) overtakes the
   AF relay by M~128. Neither K=1 nor K=4 AO-RIS catches the M-independent
   no-RIS baseline within the assignment's own M=16..256 sweep range in
   this geometry; the steep, clearly-still-climbing AO-RIS growth trend
   against a flat no-RIS line is the headline result of this plot -- it's
   the experiment that explains *why* M matters so much in exp1.
   **Bonus** (`exp2b_crossover_K1.png`, not required by the assignment,
   K=1 only): extends the M-sweep out to 1024 to show the actual
   crossover rather than leaving it as an extrapolation -- AO-RIS
   overtakes no-RIS at **M=1024** (AO-RIS=14.97 vs no-RIS=14.36
   bit/s/Hz), confirming the trend from the M=16..256 plot does continue
   and eventually wins, it just needs a genuinely large surface in this
   geometry.
3. **AO convergence** (`exp3_convergence_K1/K4.png`): the rate increases
   monotonically with AO iteration from every random initialisation and
   converges within a handful of iterations, for both K=1 (closed form)
   and K=4 (manifold optimization).
4. **EE vs SE** (`exp4_ee_vs_se.png`): larger M gives both higher SE *and*
   higher EE at a given Ptx (the RIS's extra hardware power `M*P_RIS` is
   cheap next to the rate gain it buys); each M-curve bends over at high
   Ptx as the `Ptx/eta` term in `P_total` starts to dominate the
   logarithmic rate gain.
5. **Distance sensitivity** (`exp5_distance_*.png`): rate decreases
   monotonically with both RIS-UE and BS-RIS distance for every scheme, as
   expected from free-space path loss.
6. **Frequency** (`exp6_rate_vs_frequency.png`, `exp6_absorption_vs_distance.png`):
   140 GHz and 300 GHz have substantially lower rate than 28 GHz at equal
   distance, dominated by the wavelength^2 free-space path loss term;
   the dedicated absorption panel isolates the molecular-absorption
   contribution, which is comparatively small at these (tens-of-metres)
   distances but grows with both frequency and distance, as expected.
7. **DRL demo, optional** (`exp7_drl_learning_curve.png`): SAC was trained
   for 40,000 steps (K=1, M=16), completing in 639.6 s, well inside the
   15-minute budget when run without other heavy processes competing for
   CPU. Result: AO (closed-form) oracle mean rate ≈3.09 bit/s/Hz,
   random-phase baseline ≈0.69 bit/s/Hz, SAC's final smoothed rate ≈0.63
   bit/s/Hz -- SAC did not even match the random-phase baseline, let alone
   approach the AO oracle (**≈80% gap**). The learning curve shows no
   upward trend over the full run; it just oscillates around the
   random-phase line. This is reported plainly as a negative result, not
   tuned further, per the master prompt's framing of this milestone as a
   proof of concept: each episode is a single step (no temporal credit
   assignment for the agent to exploit) and the reward landscape (coherent
   phase alignment across M=16 dimensions) is a narrow, non-convex optimum
   that off-the-shelf continuous-control RL struggles to find from scratch
   without reward shaping or curriculum. See `docs/ASSUMPTIONS.md` for the
   full writeup, including that this same qualitative finding reproduced
   across two separate training runs (before and after the channel-model
   fix below), which is further evidence it reflects the RL setup rather
   than a one-off fluke.

## Limitations

- THz molecular absorption coefficients are placeholders, not derived from
  ITU-R P.676 / HITRAN data (see [8]).
- The direct link's blockage is a fixed dB penalty, not geometric ray
  tracing against the drawn blocker.
- No channel estimation error or hardware impairments are modelled
  (perfect CSI assumed).
- The AF relay and RIS legs share a simplified single-path/sparse-SV
  path-loss model rather than independently validated measurements.
- The DRL demo (exp7, optional) is a proof of concept only, not tuned to
  match AO.

## Future work

SDR-based RIS phase optimization with Gaussian randomization; Objective B
(min-power design); realistic/estimated CSI instead of perfect CSI;
hardware impairments (phase quantization, amplifier nonlinearity); joint
multi-user DRL (beamformer + RIS phases together); DRL hyperparameter
tuning aimed at matching or beating AO; a measurement-based or ITU-R/HITRAN
molecular absorption model; a full written paper (Step 6 of the
assignment).

## References

See `docs/REFERENCES.md`.
