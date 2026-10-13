# RIS-Assisted mmWave/THz Downlink Communication (AO)

A student-explainable simulation of a Reconfigurable Intelligent Surface
(RIS) assisting a downlink mmWave/THz link whose direct line-of-sight is
blocked, solved with Alternating Optimization (AO) between BS active
beamforming and RIS passive phase shifts.

**Phase 1** (`RIS_Master_Prompt.md`) implemented a defined subset of the
full assignment (`RIS_Communication_Project_Workflow.md`) and was
presented at the mid-evaluation (tag `phase1-midsem`). **Phase 2**
(`RIS_Phase2_Prompt.md`) audited and fixed Phase 1 issues and completed
every remaining assignment milestone (Objective B, Objective C, realistic
absorption, hardware/CSI impairments, multi-K, SDR, a time-boxed DRL
study) -- **Milestones M0-M7 are done.** Milestone 8 (`paper/` draft,
`docs/FINAL_PRESENTATION_OUTLINE.md`, `docs/VIVA_QA.md`) is also written;
the LaTeX draft has been checked structurally (balanced braces/
environments, every citation resolves, every figure reference matches a
copied file) but **not compiled to PDF**, since no LaTeX distribution is
installed on the development machine -- the spec explicitly allows this
("write the .tex files even if LaTeX is not installed; compile to PDF
only if `pdflatex` is available"). See `docs/ASSUMPTIONS.md` for the
full, detailed writeup of every milestone (including bugs found and
fixed along the way) and `results/summary.csv` for the headline numbers.

## Assignment-step -> file mapping

| Step | Assignment topic | Implemented in | Status |
|---|---|---|---|
| 1 | System model & scenario definition | `config.py`, `channel.py` | Done |
| 2 | Optimization problem formulation | `metrics.py` (SINR/rate), `beamforming.py`/`ris_opt.py` (Objective A), `objective_b.py` (Objective B, min-power) | Done: Objectives A, B, and C (EE, `exp_objC_*`) |
| 3 | Algorithm design | `ao.py`, `ris_opt.py`, `beamforming.py`, `objective_b.py`, `sdr.py`, `drl_env.py` | Done: AO (MRT/ZF/RZF + closed-form/manifold RIS phases) for Objectives A/B; SDR relaxation for the max-min-SINR problem (`sdr.py`, Milestone 6b); a time-boxed DRL study, SAC/PPO (Milestone 7, Stages A/B -- Stage C, the joint K>1 agent, scoped out, see `docs/ASSUMPTIONS.md`) |
| 4 | Implementation & simulation | all of the above + `benchmarks.py`, `hardware.py`, `csi_error.py` | Done, including discrete phase shifts and imperfect-CSI impairment models |
| 5 | Result analysis & plots | `experiments/` (22 scripts), `results/summary.csv` | Done for every milestone M0-M7; reproduce everything via `./run_all.sh` (or `--fast` for a quick smoke test) |
| 6 | Documentation & paper writing | this README, `docs/`, `paper/` | Done: results + assumptions documented for M0-M7; `paper/main.tex` (IEEEtran draft, structurally checked, not PDF-compiled -- no LaTeX installed), `docs/FINAL_PRESENTATION_OUTLINE.md` (15 slides), `docs/VIVA_QA.md` (30 Q&A) all written (Milestone 8) |

## Scope

Phase 1's scope (`RIS_Master_Prompt.md` section 0): blocked-LoS, sparse
Saleh-Valenzuela + THz molecular absorption channels, sum-rate
maximization (Option A), AO (MRT/ZF/RZF + closed-form/manifold RIS
phases), three benchmarks, sensitivity sweeps over M/Ptx/distance/
frequency, an optional single-user DRL demo. Phase 2 (`RIS_Phase2_Prompt.md`,
**done through Milestone 7**) added: a post-mid-eval audit and bug-fix
pass, including a significant MRT-precoder bug affecting every K=1 result
(Milestone 0); Objective B, min-power (Milestone 1); Objective C, energy
efficiency optimization (Milestone 2); a realistic ITU-R P.676 absorption
model via the `itur` package (Milestone 3); discrete RIS phase shifts
with quantization-aware refinement (Milestone 4); an additive-Gaussian
imperfect-CSI model with a robust RZF variant (Milestone 5); multi-user
scaling and SDR for the max-min-SINR problem (Milestone 6); a time-boxed
DRL study with SAC/PPO across M=16/32/64 (Milestone 7, Stages A/B). See
`docs/ASSUMPTIONS.md` for the full writeup of every milestone, including
every bug found and fixed along the way, and `RIS_Phase2_Prompt.md` for
the original plan.

**Still out of scope**: compiling `paper/main.tex` to PDF (no LaTeX
distribution installed -- the spec explicitly allows this); Milestone 7's Stage C (joint K>1 DRL
agent -- scoped out after Stage B's own finding that DRL already strains
to scale with M under the time-boxed training budget, see
`docs/ASSUMPTIONS.md`); DRL hyperparameter tuning aimed at closing the
gap to AO (the spec explicitly time-boxes this rather than asking for a
tuned result). See the Future Work section below.

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

pytest -q                # all 38 tests

./run_all.sh              # reproduces every figure/result from a clean
                           # checkout -- takes HOURS (the DRL and SDR
                           # milestones alone took most of a day on the
                           # development machine, mostly due to unrelated
                           # background processes repeatedly triggering
                           # OS-level low-memory kills, not raw compute;
                           # see docs/ASSUMPTIONS.md)
./run_all.sh --fast       # smoke test: every script at a trivial
                           # Monte-Carlo scale, confirms everything still
                           # runs end-to-end in a few minutes -- its
                           # numbers are NOT meant to be trusted or
                           # committed
```

Or run individual milestones directly, e.g.:

```bash
python -m experiments.plot_geometry          # plots/geometry.png
python -m experiments.exp1_rate_vs_power     # Phase 1, plots/exp1_*.png
python -m experiments.exp_objB_power_vs_sinr # Milestone 1 (Objective B)
python -m experiments.exp_objC_ee_optimal    # Milestone 2 (Objective C)
python -m experiments.exp_absorption_itu     # Milestone 3 (ITU-R absorption)
python -m experiments.exp_hw_quantization    # Milestone 4 (discrete phases)
python -m experiments.exp_csi_error          # Milestone 5 (imperfect CSI)
python -m experiments.exp_rate_vs_K          # Milestone 6a (multi-K)
python -m experiments.exp_sdr_vs_manifold    # Milestone 6b (SDR vs manifold)
python -m experiments.exp_drl_stageA         # Milestone 7, Stage A (K=1, M=16)
python -m experiments.exp_drl_stageB         # Milestone 7, Stage B (K=1, M=32/64)
python -m experiments.make_summary           # rebuild results/summary.csv
```

Every experiment saves its raw curves to `results/*.npz` (so plots can be
regenerated without re-running the Monte Carlo sweeps) and its figures to
`plots/*.png` at 200 dpi. `run_all.sh` runs every script in milestone
order; see it for the complete list (22 scripts).

## Assumptions and parameters

See `docs/ASSUMPTIONS.md` for the full list with reasoning. Headline
values:

| Parameter | Value | Note |
|---|---|---|
| `fc` | 140 GHz (28 / 300 GHz also tested) | |
| `Nt`, default `M`, `K` | 8, 64 (8x8), 1 or 4 | |
| `Pmax_dBm` | 30 dBm | swept 0-30 dBm in exp1/exp4 |
| `blockage_loss_db` | **80 dB** (Phase 2 Milestone 0) | raised from Phase 1's 40 dB, which let the un-optimized direct link beat AO-RIS; 80 dB represents a genuinely severe obstruction -- the old 40 dB value stays reproducible as an explicit comparison point, see `exp_blockage_sweep.py` |
| `bs_gain_dbi` / `ue_gain_dbi` | 45 / 25 dBi (default preset); 25 / 10 dBi (`Config.realistic()`, lower-gain sensitivity point) | calibrated for ~15-25 dB mean AO SNR at K=1,M=64,Ptx=30dBm,140GHz; deliberately high -- see docs/ASSUMPTIONS.md |
| Absorption model | **`itur` (ITU-R P.676, default)**; `absorption_db_per_km` table kept as a `"placeholder"` comparison option | Phase 2 Milestone 3; see `exp_absorption_itu.py` and [8] |
| EE power model | eta=0.4, P_BS=1W, P_RIS=5mW/elem, P_UE=0.1W | see [5] |
| RIS phase quantization | continuous (default); 1-4 bit discrete with coordinate-descent refinement | Phase 2 Milestone 4, `hardware.py` |
| CSI error model | perfect CSI (default); additive Gaussian, epsilon in [0, 0.5] | Phase 2 Milestone 5, `csi_error.py` |

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

## Phase 2 results (Milestones 0-7)

*(Full writeup with every bug found and fixed, every numeric finding, and
every honest limitation is in `docs/ASSUMPTIONS.md` -- this is a brief
index. Headline numbers are in `results/summary.csv`.)*

- **Milestone 0 (audit/fixes)**: found and fixed a significant MRT-
  precoder bug (missing conjugate) affecting every K=1 result in the
  project; raised `blockage_loss_db` 40->80 dB so "blocked" is genuinely
  blocked (`exp_blockage_sweep.py`).
- **Milestone 1 (Objective B, min-power)**: closed-form (K=1) and
  manifold-descent (K>1) min-power RIS design; required power decreases
  monotonically in M, as expected (`exp_objB_*.py`).
- **Milestone 2 (Objective C, energy efficiency)**: grid search +
  golden-section refinement finds an EE-optimal Ptx distinct from the
  rate-maximizing Ptx=30dBm at every M (`exp_objC_ee_optimal.py`).
- **Milestone 3 (realistic absorption)**: real ITU-R P.676 gaseous
  attenuation via the `itur` package, replacing Phase 1's placeholder
  dB/km table (`exp_absorption_itu.py`).
- **Milestone 4 (discrete RIS phases)**: 1-4 bit quantization with
  coordinate-descent refinement; rate approaches the continuous-phase
  bound as bits increase, as expected (`exp_hw_quantization.py`).
- **Milestone 5 (imperfect CSI)**: additive-Gaussian CSI error model plus
  a robust RZF variant; K=4 (RZF) is far more CSI-error-sensitive than
  K=1 (MRT) (`exp_csi_error.py`).
- **Milestone 6 (multi-K, SDR)**: sum rate decreases monotonically with K
  at fixed Nt=8 (`exp_rate_vs_K.py`); SDR for the max-min-SINR problem,
  including two real numerical bugs found and fixed (solver scaling, then
  CLARABEL's interior-point memory/time cost at scale, fixed by switching
  to SCS with a bounded iteration count) -- manifold optimization matches
  or beats SDR+randomization at every M tested, and is much faster
  (`sdr.py`, `exp_sdr_vs_manifold.py`).
- **Milestone 7 (DRL, Stages A/B)**: SAC and PPO (3 seeds each) learn a
  non-trivial RIS phase policy purely from reward signal, landing
  28-52% short of the AO oracle depending on M -- a real, honest "DRL
  doesn't scale as gracefully as AO" finding, not a tuned result
  (`exp_drl_stageA.py`, `exp_drl_stageB.py`). Stage C (joint K>1 agent)
  was scoped out given this finding.

## Limitations

- The direct link's blockage is a fixed dB penalty, not geometric ray
  tracing against the drawn blocker.
- The AF relay and RIS legs share a simplified single-path/sparse-SV
  path-loss model rather than independently validated measurements.
- CSI error is modelled as additive Gaussian noise on the channel
  estimate, not derived from a specific estimation algorithm (e.g. pilot-
  based LS/MMSE); hardware impairments beyond RIS phase quantization
  (e.g. amplifier nonlinearity, phase noise) are not modelled.
- The DRL study (Milestone 7) is explicitly time-boxed per the spec, not
  tuned to match or beat AO; Stage C (joint multi-user agent) was not
  attempted, see `docs/ASSUMPTIONS.md`.
- SDR is not run above M=64 (the SDP does not scale well at this problem
  size even with the fixes found during Milestone 6, see
  `docs/ASSUMPTIONS.md`).

## Future work

Compiling `paper/main.tex` to PDF once a LaTeX distribution is
available (content is written and structurally checked, see Milestone
8); the DRL joint multi-user agent (Stage C, K>1); DRL hyperparameter
tuning or a longer training budget aimed at closing the gap to AO;
channel estimation modelled from a specific pilot/estimator algorithm
rather than an additive-error abstraction; a measurement-based or
HITRAN-cross-validated molecular absorption model alongside the current
ITU-R P.676 one; the DRL-initialised-AO hybrid suggested as an optional
stretch goal in the Phase 2 spec (use the trained actor's output as
AO's initial theta and measure the iteration savings vs. a random
start).

## References

See `docs/REFERENCES.md`.
