# PHASE 2 PROMPT FOR CLAUDE CODE — RIS-Assisted mmWave/THz Project (post mid-evaluation)

Paste everything below this line into Claude Code, in the existing `ris_project/` folder.

---

You are continuing an existing Python project in `ris_project/`. **Phase 1 is finished and was presented at the mid-evaluation.** Phase 2 completes the remaining parts of the original assignment (`RIS_Communication_Project_Workflow.md`) and strengthens the project toward a paper.

## 0. Ground rules

1. **Read first, then change.** Before writing anything, read: the assignment file, `RIS_Master_Prompt.md` (the Phase 1 plan), `README.md`, `docs/ASSUMPTIONS.md`, `docs/REFERENCES.md`, `config.py`, `channel.py`, `ao.py`, `beamforming.py`, `ris_opt.py`, `benchmarks.py`, `metrics.py`, and the experiment scripts. Summarise in 10 lines what exists. Do not rewrite working code; extend it, keep the existing function signatures, and keep the Phase 1 results reproducible.
2. **Version control.** If there is no git repo, run `git init`, commit the current state, and tag it `phase1-midsem`. Make one commit per milestone with a clear message.
3. **Stop-and-report.** After each milestone: run `pytest -q`, run that milestone's experiments, then print a short report (what was built, key numbers, failures, assumptions). Continue to the next milestone automatically unless a test fails or a result looks physically wrong; in that case fix it or stop and explain.
4. **Honesty.** Never fake or tune results to look good. Report negative results plainly. Separate "assumption" from "result" in the docs. Cite only verified references (see `docs/REFERENCES.md`); if you want a new reference, check that it exists via web search first and record its arXiv/DOI link; do not invent citations.
5. **Notation and style** stay the same as Phase 1 (BS, RIS, UE, H, g, Φ, θ_m, W, Nt, M, Ptx, Pmax, Lmol). Same plot style and colours per scheme in every figure. Seeded, reproducible, results saved to `results/*.npz` and plots to `plots/`.
6. **Priority tiers.** Do the milestones in order. M0–M6 are MUST. M7 (DRL) and M8 (paper) are the largest and come last. Time-box M7 (see there).

## 1. Milestone 0 — Audit and fix Phase 1 issues (MUST, do first)

Known issues found when reviewing the mid-evaluation presentation:

a. **Blocked scenario is only "weakened".** In Result 1 the no-RIS (blocked) direct link had the highest rate and even beat the AF relay, which is inconsistent with a "completely blocked" link. Investigate `blockage_loss_db`, antenna gains, noise and geometry in `config.py`. Then:
   - Add a **blockage sweep experiment** (`exp_blockage_sweep.py`): blockage loss ∈ {0, 20, 40, 60, 80, 100} dB, plotting rate of no-RIS, random RIS, AO-RIS and AF relay (M=64 and M=256, Ptx=30 dBm). Also report the blockage level at which AO-RIS first beats no-RIS and the relay.
   - Choose a documented **default for "completely blocked"** (e.g. 80 dB or more, or an explicit option `direct_link="off"` that sets the direct channel to zero and reports the rate as 0). Re-run exp1 and exp2 with that default and keep the old 40 dB results as a "partially blocked" variant in `results/`.
b. **Rate vs M sweep range.** The assignment requires M = 16, 32, 64, 128, 256. Phase 1 plotted 64–1024. Produce both: `rate_vs_M_16_256.png` (assignment range) and the extended 64–1024 crossover figure.
c. **Frequency and distance studies** were claimed but not shown. Make sure exp5 (distance) and exp6 (28 / 140 / 300 GHz) run and produce clear figures with a one-sentence takeaway each in the README.
d. **Number consistency.** Generate `results/summary.csv` that contains every number quoted in slides or README (e.g. AO gain over random at M=64, Ptx=30 dBm; crossover M vs relay and vs direct path; iterations to converge). Print them, so the student can fix any slide/text mismatch (the slides said "4.4×" in one place and "2–4×" in another, and "~10 rounds" where the plot converged in about 4–6).
e. **Realism of absolute rates.** Mean rates near 14 bit/s/Hz imply an SNR of about 40 dB. Print the mean SNR at default settings. Add a config preset `realistic` (lower antenna gains / narrower scenario so peak SE stays below about 8 bit/s/Hz) and re-run exp1/exp2 with it. Keep both presets and say which is the default and why.
f. **Initial phases.** `run_ao` starts from θ = 1 when `theta0` is None. Document this, and make the experiments that claim "random initialisations" pass random θ0 explicitly.

## 2. Milestone 1 — Objective B: minimise transmit power under an SINR target (MUST)

Assignment Option B: minimise Ptx subject to SINR_k ≥ γ for every user, |θ_m| = 1.
- **K = 1:** for fixed Φ the minimum power is P = γσ² / ‖h_eff‖². So the RIS step is the same phase alignment that maximises ‖h_eff‖ (closed form); AO converges in one pass. Implement and verify.
- **K > 1 with ZF:** with W = H_effᴴ (H_eff H_effᴴ)⁻¹ diag(√(γσ²)), the interference is zero and the required power is γσ²·trace((H_eff H_effᴴ)⁻¹). The RIS step is to minimise trace((H_eff H_effᴴ)⁻¹) over θ with Riemannian gradient descent (reuse the manifold code with the new objective; autograd as before). Add a unit test comparing the closed-form power with a direct computation.
- Report infeasibility cleanly: if the required power exceeds Pmax, the instance is "infeasible" and counted as outage.
- Experiments: required Ptx (dBm) vs target SINR (0–30 dB) for AO-RIS, random RIS, no-RIS, AF relay; required Ptx vs M; outage probability vs γ at fixed Pmax. Add a convergence plot.

## 3. Milestone 2 — Objective C: energy-efficiency maximisation (MUST)

EE = BW·R / P_total with P_total = Ptx/η + P_BS_static + M·P_RIS + K·P_UE (same model as Phase 1; see `docs/ASSUMPTIONS.md`).
- For each M, run AO at a grid of Ptx values (e.g. 40 points between −10 and 30 dBm) and choose the Ptx that maximises EE (a one-dimensional search over the transmit power; no new solver needed). Optionally refine with golden-section search.
- Plots: EE-optimal Ptx vs M; maximum EE vs M; the EE–SE trade-off curve with the EE-optimal point marked for each M; a comparison of the sum-rate-optimal point and the EE-optimal point.
- Sensitivity: EE vs RIS element power P_RIS (1, 5, 10, 20 mW) and vs static power. State clearly that the power-model numbers are assumptions (cite Huang et al., TWC 2019 for the model structure).

## 4. Milestone 3 — Realistic frequency-dependent molecular absorption (MUST)

Replace the placeholder dB/km table by a physically grounded model.
- Preferred: use the `itur` Python package (ITU-Rpy; ITU-R P.676 gaseous attenuation). Verify the API from its documentation or source before using it, and install it with pip. Compute specific attenuation γ(f) in dB/km for dry air (oxygen) and water vapour for stated conditions (e.g. T = 15 °C, p = 1013 hPa, water-vapour density ρ = 7.5 g/m³; make these config parameters).
- If `itur` cannot be installed or does not work offline, fall back to implementing the P.676 line-by-line specific attenuation from the recommendation's tables, and say clearly what you did. Do not invent coefficients; if you cannot get the real tables, keep the placeholder table and flag it loudly in the README.
- Plot specific attenuation vs frequency (20–350 GHz) and mark 28, 140 and 300 GHz. Add a humidity sweep (ρ = 2, 7.5, 15 g/m³).
- Re-run the frequency experiment (exp6) with the new model and add a distance study at 300 GHz where absorption matters most. Document the effect on the RIS-vs-relay crossover size.

## 5. Milestone 4 — Hardware limitations: discrete phase shifts (MUST)

- Implement `quantize_phase(theta, bits)` (b = 1, 2, 3, 4 bits and continuous). Evaluate the AO solution after quantisation (nearest level), and also a "quantisation-aware" variant: after the continuous solution, run a few rounds of coordinate-wise refinement over the allowed discrete levels (for each element, pick the level that maximises the rate; repeat until no change). Keep it simple.
- Optional second impairment (only if time allows): phase-dependent amplitude, using the practical reflection model from the RIS hardware literature (verify the reference before citing).
- Plots: rate vs bits (with the continuous-phase bound), and the rate-loss gap at M = 64 and 256; the effect on the RIS-vs-relay crossover size.

## 6. Milestone 5 — Imperfect CSI (MUST)

- Model: the BS knows Ĥ = H + E_H and ĝ = g + E_g where each error entry is complex Gaussian with variance σ_e² = ε × (average channel entry power), with normalised error ε ∈ {0, 0.01, 0.05, 0.1, 0.2, 0.5}. (Document this standard additive-error model; it is an assumption.)
- Run AO using the estimates (Ĥ, ĝ) only, then evaluate the rate with the true channels (H, g). Compare against a **robust-ish** variant: reduce the beamformer's power split / regularisation using the error variance (for RZF set the regulariser to account for ε) and report whether it helps.
- Plots: rate vs ε for AO-RIS, random RIS, AF relay (K=1 and K=4), with the perfect-CSI AO as an upper line. Also rate vs M for a fixed ε to show how channel error erodes the M² gain.

## 7. Milestone 6 — Multi-user extensions and SDR comparison (MUST for the first part; SDR at small M only)

a. **Rate vs number of users K** (K = 1, 2, 4, 6; Nt = 8) for ZF and RZF with AO; per-user rate and sum rate; random-phase and no-RIS for reference.
b. **SDR for the RIS phases (assignment Approach A, step 2).** Sum-rate is not directly SDR-friendly, so apply SDR to the **max-min-SINR problem with W fixed**: with u_kj = conj(g_k) ⊙ (H w_j) we have h_kᴴ w_j = θᵀ u_kj, so every SINR constraint is linear in V = θ*θᵀ (derive the exact index/conjugation convention carefully and **verify numerically against the direct computation in a test**). Solve the feasibility SDP for a target γ with `cvxpy` (SCS or Clarabel), bisect on γ, constraints diag(V) = 1, V ⪰ 0, then recover θ by Gaussian randomisation (e.g. 200 draws, keep the best). Compare against the manifold-optimisation solution on the same max-min objective at M = 16, 32, 64. Report objective value and run time. Do not run SDR above M = 64 (it scales badly); say so in the README.
c. Plot: max-min SINR (or sum rate achieved) and run-time for SDR vs manifold optimisation vs random phase vs M.

## 8. Milestone 7 — Approach B: Deep Reinforcement Learning (TIME-BOXED; do after M0–M6 are committed)

Goal: a **fair, honest** DRL study, not a guaranteed win. Time-box: stop each stage after a reasonable effort, and report what happened.

Design (follow the assignment's State / Action / Reward definition and cite Huang, Mo & Yuen, JSAC 2020):
- **Environment** (Gymnasium): each episode draws a new channel realisation (H, g) at a fixed geometry. Provide two variants:
  1. *Phase-only agent:* the BS beamformer is fixed to MRT/ZF given the current phases; the action is the vector of M RIS phases.
  2. *Joint agent:* the action also includes the BS beamformer weights (real and imaginary parts), projected to satisfy ‖W‖_F² = Pmax.
- **State:** real and imaginary parts of the channel features (e.g. the cascaded vector conj(g_m)·(H w_ref)_m for K=1, or stacked cascaded channels for K>1), normalised to zero mean and unit variance. Optionally add UE-location coordinates (assignment option) in a second experiment.
- **Action:** continuous, tanh-squashed, scaled to [−π, π] for phases.
- **Reward:** instantaneous achievable rate, normalised by the AO rate for the same instance during training (so the reward lies in [0, ~1]); log both raw and normalised.
- **Algorithms:** SAC and PPO via `stable-baselines3` (CPU is fine; use `torch`). Run at least 3 seeds for each.
- **Staging:**
  - Stage A: K = 1, M = 16. Verify the agent can learn; compare with AO and random phase. Train up to ~200k steps (cap at about 20 minutes per run).
  - Stage B: K = 1, M = 32 and 64, same settings; plot rate vs training steps for each.
  - Stage C (only if A and B work): K = 2 or 4 with the joint agent at M = 16.
- **Helpful practices:** observation and reward normalisation, a sensible network size (e.g. 2×256), a replay buffer ≥ 100k for SAC, evaluation every N steps on a fixed held-out set of channels (e.g. 500), and report mean ± std over seeds.
- **Hybrid idea (stretch, label it as exploratory):** use the DRL actor's output as the initial θ0 for AO and measure how many AO iterations are needed to converge compared with a random start.
- **Report:** learning curves (eval rate vs steps) with the AO rate and random-phase rate as horizontal reference lines; final-rate table (DRL, AO, random) per M; inference time per decision for DRL vs the run time of AO; an honest paragraph on where DRL falls short and why (sample efficiency, no convergence guarantee, scaling with M).

## 9. Milestone 8 — Documentation and paper draft (Step 6 of the assignment)

Create `paper/` with an IEEE-style LaTeX draft (`IEEEtran` class; write the .tex files even if LaTeX is not installed; compile to PDF only if `pdflatex` is available):
- `main.tex`, `refs.bib` (verified references only: the nine in `docs/REFERENCES.md` plus any new ones you verified), and `figures/` (copied from `plots/` at the end of the run via a script).
- Sections per the assignment: Abstract & Introduction (motivation: blockage in 6G mmWave/THz; proposed solution: RIS with AO, plus the DRL comparison); System & Channel Model (equations for H, g, Φ, the absorption loss and the SINR); Problem Formulation and Proposed Method (objectives A, B, C; the AO algorithm with a numbered algorithm box; the closed-form K = 1 phase step and the manifold step with the tangent projection and retraction; convergence argument: monotone and bounded; limitations: converges to a stationary point, not necessarily the global optimum for K > 1; DRL architecture diagram); Simulation Setup (parameter table with all assumptions); Results and Discussion (every figure with one paragraph of physical insight); Conclusion and Future Scope.
- **State the contribution honestly.** Proposed claims (edit them to match what the results support; do not overclaim): (C1) a RIS-size crossover analysis versus an AF relay and the direct path across 28 / 140 / 300 GHz with a standards-based absorption model; (C2) a practical-impairment study (discrete phases and imperfect CSI) of the AO scheme; (C3, exploratory) AO versus DRL, including the DRL-initialised AO hybrid if it was run. Say clearly that the AO algorithm itself follows the literature and is not claimed as new.
- Also produce `docs/FINAL_PRESENTATION_OUTLINE.md`: a slide-by-slide outline (about 15 slides) for the final evaluation with the figure to show on each slide, plus `docs/VIVA_QA.md` with 30 likely questions and short answers based on the final results.

## 10. Tests and quality gates (apply to every milestone)

- Keep all Phase 1 tests passing. Add tests for: Objective B closed-form power vs direct computation; EE formula; quantiser (levels, |θ|=1); CSI error model statistics (variance matches ε); SDR constraint algebra vs direct SINR; DRL environment reward equals the rate function; seeds give reproducible results.
- Sanity checks that must hold (treat a violation as a bug): AO rate is non-decreasing; rate is non-decreasing in Ptx; required power is non-increasing in M; perfect-alignment bound ≥ any scheme for K=1; quantised rate ≤ continuous rate (on average) and approaches it as bits increase; rate with CSI error ≤ rate with perfect CSI (on average).
- `make all` (or `run_all.sh`) must reproduce every figure from a clean checkout in a reasonable time; provide a `--fast` flag with fewer Monte-Carlo trials for quick checks.

## 11. Final deliverables

1. Updated code, tests passing, tagged `phase2-final`.
2. All new plots in `plots/`, data in `results/`, and a refreshed `results/summary.csv`.
3. Updated `README.md` (assignment-step table now showing which steps are done, partial or deferred), `docs/ASSUMPTIONS.md`, `docs/REFERENCES.md`, `docs/EXPLAIN.md`.
4. `paper/` draft, `docs/FINAL_PRESENTATION_OUTLINE.md`, `docs/VIVA_QA.md`.
5. A final printed summary for the student: what is done, the 10 headline numbers, what failed or was skipped, and the three weakest points a reviewer would attack.

Begin with step 0.1 (read everything and summarise), then Milestone 0.
