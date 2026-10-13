# Final presentation outline

Slide-by-slide outline for the final evaluation (Phase 2 spec, Milestone
8), ~15 slides. Each slide names the figure to show (from `plots/`,
unless noted) and the one or two sentences to say. Numbers are pulled
from `results/summary.csv` and `docs/ASSUMPTIONS.md`; see those for
full detail and `docs/VIVA_QA.md` for anticipated questions.

---

**1. Title**
No figure. Project title, your name, course/institution, date. One line
framing the problem: "RIS-assisted mmWave/THz links: can a passive
reflecting surface route around a blocked direct path, and how far can
cheaper alternatives (DRL) go toward the same answer?"

**2. Motivation**
Figure: `geometry.png`. mmWave/THz bands offer huge bandwidth but are
easily blocked (near-zero diffraction) and suffer severe path loss. An
RIS placed with line of sight to both BS and UE offers a low-cost,
passive way to route around the blockage. State the scenario: BS, RIS,
blocked UE, as drawn.

**3. System model**
No figure (or a simple equation slide). Show the cascaded channel
$h_{\mathrm{eff},k}^H = g_k^H \Phi H$ and the SINR expression. One
sentence: "RIS applies a per-element phase shift $\theta_m$, $|\theta_m|=1$
-- it reflects, it does not amplify."

**4. The three objectives**
No figure. Objective A (max sum rate), B (min power s.t. SINR target),
C (max energy efficiency). One line each; say these are all solved by
the same underlying AO machinery with a different per-step objective.

**5. The AO algorithm**
No figure (or the algorithm box from `paper/main.tex`). Alternate: fix
phases, solve beamformer (closed form); fix beamformer, solve phases
(closed form for K=1, Riemannian manifold ascent for K>1). Each
half-step only accepted if it doesn't decrease the rate -> guaranteed
monotone convergence to a stationary point.

**6. Milestone 0: a bug we found and fixed**
Figure: `exp_blockage_sweep_M64.png`. The mid-eval's "blocked" link
wasn't blocked enough (40 dB let it beat RIS); fixed by raising to 80 dB
after this sweep. Also found and fixed a more consequential bug: the
MRT precoder was missing a conjugate, affecting every K=1 result before
the fix -- say this plainly, it's a strength (rigor), not a weakness.

**7. Objective A results: RIS helps, but path loss is brutal**
Figures: `exp1_rate_vs_power_K1.png`, `exp2_rate_vs_M_K1.png`. RIS beats
random phases at every power level (2.16x at 30 dBm) -- but the
un-optimized direct link still wins at M=64, because a cascaded 2-hop
path loss is a *product*, not a sum. RIS only overtakes direct once M
is large enough (array gain ~M^2, confirmed 15.5x vs theoretical 16x
for M: 64->256).

**8. Objective B & C: power and energy efficiency**
Figures: `exp_objB_power_vs_sinr.png`, `exp_objC_ee_se_tradeoff.png`.
Same coherent-combining gain, read as a power saving instead of a rate
gain (63x less power at a 15 dB target). EE-optimal transmit power
(20.6 dBm) is below the rate-maximizing budget (30 dBm) -- past that
point, extra power costs more than the log-scaled rate gain is worth.

**9. Milestone 3: a real absorption model**
Figure: `exp_absorption_vs_frequency.png`. Replaced a placeholder dB/km
table with the real ITU-R P.676 gaseous attenuation model. At the
distances tested here it's a small correction (<1% rate difference) but
it's now the physically correct model, not a guess.

**10. Milestones 4 & 5: what happens with cheap hardware and noisy CSI**
Figures: `exp_hw_quantization_M64.png`, `exp_csi_error_vs_epsilon_K4.png`.
4-bit RIS phases recover 99.9% of the continuous-phase rate -- discrete
hardware is nearly free. Imperfect CSI is a different story: K=4 (RZF)
loses 76% of its rate at the worst tested error level, far worse than
K=1's 5% loss, because RZF needs an accurate estimate to null
interference, not just to steer.

**11. Milestone 6: more users, and a semidefinite-relaxation detour**
Figures: `exp_rate_vs_K_sum.png`, `exp_sdr_vs_manifold_sinr.png`. Sum
rate falls as K grows (not a bug -- ZF/RZF spends more of a fixed
antenna budget nulling interference). SDR is a classical convex
relaxation for the RIS-phase problem; its bound is correct, but
practical recovery (Gaussian randomization) loses to direct manifold
optimization by ~9x at M=64 -- a known property of SDR, not an
implementation error (two real numerical bugs were found and fixed
getting to this clean comparison, see backup slide / Q&A).

**12. Milestone 7: can DRL learn this instead?**
Figure: `exp_drl_stageA_learning_curve.png`. Framed the problem as a
one-step contextual bandit (new channel each episode); trained SAC and
PPO. Both clearly learn -- but land 28-52% short of the AO oracle
depending on M, and that gap *widens* as M grows. This is the honest
result the assignment asked for, not a tuned win.

**13. Headline numbers (summary table)**
No new figure -- a table slide with the 10 headline numbers from
`docs/FINAL_SUMMARY.md`. Use this as the "if you remember one slide"
slide.

**14. Limitations and what a reviewer would attack**
No figure. State the three weakest points plainly (from
`docs/FINAL_SUMMARY.md`): calibrated antenna gains, DRL's time-boxed
budget, and the abstracted CSI-error/blockage models. Shows you know
where the project is strongest and weakest.

**15. Conclusion and future work**
No figure. One line per milestone's headline finding; future work: the
DRL-initialized-AO hybrid (not run), the joint multi-user DRL agent
(Stage C, not run), longer/tuned DRL training, a pilot-based CSI-error
model, HITRAN cross-validation for absorption.

---

## Backup slides (if time / questions permit)

- **B1. SDR derivation detail** -- the $u_{kj}$/$V$ algebra, for anyone
  who asks "how does SDR even apply to a problem with $|\theta_m|=1$
  constraints."
- **B2. The two SDR bugs** -- scaling (solver tolerance vs. the
  problem's actual $10^{-12}$ W scale) and CLARABEL's $O(n^4)$
  interior-point memory cost at M=64, fixed by switching to SCS.
- **B3. Full DRL cross-M table** -- the complete SAC/PPO vs. AO/random
  numbers at M=16/32/64 (Table in `paper/main.tex`, Section Results).
- **B4. Test coverage** -- 39 automated tests, covering all 6 required
  sanity checks (non-decreasing AO rate, etc.) and all 7 spec-named
  required tests.
