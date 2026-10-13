# Viva Q&A

30 likely questions and short answers, based on the final Phase 2
results (`docs/FINAL_SUMMARY.md`, `docs/ASSUMPTIONS.md`,
`paper/main.tex`). Questions 1-7 extend `docs/EXPLAIN.md`'s Phase 1 set;
8-30 cover Phase 2.

## General algorithm / Phase 1 foundations

**1. Why does the RIS phase have a closed-form solution for K=1 but not K>1?**
For K=1 the objective is `|sum_m c_m*theta_m|^2` for fixed complex
`c_m = conj(g_m)*a_m` -- a phase-weighted sum whose magnitude is
maximised exactly by aligning every term's phase (triangle-inequality
equality case). For K>1 the objective is a sum of coupled SINR ratios
through the same `theta` -- no closed form, so we optimise numerically
on the unit-modulus manifold.

**2. Why a manifold optimizer instead of a gradient step then normalize?**
Normalize-after-step ignores the constraint set's curvature and can
waste steps. The Riemannian version projects the gradient onto the
tangent space first (removing the radial component normalization would
discard anyway), so every step moves along the manifold, with an Armijo
line search giving a principled, monotone-improving accept/reject rule.

**3. Why is AO's rate history guaranteed non-decreasing?**
Each half-step only replaces the current W or theta if the candidate's
rate is >= the best rate seen so far; otherwise the previous iterate is
kept (safeguarded AO). Combined with a finite power/finite-M upper
bound, this guarantees convergence, though only to a stationary point
for K>1, not necessarily the global optimum.

**4. Why can RIS (even AO-optimised) be worse than no RIS at all?**
A RIS-aided link is cascaded: its path loss is the *product* of the two
hops' losses, much larger than a single-hop link. Coherent phases buy
back an M^2 array gain, but M^2 isn't unconditionally "big enough" --
at M=64 in this geometry, the un-optimized direct link still wins; RIS
only overtakes it once M grows large (M=1024 in our extended sweep, or
M>=256-ish once blockage is severe enough, see Q9).

**5. Why does ZF need Nt>=K, and what does RZF add?**
ZF inverts the K x K Gram matrix H_eff H_eff^H, requiring H_eff to have
rank K (so Nt antennas and enough multipath). RZF adds a
`(K*sigma2/P)*I` regulariser, trading a little interference leakage for
much better conditioning -- it never fails outright the way plain ZF can
on a near-singular channel.

**6. Why doesn't energy efficiency just increase monotonically with Ptx?**
EE = BW*rate/P_total, and P_total includes Ptx/eta, growing linearly in
Ptx, while rate grows only logarithmically (log2(1+SNR)). EE rises,
peaks, then falls once the marginal rate gain no longer justifies the
extra power -- confirmed numerically: the EE-optimal Ptx at M=64 is
20.6 dBm, below the rate-maximizing 30 dBm budget.

**7. Why is the direct link modelled with a fixed blockage_loss_db instead of ray tracing?**
The assignment asks for a "completely blocked" direct LoS while keeping
the no-RIS curve finite and plottable. A fixed penetration-loss penalty
is the simplest model meeting both requirements without implementing
full geometric ray tracing, which is out of scope.

## Milestone 0: audit and fixes

**8. What was the MRT precoder bug, and how did you find it?**
`w = h_eff/||h_eff||` (no conjugate) instead of the correct
`w = conj(h_eff)/||h_eff||` -- by Cauchy-Schwarz, the optimal matched
filter is proportional to the conjugate, not the raw vector. Found while
investigating the blockage-realism issue (Q9): fixing the "blocked"
direct link exposed that AO-RIS's own numbers looked suspiciously low,
which led to re-deriving the MRT optimum from scratch and catching the
missing conjugate. Verified by brute-force search and a closed-form
Cauchy-Schwarz check; it affected every K=1 result in the project.

**9. Why 80 dB for "completely blocked," and why not just set the direct link to zero?**
A blockage sweep (0-100 dB) showed the AO-RIS/no-RIS crossover sits at
60 dB in this geometry; 80 dB represents a genuinely severe obstruction
while keeping the no-RIS curve finite and plottable (the assignment
wants a comparable baseline, not a literal zero). The old 40 dB value is
kept reproducible as an explicit "partially blocked" comparison point.

## Milestone 1: Objective B (min power)

**10. How does the min-power problem relate to the max-rate problem for K=1?**
For fixed phases, the K=1 required power is `P = gamma*sigma2/||h_eff||^2`
-- minimized by *maximizing* `||h_eff||`, which is exactly the same
phase-alignment step that Objective A uses to maximize rate. So for
K=1, AO for Objective B converges in a single pass, reusing Objective
A's RIS step unchanged.

**11. What does "outage" mean in Objective B, and how is it detected?**
If the power required to hit the target SINR exceeds the budget Pmax,
the instance is infeasible under that budget; we flag it as an outage
and count the outage rate across draws, rather than silently clipping
power to Pmax (which would misrepresent the SINR actually achieved).

**12. Why does required power decrease monotonically with M?**
More RIS elements means a larger `||h_eff||` once phases are optimally
aligned (every added element contributes constructively), so the power
needed to reach a fixed SINR target strictly falls as M grows --
confirmed numerically (0.066x going M=64->256) and enforced as an
automated test.

## Milestone 2: Objective C (energy efficiency)

**13. Why is there an EE-optimal power distinct from the rate-maximizing power?**
See Q6 -- EE's power term grows linearly, rate only logarithmically, so
EE peaks before Ptx reaches its maximum budget. We find this optimum by
a grid search over Ptx (refined with golden-section search), not a new
solver.

**14. Does more RIS (larger M) always help EE?**
Larger M raises both SE and EE at a fixed Ptx -- the extra element
power (`M*P_RIS`, 5 mW/element) is cheap next to the rate gain it buys.
But the EE-*optimal* Ptx itself shifts down as M grows (more array gain
means less transmit power is needed to reach the point of diminishing
marginal return).

## Milestone 3: realistic absorption

**15. Why does molecular absorption matter more at higher frequency and longer distance?**
Absorption is a multiplicative `10^(-k(f)*d/10000)` factor: `k(f)`
itself grows with frequency (closer to oxygen/water-vapour resonance
lines), and the exponent grows linearly with distance -- so the effect
compounds with both.

**16. How much did switching from the placeholder table to ITU-R P.676 change your results?**
Small at the distances tested here: 3.558 vs. 3.526 bit/s/Hz at 300 GHz,
d=20m (<1% difference) -- free-space path loss dominates at these
tens-of-metres distances. The real model is still the physically
correct one to use, and the difference would grow at longer range or
higher humidity (see the humidity-sweep plot).

## Milestone 4: discrete phase shifts

**17. Why does even 1-bit quantization recover most of the rate?**
1-bit quantization still lets every element pick the *closer* of two
phases to its ideal continuous value -- for a coherent-combining
objective, being roughly right on every element beats being exactly
right on none. We measured 89.3% of the continuous-phase rate recovered
at 1 bit, rising to 99.9% at 4 bits.

**18. What is "quantization-aware refinement," and why does it help?**
After quantizing the continuous-optimal phases to the nearest discrete
level, we run coordinate-descent: for each element in turn, try every
allowed discrete level and keep the one that maximizes rate, repeating
until no element changes. This recovers some of the loss from naive
nearest-level rounding, since the "nearest in phase" level isn't always
the "best for rate" level once other elements are already quantized.

## Milestone 5: imperfect CSI

**19. Why is K=4 (RZF) so much more sensitive to CSI error than K=1 (MRT)?**
MRT only needs the channel estimate to *steer* toward one user -- a
noisy estimate causes mis-steering, a graceful degradation. RZF
additionally needs the estimate to *null* interference between users; a
noisy estimate both mis-steers and fails to null, and the nulling error
compounds across users. Measured: K=1 loses ~5% of its rate at the
worst tested error level; K=4 loses ~76%.

**20. What would a "robust" precoder do differently, and did it help?**
The robust RZF variant inflates the usual regularizer by the estimated
channel-error power, trusting a noisier estimate less (treating
estimation error like extra effective noise). It stayed consistently a
few percent above plain RZF at every error level, but could not fully
compensate for severe error -- a partial, not complete, mitigation.

## Milestone 6: multi-user scaling and SDR

**21. Why does sum rate decrease with K instead of increase?**
At fixed Nt=8 antennas, going from K=1 to higher K means ZF/RZF spends
more of that fixed spatial budget nulling inter-user interference rather
than steering array gain to any one user (the "price of orthogonality").
We measured a monotonic decrease (11.58, 11.54, 10.29, 9.65 bit/s/Hz at
K=1,2,4,6) -- an earlier draft of our own writeup wrongly claimed a peak
at K=4; we caught and corrected this against the actual saved data.

**22. What is semidefinite relaxation, intuitively, and why is it needed for RIS phases?**
The unit-modulus constraint `|theta_m|=1` makes the phase problem
non-convex. SDR lifts theta into a higher-dimensional PSD matrix
variable V = conj(theta)*theta^T and relaxes the "V has rank 1"
requirement away, leaving a convex semidefinite program that every SINR
constraint becomes *linear* in. It's a standard technique for this class
of non-convex QCQP, not something we invented.

**23. Why does Gaussian randomization lose to manifold optimization if the SDR bound is correct?**
We independently confirmed the SDR relaxation bound itself is correct
(the manifold result lands inside the SDR-bisected feasible region). The
gap is in *recovery*: Gaussian randomization draws candidate thetas from
V's distribution and keeps the best of ~200-3000 tried, but for this
problem that practical search undershoots the bound noticeably --
0.11 vs. 1.01 at M=64 -- a known, literature-documented SDR limitation
for this class of problem, not a bug in our implementation.

**24. What numerical bugs did you find in the SDR implementation?**
Two. First, an unscaled solve (the problem's natural scale, sigma2 ~
1e-12 W, is far below typical solver tolerances) reported a spuriously
"feasible" gamma that actually violated a constraint when checked
directly -- fixed by rescaling u by 1/sqrt(sigma2) before solving
(exact, not approximate, since gamma is a dimensionless ratio). Second,
the interior-point solver CLARABEL cost ~3.9 GB and 20-25s for a single
M=64 solve (architecturally O(n^4) memory for an nxn PSD cone),
making the full sweep impractical -- fixed by switching to the
first-order solver SCS with a bounded iteration count, verified to agree
with CLARABEL on every feasibility decision tested.

## Milestone 7: Deep Reinforcement Learning

**25. Why is the DRL episode just one step instead of a multi-step trajectory?**
Each episode draws an independent, i.i.d. channel realization -- there
is no state that carries meaningfully from one channel to the next, so
a multi-step formulation would have nothing for the agent to learn
across steps. This is the standard "learn a per-instance configuration
policy" framing used in the cited reference (Huang, Mo & Yuen, JSAC
2020), not a simplification we introduced.

**26. What is the reward function, and why normalize by the AO rate?**
Reward = achieved rate for the chosen action, divided by the AO rate for
the *same* channel instance, so reward lies in roughly [0, ~1] across
very different channel conditions -- without this, a lucky high-SNR
draw would dominate the training signal over genuinely good phase
choices on harder draws.

**27. Why does SAC do worse than PPO as M grows?**
Both are given the same fixed 20-minute CPU-time budget per run. SAC's
per-step cost (replay-buffer gradient updates) rises faster with M than
PPO's does, so SAC completes far fewer effective training steps at
larger M within the same budget -- confirmed directly: SAC's step count
roughly halves from M=16 to M=64 for the same CPU budget, while PPO
(step-capped at 200k, not time-capped, at nearly every M tested) is
much less affected.

**28. Is DRL ever better than AO in your experiments? If not, why report it at all?**
No -- DRL (PPO, the better of the two) reaches at most 72% of the AO
rate (M=16), falling to 58% at M=64. We report it because the
assignment explicitly asks for a "fair, honest" comparison, not a
guaranteed win, and the *shape* of the result (DRL learns something
real, but its relative quality degrades as the problem scales) is itself
an informative finding about where model-free RL struggles against a
well-understood closed-form/iterative baseline.

**29. What would you need to do to make DRL competitive with AO?**
Per our own limitations writeup: a much longer or better-tuned training
budget (ours is explicitly time-boxed per the assignment), likely
reward shaping or a curriculum for larger M, and possibly the
DRL-initialized-AO hybrid suggested as an optional stretch goal (use the
trained actor's output as AO's initial theta and measure iteration
savings) -- none of which was run here, left as future work.

## Methodology / reproducibility

**30. How do you know your results are reproducible and correct?**
Every experiment is seeded via a single `Config.rng(seed)` factory, so
re-running any script reproduces identical numbers. 39 automated tests
(`pytest -q`) cover every sanity check the spec requires (e.g. AO rate
non-decreasing, required power non-increasing in M) as hard regression
tests, not just visual plot inspection, plus every spec-named required
test (Objective B closed-form, EE formula, quantiser, CSI-error
variance, SDR algebra, DRL reward-matches-rate, seed reproducibility).
`./run_all.sh --fast` smoke-tests the entire pipeline end-to-end in
minutes; `./run_all.sh` (no flag) regenerates every real figure/result
from a clean checkout.
