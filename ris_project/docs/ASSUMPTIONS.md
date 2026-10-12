# Assumptions

Every numeric choice here that is not a standard physical constant is an
assumption, made explicit so a reader can tell "assumption" from "result".
Corresponding code lives mostly in `config.py` and `channel.py`.

## Phase 2, Milestone 0: post-mid-eval audit and fixes

The mid-evaluation review of Phase 1 flagged several issues. This section
records what was found and what changed; see `RIS_Phase2_Prompt.md` for
the original audit instructions.

**(a) "Blocked" wasn't blocked enough.** At Phase 1's `blockage_loss_db=40`,
the un-optimized direct link (no-RIS) had the *highest* rate of all four
schemes at K=1, M=64, Ptx=30 dBm -- beating both AO-RIS and the AF relay.
That's inconsistent with "completely blocked." `experiments/exp_blockage_sweep.py`
sweeps blockage_loss_db in {0,20,40,60,80,100} dB at M in {64,256} and
finds the AO-RIS/no-RIS crossover sits at **80 dB for M=64** and **60 dB
for M=256** (full 200-draw results; see `plots/exp_blockage_sweep_M{64,256}.png`).
The AF relay crossover essentially never happens vs blockage, by
construction -- the relay sits at the RIS location, not behind the
blocker, so its rate is blockage-independent; only no-RIS responds to this
parameter. **New default: `blockage_loss_db=80`**, representing a
genuinely severe obstruction. The old 40 dB value is preserved and
reproducible, not deleted -- Phase 1's exact numbers live on as the
`*_blocked40.npz`/`.png` files in `results/`/`plots/` (`Config(blockage_loss_db=40.0)`
regenerates them). "Partially blocked" (40 dB) vs "completely blocked"
(80 dB, new default) are both legitimate scenarios; only the *label* and
default changed, not the modelling capability.

**(b) M-sweep range.** The assignment's own M range is 16/32/64/128/256;
`experiments/exp2_rate_vs_M.py` already used exactly that (Phase 1 was
correct here) and is unchanged. The *extended* 64-1024 crossover view
(`experiments/exp2b_crossover.py`, added for the mid-eval panel) is kept
as a clearly-separate, clearly-labelled "bonus" experiment -- it was never
presented as satisfying the assignment's own M range requirement, which
exp2 already does on its own.

**(c) exp5/exp6 status.** Both already run and produce figures with a
one-sentence physical-interpretation takeaway in the README's Results
section (items 5 and 6) -- confirmed present, no gap found here.

**(d) Number consistency.** `results/summary.csv` is the single source of
truth for every headline number; any number quoted in slides or the
README should be checked against it, not re-derived or approximated from
memory. (We don't have access to the actual mid-eval slide deck to check
specific quoted figures like "4.4x" vs "2-4x" against -- if a
specific discrepancy is found, trace it to the exact summary.csv row
and Ptx/M operating point before changing either the slide or the code.)

**(e) Absolute-rate realism / gain calibration.** Phase 1's calibrated
gains (`bs_gain_dbi=45`, `ue_gain_dbi=25`) are deliberately high (see
below) to hit a 15-25 dB SNR target at M=64 given the severe cascaded RIS
path loss -- but that also meant the *no-RIS* link (which doesn't pay the
cascaded-path-loss penalty) could reach ~14 bit/s/Hz at 40 dB blockage,
implying a ~40 dB link SNR, unrealistically high for a real deployment. A
new **`Config.realistic()`** preset (`bs_gain_dbi=25`, `ue_gain_dbi=10`,
closer to typical real hardware) is added alongside the default, not
instead of it -- see the classmethod's docstring in `config.py` for the
full reasoning and the measured peak-rate numbers for each preset.

**(f) Initial RIS phase.** `run_ao`'s `theta0=None` default starts AO from
`theta = all-ones` (zero phase everywhere), *not* a random initialisation.
Only `experiments/exp3_convergence.py` genuinely uses random per-trial
initialisations (and correctly labels them as such); no other experiment
claims "random initialisation" while relying on the default -- confirmed
by inspection, no code change was needed here beyond documenting the
default explicitly in `ao.py`'s docstring.

## A second, more severe bug found while fixing Milestone 0 (MRT precoder)

While investigating item (a) above, a second, independent, and more
consequential bug was found and fixed: `beamforming.py`'s `mrt()` computed
`w = h_eff / ||h_eff||` (no conjugate). Since `h_eff` (per
`effective_channel`'s convention) already stores `h^H`'s entries, the
textbook matched-filter result (Cauchy-Schwarz: `|h^H w| <= ||h||*||w||`,
equality iff `w` is proportional to `h` itself) requires
`w = conj(h_eff) / ||h_eff||`. The unconjugated version is a genuinely
suboptimal direction, not just a different convention -- verified three
ways: (1) direct comparison against the closed-form optimum
`P*||h||^2`, (2) a brute-force search over 200,000 random candidate
directions, which the "fixed" direction beat and the buggy one did not,
and (3) a Monte Carlo estimate that the buggy direction achieves only
**~22% of the optimal SINR on average** (~6-7 dB loss), a ~30%+ rate
underestimate at typical operating SNRs. This affected **every K=1 (MRT)
result across the entire project** -- K>1 (ZF/RZF, which correctly used
`.conj().T`) and the manifold phase optimizer were unaffected (confirmed
by inspection and by the fact that K=4 results didn't move when this fix
was applied). Fixed in `beamforming.py`; locked in by
`test_mrt_achieves_optimal_snr` (exact closed-form check, not just a power-
constraint check). All K=1 experiments were re-run; see Milestone 0 above
for the combined effect with the blockage fix -- together they change the
exp1/exp2 K=1 headline story substantially: AO-RIS now wins decisively at
*every* M tested (64 through 1024), not just at very large M, which is a
cleaner and more reassuring result than either fix alone would have given.

## Phase 2, Milestone 3: realistic ITU-R P.676 molecular absorption

- Replaced the Phase 1 placeholder dB/km table with the real ITU-R P.676
  gaseous (oxygen + water vapour) attenuation model, via the `itur`
  (ITU-Rpy) package. **API verified from its own docstring before use**
  (`itur.gaseous_attenuation_terrestrial_path(r, f, el, rho, P, T, mode)`,
  confirmed to scale exactly linearly with path length r, so calling it
  with r=1 km returns the specific attenuation directly in dB/km) --
  not guessed or invented. Used `mode="exact"` (line-by-line summation,
  valid 1-1000 GHz) rather than `"approx"`, which itur itself warns is
  only valid for elevation angles 5-90 degrees -- our terrestrial path is
  horizontal (el=0).
- Default conditions (now `Config` fields): `temperature_c=15`,
  `pressure_hpa=1013`, `water_vapour_density=7.5` (g/m^3), exactly the
  spec's stated defaults.
- **New default**: `Config.absorption_model = "itu"`. The Phase 1
  placeholder table is kept, unmodified, as `absorption_model="placeholder"`
  for reproducibility (ground rule: never break Phase 1 reproducibility).
- Sanity-checked against well-known atmospheric physics (and locked in by
  `test_itu_absorption_model_matches_known_physics`): a strong oxygen
  absorption line near 60 GHz, a window of lower attenuation 70-150 GHz,
  water-vapour lines near 180 and 325 GHz that scale with humidity while
  the 60 GHz oxygen line does not (confirmed <5% change across
  rho=2->15 g/m^3) -- see `plots/exp_absorption_vs_frequency.png` and
  `plots/exp_absorption_humidity_sweep.png`, both of which visually
  reproduce the textbook ITU-R P.676 spectrum shape.
- Measured values at the three key frequencies (default conditions):
  28 GHz -> 0.10 dB/km, 140 GHz -> 0.92 dB/km, 300 GHz -> 5.25 dB/km.
  These replace Phase 1's hand-picked placeholders (0.1, 2.0, 8.0
  dB/km respectively) with physically grounded numbers -- all in the
  same order of magnitude, but not identical, confirming the
  placeholders were reasonable guesses, not wildly off.
- **Effect on existing results**: negligible at the project's geometry
  (tens of metres). The 300 GHz distance study
  (`exp_absorption_300ghz_distance.png`) overlays the ITU and placeholder
  models directly and they are visually indistinguishable; both give the
  *same* AO-RIS/no-RIS and AO-RIS/AF-relay crossover point (2 m, the
  shortest distance tested -- AO-RIS already wins at every tested
  distance under both models). **Because of this negligible effect,
  only exp6 (the experiment this milestone directly targets) was
  re-run** with the new default; exp1/exp2/exp_objB/exp_objC were not
  re-run for this change alone, since a sub-dB absorption difference at
  these distances cannot plausibly change any of their qualitative
  conclusions (verified by this same overlay comparison) -- re-running
  them would cost significant compute for a change admitted, in advance,
  to be undetectable in the result. If a future milestone moves the
  scenario to much longer ranges (where Milestone 3's own frequency
  sweep shows the models diverge more), this decision should be revisited.

## Phase 2, Milestone 4: discrete RIS phase shifts (hardware limitation)

- `hardware.py`: `quantize_phase` rounds to the nearest of 2^bits levels;
  `quantized_coordinate_refine` does a simple per-element coordinate
  descent over the discrete levels afterward (a few rounds, stops early
  if nothing changes) to recover some of the quantization loss. Verified:
  quantized rate <= continuous bound always, gap shrinks monotonically as
  bits increase (1->4), refinement never makes it worse.
- Measured gap to the continuous bound (K=1, Ptx=30 dBm, 200 draws):
  at M=64, 1-bit costs ~1.1 bit/s/Hz (continuous ≈11.6), 4-bit costs
  ~0.02; at M=256, the pattern is the same shape at a higher baseline.
  Refinement recovers only a small fraction of the plain-quantization
  gap (a few hundredths of a bit/s/Hz) -- most of the loss is inherent
  to the coarse angle grid itself, not to a bad choice of which grid
  point to round to.
- **Effect on the RIS-vs-relay crossover: none in practice.** Even at the
  coarsest tested setting (1-bit, M=64), AO-RIS's rate (~9.5-10.3
  bit/s/Hz depending on exact draw) still comfortably exceeds the AF
  relay's (~7.7-8.0) -- quantization narrows the margin but the margin
  established in Milestone 0 (continuous phases) was large enough that
  1-bit control doesn't erase it at these M values. See
  `plots/exp_hw_quantization_M{64,256}.png`.

## Phase 2, Milestone 5: imperfect CSI

- Standard additive-error model (assumption): Hhat = H + E_H,
  ghat = g + E_g, each error entry i.i.d. complex Gaussian with variance
  epsilon * mean(|true entry|^2) (computed separately for H and g, since
  they have different power levels) -- see `csi_error.py`.
- AO is run using ONLY the estimates (Hhat, ghat); the resulting (theta,
  W) is then evaluated against the TRUE (H, g) to get the actually
  achieved rate. Verified the expected (and required, per the Phase 2
  spec's sanity checks) behaviour: the rate AO *thinks* it achieved (on
  the noisy estimate) actually INCREASES with epsilon (the optimizer is
  fooled by channel-shaped noise into overestimating its own performance
  -- consistent with how any optimizer overfits the information it is
  given), while the TRUE achieved rate (what you actually get) decreases
  monotonically with epsilon, confirmed numerically (K=1, M=64, Ptx=30
  dBm, a representative single instance: estimate-rate rises 10.58 ->
  10.99 from epsilon=0 to 0.5, true rate falls 10.58 -> 10.10 over the
  same range).
- "Robust-ish" RZF (`csi_error.rzf_robust`, used via `ao.run_ao(...,
  mode="rzf_robust", epsilon=...)`, extending `ao.py`'s existing mode
  dispatch rather than duplicating the AO loop): inflates the usual RZF
  regulariser by the estimated channel-error power
  (`K*(sigma2 + epsilon*mean(|Hhat_eff|^2))/P`), a standard MMSE-under-
  estimation-error heuristic (the error behaves like extra effective
  noise). Verified it helps: at K=4, M=64, epsilon=0.1, true rate 4.21
  (plain RZF) vs 4.33 (robust); at epsilon=0.3, 2.61 vs 2.75.
- Rate-vs-M at a fixed epsilon=0.1 shows CSI error erodes, but does not
  eliminate, the ~M^2 scaling established in Milestone 0 -- larger M
  still helps, just less than the perfect-CSI curve predicts (the error
  variance itself doesn't shrink with M, so a larger RIS gives AO *more*
  parameters to get subtly wrong, while also providing more coherent-
  combining gain; the net effect in this setup is still a net positive
  from larger M, just at a discount).
- **K=4 is far more CSI-error-sensitive than K=1** (full 200-draw runs,
  see `plots/exp_csi_error_vs_epsilon_K{1,4}.png`): at K=1 (MRT, no
  interference to manage), rate only drops from 11.58 to 11.01 bit/s/Hz
  across epsilon=0->0.5 (~5% loss). At K=4 (RZF, actively nulling
  inter-user interference), rate collapses from 10.3 to 2.5 bit/s/Hz over
  the same range (~75% loss) and drops *below* the random-phase reference
  by epsilon~0.1 -- interference-nulling precoders need much more
  accurate CSI than a simple matched filter, since a wrong null placement
  actively creates interference rather than just missing some combining
  gain. The "robust" RZF variant tracks a few percent above plain RZF
  throughout but doesn't change this qualitative picture -- CSI accuracy
  matters more than precoder regularisation choice at high epsilon.

## Phase 2, Milestone 6: multi-user extensions and SDR

**6(a) Rate vs K.** `exp_rate_vs_K.py`, K in {1,2,4,6}, Nt=8, M=64. Sum
rate peaks around K=4 and falls off toward K=6 (close to the Nt=8
antenna-count limit): more users means more multiplexing gain but also
less room for ZF/RZF to null interference without noise enhancement --
the classic massive-MIMO trade-off, reproduced here as expected, not an
assumption we had to tune in.

**6(b) SDR for RIS phases -- a real bug found and fixed during
verification.** Applied SDR to the max-min-SINR problem with W fixed
(sum-rate is not SDR-friendly, per the spec). Derivation (see `sdr.py`'s
module docstring for the full derivation): `u_kj = conj(g_k) * (H w_j)`
(elementwise), `h_k^H w_j = theta^T u_kj` (theta NOT conjugated -- a
convention that is easy to get backwards), `|h_k^H w_j|^2 = trace(A_kj V)`
with `A_kj = u_kj u_kj^H` and `V = conj(theta) theta^T`. **Verified
numerically against a direct computation** (both the linear and the
quadratic identity, to 1e-9) -- the spec-required check.

While verifying the SDP solve itself (not just the algebra), found a
second real bug: the problem's natural numerical scale is tiny
(sigma2 ~1e-12 W, |u_kj| ~1e-7), well below the SDP solver's absolute
feasibility tolerance. An unscaled solve reported a target gamma as
"feasible" with a solution that, checked directly against the
constraints, actually *violated* one of them (by a small absolute but
large relative margin) -- the solver's "Solution may be inaccurate"
warning was exactly right, and trusting the `status` string alone would
have silently accepted a wrong answer. **Fixed by rescaling** (divide
`u_kj` by sqrt(sigma2), set sigma2=1 for the solve) before bisecting on
gamma; gamma is a dimensionless SINR ratio so this rescaling is an exact
equivalence, not an approximation. Verified three independent ways: (1)
the rescaled solution's constraints hold when checked directly against
the *original* (unscaled) u/sigma2, (2) CLARABEL and SCS agree once both
are rescaled, (3) an independently-coded manifold optimizer (see below)
converges to a min-SINR *inside* the SDR-bisected feasible region,
confirming the relaxation bound itself was correct all along.

**SDR's Gaussian-randomization recovery underperforms manifold
optimization in practice -- a genuine finding, not a bug.** Even after
fixing the scaling bug, Gaussian randomization (up to 3000 draws tried)
recovered a theta achieving a visibly lower min-SINR than the correct
relaxation bound -- e.g., one representative instance (K=4, M=16): SDR
bound ≈0.2-0.3 (confirmed correct, see above), but the best theta found
by randomization only reached ≈0.15-0.16, actually *below* the trivial
all-ones-phase baseline (≈0.179) and a 500-draw random-phase search
(≈0.11-0.14, comparable). A manifold optimizer on the same objective
(Riemannian gradient ascent on a smooth soft-min surrogate of the true,
non-relaxed SINR, reusing the tangent-projection/Armijo/retraction
structure from `ris_opt.py`) converged from 5 different random inits to
the *same* value (≈0.211) every time, landing inside the SDR-confirmed
feasible region and comfortably beating SDR's practical (post-
randomization) result -- while also running roughly 5-10x faster at
these M. This is a known, literature-documented property of SDR for
RIS phase problems (the relaxation bound can be loose relative to what
Gaussian randomization recovers in practice for small-to-moderate
problem sizes) and is exactly the motivation cited for manifold
optimization in docs/REFERENCES.md [4] -- both methods are correctly
implemented and verified here; the comparison result itself is the
finding, reported honestly rather than tuned away.

**Compute cost / scope**: SDR is not run above M=64 (per the spec --
the SDP scales badly). `exp_sdr_vs_manifold.py` therefore uses a reduced
Monte-Carlo count (20 draws, not 200) at M in {16,32,64} -- a deliberate,
documented trade-off given SDP solve cost, not an oversight.

**A third real bug: CLARABEL at M=64 cost ~3.9 GB per solve and got the
whole sweep OS-killed for running out of memory.** The full-scale
`exp_sdr_vs_manifold.py` run (20 draws x M in {16,32,64}) was killed by
the OS mid-run on low memory. Initial hypothesis (rebuilding a fresh
`cp.Problem`/`cp.Variable` on every one of the ~20 bisection calls per
draw was leaking memory) turned out to be wrong -- refactoring to
canonicalize the SDP once per draw and bisect via a `cp.Parameter` (the
diagnostically-correct fix for that *particular* inefficiency) did not
fix the OOM. A standalone single-call probe isolated the real cause:
CLARABEL (interior-point) solves ONE M=64 feasibility SDP (64x64
complex-Hermitian V, embedded as a 128x128 real PSD cone) in ~20-25s
using ~3.9 GB RSS, and that cost does not accumulate across calls --
it is a single large working set per solve, consistent with the
architectural fact that a dense interior-point method's per-iteration
barrier Hessian for an nxn PSD cone is an n^2 x n^2 object (O(n^4)
memory; n=128 here). At M=16 CLARABEL is fine (~0.1s, no memory issue);
the cost explodes between M=32 (~1s) and M=64 (~20-25s) -- consistent
with that quartic scaling. With bisection_iters=20 and N_DRAWS=20 at
M=64, CLARABEL was asking for a sustained ~3.9 GB for roughly 400 solves
-- this (combined with other processes on the machine) is what the OS
killed. **Fixed by switching the default solver to SCS** (a first-order
operator-splitting method, no dense barrier Hessian): the identical
M=64 problem solves in ~0.4s using ~0.3 GB. Verified SCS is not just
faster but *correct*: at M=32, SCS and CLARABEL agree on every
feasible/infeasible decision tested, including right at the feasibility
boundary (gamma=5.0: CLARABEL reports `infeasible_inaccurate`, SCS
reports a clean `infeasible` -- same decision, SCS's status is actually
more decisive). CLARABEL remains the solver used during the M=16-scale
algebra/scaling verification above (point-checks, not a 20x20-draw
sweep), where it is fast and was never in question.

**A fourth real bug: the SCS fix solved memory but not time -- bisection
itself gets slow right where it matters most.** Even after the SCS
switch, the full M6(b) sweep at M=64 still took hours (one single draw
observed taking 3+ hours via the run's tqdm progress log), while the
isolated single-call SCS probe above (a synthetic, randomly-generated
`u`) had suggested ~0.4s per solve -- a ~100x discrepancy that pointed
at something specific to the *real*, channel-derived problem rather
than SCS itself. Root-caused by tracing every bisection iteration of a
real (slow) draw individually: solves far from the feasibility boundary
are fast (~0.6-1s, matching the earlier estimate), but as bisection
necessarily narrows in on the true boundary, the SDP becomes
near-degenerate and SCS's convergence slows sharply and progressively
(one iteration next to the optimum took 40s with SCS's default
max_iters, worse for the next, closer iterations) -- this is *why*
bisection is especially expensive: by construction it spends most of
its iterations exactly in the hardest region of the problem. This is
not special to SCS; CLARABEL would hit the same near-degenerate
conditioning, just paying for it in memory/interior-point iterations
rather than first-order iterations. **Fixed by capping SCS at
max_iters=1000** per solve: the pathological-case solve above drops
from 40s+ (and climbing further for closer iterations) to a few
seconds, reported as `optimal_inaccurate` rather than `optimal` --
already treated as feasible by the existing status check, and harmless
to correctness because the *final* reported SINR always comes from
`actual_min_sinr` (direct evaluation against the true, non-relaxed
objective, after Gaussian randomization), never from the SDP's own
near-boundary objective value. Verified end-to-end on 4 real M=64
draws: total per-draw SDR time dropped from 802.69s (unbounded) to
27.69-101.10s (bounded), with the achieved min-SINR essentially
unchanged on the one draw checked both ways (0.1890 unbounded vs 0.1843
bounded, a difference well within Monte-Carlo noise). On these same 4
draws, manifold optimization beat SDR+randomization by a much larger
margin than the M=16 case documented above (e.g. 1.20 vs 0.18, 1.86 vs
0.064) -- consistent with, and a sharper version of, the already-
documented finding that Gaussian randomization's practical gap behind
the true relaxation bound worsens as M grows (a unit-modulus vector
becomes exponentially harder to hit by random chance in higher
dimensions), while manifold optimization's direct, non-relaxed
Riemannian ascent does not share that scaling disadvantage.

**Final full-scale result** (`exp_sdr_vs_manifold.py`, all fixes above
applied, N_DRAWS=20, M in {16,32,64}):

```
   M   SDR SINR   Manifold SINR   Random SINR  SDR time[s]  Manifold time[s]
  16     0.0863          0.1902        0.1196        3.422             0.781
  32     0.0939          0.4833        0.1572       20.753             0.598
  64     0.1127          1.0099        0.1715      135.279             0.626
```

Confirms the finding at full scale, not just on a handful of point-
checked draws: manifold beats both SDR+randomization and random-phase
at every M, with the margin widening sharply as M grows (manifold
improves ~5.3x from M=16 to M=64, while SDR+randomization barely moves,
0.086->0.113) -- exactly the "randomization gets relatively worse in
higher dimensions" mechanism above. SDR's average per-draw time also
confirms the near-boundary cost is real but now bounded rather than
catastrophic (3.4s -> 20.8s -> 135.3s, roughly an order of magnitude per
M-doubling, consistent with bisection spending more of its iterations
near an increasingly delicate boundary as M grows) -- vs. manifold's
flat ~0.6-0.8s regardless of M, a direct illustration of why manifold
optimization, not SDR, is the practical choice at scale.

**A fifth, non-algorithmic lesson: checkpoint anything that takes this
long.** This sweep was killed by the OS for system-wide low memory
*twice* after the above fixes, each time with 16-20 of 20 M=64 draws
already done -- not from this script's own footprint (a flat ~0.3 GB
throughout, confirmed via process monitoring) but from other
processes on the machine (a single Chrome process alone was observed
at 1.5 GB). `exp_sdr_vs_manifold.py` now checkpoints every single
draw's result to disk immediately (`results/exp_sdr_vs_manifold_
checkpoint_M{M}.npz`) and resumes from the first incomplete draw per M
on re-run, since each draw's randomness is keyed only by its own index
and is independent of every other draw. The run that finally completed
end-to-end took ~53 minutes of wall-clock compute (M=16: ~16s, M=32:
~7 min, M=64: ~45 min) -- a real cost worth protecting against being
silently discarded by something entirely outside this script's control.

## Phase 2, Milestone 7: DRL (Approach B), Stage A

**Environment design** (`drl_env.py`): per the spec's own framing (citing
Huang, Mo & Yuen, JSAC 2020), each episode draws one fresh channel
realisation and is exactly one step long -- this is NOT a simplification
we introduced; the spec's episode definition ("each episode draws a new
channel realisation") only makes sense this way for a problem with no
state carrying across time. `RISPhaseEnv` (Stages A/B) supersedes the
Phase 1 proof-of-concept `exp7_drl_demo.py` (kept, not modified) for M7's
purposes: that demo fixed its reference beamformer from a RANDOM phase
independent of the action, so its reward didn't track the agent's own
phase choice; here the beamformer is recomputed from the CURRENT action
every step (mirroring AO's own "fix theta, update W" half-step) and the
reward is the resulting rate normalised by the AO rate for the SAME
channel instance, as the spec requires.

**A real bug found before any training even started: wall-clock time-
boxing is not safe on a machine that can sleep.** The first full Stage A
attempt used `time.time()` to enforce the spec's "cap at about 20 minutes
per run." The machine went idle for an extended stretch overnight mid-run
(confirmed: SAC seed=0 completed a reasonable 16,059 steps in 1775.8s,
but seed=1 managed only 261 steps over 12,185.6s and seed=2 only 1,010
steps over 13,526.6s -- wall-clock time that had mostly elapsed while the
process was suspended, not computing). Comparing a seed that got ~60x
more real training than its siblings would have made any seed-to-seed
comparison meaningless. **Fixed by switching to `time.process_time()`**
(this process's own CPU time, immune to both sleep and -- to first order
-- contention from unrelated processes) for the training time budget.
Verified the fix directly: the corrected run hit exactly 1200.0s CPU on
every one of the 3 SAC runs regardless of wall-clock (one of the 3 PPO
runs, seed=1, again straddled a sleep event -- 426.8s CPU vs 2384.4s
wall-clock -- and still completed its full step budget correctly,
confirming the fix is robust to recurring sleep, not just a one-off).

**Result** (K=1, M=16, SAC and PPO via stable-baselines3, 3 seeds each,
net_arch=[256,256], eval every 5000 steps on a fixed 500-channel held-out
set):

```
  Algo   Final rate   +/- std   Infer ms/dec   Train time[s]
    AO       7.7906                   0.8846   (closed-form)
Random       3.7380
   SAC       5.3631    0.0738         0.2814          1200.0
   PPO       5.6255    0.0350         0.2214           403.8
```

Both agents clearly learn: both rise from the random-phase baseline
(3.74 bit/s/Hz) early in training and keep improving, reaching 5.36
(SAC) and 5.63 (PPO) bit/s/Hz -- a genuine ~44-50% improvement over
random phase -- while falling short of the AO closed-form oracle (7.79,
a ~28-31% gap). Neither learning curve has fully plateaued by its budget
cutoff (see `plots/exp_drl_stageA_learning_curve.png`), so more training
budget would likely narrow the gap further, though by how much is not
established here. Seed-to-seed variance is small (std 0.07/0.03 vs a
~2 bit/s/Hz spread between random and AO), i.e. this is a consistent
effect across seeds, not one lucky run.

PPO reached a slightly *better* final rate than SAC while using roughly
3x less CPU time (403.8s vs SAC's full 1200.0s budget -- PPO finished its
entire 200k-step allowance well within the time cap, SAC did not: SAC
only reached 63,821-77,962 steps in the same CPU budget, since each SAC
step costs far more compute than each PPO step, mostly from its replay-
buffer gradient updates). This runs somewhat against the common intuition
that off-policy methods like SAC are more sample-efficient than on-policy
PPO in continuous control -- plausibly because this is a stateless,
one-step-per-episode (contextual-bandit-style) problem rather than a
long-horizon sequential-decision MDP, which is exactly the regime where
off-policy replay's main advantage (reusing past transitions for
multi-step credit assignment) does not apply; PPO's simpler on-policy
updates may just suit this problem's structure better. This is reported
as an observation, not a general claim about SAC vs PPO.

DRL inference (0.22-0.28 ms/decision) is faster than AO's own closed-form
run time (0.88 ms/instance) at this M=16, K=1 scale -- a real, measured
result, though it should be read cautiously: AO's K=1 closed-form step is
already about as cheap as a single iterative algorithm step can be: this
comparison is far more likely to favour DRL as M grows or K>1 requires
AO's iterative manifold optimisation (not yet measured here; Stage B/the
hybrid idea would be the place to check that trend).

**What this Stage A result does and does not show** (per the spec's own
"fair, honest, not a guaranteed win" framing): it shows DRL CAN learn a
non-trivial RIS phase policy from scratch, purely from reward signal, in
a time-boxed budget -- a real, positive, reproducible result. It does NOT
show DRL matches or beats AO at this scale, nor does it establish whether
further training would close the gap, nor does it say anything yet about
how DRL scales with M (Stage B) or with a learned joint beamformer at
K>1 (Stage C) -- both remain open questions this stage was not designed
to answer.

## Phase 2, Milestone 7: DRL (Approach B), Stage B

K=1, M in {32, 64} (`exp_drl_stageB.py`), same settings as Stage A, per
the spec ("Stage B ... same settings"). Run only because Stage A
succeeded.

**A resume bug found while running this: a completed stage silently got
retrained from zero.** `run_stage()`'s checkpoint is deleted on success
(it's scratch, not a deliverable), so when a later M=64 attempt was
killed by the machine's low-memory condition and the whole script was
re-run, it had no way to tell "M=32 already finished" from "M=32 never
started," and began retraining all 6 of M=32's runs from scratch --
confirmed in the log, and caught before it could finish (the retrain was
itself killed partway through, by the same recurring low-memory
condition, before it could overwrite the good M=32 results on disk).
Fixed by having `run_stage()` skip immediately if its results file
already exists and no checkpoint is pending for it. This machine's
tendency for unrelated background processes (observed: Chrome processes
individually growing past 500-800 MB, fluctuating) to trigger OS-level
low-memory kills of long-running jobs was a recurring theme across M6
and M7 -- in this run specifically, the single PPO run that remained
after the fix needed **7 consecutive kill-and-resume cycles** before
finally completing (each kill lost nothing beyond that one in-progress
run, confirming the checkpoint design held up under far more repeated
interruption than any other experiment in this project saw).

**Result** -- full cross-M picture (Stage A's M=16 row repeated for
comparison):

```
   M      AO    Random      SAC (+/-std)      PPO (+/-std)   SAC/AO  PPO/AO  SAC/Rand PPO/Rand
  16   7.7906    3.7380   5.3631 (0.0738)   5.6255 (0.0350)   68.8%   72.2%    143.5%   150.5%
  32   9.7514    4.5597   4.6591 (0.0951)   6.2636 (0.1550)   47.8%   64.2%    102.2%   137.4%
  64  11.7443    5.4928   5.6529 (0.0411)   6.8260 (0.1070)   48.1%   58.2%    102.9%   124.3%
```

**DRL's relative advantage over AO shrinks as M grows, for both
algorithms** -- the honest answer Stage B was designed to get. PPO's
share of the AO rate drops steadily (72.2% -> 64.2% -> 58.2%) as M
doubles twice, though it still clears random phase by a healthy,
if also shrinking, margin (150.5% -> 137.4% -> 124.3%) and its learning
curve (`plots/exp_drl_stageB_M64_learning_curve.png`) is still rising at
the time-budget cutoff, so the M=64 gap may not be its true floor.

**SAC specifically collapses to barely-better-than-random at M=32 and
M=64** (102.2%, 102.9% of random phase -- essentially no learning beyond
the random baseline, visible directly in both learning-curve plots as an
early plateau/dip rather than continued improvement), a much sharper
degradation than PPO's. The mechanism is visible in the raw step counts:
because SAC's budget is CPU-time-capped (not step-capped) and SAC's
per-step cost rises with the environment's action/observation dimension
(M), SAC completed far fewer effective gradient updates at larger M
within the SAME 1200s CPU budget (M=16: 63.8k-78.0k steps; M=32: did not
even log a consistent count above in this writeup's table but was
visibly capped well under its M=16 step counts; M=64: ~31-35k steps,
roughly half of M=16's), compounding an already-harder learning problem
with proportionally less training. PPO, by contrast, is step-capped
(TOTAL_TIMESTEPS=200,000) rather than time-capped at every M tested here
except M=64 seed=0/1 (which hit the time cap at 194,561/200,704 steps,
i.e. barely affected) -- so PPO's training budget degrades far more
gracefully with M than SAC's does, which plausibly explains most of the
SAC-vs-PPO gap widening at larger M, on top of whatever genuine
algorithmic difference exists between them for this problem (see the
Stage A write-up's note on off-policy vs on-policy fit for this
stateless, one-step-episode formulation).

**What this does and does not show**: it shows, honestly, that this
particular time-boxed DRL recipe does not scale as gracefully as AO as M
grows -- a real, if unflattering, finding, exactly the kind the spec's
"fair, honest, not a guaranteed win" framing asks for. It does NOT show
DRL is fundamentally incapable of scaling to larger M (a larger or
differently-tuned time budget, a different SAC learning-rate/buffer
schedule, or simply more wall-clock patience could plausibly change this
picture -- none of that was explored here, consistent with the spec's
time-boxing). Stage C (joint agent, K=2 or 4, M=16) was not attempted:
given Stage B's own finding that scaling M already strains this training
budget, and that Stage C adds both a higher-dimensional action space
(the beamformer) AND a harder reward landscape (K>1 requires AO's
iterative manifold optimisation as the per-episode baseline, not the
K=1 closed form), it was judged very unlikely to produce a meaningful
result within the same time-boxed budget, and was not run -- reported
here as a scoping decision, not a result.

## Geometry

- BS at (0, 0, 10) m, RIS at (20, 10, 10) m (given in the master prompt) ->
  BS-RIS distance ≈ 22.36 m.
- UEs are placed at height `ue_height = 7 m` (not street level), uniformly
  random azimuth around the RIS, at a *3D* distance in [3, 8] m from the
  RIS. This height was chosen (rather than a literal 1.5 m "street level"
  UE) purely so that a 3-8 m 3D distance from a RIS mounted at 10 m is
  geometrically reachable without degenerate (near-zero horizontal offset)
  placements for most of that range -- e.g. representing UEs on elevated
  balconies/walkways, a scenario consistent with a building-mounted RIS.
- The blocker rectangle drawn in `plots/geometry.png` is purely
  illustrative. The direct BS-UE link is **not** geometrically ray-traced;
  it is penalised by a fixed `blockage_loss_db` instead (see below). So the
  blocker's exact position/size does not affect any numerical result.
- `experiments/exp5_distance.py` and `exp6_frequency.py` override
  `ue_height` to equal the RIS height for their distance sweeps, so that
  horizontal distance = 3D distance and the swept "distance" parameter is
  unambiguous. This is a deliberate deviation from the default placement,
  local to those two scripts.

## Blocked direct link

- `blockage_loss_db = 40 dB`, a typical building-wall / foliage NLoS
  penetration loss at mmWave/THz, added on top of ordinary free-space path
  loss for the BS->UE link. This keeps the "no-RIS" curve finite and
  plottable instead of literally zero, per the master prompt.

## Antenna gains (calibrated)

- `bs_gain_dbi = 45`, `ue_gain_dbi = 25`. Calibrated by running AO for
  K=1, M=64, Ptx=30 dBm at 140 GHz over 30 channel draws and sweeping
  candidate gain pairs until the mean post-AO SNR landed in the requested
  15-25 dB band. Measured result at these values: **mean SNR ≈ 20.2 dB**
  (std ≈ 6.9 dB across draws, reflecting the random Saleh-Valenzuela path
  gains and NLoS angles -- the calibration targets the *mean*, not every
  draw).
- These are deliberately high, directional/phased-array-grade gains (a
  combined 70 dB). That is a direct, honest consequence of the corrected
  channel model below: the cascaded (two-hop) RIS link's path loss at
  140 GHz over these distances is severe even after the RIS's M^2 coherent
  combining gain, so hitting a workable 15-25 dB SNR at the assignment's
  default M=64 needs either very large M or high-gain antennas. We kept
  M=64 as specified and raised the gains, per the master prompt's
  instruction to calibrate `bs_gain_dbi`/`ue_gain_dbi` for this purpose.
- RIS elements are assumed unity gain (0 dBi); only the BS and UE ends
  carry an explicit antenna gain. This is a standard simplification in the
  RIS literature (the RIS's "gain" comes from the M-element coherent
  combining, not element directivity).

## Molecular absorption (THz)

- `k(28 GHz) = 0.1 dB/km`, `k(140 GHz) = 2 dB/km`, `k(300 GHz) = 8 dB/km`.
  Explicitly labelled as **approximate placeholders**, not derived from a
  physical absorption-line database. A proper model would use ITU-R
  P.676 / HITRAN line data (see docs/REFERENCES.md, [8]); left as future
  work. At the scenario's short (tens-of-metres) distances, this
  absorption factor is a small effect (well under 1 dB) compared to the
  free-space path loss difference between bands (which scales with
  wavelength^2, i.e. ~14 dB between 28 and 140 GHz) -- see
  `exp6_absorption_vs_distance.png` for the absorption factor in isolation.

## Power model (for energy efficiency)

- `P_total = Ptx/eta + P_BS_static + M*P_RIS + K*P_UE`, with
  `eta = 0.4`, `P_BS_static = 1 W`, `P_RIS = 5 mW/element`, `P_UE = 0.1 W`.
  These are illustrative constants in the range typically cited for RIS
  energy-efficiency studies (see docs/REFERENCES.md, [5]), not
  measurements of real hardware.

## Saleh-Valenzuela path count vs K

- Default `sv_num_paths = 3` (1 LoS + 2 NLoS) for K=1, as specified.
- **For K>1, `sv_num_paths` is automatically bumped to `K+2`** in
  `Config.__post_init__`. Reason: H (BS->RIS) is built as a sum of
  `sv_num_paths` rank-1 terms, so `rank(H) <= sv_num_paths`; every user's
  effective channel row `g_k^H Phi H` lies in the (<= sv_num_paths)-
  dimensional row space of H. With the K=1 default of 3 paths, a K=4
  scenario would hand ZF/RZF a provably rank-deficient effective channel
  (rank <= 3 < 4) -- confirmed empirically: `H_eff H_eff^H` had condition
  number ~1e17 (effectively exactly singular) with 3 paths, vs ~1e2-1e3
  with 6. This caused outright `numpy.linalg.LinAlgError: Singular matrix`
  crashes in early experiment runs for K=4 before the fix. This is a
  standard sparse-mmWave-MIMO consideration, not a special-cased hack.

## AF relay benchmark

- The relay has a single antenna, located exactly at the RIS position.
  Both hops (BS->relay, relay->UE) use a single-path Rayleigh channel with
  the same free-space-path-loss + molecular-absorption model as the rest
  of the link, and **no** blockage penalty (the relay is not behind the
  blocker -- it sits where the RIS is, in the clear).
- Hop 1 (BS->relay) transmits at the swept Ptx (same power as the other
  curves on that axis, for a fair comparison); hop 2 (relay->UE) always
  transmits at the relay's own fixed power budget `Pmax`, per the
  assignment ("relay transmit power = Pmax").
- gamma = gamma1*gamma2/(gamma1+gamma2+1) is the standard AF end-to-end
  SNR approximation; rate = 0.5*log2(1+gamma) (half-duplex pre-log factor).

## A channel-model bug we caught via the required M^2 sanity check (and fixed)

Section 8 of the master prompt requires checking "with M->large the RIS
gain should grow roughly as M^2; if a curve violates these, treat it as a
bug." That check caught a real bug during development, worth recording:

- **The bug**: `sv_channel_H` and `sv_channel_g` originally included an
  extra `sqrt(Nt*M/L)` / `sqrt(M/L)` size-dependent prefactor, copied from
  the standard point-to-point mmWave MIMO normalisation of El Ayach et al.
  [7] (`H = sqrt(Nt*Nr/L) * sum alpha a_r a_t^H`). That convention is
  correct for a *single* link about to be collapsed by an Nr-branch
  receive combiner -- but M here is a **cascaded, non-collapsing**
  dimension (the same RIS elements appear in both H and g via
  `diag(theta)`), so applying it to *both* legs double-counted the RIS's
  array gain. The correct M^2 SNR scaling (from AO's coherent phase
  alignment: M unit-modulus terms summed then squared for power) already
  emerges from the steering-vector algebra alone; the extra prefactors
  inflated it further, to a measured **M^4**-ish scaling (summary metric
  "AO-RIS array-gain scaling M=64->256" read ≈253x against a ~16x
  theoretical expectation for M^2 over a 4x change in M -- a clear, large
  violation of the required sanity check).
- **The fix**: removed both prefactors; H and g are now scaled by the
  path-loss/absorption/antenna-gain amplitude only (see the channel.py
  module docstring for the corrected formula and the full derivation of
  why this is the physically correct convention for a cascaded RIS link).
  Re-verified with `test_rate_scales_roughly_as_M_squared` (asserts the
  SNR ratio for a 4x change in M lands in [8x, 32x], i.e. around the
  theoretical 16x) and by re-deriving the scaling by hand for a toy L=1,
  Nt=1 case (included in the channel.py docstring reasoning).
- **Consequence**: this was caught *before* any numbers were reported as
  final, but *after* a first full round of Monte-Carlo experiments had
  already been run and antenna gains calibrated against the buggy model.
  Both were redone from scratch against the corrected model (recalibrated
  gains above; all `exp1`-`exp7` re-run). This is flagged here rather than
  silently fixed, per the "never hide a failed run" principle -- the
  numbers reported everywhere else in this project are from the corrected
  model.

## A result worth flagging explicitly (not a bug)

With the corrected channel model, the RIS's handicap at the assignment's
default M=64 is more pronounced than it first appeared: **No-RIS clearly
beats AO-RIS across the entire Ptx=0-30 dBm range for K=1** (not just at
high Ptx), and random-phase RIS is far below both. AO-RIS only starts to
win against No-RIS, for K=1, once M grows into the hundreds (empirically,
win rate over 20 draws: M=64 -> 1/20, M=256 -> 2/20, M=1024 -> 12/20).
The exact crossover point (200-draw mean, not just a per-draw win rate)
was pinned down directly with a bonus experiment,
`experiments/exp2b_crossover.py` (K=1 only, M up to 1024, not required by
the assignment -- see `plots/exp2b_crossover_K1.png`): AO-RIS overtakes
no-RIS at **M=1024** (AO-RIS=14.97 vs no-RIS=14.36 bit/s/Hz; still behind
at M=512: 12.94 vs 14.36). This confirms directly (not just by
extrapolating the M=16..256 trend) that AO-RIS does eventually win in
this geometry, given enough elements.
This is the well-documented "multiplicative path loss" property of
RIS-aided links (a cascaded two-hop link's path loss is the *product*,
not the sum, of the two legs' losses, so even M^2 coherent combining can
leave a net deficit against a merely-blocked-not-obstructed single-hop
link) -- it is why RIS papers emphasise that large M and/or careful
placement are needed, and it is exactly the limitation flagged for the
simplified cascaded-channel model in docs/REFERENCES.md, [3] and [8].
The K=4 case shows the same pattern and, like K=1, does **not** cross
within the assignment's own M=16..256 sweep range either (at M=256,
AO-RZF ≈17.6 bit/s/Hz vs no-RIS ≈58.0 bit/s/Hz -- no-RIS benefits from
ZF across Nt=8 antennas directly serving 4 users over a single, much
shorter-effective-path-loss hop). Both the exp1 and exp2 plots show this
honestly: AO-RIS/AO-RZF climbing steeply with M (confirming the ~M^2
mechanism works as expected) against a flat, M-independent no-RIS line
that it has not yet caught by M=256 in either K setting, in this
geometry.

## exp7 (optional DRL demo) -- actual result

Trained SAC for 40,000 steps, against the corrected channel model and
recalibrated gains, running alone (no concurrent heavy processes).
Measured: AO (closed-form) oracle mean rate ≈3.092 bit/s/Hz, random-phase
baseline ≈0.694 bit/s/Hz, SAC's final smoothed rate ≈0.632 bit/s/Hz --
SAC did not even match the random-phase baseline, let alone approach the
AO oracle (**≈80% gap to the AO oracle**). `plots/exp7_drl_learning_curve.png`
shows the smoothed curve oscillating around the random-phase line for the
full 40k steps with no visible upward trend. This is an honestly-reported
negative result, not a bug -- and it reproduces across both the pre-fix
and post-fix channel models (absolute numbers changed, the qualitative
"SAC does not learn" finding did not), which is further evidence it's a
genuine property of this RL setup rather than an artifact of the channel
bug. Root cause (not pursued further, per "do not tune further"): the
action space is the full M=16-dimensional phase vector, each episode is a
single step (no temporal credit assignment to lean on), and the reward
landscape (coherent phase alignment) is exactly the kind of narrow,
non-convex optimum that off-the-shelf continuous-control RL struggles
with when trained from scratch without reward shaping or curriculum --
consistent with the master prompt's framing of this milestone as a proof
of concept, not a tuned baseline meant to compete with AO.
- Training wall-clock time: **639.6 s (≈10.7 min)**, inside the master
  prompt's 15-minute budget -- this run was deliberately launched alone
  (nothing else heavy running concurrently). An earlier attempt that ran
  concurrently with the heavy K=4 AO-manifold sweeps in exp1/exp2 took
  2363 s (≈39 min) purely from CPU contention, which is why this one was
  re-run in isolation -- a legitimate redo to get an honest read on the
  actual 15-minute budget, not "tuning the result."

## exp4 (EE vs SE)

- Uses the K=1 AO-MRT scheme as the representative single-user operating
  point, swept over Ptx and M. A multi-user EE/SE curve would also need to
  fix K in the power model's `K*P_UE` term and is a straightforward
  extension, left out to keep the figure to the one sweep the assignment
  asks for.

## Phase 2, Milestone 1: Objective B (min-power)

- K=1's "RIS step = same phase alignment that maximises ||h_eff||" is
  justified by an exact algebraic argument (max-rate for fixed P and
  min-power for fixed SINR share the same optimal theta/W-direction for
  K=1 -- see `objective_b.py`'s module docstring), not just an assumption;
  cross-validated against a direct Riemannian maximisation of ||h_eff||^2
  and found to agree within 0.3%.
- K>1 ZF's closed-form power formula
  (`P = gamma*sigma2*trace((H_eff H_eff^H)^-1)`) and precoder were both
  verified three ways: achieved SINR equals gamma exactly, ||W||_F^2
  equals the closed-form power exactly, and the closed-form power matches
  a from-scratch direct computation of the same trace (the spec-required
  test, `test_objective_b_zf_closed_form_matches_direct_computation`).
- AF relay's required-power formula (inverting
  `gamma = gamma1*gamma2/(gamma1+gamma2+1)` for gamma1, gamma2 fixed by
  the relay's own Pmax) returns `+inf` (reported as outage, not an error
  or a silently wrong number) when gamma2 <= the target -- the second hop
  alone cannot reach that SINR at *any* first-hop power. Verified: a
  forward rate computation at the inverted power reproduces the target
  SINR exactly for every tested value.
- Required power is confirmed non-increasing in M (sanity check, exp
  `exp_objB_power_vs_M.py`): a 4x increase in M roughly quarters the
  required power, consistent with the ~M^2 SNR-scaling law established in
  Milestone 0 (power needed scales as ~1/M^2 for a fixed SINR target).
- Outage probability vs target SINR (`exp_objB_outage.py`, K=1, M=64,
  Pmax=30 dBm default) shows the expected ordering: AF relay has by far
  the lowest outage (its second hop, at the relay's own full Pmax, does
  most of the work), then AO-RIS, then no-RIS and random-phase RIS
  (worst, since they must supply the whole link from a single BS-side
  power budget with a weaker or unoptimised channel).

## Phase 2, Milestone 2: Objective C (energy efficiency)

- EE-optimal Ptx sits well below the 30 dBm power budget at every tested
  M (20-22 dBm, vs the sum-rate-optimal 30 dBm) -- expected, since
  `P_total` includes `Ptx/eta` (linear in Ptx) while rate grows only
  logarithmically, so EE trades off rate against power cost and peaks at
  a genuine interior point (confirmed visually: `exp_objC_ee_se_tradeoff.png`
  shows a concave EE-vs-SE curve per M with a clear interior maximum, not
  a boundary solution).
- **Maximum achievable EE vs M has its own interior peak, at M=128**
  (not M=256): more RIS elements raise the achievable rate but also cost
  more static hardware power (`M*P_RIS` in the power model), so EE
  eventually turns over once the marginal rate gain from more elements
  stops paying for their own power draw. This is a genuine finding of the
  power-model assumptions in docs/REFERENCES.md [5], not built in by
  construction -- a different `P_RIS` would shift (or remove) the peak,
  which is exactly what the P_RIS sensitivity sweep demonstrates.
- Golden-section refinement around the coarse-grid argmax was used after
  the 41-point grid search (the spec allows a plain grid search; the
  refinement is a small, optional accuracy improvement, not a different
  method -- see `_golden_section_refine` in `exp_objC_ee_optimal.py`).
