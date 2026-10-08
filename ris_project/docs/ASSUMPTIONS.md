# Assumptions

Every numeric choice here that is not a standard physical constant is an
assumption, made explicit so a reader can tell "assumption" from "result".
Corresponding code lives mostly in `config.py` and `channel.py`.

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
