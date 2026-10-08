# Final summary

Per the master prompt's closing instruction ("print a short summary for
the student: what was built, headline numbers, anything that failed or
was skipped").

## What was built

A complete RIS-assisted mmWave/THz downlink simulation solved with
Alternating Optimization (AO): sparse Saleh-Valenzuela + molecular-
absorption channel model, AO alternating MRT/ZF/RZF active beamforming
with closed-form (K=1) / Riemannian-manifold (K>1) RIS phase optimization,
three benchmarks (no-RIS, random-phase RIS, AF relay), all 6 required
sensitivity experiments plus the optional DRL demo (exp7) and one bonus
experiment (exp2b), full documentation (README, ASSUMPTIONS, EXPLAIN,
REFERENCES), and a 16/16-passing test suite. See README.md for the full
write-up and `docs/` for supporting detail.

## Headline numbers

(K=1, M=64, Ptx=30 dBm, 140 GHz unless noted; full table in
`results/summary.csv`)

- AO-RIS beats random-phase RIS by **4.4x** (K=1) / **2.1x** (K=4, RZF) —
  the core required comparison, holds reliably everywhere tested.
- RIS array gain scales as **M^2**: measured 15.8x vs a 16x theoretical
  expectation for a 4x change in M (64->256).
- AO-RIS does **not** beat the blocked direct link (no-RIS) at the
  assignment's default M=64 (6.9 vs 14.4 bit/s/Hz) -- but does at
  **M=1024** (14.97 vs 14.36), confirmed directly with a bonus experiment,
  not just extrapolated.
- AO converges within ~3 iterations (K=1, closed form) to ~10-12
  iterations (K=4, manifold optimization), monotonically, as required.
- Optional DRL demo (SAC, 40k steps, 639.6s): final rate 0.63 bit/s/Hz,
  below the random-phase baseline (0.69) and ~80% behind the AO oracle
  (3.09) -- an honestly-reported negative result.

## What failed or was skipped

**Nothing failed outright.** One real bug was caught and fixed during
development: the channel model was double-counting the RIS's array gain
(SNR scaling as M^4 instead of the theoretically correct M^2), caught via
the sanity check the master prompt itself requires, fixed, and
numerically re-verified (15.8x vs 16x). All experiment data was re-run
against the corrected model. See `docs/ASSUMPTIONS.md` for the full
writeup.

The optional DRL demo (exp7) underperforms by design-of-experiment --
reported plainly, not tuned further, per the master prompt's own framing
of that milestone as a proof of concept.

Everything explicitly out of scope per the master prompt (SDR with
Gaussian randomization, Objective B min-power design, realistic channel
estimation, hardware impairments, multi-user DRL, DRL tuning to beat AO,
a full written paper) was deliberately left out, not abandoned -- see
README.md's "Future work" section.

## What's not yet done

Two small, no-new-code items, both low-risk to close at any time:
1. A from-scratch clean-environment re-verification (delete `.venv`,
   reinstall from `requirements.txt`, re-run `pytest -q` and every
   experiment). Reproducibility has been substantively proven by
   repeatedly re-running everything fresh in the existing venv, but the
   literal teardown-and-rebuild hasn't been performed.
2. This file itself was the last missing explicit deliverable (the
   master prompt's closing "print a summary" instruction) -- now done.
