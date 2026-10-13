# Phase 2 final summary

Per `RIS_Phase2_Prompt.md` section 11, item 5. Updated at the point
where Milestones 0-8 are all complete and committed: Section 10
(tests/quality gates, `run_all.sh`), the rest of Section 11
(summary.csv refresh, README/REFERENCES/EXPLAIN updates), and Milestone
8 (`paper/main.tex`, `docs/FINAL_PRESENTATION_OUTLINE.md`,
`docs/VIVA_QA.md`) are all done. The one open item is compiling
`paper/main.tex` to PDF, which the spec explicitly allows deferring when
no LaTeX distribution is installed (the case here) -- the source has
been checked structurally instead (balanced braces/environments, every
citation resolves against `refs.bib`, every figure reference matches a
file actually copied by `paper/copy_figures.py`).

## What is done

- **Milestone 0**: post-mid-eval audit. Found and fixed a significant bug
  in the MRT precoder (missing conjugate) affecting every K=1 result in
  the project, and raised `blockage_loss_db` 40->80 dB so the assignment's
  "completely blocked" direct link is actually modelled as such.
- **Milestone 1**: Objective B (min transmit power s.t. SINR >= target),
  closed-form K=1 and manifold-descent K>1.
- **Milestone 2**: Objective C (energy efficiency), grid search +
  golden-section refinement over Ptx, per M.
- **Milestone 3**: real ITU-R P.676 molecular absorption via the `itur`
  package, replacing Phase 1's placeholder dB/km table (kept as an
  explicit comparison option).
- **Milestone 4**: discrete RIS phase shifts (1-4 bit) with
  coordinate-descent quantization-aware refinement.
- **Milestone 5**: imperfect CSI (additive Gaussian error model) and a
  robust-RZF precoder variant.
- **Milestone 6**: multi-user scaling (K in {1,2,4,6}) and SDR for the
  max-min-SINR RIS-phase problem, including two real numerical bugs found
  and fixed (a solver-scaling bug, then CLARABEL's interior-point
  memory/time cost at M=64, fixed by switching to SCS with a bounded
  iteration count).
- **Milestone 7**: a time-boxed DRL study (SAC, PPO; 3 seeds each) for
  RIS phase control, Stages A (K=1, M=16) and B (K=1, M=32/64). Stage C
  (joint K>1 agent) was scoped out, see below.
- **Section 10/11 housekeeping**: `run_all.sh` (full and `--fast`
  smoke-test modes), a refreshed `results/summary.csv` covering every
  milestone, updated `README.md`/`docs/REFERENCES.md`/`docs/EXPLAIN.md`,
  this document, and the `phase2-final` git tag. A full end-to-end
  `--fast` smoke test of all 22 experiment scripts plus the test suite
  passed cleanly (two real bugs were found and fixed getting it to that
  state -- see "What failed" below). Two further test-coverage gaps
  (a missing CSI-error sanity-check test and a missing EE-formula test,
  both explicitly named in the spec's Section 10) were found and closed
  after the initial pass, bringing the suite to 39 tests.
- **Milestone 8**: `paper/main.tex` (IEEEtran draft covering System
  Model, Objectives A/B/C, the AO algorithm with a numbered algorithm
  box and convergence argument, the SDR derivation, the DRL design, a
  full parameter table, and a Results section with every included
  figure discussed), `paper/refs.bib` (14 verified references -- the
  original 9 plus 5 newly verified: SAC, PPO, the SDR survey paper,
  Stable-Baselines3, and ITU-R P.676 itself), `paper/copy_figures.py`,
  `docs/FINAL_PRESENTATION_OUTLINE.md` (15 slides + 4 backup slides),
  and `docs/VIVA_QA.md` (30 Q&A).

**Not done**: compiling `paper/main.tex` to PDF (no LaTeX distribution
installed; the spec explicitly allows deferring this).

## Ten headline numbers

| # | Metric | Value |
|---|---|---|
| 1 | AO-RIS gain over random-phase, K=1, Ptx=30dBm | **2.16x** |
| 2 | AO-RZF gain over random-phase, K=4, Ptx=30dBm | **2.11x** |
| 3 | AO-RIS array-gain scaling, M=64->256 (expect ~(256/64)^2=16x) | **15.5x** |
| 4 | Min Tx power, AO-RIS vs No-RIS @ target SINR=15dB, K=1 | **0.033 W vs 2.11 W (~63x less)** |
| 5 | EE-optimal Ptx @ M=64 (vs the rate-maximizing 30 dBm) | **20.6 dBm** |
| 6 | Blockage loss at which AO-RIS overtakes the un-optimized direct link, M=64 | **60 dB** |
| 7 | 4-bit quantized rate vs continuous phase, M=64 | **11.57 vs 11.58 bit/s/Hz (99.9%)** |
| 8 | AO-RZF rate, perfect CSI vs eps=0.5 CSI error, K=4 | **10.29 -> 2.49 bit/s/Hz (-76%)** |
| 9 | Max-min SINR, manifold optimization vs SDR+randomization, M=64, K=4 | **1.01 vs 0.11 (~9x)** |
| 10 | PPO (DRL) final rate vs the AO oracle, K=1, M=16 | **5.63 vs 7.79 bit/s/Hz (72%)** |

Full detail and every other computed metric: `results/summary.csv`.

## What failed or was skipped

- **DRL does not match or beat AO** at any M tested (72% of AO at M=16,
  falling to 58% at M=64 for the better of the two algorithms, PPO) --
  reported as a genuine, honest finding per the spec's own "fair, not a
  guaranteed win" framing for Milestone 7, not a bug.
- **Milestone 7 Stage C** (joint RIS-phase + beamformer DRL agent, K>1)
  was not attempted: Stage B's own finding that DRL already strains to
  scale with M under the time-boxed training budget made Stage C's
  strictly harder problem (higher-dimensional action space, iterative
  AO baseline instead of a closed form) judged very unlikely to produce
  a meaningful result in the same budget. A scoping decision, not a
  result.
- **SDR is not run above M=64**: the SDP does not scale well at this
  problem size even after the Milestone 6 fixes (a single M=64 solve
  still costs noticeably more than M=16/32).
- **Two real bugs were found (and fixed) while building this pass's
  `run_all.sh --fast` smoke test**, not before: (1) a resume-logic bug
  where re-running a multi-stage DRL script after a later stage was
  killed would silently retrain an already-completed earlier stage from
  zero; (2) the `--fast` smoke test for the DRL scripts was silently
  skipping itself, because the "skip if already done" check from fix (1)
  also fired against the real full-scale results sitting on disk. Both
  are now fixed and the full smoke test passes cleanly; see
  `docs/ASSUMPTIONS.md` for the complete account of every bug found this
  phase, including several unrelated to this final pass.
- **One factual error was caught and corrected** while building the
  refreshed summary: `docs/ASSUMPTIONS.md` previously claimed Milestone
  6(a)'s sum rate "peaks around K=4" -- checked directly against the
  saved data and found to be wrong (sum rate decreases monotonically
  with K); corrected with the right underlying mechanism.
- **`paper/main.tex` has not been compiled to PDF**: no LaTeX
  distribution is installed on the development machine. The spec
  explicitly allows this ("write the .tex files even if LaTeX is not
  installed; compile to PDF only if `pdflatex` is available"); the
  source was instead checked structurally (balanced braces and
  `\begin`/`\end` environments, every `\cite` key resolves in
  `refs.bib` with none left unused, every `\includegraphics` path
  matches a figure `copy_figures.py` actually copies) but has not been
  run through an actual LaTeX engine, so page layout, float placement,
  and any LaTeX-syntax errors outside what static checking can catch
  remain unverified.

## Three weakest points a reviewer would attack

1. **The antenna gains (45/25 dBi default preset) are calibrated, not
   independently justified.** They were deliberately chosen so AO-RIS's
   SNR lands in a target 15-25 dB band at the assignment's default
   M=64 -- a reasonable way to make a toy scenario well-posed, but a
   reviewer could fairly say the headline "RIS helps" numbers are shaped
   by a tuned parameter rather than a hardware spec. Mitigation already
   in place: `Config.realistic()` (25/10 dBi, a lower, more plausible
   mmWave sector-antenna preset) is kept as an explicit, re-run
   sensitivity point precisely so this isn't hidden -- see
   `exp1_rate_vs_power_K1_realistic.npz` and `docs/ASSUMPTIONS.md`.
2. **The DRL study's negative-ish result could be read as "didn't try
   hard enough" rather than "DRL genuinely struggles here.** Training
   is explicitly time-boxed (20 min CPU-time cap, 200k step cap, 3
   seeds), per the spec's own instruction -- but that means the gap to
   AO reported here is an upper bound on what a longer or better-tuned
   training run might close, not a ceiling on DRL's true potential for
   this problem. The honest framing (stated plainly in
   `docs/ASSUMPTIONS.md` and the README) is that this result shows DRL
   *can* learn something non-trivial in this budget and that the gap to
   AO *widens* with M within this budget -- not that DRL cannot ever
   compete with AO on this problem.
3. **The CSI-error and blockage models are both abstractions, not
   derived from a specific physical mechanism.** CSI error is additive
   Gaussian noise on the channel estimate, parameterized by a dimension-
   less epsilon -- not tied to a specific pilot/estimation algorithm, so
   there's no direct mapping from epsilon to e.g. pilot SNR or training
   overhead. Similarly, `blockage_loss_db` is a single fixed dB penalty,
   not geometric ray-tracing against the obstacle actually drawn in
   `plots/geometry.png`. Both are explicitly documented as
   simplifications (not hidden) in `docs/ASSUMPTIONS.md` and the
   README's Limitations section, but a reviewer could reasonably ask for
   a more physically-grounded version of either.
