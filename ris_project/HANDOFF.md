# Handoff / context for the next agent (human or LLM)

This file exists so any agent (Claude, another LLM, or a human) can pick up
this project cold and continue it correctly. Read this fully before
touching code. It is a snapshot as of the point it was written — update it
(or delete it once the project is fully finished and handed to the user)
rather than letting it silently go stale.

## What this project is

A Python simulation of RIS (Reconfigurable Intelligent Surface)-assisted
mmWave/THz downlink communication with a blocked direct link, solved with
Alternating Optimization (AO). Built from two spec files in the parent
directory (`../RIS_Master_Prompt (1).md`, the implementation plan, and
`../RIS_Communication_Project_Workflow.md`, the original student
assignment — the master prompt is a deliberate *subset* of the assignment;
read both, master prompt section 0 states the scope precisely). **Read the
master prompt in full before changing anything** — it is the source of
truth for file structure, algorithms, and deliverables.

Do not re-derive things you can find already written down: `README.md`
(project overview, how to run, results summary), `docs/ASSUMPTIONS.md`
(every non-obvious numeric/modelling choice and why), `docs/EXPLAIN.md`
(pipeline walkthrough + viva Q&A), `docs/REFERENCES.md` (bibliography).
This file is specifically about *process state* — what's done, what's
running, what's left — not the technical content, which lives in those
other docs.

## Current status (update this section as you go)

**All of Milestones 1-4 are implemented and all 16 tests in
`tests/test_core.py` pass.** Milestone 5 (exp7, optional DRL demo) is
implemented and has been run once.

**A significant channel-model bug was found and fixed partway through
Milestone 4** (see "The bug" below). All code is now correct and
consistent with the fix. However, as of this snapshot, **the experiment
*data* (`results/*.npz`, `plots/*.png`) is a mix of pre-fix and post-fix
runs** — re-running everything from a clean state was in progress when
this file was written. Check each experiment's completion status below
before trusting its plot/npz.

| Experiment | Code status | Data status (as of this snapshot) |
|---|---|---|
| `plot_geometry.py` | done | valid (geometry doesn't depend on the channel-amplitude bug) |
| `exp1_rate_vs_power` | done | **re-run in progress** — K=1 panel done post-fix; K=4 panel was at ~5/7 Ptx points when this file was written. Check `experiments_exp1.log` / `results/exp1_rate_vs_power_K4.npz` mtime vs the fix time before trusting it. |
| `exp2_rate_vs_M` | done | **STALE — pre-fix data.** Must be re-run after exp1 finishes (both are heavy K=4 manifold-optimization runs; avoid running them concurrently, see "Performance notes"). |
| `exp3_convergence` | done | **valid, post-fix** (re-run already completed after the fix) |
| `exp4_ee_vs_se` | done | **valid, post-fix** (re-run already completed after the fix) |
| `exp5_distance` | done | **valid, post-fix** (re-run already completed after the fix) |
| `exp6_frequency` | done | **valid, post-fix** (re-run already completed after the fix) |
| `exp7_drl_demo` (optional) | done | **STALE — pre-fix data AND calibrated against old gains.** Should be re-run alone (not concurrently with exp1/exp2) so it can respect its 15-minute time budget — see "Performance notes", it blew the budget (39 min) last time purely due to CPU contention. |
| `make_summary.py` | done | **must be re-run last**, after every exp1-exp6 npz is fresh, to regenerate `results/summary.csv` |

### Immediate next steps, in order

1. Wait for / check on the exp1 K=4 background run (if still running — check
   `experiments_exp1.log` in this directory, or just re-run
   `python -m experiments.exp1_rate_vs_power` if unsure of its state; it's
   idempotent, just slow — ~10-12 min total on this machine).
2. Re-run `python -m experiments.exp2_rate_vs_M` (the other heavy one,
   ~15-20 min — its K=4 panel is the slow part).
3. Re-run `python -m experiments.exp7_drl_demo` **alone** (nothing else
   heavy running concurrently) so it can finish in its 15-minute budget.
4. Re-run `python -m experiments.make_summary` to regenerate
   `results/summary.csv` from the fresh data.
5. Re-run `pytest -q` once more as a final check (should already pass, but
   confirm after the data refresh).
6. Update `README.md`'s "Results" section and `docs/ASSUMPTIONS.md` with
   the actual final numbers (the current numbers in those files reference
   some pre-fix results — e.g. the exp2 M-scaling example in
   ASSUMPTIONS.md's "A result worth flagging" section was computed
   pre-fix for K=4 and should be re-verified against the fresh exp2 data;
   the K=1 finding in the same section IS already post-fix and correct).
7. Visually sanity-check every plot in `plots/` (open and look — matplotlib
   won't error on a nonsensical curve, only your eyes will catch it).
8. Only then consider the deliverable finished. At that point, this
   HANDOFF.md file has served its purpose — either delete it or trim it to
   a short "project complete, see README" note so it doesn't confuse
   someone into thinking the project is still in progress.

## The bug (read this before touching channel.py)

`channel.py`'s `sv_channel_H` and `sv_channel_g` originally multiplied the
channel by an extra `sqrt(Nt*M/L)` / `sqrt(M/L)` size-dependent prefactor,
copied from the standard point-to-point mmWave MIMO normalisation
convention (El Ayach et al., used when a channel is about to be collapsed
by an *Nr*-branch receive combiner). That's wrong here: M is a **cascaded,
non-collapsing** dimension in a RIS link (the same M RIS elements appear in
both the BS→RIS leg and the RIS→UE leg via `diag(theta)`), so applying that
prefactor to *both* legs double-counted the RIS's array gain. Measured
effect: AO-RIS SNR scaled as roughly **M^4** instead of the textbook
**M^2**. This was caught by exactly the sanity check the master prompt
requires in section 8 ("with M→large the RIS gain should grow roughly as
M²; if a curve violates these, treat it as a bug") — caught via the
`make_summary.py` output, specifically a metric computing the SNR ratio
between M=64 and M=256, which read ~253x against a ~16x theoretical
expectation.

**The fix**: removed both prefactors. H and g are now scaled by the
path-loss/absorption/antenna-gain amplitude only — see `channel.py`'s
module docstring for the full corrected formula and the derivation of why
this is correct for a cascaded (not point-to-point) link. A regression
test (`test_rate_scales_roughly_as_M_squared` in `tests/test_core.py`)
locks in the ~M² scaling going forward.

**Downstream consequences of the fix** (already applied in code, just
listed here so you understand why numbers changed):
- Antenna gains (`bs_gain_dbi`, `ue_gain_dbi` in `config.py`) had to be
  recalibrated — the channel is now much weaker (correctly so), so hitting
  the master prompt's target of "~15-25 dB mean AO SNR at K=1, M=64,
  Ptx=30dBm, 140GHz" needs much higher gains than before (now 45/25 dBi,
  was 25/10 dBi — see `docs/ASSUMPTIONS.md` "Antenna gains" section for
  the calibration run).
- A self-added test (`test_ao_beats_no_ris_most_of_the_time`, which
  asserted AO-RIS beats the blocked direct link 90% of the time at the
  default M=64) started failing under the corrected model — not a test
  bug, a real finding: at M=64 in this geometry, the un-optimized blocked
  direct link now genuinely beats AO-RIS most of the time (the RIS's
  cascaded/multiplicative path loss is a real, well-documented handicap
  that the old bug was papering over). Replaced with
  `test_ao_vs_no_ris_crosses_over_with_M`, which asserts the *win rate*
  increases with M instead of asserting an unconditional win — this is
  both true and is the actual point of exp2's M-sweep. Do not try to
  "fix" this by reverting the channel-model correction or by further
  tuning gains/geometry to force AO-RIS to win at M=64 — that would be
  re-introducing the bug by another path. The honest result (RIS needs
  large M here) is reported in `docs/ASSUMPTIONS.md` and should be
  reflected in the README results section.

## Performance notes (read before re-running experiments)

- K=1 scenarios are cheap (closed-form phase update, no PyTorch): a full
  200-draw, 7-point sweep takes ~10-15 seconds.
- K=4 scenarios use the Riemannian-manifold phase optimizer (PyTorch
  autograd) and are the bottleneck: roughly 80-90s **per sweep point** when
  running alone on this 16-core machine (`torch.set_num_threads(1)` is set
  in `ris_opt.py` deliberately — these are tiny tensors, so threading
  overhead dominated single-call latency with the default thread count;
  single-threaded is both faster per-call AND avoids oversubscription when
  multiple experiment scripts run concurrently).
- **Running two K=4-heavy scripts concurrently roughly triples their
  individual runtime** (observed: a K=4 sweep point went from ~235s to
  ~700s when exp1 and exp7 were both running). If you background multiple
  experiment scripts, know what you're trading off. exp1 and exp2 are the
  two heavy ones (each has a K=4 panel); exp3-exp6 are cheap (K=1 only, or
  trivial sweeps); exp7 is heavy in a different way (wall-clock from SAC
  training, time-boxed to 15 min **only if it's not fighting other
  processes for CPU** — it is not itself K=4/manifold-heavy, M=16 and K=1
  there, it's just a lot of sequential environment steps).
- Background a heavy script with the Bash tool's `run_in_background: true`,
  then either poll its output file or (better) use the `Monitor` tool with
  an until-loop watching the log file for a known completion string (e.g.
  `"exp1 done."`) — this avoids wasting turns on repeated manual polling.
  Each `expN_*.py` script tees its own output to `experiments_expN.log` in
  this directory if you invoke it with `| tee experiments_expN.log`;
  that's the convention used so far, keep it for consistency and so the
  Monitor pattern above keeps working.

## Architecture quick-reference

```
config.py       Config dataclass: every parameter, derived quantities (M, wavelength,
                Pmax_w, noise_power_w, antenna gain linear values), seeded RNG factory.
                __post_init__ auto-bumps sv_num_paths for K>1 (see "rank" note below).
channel.py      ula_steering, upa_steering (array responses), free_space_pathloss,
                molecular_absorption, get_ue_positions, sv_channel_H/g (sparse SV model),
                direct_channel (blocked link, Rayleigh + blockage_loss_db).
beamforming.py  mrt (K=1), zf, rzf (K>1) -- BS active precoders, ||W||_F^2 = P exactly.
ris_opt.py      phase_update_single_user (K=1 closed form), phase_update_manifold
                (K>1, Riemannian gradient ascent via torch autograd; see docstring for
                why the search direction is normalised before the Armijo line search --
                a correctness-affecting detail, not cosmetic).
ao.py           run_ao: alternates the two halves above, safeguarded (never accepts a
                rate-decreasing update), returns (theta, W, rate_history).
metrics.py      effective_channel, sinr_per_user, sum_rate, energy_efficiency.
benchmarks.py   no_ris_rate, random_phase_rate, af_relay_rate.
experiments/    exp1-exp7 (+ plot_geometry.py, make_summary.py, _common.py for shared
                plotting/result-saving helpers). Each is runnable standalone via
                `python -m experiments.exp<N>_<name>` from this directory.
tests/test_core.py   16 tests, organised by milestone; run `pytest -q`.
docs/           ASSUMPTIONS.md, EXPLAIN.md, REFERENCES.md (see top of this file).
```

Key non-obvious facts worth knowing before you edit:
- **`sv_num_paths` is auto-bumped for K>1** (`Config.__post_init__`,
  `sv_num_paths = K+2` if the current value is `<= K`). Reason: H's rank is
  bounded by the path count, and ZF/RZF need the K-user effective channel
  to have rank ≥ K, or you get an exactly-singular Gram matrix (this was
  also caught the hard way, via a real `LinAlgError: Singular matrix`
  crash during an early K=4 experiment run — see
  `docs/ASSUMPTIONS.md`).
- **AO's rate history is monotonic by construction**, not by luck: every
  half-step in `ao.py` only commits a candidate `W`/`theta` if its rate is
  ≥ the best rate seen so far, else keeps the previous iterate. Don't
  "simplify" this away — removing the safeguard reintroduces the
  possibility of the rate decreasing (ZF/RZF aren't exactly rate-optimal
  for a fixed theta, and the manifold solver isn't guaranteed to improve
  on a bad random re-init in one step).
- **No-RIS can beat AO-RIS, and random-phase-RIS can beat no-RIS's
  opposite too (random can be worse than no-RIS)** — both are real,
  already-investigated, already-documented findings about this specific
  scenario's link budget (see `docs/ASSUMPTIONS.md`), not bugs to chase.
  If a future change makes a *new* curve look physically wrong, cross-
  check against the sanity criteria in master-prompt section 8 item 4
  (non-decreasing AO rate; ~M² scaling; rate increases with Ptx) before
  assuming it's correct OR assuming it's a bug — verify, as that section
  demands, don't guess either way.

## Environment

- Windows machine. Primary shell for this project's commands has been the
  Bash tool (Git Bash / MSYS), not PowerShell, though both are available.
- Python: a **virtualenv at `ris_project/.venv`** (created from the
  system's Python 3.13 install) has all dependencies installed
  (numpy, scipy, matplotlib, torch CPU, stable-baselines3, gymnasium,
  tqdm, pytest). **Use `./.venv/Scripts/python.exe` explicitly** for every
  command in this directory — there are multiple other Pythons on PATH
  (an MSYS one with no packages, a bare Python 3.13 with only numpy) that
  will fail confusingly if invoked by plain `python`.
- `requirements.txt` lists the packages if the venv needs recreating.

## Git / repo state

This project was just (as of this handoff) pushed to
`https://github.com/chirag9935/Minor.git` for the first time — the repo
was empty before this push. The git repo root is the **parent** directory
(`C:\Users\...\Desktop\Projects\Minor`, one level above `ris_project/`),
not `ris_project/` itself, so that the two spec markdown files
(`RIS_Master_Prompt (1).md`, `RIS_Communication_Project_Workflow.md`) are
included for context. `.venv/`, `__pycache__/`, `.pytest_cache/`, and the
loose `experiments_exp*.log` tee-output files are gitignored (see
`.gitignore` at the repo root) — they're either huge/regeneratable or
transient run logs, not deliverables.

When you make further changes: commit with a message describing *why*,
not just *what* (standard practice, also just good practice generally).
Don't force-push. If you're an LLM agent continuing this and the user
hasn't explicitly asked you to commit/push, ask first or at least flag
that you're about to, per normal "confirm before actions with external
effects" practice — don't assume every edit should go straight to the
remote.
