# One-page walkthrough (for the oral evaluation)

## The pipeline, in order

1. **config.py** fixes every number the rest of the pipeline needs: carrier
   frequency, array sizes (Nt, M=Mx*My, K), power budget, geometry,
   blockage loss, antenna gains, absorption coefficients, the power model
   for EE, and a seeded RNG factory (`cfg.rng(extra_seed)`).

2. **channel.py** draws three channels from that config:
   - `H` (BS->RIS, M x Nt): sparse Saleh-Valenzuela (SV) model, 1 LoS +
     (L-1) NLoS paths, each path a random complex gain times a BS ULA
     steering vector and a RIS UPA steering vector (outer product).
   - `g` (RIS->UE, M x K): same SV structure, RIS steering vector only
     (the UE has a single antenna).
   - `d` (BS->UE direct, K x Nt): plain Rayleigh fading, scaled by free-
     space path loss **plus** a fixed `blockage_loss_db` (the direct link
     is blocked).
   All three are scaled by `sqrt(pathloss * Lmol(fc,d) * antenna_gains)`.

3. **beamforming.py** turns an *effective* channel `H_eff` (K x Nt, one row
   per user, already folding in the RIS) into a transmit precoder `W`
   (Nt x K) under the power constraint `||W||_F^2 = P`: `mrt` for K=1,
   `zf`/`rzf` for K>1.

4. **ris_opt.py** does the other half: given a fixed `W`, choose the RIS
   phases `theta` (|theta_m|=1) to maximise the rate. K=1 has a closed
   form (align every reflected path's phase). K>1 has no closed form, so
   `phase_update_manifold` runs Riemannian gradient ascent on the unit-
   modulus manifold (autograd gradient -> tangent projection -> normalise
   -> Armijo line search -> retract to the unit circle).

5. **ao.py** alternates steps 3 and 4 (`run_ao`): fix theta, solve for W;
   fix W, solve for theta; repeat until the rate stops improving. Each
   half-step is only accepted if it does not decrease the rate, so the
   rate history is non-decreasing by construction.

6. **metrics.py** computes the sum rate (`sum_k log2(1+SINR_k)`) and energy
   efficiency (`BW*rate / P_total`) from any `(H_eff, W)` pair.

7. **benchmarks.py** computes the three baselines with the same channels:
   no-RIS (direct link only), random-phase RIS (200 random `theta` draws,
   averaged), and a half-duplex AF relay (two single-hop Rayleigh legs
   through the RIS location).

8. **experiments/** run all of the above over many random channel draws
   (Monte Carlo) and sweep one parameter at a time (Ptx, M, distance,
   frequency), saving the mean curves to `results/*.npz` and `plots/*.png`.

## 10 likely viva questions

**1. Why does the RIS phase have a closed-form solution for K=1 but not K>1?**
For K=1 the objective is `|sum_m c_m*theta_m|^2` for fixed complex numbers
`c_m` (c_m = conj(g_m)*a_m) -- a magnitude of a phase-weighted sum, which is
maximised exactly by aligning every term's phase (triangle inequality,
equality case). For K>1 the objective is a sum of *ratios* (SINR terms)
that all depend on the same `theta` through different, coupled
expressions -- there is no closed form, so we optimise numerically on the
manifold.

**2. Why a manifold optimizer instead of just projecting onto the unit circle after an ordinary gradient step?**
A naive "take a gradient step then normalise" does work in practice for
this constraint set (it's one common heuristic), but it ignores the
curvature of the constraint set and can take wasteful steps. The
Riemannian version projects the gradient onto the *tangent space* first
(removing the radial component that a normalisation step would just
throw away) so the step moves along the manifold from the start, and the
Armijo line search gives a principled accept/reject rule with a
guaranteed-non-decreasing objective.

**3. Why does the tangent-space gradient sometimes need to be normalised before the line search?**
Far from the optimum (e.g. a random initial `theta` with M large), the raw
gradient can be numerically tiny even though the objective is far from
its maximum, which would force many iterations of negligible progress
with a fixed step size. Normalising the direction and starting the line
search from a fixed step (`pi` radians) decouples the step size from the
gradient's magnitude.

**4. Why is AO's rate history guaranteed non-decreasing?**
Because each half-step (`ao.py`) only replaces the current `W` or `theta`
if the candidate's rate is >= the best rate seen so far; otherwise the
previous iterate is kept. This is a standard "safeguarded AO" trick.

**5. Why does ZF fail (singular matrix) for K=4 with only 3 multipath components?**
`H` (BS->RIS) is built as a sum of `sv_num_paths` rank-1 terms, so
`rank(H) <= sv_num_paths`. Every user's effective row `g_k^H*Phi*H` lives
in that same low-rank row space, so `rank(H_eff) <= sv_num_paths`
regardless of K. With only 3 paths and 4 users, `H_eff` (4 x Nt) can have
rank at most 3 -- its Gram matrix is exactly singular, not just
ill-conditioned. We fix this by using more paths (`K+2`) whenever K>1 (see
`Config.__post_init__`).

**6. Why can RIS (even AO-optimised) be *worse* than no RIS at all?**
A RIS-aided link is a cascaded (two-hop) channel: its path loss is the
*product* of the BS-RIS and RIS-UE path losses, not a sum -- much larger
than a single-hop link's loss. Coherent (AO-optimised) phases buy back an
M^2 array gain that can compensate, but M^2 is not unconditionally "big
enough" -- at the assignment's default M=64 in this project's geometry,
it is not: no-RIS beats AO-RIS across the whole Ptx range for K=1, and
only starts losing to AO-RIS once M grows into the hundreds (random
phases, which only give an incoherent ~M-fold gain instead of M^2, do
even worse). This is a well-documented RIS limitation, not a bug -- see
docs/ASSUMPTIONS.md for the measured crossover points and exp2's M-sweep
for where AO-RIS catches up.

**7. What does the Rician K-factor control in the channel model?**
It sets the power split between the LoS path and the NLoS paths:
`p_los = K_lin/(K_lin+1)`, with the remaining power split equally among
the NLoS paths. A higher K-factor means a more deterministic, LoS-
dominated channel; lower means more Rayleigh-like fading.

**8. Why does ZF need Nt >= K, and what does RZF buy over ZF?**
ZF inverts the K x K Gram matrix `H_eff H_eff^H`; this requires `H_eff`
to have rank K, which needs at least K antennas/degrees of freedom (and,
per Q5, enough multipath). RZF adds a `(K*sigma2/P)*I` regularisation
term, trading a little interference leakage for much better conditioning
and noise robustness -- it never fails outright the way plain ZF can.

**9. Why does EE not simply increase monotonically with Ptx?**
`EE = BW*rate / P_total`, and `P_total` includes `Ptx/eta` -- a term that
grows linearly with Ptx while `rate` grows only logarithmically
(`log2(1+SNR)`). So EE typically rises, peaks, and then falls as Ptx
increases past the point where the marginal rate gain no longer justifies
the extra transmit power -- this is exactly what the EE-vs-SE curve
(exp4) is designed to show.

**10. Why is the direct link modelled with a fixed `blockage_loss_db` instead of real geometry/ray tracing?**
The assignment specifies the direct LoS is "completely blocked" and asks
for a scenario where the no-RIS curve is still finite and plottable. A
fixed penetration-loss penalty is the simplest model that satisfies both
requirements without implementing full ray tracing, which is out of
scope for this project (see Scope / Future work in the README).
