"""
Global configuration for the RIS-assisted mmWave/THz link-level simulation.

Every numeric default here is either (a) a standard constant, (b) taken
directly from the assignment (RIS_Communication_Project_Workflow.md), or
(c) an explicitly labelled *assumption* needed to make the toy scenario
well-posed (e.g. antenna gains, absorption coefficients, power-model
constants). Assumptions are also recorded in docs/ASSUMPTIONS.md.

Notation follows the assignment: BS, RIS, UE, H (BS->RIS), g (RIS->UE),
Phi (RIS phase matrix), theta_m (RIS element phase), W (BS beamformer),
Nt (BS antennas), M (RIS elements), Ptx (transmit power), Pmax (BS power
budget), Lmol (molecular absorption loss factor).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import warnings
import numpy as np

try:
    import itur as _itur
    _ITUR_AVAILABLE = True
except ImportError:
    _ITUR_AVAILABLE = False
    warnings.warn(
        "itur (ITU-Rpy) is not installed; Config.absorption_model='itu' will "
        "fall back to the Phase 1 placeholder dB/km table instead. Install "
        "with `pip install itur` for the real ITU-R P.676 model (Phase 2 "
        "Milestone 3). See docs/ASSUMPTIONS.md."
    )

# One ITU-R P.676 call per distinct (fc, T, P, rho) combo -- these repeat
# constantly across Monte Carlo draws at fixed sweep points, so caching
# avoids redundant calls (each ~0.2 ms, cheap but non-zero over thousands
# of channel draws).
@lru_cache(maxsize=4096)
def _itu_absorption_db_per_km(fc: float, temperature_c: float, pressure_hpa: float,
                               water_vapour_density: float) -> float:
    """Real ITU-R P.676 gaseous (oxygen + water vapour) specific attenuation,
    dB/km, via itur.gaseous_attenuation_terrestrial_path (verified API: see
    docs/ASSUMPTIONS.md). r=1 km -> the returned dB value IS dB/km, since
    the terrestrial-path attenuation scales exactly linearly with path
    length (confirmed numerically: r=2km gives exactly 2x the r=1km
    attenuation). el=0 (horizontal path) -> use mode="exact" (the "approx"
    mode in itur warns that it is only valid for elevation 5-90 degrees).
    Falls back to the nearest Phase 1 placeholder value if itur is not
    installed (see docs/ASSUMPTIONS.md)."""
    if not _ITUR_AVAILABLE:
        placeholders = {28e9: 0.1, 140e9: 2.0, 300e9: 8.0}
        freqs = np.array(list(placeholders.keys()))
        nearest = freqs[np.argmin(np.abs(freqs - fc))]
        return placeholders[nearest]
    att = _itur.gaseous_attenuation_terrestrial_path(
        r=1.0, f=fc / 1e9, el=0, rho=water_vapour_density,
        P=pressure_hpa, T=temperature_c + 273.15, mode="exact")
    return float(att.value)


@dataclass
class Config:
    # ----- reproducibility -----
    seed: int = 42

    # ----- carrier / bandwidth / noise -----
    fc: float = 140e9          # Hz, carrier frequency (also test with 28e9, 300e9)
    c: float = 3e8             # m/s, speed of light
    BW: float = 100e6          # Hz, system bandwidth
    noise_psd_dbm_hz: float = -174.0   # dBm/Hz, thermal noise PSD (kT at room temp)
    NF_db: float = 8.0         # dB, receiver noise figure

    # ----- array sizes -----
    Nt: int = 8                # BS ULA elements (half-wavelength spacing)
    Mx: int = 8                # RIS UPA columns
    My: int = 8                # RIS UPA rows (default Mx*My = 64)
    K: int = 1                 # number of single-antenna UEs (1 or 4)

    # ----- power budget -----
    Pmax_dBm: float = 30.0      # dBm, BS transmit power budget (||W||_F^2 <= Pmax)

    # ----- geometry (metres) -----
    bs_pos: tuple = (0.0, 0.0, 10.0)
    ris_pos: tuple = (20.0, 10.0, 10.0)
    # UEs are placed at this height (assumption: elevated users / balconies,
    # consistent with a RIS mounted at building height) so that a 3-8 m
    # *3D* distance from the RIS is geometrically reachable.
    ue_height: float = 7.0
    ue_min_dist_from_ris: float = 3.0
    ue_max_dist_from_ris: float = 8.0

    # Blocker rectangle drawn in plots/geometry.png, between BS and UEs
    # (x_min, x_max, y_min, y_max, z_min, z_max). Purely illustrative: the
    # direct link is *not* geometrically ray-traced, it is penalised by a
    # fixed blockage_loss_db instead (see below).
    blocker_box: tuple = (8.0, 12.0, -5.0, 15.0, 0.0, 12.0)

    # ----- blocked direct link (assumption) -----
    # The BS->UE direct link is "completely blocked" per the assignment.
    # We keep it finite (so the no-RIS benchmark is plottable, rather than
    # exactly zero) by adding a fixed penetration/blockage loss on top of
    # ordinary free-space path loss.
    #
    # PHASE 2 CHANGE (see docs/ASSUMPTIONS.md "Milestone 0"): the Phase 1
    # default of 40 dB was too weak to represent "completely blocked" --
    # at 40 dB the un-optimized direct link (no-RIS) still beat both
    # AO-RIS and the AF relay (see exp_blockage_sweep.py). A sweep over
    # 0-100 dB shows the AO-RIS/no-RIS crossover sits between 60-80 dB in
    # this geometry; 80 dB is the new default, representing a genuinely
    # severe obstruction (thick wall / deep NLoS), consistent with
    # "completely blocked." The old 40 dB value is kept reproducible as an
    # explicit "partially blocked" comparison point (see
    # exp_blockage_sweep.py and the *_blocked40 result/plot files), not
    # deleted, so Phase 1's numbers can still be regenerated.
    blockage_loss_db: float = 80.0

    # ----- antenna gains (assumption, calibrated) -----
    # Chosen so that for K=1, M=64, Ptx=30 dBm at 140 GHz, the mean AO SNR
    # lands roughly in 15-25 dB (measured: ~20.2 dB mean over 30 draws).
    # Deliberately high (directional/phased-array-grade) gains: the cascaded
    # (two-hop) RIS link's path loss at 140 GHz over these distances is
    # severe even after M^2 coherent combining, so a workable SNR needs
    # either very large M or high-gain antennas -- see docs/ASSUMPTIONS.md.
    bs_gain_dbi: float = 45.0
    ue_gain_dbi: float = 25.0

    # ----- molecular absorption (Phase 2 Milestone 3) -----
    # Default model is now "itu": real ITU-R P.676 gaseous attenuation
    # (oxygen + water vapour), via the itur (ITU-Rpy) package, verified
    # against its own documented API (gaseous_attenuation_terrestrial_path)
    # before use -- see docs/ASSUMPTIONS.md. "placeholder" keeps Phase 1's
    # illustrative, hand-picked dB/km table (absorption_db_per_km below)
    # reproducible for comparison; it is not deleted.
    absorption_model: str = "itu"        # "itu" or "placeholder"
    temperature_c: float = 15.0          # deg C, ITU-R P.676 default condition
    pressure_hpa: float = 1013.0         # hPa, ITU-R P.676 default condition
    water_vapour_density: float = 7.5    # g/m^3, ITU-R P.676 default condition

    # k(f) in dB/km. Phase 1 placeholders, kept for the absorption_model=
    # "placeholder" fallback/comparison path only (see docs/REFERENCES.md, [8]).
    absorption_db_per_km: dict = field(default_factory=lambda: {
        28e9: 0.1,
        140e9: 2.0,
        300e9: 8.0,
    })

    # ----- power model for energy efficiency (assumptions, see [5]) -----
    pa_efficiency: float = 0.4          # eta, BS power-amplifier efficiency
    p_bs_static_w: float = 1.0          # W, BS static (circuit) power
    p_ris_elem_w: float = 5e-3          # W, power per RIS element
    p_ue_w: float = 0.1                 # W, power per UE receiver chain

    # ----- Saleh-Valenzuela channel model -----
    sv_num_paths: int = 3                # 1 LoS + 2 NLoS (K=1 default)
    sv_rician_k_db: float = 10.0         # K-factor of the LoS path

    def __post_init__(self):
        # H (BS->RIS) is a sum of sv_num_paths rank-1 terms, so rank(H) <=
        # sv_num_paths; ZF/RZF need the K-user effective channel to have
        # rank >= K, which is only possible if sv_num_paths >= K. With the
        # K=1 default (3 paths) this is automatically satisfied; for K>1 we
        # bump the path count so a K=4 scenario isn't handed a structurally
        # rank-deficient (exactly singular) channel. See docs/ASSUMPTIONS.md.
        if self.sv_num_paths <= self.K:
            self.sv_num_paths = self.K + 2

    # ---------------------------------------------------------------
    # derived quantities
    # ---------------------------------------------------------------
    @property
    def M(self) -> int:
        return self.Mx * self.My

    @property
    def wavelength(self) -> float:
        return self.c / self.fc

    @property
    def Pmax_w(self) -> float:
        return 10 ** ((self.Pmax_dBm - 30) / 10)

    @property
    def noise_power_w(self) -> float:
        """sigma^2 = N0 * BW * NF, with N0 the thermal noise PSD."""
        noise_dbm = self.noise_psd_dbm_hz + 10 * np.log10(self.BW) + self.NF_db
        return 10 ** ((noise_dbm - 30) / 10)

    def absorption_coeff(self, fc: float | None = None) -> float:
        """k(fc) in dB/km. Dispatches on self.absorption_model:
        "itu" (default) -> real ITU-R P.676 gaseous attenuation via itur,
        at this Config's temperature_c/pressure_hpa/water_vapour_density;
        "placeholder" -> Phase 1's hand-picked dB/km table (nearest
        tabulated frequency in absorption_db_per_km)."""
        fc = self.fc if fc is None else fc
        if self.absorption_model == "itu":
            return _itu_absorption_db_per_km(fc, self.temperature_c, self.pressure_hpa,
                                              self.water_vapour_density)
        freqs = np.array(list(self.absorption_db_per_km.keys()))
        nearest = freqs[np.argmin(np.abs(freqs - fc))]
        return self.absorption_db_per_km[nearest]

    @property
    def bs_gain_lin(self) -> float:
        return 10 ** (self.bs_gain_dbi / 10)

    @property
    def ue_gain_lin(self) -> float:
        return 10 ** (self.ue_gain_dbi / 10)

    def rng(self, extra_seed: int = 0) -> np.random.Generator:
        """Seeded RNG. Pass extra_seed to get an independent-but-reproducible stream."""
        return np.random.default_rng(self.seed + extra_seed)

    @classmethod
    def realistic(cls, **kwargs) -> "Config":
        """Alternative preset (Phase 2, Milestone 0 item e): more modest,
        typical-hardware antenna gains (bs_gain_dbi=25, ue_gain_dbi=10 --
        closer to a real mmWave sector antenna / UE phased array) instead
        of the default preset's deliberately high gains (45/25 dBi, chosen
        so AO-RIS reaches ~20 dB SNR at the assignment's default M=64 given
        the severe cascaded RIS path loss -- see docs/ASSUMPTIONS.md).
        Peak sum rate across schemes at K=1, M=64, Ptx=30 dBm, 140 GHz is
        ~5.2 bit/s/Hz under this preset (vs ~7.7 bit/s/Hz for the default
        preset), i.e. a plausible-SNR regime (well under the ~40 dB SNR
        implied by the Phase 1 mid-eval's ~14 bit/s/Hz rates).

        The *default* preset (plain `Config(...)`) stays the project's
        default because it is the one calibrated, per the master prompt's
        explicit instruction, to land AO-RIS's SNR in the requested
        15-25 dB band at the assignment's own M=64 -- `realistic` is kept
        alongside it as an explicit lower-gain sensitivity point, not a
        replacement; both are re-run and both results are kept in
        results/ (see exp1/exp2's *_realistic variants)."""
        kwargs.setdefault("bs_gain_dbi", 25.0)
        kwargs.setdefault("ue_gain_dbi", 10.0)
        return cls(**kwargs)
