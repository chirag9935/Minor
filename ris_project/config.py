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
import numpy as np


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
    # We keep it finite (so the no-RIS benchmark is plottable) by adding a
    # fixed penetration/blockage loss on top of ordinary free-space path
    # loss. 40 dB is a typical value for a building-wall / foliage NLoS
    # penetration loss at mmWave/THz frequencies.
    blockage_loss_db: float = 40.0

    # ----- antenna gains (assumption, calibrated) -----
    # Chosen so that for K=1, M=64, Ptx=30 dBm at 140 GHz, the mean AO SNR
    # lands roughly in 15-25 dB (measured: ~20.2 dB mean over 30 draws).
    # Deliberately high (directional/phased-array-grade) gains: the cascaded
    # (two-hop) RIS link's path loss at 140 GHz over these distances is
    # severe even after M^2 coherent combining, so a workable SNR needs
    # either very large M or high-gain antennas -- see docs/ASSUMPTIONS.md.
    bs_gain_dbi: float = 45.0
    ue_gain_dbi: float = 25.0

    # ----- molecular absorption coefficients (approximate placeholders) -----
    # k(f) in dB/km. A proper model would use ITU-R P.676 / HITRAN line
    # data; these are illustrative placeholders only (see docs/REFERENCES.md, [8]).
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
        """k(fc) in dB/km, nearest tabulated frequency (see absorption_db_per_km)."""
        fc = self.fc if fc is None else fc
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
