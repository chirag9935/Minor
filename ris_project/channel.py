"""
Channel generation for the RIS-assisted link.

Links modelled
--------------
H  : BS -> RIS,  shape (M, Nt)   sparse Saleh-Valenzuela (SV), Rician LoS + NLoS
g  : RIS -> UE,  shape (M, K)    sparse SV per user, Rician LoS + NLoS
d  : BS -> UE,   shape (K, Nt)   Rayleigh (direct link, blocked)

Saleh-Valenzuela model (H, g)
------------------------------
For L paths (1 LoS + L-1 NLoS), path l has complex gain alpha_l ~ CN(0, p_l),
with power split p_los = Kr/(Kr+1) for the LoS path (Kr = Rician K-factor,
linear) and p_nlos = (1-p_los)/(L-1) split equally among NLoS paths (so
sum_l p_l = 1). NLoS angles are the LoS angle plus a random angular offset
(angular spread).

    H = sqrt(pathloss(d) * Lmol(fc,d) * Gbs) * sum_l alpha_l * a_RIS(l) a_BS(l)^H
    g_k = sqrt(pathloss(d_k) * Lmol(fc,d_k) * Gue) * sum_l alpha_l * a_RIS(l)

where a_BS is the BS ULA steering vector (azimuth AoD), a_RIS is the RIS
UPA steering vector (azimuth/elevation AoA or AoD, depending on the link,
entries of unit modulus), and Gbs/Gue are the linear BS/UE antenna gains
(RIS elements are assumed unity-gain). With sum_l p_l = 1 and unit-modulus
steering entries, E[|H_mn|^2] = pathloss*Lmol*Gbs exactly -- i.e. each
BS-antenna-to-RIS-element sub-channel has the physically correct path-loss-
scaled power, independent of Nt, M or L.

Note: there is deliberately **no** extra sqrt(Nt*M) / sqrt(M) size-dependent
prefactor here (unlike the standard point-to-point mmWave MIMO convention
of El Ayach et al. [7], H = sqrt(Nt*Nr/L) sum alpha a_r a_t^H). That
convention normalises a channel meant to be immediately *collapsed* by an
Nr-branch receive combiner. Here M is a cascaded, non-collapsing dimension
(the same RIS elements appear in both H and g via diag(theta)), so copying
that prefactor into both legs double-counts the array gain: BS-side MRT
gain (~Nt) and RIS coherent-phase gain (~M^2, from summing M unit-modulus
terms then squaring for power) already emerge correctly from the steering-
vector algebra itself once theta is optimised -- adding Nt/M prefactors on
top of that inflated the simulated SNR by roughly (Nt*M) extra, i.e. the
RIS gain was scaling like M^4 instead of the textbook M^2 (see
docs/ASSUMPTIONS.md for the sanity check that caught this).

Direct link (blocked)
----------------------
    d_k = sqrt(pathloss(d_k) * Lmol(fc,d_k) * Gbs * Gue * 10^(-blockage_loss_db/10)) * CN(0, I_Nt)

Free-space path loss (power ratio): pathloss(d) = (wavelength / (4*pi*d))^2
Molecular absorption factor:        Lmol(fc,d)  = 10^(-k(fc)*d / 10000), k in dB/km, d in metres.
"""
from __future__ import annotations

import numpy as np

from config import Config


def ula_steering(N: int, angle: float) -> np.ndarray:
    """ULA steering vector, half-wavelength spacing: a_n = exp(j*pi*n*sin(angle)), n=0..N-1."""
    n = np.arange(N)
    return np.exp(1j * np.pi * n * np.sin(angle))


def upa_steering(Mx: int, My: int, az: float, el: float) -> np.ndarray:
    """UPA steering vector, half-wavelength spacing in both dimensions, flattened (Mx*My,).

    a_{mx,my} = exp(j*pi*(mx*sin(el)*cos(az) + my*sin(el)*sin(az))), mx=0..Mx-1, my=0..My-1.
    """
    mx = np.arange(Mx)
    my = np.arange(My)
    MX, MY = np.meshgrid(mx, my, indexing="ij")
    phase = np.pi * (MX * np.sin(el) * np.cos(az) + MY * np.sin(el) * np.sin(az))
    return np.exp(1j * phase).reshape(-1)


def free_space_pathloss(d: float, wavelength: float) -> float:
    """Free-space path loss as a power ratio: (wavelength / (4*pi*d))^2."""
    return (wavelength / (4 * np.pi * d)) ** 2


def molecular_absorption(k_db_per_km: float, d_m: float) -> float:
    """Lmol = 10^(-k*d/10000), k in dB/km, d in metres."""
    return 10 ** (-k_db_per_km * d_m / 10000.0)


def _azimuth(vec: np.ndarray) -> float:
    """Azimuth angle (x-y plane) of a direction vector, used for the BS ULA."""
    return float(np.arctan2(vec[1], vec[0]))


def _azimuth_elevation(vec: np.ndarray) -> tuple:
    """Azimuth and elevation angles of a direction vector, used for the RIS UPA."""
    az = float(np.arctan2(vec[1], vec[0]))
    horiz = np.hypot(vec[0], vec[1])
    el = float(np.arctan2(vec[2], horiz))
    return az, el


def get_ue_positions(cfg: Config, rng: np.random.Generator, K: int | None = None) -> np.ndarray:
    """Place K UEs at a 3D distance in [ue_min_dist_from_ris, ue_max_dist_from_ris] from the
    RIS, at fixed height cfg.ue_height, uniformly random azimuth around the RIS."""
    K = cfg.K if K is None else K
    ris = np.array(cfg.ris_pos)
    dz = cfg.ue_height - ris[2]
    positions = np.zeros((K, 3))
    for k in range(K):
        dist = rng.uniform(cfg.ue_min_dist_from_ris, cfg.ue_max_dist_from_ris)
        angle = rng.uniform(0, 2 * np.pi)
        horiz = np.sqrt(max(dist ** 2 - dz ** 2, 0.0))
        x = ris[0] + horiz * np.cos(angle)
        y = ris[1] + horiz * np.sin(angle)
        positions[k] = [x, y, cfg.ue_height]
    return positions


def _path_powers(L: int, rician_k_db: float) -> list:
    k_lin = 10 ** (rician_k_db / 10)
    p_los = k_lin / (k_lin + 1)
    p_nlos = (1 - p_los) / (L - 1)
    return [p_los] + [p_nlos] * (L - 1)


def sv_channel_H(cfg: Config, rng: np.random.Generator, fc: float | None = None) -> np.ndarray:
    """BS -> RIS channel, shape (M, Nt). See module docstring for the formula."""
    fc = cfg.fc if fc is None else fc
    wavelength = cfg.c / fc
    bs = np.array(cfg.bs_pos)
    ris = np.array(cfg.ris_pos)
    d = np.linalg.norm(ris - bs)
    amp = np.sqrt(free_space_pathloss(d, wavelength) * molecular_absorption(cfg.absorption_coeff(fc), d) * cfg.bs_gain_lin)

    L = cfg.sv_num_paths
    powers = _path_powers(L, cfg.sv_rician_k_db)
    az_bs_los = _azimuth(ris - bs)
    az_ris_los, el_ris_los = _azimuth_elevation(bs - ris)

    H = np.zeros((cfg.M, cfg.Nt), dtype=complex)
    for l in range(L):
        if l == 0:
            theta_bs, az_ris, el_ris = az_bs_los, az_ris_los, el_ris_los
        else:
            theta_bs = az_bs_los + rng.uniform(-np.pi / 3, np.pi / 3)
            az_ris = az_ris_los + rng.uniform(-np.pi / 3, np.pi / 3)
            el_ris = el_ris_los + rng.uniform(-np.pi / 6, np.pi / 6)
        alpha = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2) * np.sqrt(powers[l])
        a_bs = ula_steering(cfg.Nt, theta_bs)
        a_ris = upa_steering(cfg.Mx, cfg.My, az_ris, el_ris)
        H += alpha * np.outer(a_ris, np.conj(a_bs))
    H *= amp
    return H


def sv_channel_g(cfg: Config, rng: np.random.Generator, ue_positions: np.ndarray,
                  fc: float | None = None) -> np.ndarray:
    """RIS -> UE channel(s), shape (M, K). See module docstring for the formula."""
    fc = cfg.fc if fc is None else fc
    wavelength = cfg.c / fc
    ris = np.array(cfg.ris_pos)
    K = ue_positions.shape[0]
    L = cfg.sv_num_paths
    powers = _path_powers(L, cfg.sv_rician_k_db)

    g = np.zeros((cfg.M, K), dtype=complex)
    for k in range(K):
        ue = ue_positions[k]
        d = np.linalg.norm(ue - ris)
        amp = np.sqrt(free_space_pathloss(d, wavelength) * molecular_absorption(cfg.absorption_coeff(fc), d) * cfg.ue_gain_lin)
        az_los, el_los = _azimuth_elevation(ue - ris)
        gk = np.zeros(cfg.M, dtype=complex)
        for l in range(L):
            if l == 0:
                az, el = az_los, el_los
            else:
                az = az_los + rng.uniform(-np.pi / 3, np.pi / 3)
                el = el_los + rng.uniform(-np.pi / 6, np.pi / 6)
            alpha = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2) * np.sqrt(powers[l])
            gk += alpha * upa_steering(cfg.Mx, cfg.My, az, el)
        g[:, k] = gk * amp
    return g


def direct_channel(cfg: Config, rng: np.random.Generator, ue_positions: np.ndarray,
                    fc: float | None = None) -> np.ndarray:
    """BS -> UE direct (blocked) channel, shape (K, Nt). See module docstring for the formula."""
    fc = cfg.fc if fc is None else fc
    wavelength = cfg.c / fc
    bs = np.array(cfg.bs_pos)
    K = ue_positions.shape[0]
    blockage_lin = 10 ** (-cfg.blockage_loss_db / 10)

    d_mat = np.zeros((K, cfg.Nt), dtype=complex)
    for k in range(K):
        ue = ue_positions[k]
        d = np.linalg.norm(ue - bs)
        amp = np.sqrt(free_space_pathloss(d, wavelength) * molecular_absorption(cfg.absorption_coeff(fc), d)
                      * cfg.bs_gain_lin * cfg.ue_gain_lin * blockage_lin)
        d_mat[k, :] = amp * (rng.standard_normal(cfg.Nt) + 1j * rng.standard_normal(cfg.Nt)) / np.sqrt(2)
    return d_mat
