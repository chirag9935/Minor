# References

All verified to exist; cited only where actually used in this project's
design choices. See the README bibliography for inline citation numbers.

| # | Reference | Use it for |
|---|---|---|
| [1] | Q. Wu and R. Zhang, "Intelligent Reflecting Surface Enhanced Wireless Network via Joint Active and Passive Beamforming," IEEE Trans. Wireless Commun., 2019. arXiv:1810.03961 | Core AO structure: alternate active (BS) and passive (RIS) beamforming; unit-modulus constraint; rate-vs-M scaling |
| [2] | Q. Wu and R. Zhang, "Towards Smart and Reconfigurable Environment: Intelligent Reflecting Surface Aided Wireless Network," IEEE Commun. Mag., 2020. arXiv:1905.00152 | Overview and motivation, blocked-link use case |
| [3] | M. Di Renzo et al., "Smart Radio Environments Empowered by Reconfigurable Intelligent Surfaces: How it Works, State of Research, and Road Ahead," IEEE JSAC, 2020. arXiv:2004.09352 | Background, RIS vs relay comparison (Benchmark 3), multiplicative path loss discussion |
| [4] | X. Yu, D. Xu, R. Schober, "MISO Wireless Communication Systems via Intelligent Reflecting Surfaces," IEEE ICCC, 2019. arXiv:1904.12199 | Manifold optimization for RIS phases, MISO setting |
| [5] | C. Huang, A. Zappone, G. C. Alexandropoulos, M. Debbah, C. Yuen, "Reconfigurable Intelligent Surfaces for Energy Efficiency in Wireless Communication," IEEE Trans. Wireless Commun., 2019. arXiv:1810.06934 | Power-consumption model and energy-efficiency metric |
| [6] | C. Huang, R. Mo, C. Yuen, "Reconfigurable Intelligent Surface Assisted Multiuser MISO Systems Exploiting Deep Reinforcement Learning," IEEE JSAC, 2020. arXiv:2002.10072 | State/action/reward design for the optional DRL demo and future work |
| [7] | O. El Ayach et al., "Spatially Sparse Precoding in Millimeter Wave MIMO Systems," IEEE Trans. Wireless Commun., 2014. arXiv:1305.2460 | Sparse Saleh-Valenzuela-style mmWave channel and array-response model |
| [8] | W. Tang et al., "Wireless Communications with Reconfigurable Intelligent Surface: Path Loss Modeling and Experimental Measurement," IEEE Trans. Wireless Commun., 2021. arXiv:1911.05326 | Realistic RIS path-loss modelling; cited as a limitation of this project's simplified cascaded model |
| [9] | N. Boumal, B. Mishra, P.-A. Absil, R. Sepulchre, "Manopt, a Matlab Toolbox for Optimization on Manifolds," JMLR, 2014 (and the Pymanopt Python port) | Riemannian optimization background (tangent-space projection, retraction, Armijo line search) |

[1] and [4] justify the AO/manifold algorithm, [7] the channel model, [5]
the EE model, [3] the relay comparison and the multiplicative-path-loss
discussion, [6] the future DRL extension, [8] the path-loss-model
limitation, [9] the Riemannian optimization mechanics. THz absorption
values in `config.py` are approximate placeholders (see
docs/ASSUMPTIONS.md); a proper model would use ITU-R P.676 / HITRAN data,
left as future work.
