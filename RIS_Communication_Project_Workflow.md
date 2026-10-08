# **PROJECT WORKFLOW & RESEARCH ROADMAP** 

**RIS/IRS-Assisted mmWave/THz Communication Systems** 

_Comprehensive Student Guide | 6G Optical & Wireless Networks_ 

## **Step 1: System Model & Scenario Definition** 

### **Environment Setup** 

Design a scenario where the direct Line-of-Sight (LoS) path between the Base Station (BS) and the User Equipment (UE) is completely blocked by obstacles (e.g., buildings, walls, or foliage). 

### **Setup Coordinates & Network Architecture** 

- Base Station equipped with Nt transmit antennas. 

- Reconfigurable Intelligent Surface (RIS) equipped with M reflecting elements (e.g., an 8 × 8 = 64 or 16 × 16 = 256 uniform planar array). 

- User Equipment (UE) equipped with single or multiple receive antennas (Nr). 

### **Channel Model Formulation** 

- BS-to-RIS Channel (H): Modeled using the sparse mmWave/THz Saleh-Valenzuela channel model (dominant LoS path + few NLoS paths). 

- RIS-to-UE Channel (g): Similar sparse mmWave/THz channel incorporating the molecular absorption loss factor Lmol(f, d) for THz band frequencies. 

## **Step 2: Optimization Problem Formulation** 

### **Objective Function (Select One)** 

- Option A: Maximize Achievable Rate / Spectral Efficiency (R = log₂(1 + SINR)). 

- Option B: Minimize Total Transmit Power (Ptx) subject to a minimum required SINR constraint. 

- Option C: Maximize Energy Efficiency (EE = Data Rate / Total Power Consumption). 

### **Constraints** 

- BS Transmit Power Limit: ||W||F² ≤ Pmax. 

- RIS Unit-Modulus Constraint (Phase-only control): |θm| = 1, m  {1, ..., M}.∀ ∈ 

## **Step 3: Proposed Algorithm Design (Novel Contribution)** 

Choose one of the following research approaches: 

### **Approach A: Mathematical Optimization (Alternating Optimization - AO)** 

Decouple the joint optimization problem into two iterative sub-problems: 

1. Fix the RIS Phase Matrix (Φ) and optimize the BS Beamforming Matrix (W) using Zero-Forcing (ZF) or Minimum Mean Square Error (MMSE). 

2. Fix W and optimize Φ using Semidefinite Relaxation (SDR) or Manifold Optimization (MO). 

3. Iterate between the sub-problems until convergence is reached. 

### **Approach B: AI/ML-Based Optimization (Deep Reinforcement Learning - DRL)** 

• State (St): Estimated Channel State Information (CSI) / Received Signal Strength Indicator (RSSI) / UE location coordinates. 

- Action (At): Continuous phase shift values for the RIS elements (θ₁, θ₂, ..., θM) alongside BS beamformer weights. 

- Reward (Rt): Instantaneous Achievable Rate or SINR. 

- Algorithm: Implement model-free DRL algorithms like Proximal Policy Optimization (PPO) or Soft Actor-Critic (SAC). 

## **Step 4: Implementation & Simulation (Code Execution)** 

### **Software Tools** 

MATLAB (using CVX Toolbox for optimization) or Python (using PyTorch/TensorFlow and NumPy). 

### **Baseline Benchmarks (For Performance Evaluation)** 

- Benchmark 1: Direct link only (No RIS — Blocked scenario). 

- Benchmark 2: RIS with Random Phase Shifts. 

- Benchmark 3: Conventional Amplify-and-Forward (AF) Relay system. 

### **Parameter Tuning & Sensitivity Analysis** 

- Number of RIS reflecting elements (M = 16, 32, 64, 128, 256). 

- Transmit Power (Ptx = 0 to 30 dBm). 

- Varying distance and geometry between BS, RIS, and UE. 

## **Step 5: Result Analysis & Performance Metrics** 

Generate and analyze the following key plots: 

1. Achievable Rate vs. Transmit Power (Ptx) 

2. Achievable Rate vs. Number of RIS Elements (M) 

3. Convergence Plot: Cumulative Reward / Objective Function Value vs. Iterations or Epochs (Proof of convergence for AO/DRL). 

4. Energy Efficiency vs. Spectral Efficiency Trade-off Curve. 

## **Step 6: Documentation & Paper Writing** 

- Abstract & Introduction: Background and motivation (6G mmWave/THz blockage issues) + Proposed Solution (RIS + Novel DRL/AO approach). 

- System & Channel Model: Complete mathematical formulations for channels H and g, and the RIS interaction matrix Φ. 

- Problem Formulation & Proposed Method: Mathematical derivations / DRL architecture framework diagram. 

- Simulation Results & Discussion: Comparative plots accompanied by physical insights. 

- Conclusion & Future Scope: Key findings and potential future extensions. 

## **APPENDIX: Updated Multiplexer Implementation Table** 

Configuration: Select inputs (S₁, S₀) = (C, D) and input variables = (A, B). 

|**MUX Input**|**Selected**<br>**(C, D)**|**Minterms**<br>**(A, B)**|**AB = 00**|**AB = 01**|**AB = 10**|**AB = 11**|**Output**<br>**Expressio**<br>**n**|
|---|---|---|---|---|---|---|---|
|I₀|0 0|m₀, m₄, m₈,<br>m₁₂|0 (m₀)|1 (m₄)|0 (m₈)|0 (m₁₂)|A'B|
|I₁|0 1|m₁, m₅, m₉,<br>m₁₃|1 (m₁)|0 (m₅)|0 (m₉)|1 (m₁₃)|A'B' + AB|
|I₂|1 0|m₂, m₆, m₁₀,<br>m₁₄|0 (m₂)|0 (m₆)|1 (m₁₀)|1 (m₁₄)|A|
|I₃|1 1|m₃, m<br>, m₁₁,<br>⇇<br>m₁₅|1 (m₃)|1 (m<br>)<br>⇇|0 (m₁₁)|1 (m₁₅)|A' + B|



