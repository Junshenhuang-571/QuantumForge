# Variational / near-term quantum PDE solvers: a primary-source survey

## Scope

This report records what primary papers actually demonstrate when a PDE solution is mapped onto a parameterized quantum circuit and trained in a hybrid loop. It covers: (i) the Variational Quantum Linear Solver (VQLS) applied to discretized PDE systems; (ii) differentiable quantum circuits (DQC) and quantum-circuit-Born-machine (QCBM)-style trial functions; (iii) quantum-informed / physics-informed neural network (PINN) hybrids; and (iv) variational quantum computational fluid dynamics (VQCFD). Only arXiv preprints, peer-reviewed papers, and paper text read directly are used; blog summaries and encyclopedic pages were not used as authority. Each URL was registered in the citation ledger at retrieval time and each sourced sentence carries the resulting inline id. Version numbers and dates are given as stated in the papers. This report holds recommendations: it records demonstrations, numbers, and stated limitations only.

Where a paper reports an asymptotic claim without a hardware or numerical demonstration at that scale, the claim is labelled as a claim, not as a demonstrated result. Claims that could not be checked against the fetched paper text are labelled `[unverified]` and are not counted as demonstrated.

## Approaches, ansatz design, and demonstrated problem sizes

### 1. VQLS and linear systems arising from PDE discretizations

Bravo-Prieto, LaRose, Cerezo, Subaşı, Cincio, and Coles introduced VQLS (arXiv:1909.05820, v4 dated 21 Nov 2023; accepted in *Quantum* 2023-11-15). The ansatz is `|x(α)⟩ = V(α)|0⟩`; the matrix is given as a linear combination of unitaries `A = Σ c_l A_l` and the cost is a global or local overlap between `A|x⟩` and `|b⟩`, estimated with Hadamard tests. The paper proves the termination condition `C ≥ ε²/κ²` and states that the cost cannot be efficiently estimated classically under typical complexity assumptions.[26][1]

Demonstrated scale in the original paper: VQLS was implemented on Rigetti hardware for a problem of size 1024 × 1024, and numerical simulations were run up to 2^50 × 2^50 (i.e., 50 qubits) using the local cost; the authors report that with `n = 50` the global cost could not be lowered significantly below one, while the local cost remained trainable.[26]

The reported scaling is heuristic, not a proof: the authors state evidence of at worst linear scaling in the condition number κ, logarithmic scaling in 1/ε, and polylogarithmic scaling in N, and explicitly note that finite sampling was excluded from those simulations and that their verification procedure requires shot noise to scale with ε and κ.[26] The paper also notes that global cost functions can exhibit barren plateaus with gradients vanishing exponentially in qubit number n.[26]

Subsequent work ties VQLS more directly to discretized PDEs. Dsouza and Singhal (arXiv:2608.19655, v1 20 Aug 2026) build an exact block encoding of the discrete Laplacian supporting Dirichlet, Neumann, Robin, and periodic conditions with `m = 3 + ⌈log₂D⌉` ancilla qubits, and evaluate VQLS on system sizes from N = 4 to N = 1024.[21] They measure the barren plateau directly through the decay of `Var_θ[∂_μ C_G]` with qubit number and report that CMA-ES delays stagnation in regimes where COBYLA makes no progress; they also state that extracting the full classical solution vector from the output state costs O(N) measurements, eliminating any potential exponential speedup.[21]

García-Molina, Rodríguez-Mediavilla, and García-Ripoll (arXiv:2104.02668, v2 13 Apr 2021) use a space-efficient ansatz combining structured coarse-graining circuits with Fourier interpolation, with ZGR and R_Y ansätze at depths 1 and 2, on 2, 3, 4, 5, and 6 qubits.[22] They compare COBYLA (gradient-free), SPSA (numerical gradient), ADAM (analytic gradient), and L-BFGS-B with infinite measurements, using 8192 evaluations in the harmonic-oscillator study, and report infidelities of 10⁻⁵–10⁻⁴ for the harmonic oscillator and transmon qubit equations.[22] Their negative finding is that observable properties such as eigenvalues remain limited by finite precision, requiring roughly 10⁴–10⁵ measurements or gate precisions beyond current devices.[22]

Surana and Gnanasekaran (arXiv:2405.16651, v2 10 Jun 2024) present a variational quantum framework for PDE-constrained optimization using VQLS, with hand-crafted controlled-rotation state-preparation circuits for 3-qubit (`n_x = 2, n_t = 4`) and 4-qubit (`n_x = 4, n_t = 4`) examples, compared to a deeper Møttonen preparation circuit.[23] Their crossover estimates, stated under fixed sparsity `s = 0.5`, precision `ε = 0.01`, and κ = 1, are N > 750 (>9 qubits) versus Gaussian elimination, N > 2×10⁵ (>17 qubits) versus GMRES, and N > 2×10¹² (>41 qubits) versus conjugate gradient; these are analytic comparisons, not demonstrated solve sizes.[23]

**Benchmark and replication papers that challenge early VQLS claims.** Turati, Marruzzo, Ferrari Dacrema, et al. (arXiv:2409.06339, v1 10 Sep 2024) test VQLS at 2, 4, 6, 8, and 10 qubits on Ising-conditioned systems (condition numbers 2.34, 9.08, 13.62, 20.16, 30.38), random Pauli systems (condition numbers up to 219.21), and Darcy flow systems.[19] They report high-quality results on Ising instances but poor results on structured random Pauli and Darcy problems: for random Pauli, cosine similarity falls below 0.75 at n = 8 and below 0.50 at n = 10, and Darcy cosine values stay below 0.83, with tables showing up to 100000 cost evaluations.[19] They identify ansatz expressivity, the number of circuit executions needed to estimate the cost, and high gate counts as critical challenges, and note that the local cost costs O(nL²) circuit executions versus O(L²) for the global cost.[19]

Ali and Kabel (arXiv:2211.14064, v2 24 May 2023) ran an extensive Poisson-equation study on state-vector simulators, shot-noise simulators, and IBM superconducting hardware; their stated conclusion is that hybrid solvers for PDEs are not a promising route for quantum advantage in the short to near term.[20] With BFGS on a noiseless simulator they obtain over 98% fidelity at 10000 shots, but on `ibmq_ehningen` BFGS reaches roughly 17% fidelity and stops improving; NFT reaches almost 98% fidelity with 1000 shots per circuit on a simulator.[20] They report that the convergence rate of estimation error with shot count is at best square-root, i.e., analogous to Monte Carlo, and that for n ≥ 20 qubits they consider other classical high-dimensional PDE methods more promising.[20]

Özgüler (arXiv:2503.24045, v3 01 Sep 2026; dated 8 June 2026 in the abstract block) benchmarks VQE cast as a variational quantum linear solver on the advection–diffusion equation against Trotterization, VarQTE, and adaptive variational quantum dynamics at matched grid size.[24] On a noiseless state-vector simulator the N = 4 solver reaches an infidelity floor of ~10⁻¹⁴ at depth L ≈ 5; with finite shots the infidelity follows `1 − f ≈ c/S`, while the corresponding noisy hardware runs reported by Alipanah et al. exceeded 10⁻¹.[24] The same formulation is applied at N = 4, 5, 6 and to a two-dimensional eight-qubit, 16 × 16 problem at t = 1, where state-vector infidelity is ~10⁻⁷ versus ~10⁻⁵ for the shot-based simulation; the authors describe this gap as a difference of measurement regime, not algorithmic superiority, and identify an expressivity wall at N = 6 where L = 15 layers become necessary.[24]

### 2. Differentiable quantum circuits (feature-map trial functions)

Jaffali et al. (arXiv:2410.01130, v2 10 Oct 2025) define H-DES through a spectral decomposition of trial functions encoded in VQC amplitudes, with loss `L(θ) = L_diff(θ) + η·L_boundaries(θ)` and floating boundary handling as an alternative to explicit penalties.[5]

Kyriienko, Paine, and Kiffner (arXiv:2011.10395, v2 18 May 2021) define functions as expectation values of parameterized circuits with a data feature map `U(x)` and trainable circuit `V(α)`, differentiating with respect to both x and α to obtain derivatives for the PDE residual; the trial function is read out as a single observable expectation value, avoiding amplitude encoding.[3] Their optimization uses Adam with automatic differentiation, hardware-efficient ansätze at depth d = 5 or d = 6, learning rates 0.01–0.02, and approximately 200 iterations on collocation sets (e.g., 100 or 20 training points), with a six-qubit register in the differential-equation example and a Chebyshev tower feature map.[3] Boundary handling is compared as pinned versus floating; the paper covers nonlinear ODE/PDE systems including material-deformation curves.[3]

Paine, Gyurik, Gentile, et al. (arXiv:2503.22656, v1 28 Mar 2025) address the cost of that same DQC family. The original protocol requires O(Lm) circuit evaluations for L training steps and m collocation points; their "trainable observable" variant scales as O(dm) with `d ∈ poly(n), d ≪ L`, and their "flipped model" scales as O(L log m), with up to ~100-fold reductions in circuit evaluations in one- and two-dimensional benchmark simulations.[7] The savings are in circuit evaluations, not in a demonstrated hardware speedup.[7]

### 3. Born-machine-style and structured ansätze for nonlinear PDEs

Sarma, Watts, Moosa, Liu, and McMahon (arXiv:2311.01531, v2 24 Jun 2024) introduce a ZGR-QFT ansatz whose expressibility as a function of parameter count is stated to be invariant with respect to qubit number n for a fixed number of Fourier modes m, and benchmark several out-of-the-box optimizers.[8]
Their classical simulations use up to n = 12 ansatz qubits with 2^n grid points: nonlinear Black–Scholes with 8 ansatz qubits, the two-dimensional linear Black–Scholes equation with 6 ansatz qubits per degree of freedom (12 total), the Buckmaster equation with 5, and the deterministic KPZ equation with 5; a proof-of-concept on an 11-qubit IonQ trapped-ion device used only 2 ansatz qubits to estimate two expectation values for one timestep.[8]
They report that BFGS, truncated Newton, and ADAM require a budget on the order of `p·10⁴` cost-function evaluations (p = parameter count), that Hadamard-test shot error scales as 1/√M, and that obtaining exponential precision with the Hadamard test makes n > 10 impractical on current hardware and likely rules out quantum advantage.[8]
They list as open problems the design of ansätze that are unchaotic, low-parameter, expressive, trainable, and classically intractable.[8]

Joo and Moon (arXiv:2109.09216, v1 19 Sep 2021) describe a QuVa PDE solver in which ansatz parameters and PDE information are passed to a quantum part using system, control, and ancillary qubits, with control-qubit expectation values feeding a machine-learning classical part.[12]

Wang, Xue, et al. (arXiv:2507.19861, v2 01 Aug 2025) use a QCBM not as a direct PDE solution state but as a "Q-prior" inside a quantum-informed machine-learning architecture for chaotic dynamical systems, trained on superconducting hardware with Adam and an MMD-based loss.[13] Their QCBM instances use 10 qubits (1024 basis states) with 120 trainable parameters in one experiment and 180 in another, over 500 training epochs, with reported memory efficiency from representing invariant statistics in a compact parameter set.[13] The paper states that the QCBM avoids the data-loading bottleneck because it does not encode raw data into quantum states.[13]

### 4. Quantum-informed / PINN hybrids

Farea, Khan, et al. (arXiv:2503.16678, v5 05 Aug 2025) propose QCPINN, combining discrete-variable qubit circuits (alternate, cascade, cross-mesh, layered topologies; amplitude and angle embedding) and continuous-variable circuits with a classical PINN backbone, evaluated on five benchmark PDEs.[14] They report comparable accuracy and stable convergence with approximately 10% of the trainable parameters of classical PINNs, a 40% reduction in relative L₂ error for the convection–diffusion equation, and losses 2–3 times lower in some configurations; their physics loss combines PDE residual, boundary, and initial-condition terms with weights λ₁ = 1.0, λ₂ = 10.0, λ₃ = 10.0.[14]

Zabihi, Montazeri, and Suiker (arXiv:2606.04679) report that HQPINN advantages over classical PINNs are problem-dependent and explicitly limit their conclusions to one PQC family, one encoding strategy, and simulation without calibrated hardware noise.[18]

Leong, Ewe, et al. (arXiv:2503.02202, v2 01 Aug 2025) benchmark hybrid quantum-classical PINNs (HQPINN) on high-speed flows, with a parameterized quantum circuit in parallel with a classical network under a physics-informed loss.[15] Their stated negative result is that quantum-only PINNs are efficient on harmonic problems (lowest loss with minimal parameters, attributed to Fourier structure) but struggle to generalize to non-harmonic settings with shocks and discontinuities, exhibiting overshoot and under-damped oscillations; the hybrid model reduces those artifacts and matches classical models with sufficient parameterization.[15]

Klement, Eyring, and Schwabe (arXiv:2601.15046, v2 16 Apr 2026) test simulated qPINNs against fully classical PINNs with 150 trainable parameters and extend training to 2×10⁴ epochs for qPINNs and 10⁶ for classical PINNs.[16] Their stated finding is that the quantum advantage in state-vector simulation is faster convergence in training epochs, not a lower asymptotic error: both networks reach comparable approximations after sufficient epochs.[16]

A hydrological HQC-PINN (arXiv:2604.09374, v1 10 Apr 2026) encodes 25 remote-sensing features by angle encoding into n_q ∈ {4, 8} qubits, uses a hardware-efficient variational ansatz with `G = L(3n_q − 1)` gates (69 gates for n_q = 8, L = 3), and constrains the output with Saint-Venant shallow-water and Manning differentiable loss terms plus a data loss.[17]

### 5. Variational quantum CFD

Rung (arXiv:2209.04915, v1 11 Sep 2022) is a perspective on VQCFD built on the QNPU of Lubasch et al.: multigrid renormalization encodes a coordinate-space trial function with M = 2^N discrete values into N qubits, with boundary and design conditions encoded in the PDE formulation and cost function.[2] The paper states that VQCFD scales at least polynomially better with Reynolds number than corresponding classical Navier–Stokes simulations for paradigmatic two- and three-dimensional flows, while explicitly stating that an exponential quantum speedup may be possible but has not been proven.[2] It describes the method as a structure-resolving reduced-order model whose restricted trial-function set is physics-based rather than data-driven.[2]

Song, Deaton, et al. (arXiv:2406.00280, v2 09 Oct 2024) combine Chorin pressure projection with a variational solve of the pressure Poisson equation, using a VQE formulation with HTree readout stated to scale linearly in qubit count for real-valued problems.[4] In state-vector simulation at Re = 100 they report a cost of 10⁻⁴ and 99.99% state fidelity within 700 iterations, and 4×10⁻⁸ cost with 1 − F = 4.14×10⁻⁸ at 1600 iterations; L-BFGS-B consistently outperformed COBYLA and Adam in state-vector simulation, while COBYLA was used for noisy QASM simulations.[4] The hardware result is negative: on the 133-qubit `ibm_torino` processor the maximal residual stayed around 10⁻¹ and convergence was not reached, and results on `Fake_Kolkata` likewise showed noise-limited behavior.[4]

Baumann, Mohdeb, Randrianarisoa, Katz, Boyle, and Holweck (arXiv:2601.04439, v1 7 Jan 2026) report H-DES runs on IBM Heron R2 devices (`ibm_fez`, `ibm_marrakesh`) in the 156-qubit class, solving a one-dimensional material-deformation problem and the inviscid Burgers equation.[9] The ansatz is a stacked hardware-efficient circuit of Rx rotations with CZ/ECR entanglers, and the solution is encoded by a Chebyshev spectral decomposition; loss combines the differential-equation residual with weighted boundary-condition penalties, with a floating-boundary shift as an alternative.[9] Case 1 used 40 qubits with 30 collocation points in x and 51 in t under a four-stage shot schedule; case 2 used 50 qubits, four ansatz layers, and 10000 shots per function evaluation, reaching a best loss of 5.27×10⁻² after 10 iterations (160 evaluations).[9]

A 2026 review by Cecile, van Hülst, et al. (arXiv:2609.39962, v1 30 Sep 2026) collects the resource estimates and states that quantum advantage in CFD cannot be assessed from asymptotic complexity alone because the full workflow must include state preparation, operator construction, circuit width and depth, measurement cost, and output readout.[10] It reports that a fully fault-tolerant QLSA-based Navier–Stokes study (Zhuang et al. 2025) projects roughly 8.71×10⁶ physical qubits and 42.6 days for a 2^40 × 2^40 grid (~1.21×10²⁴ cells) at a physical error rate of 5×10⁻⁴, against an estimated classical runtime of ~130 years, while VQA work on early fault-tolerant architectures assumes ~10⁴ physical qubits and two-qubit error rates around 10⁻³.[10] The review's stated position is that current VQCFD performance models still require substantial reductions in circuit-execution, measurement, and optimization cost before an end-to-end crossover.[10]

A block-wise VQA variant (arXiv:2609.36710, v1 29 Sep 2026) decomposes the domain into independently trained blocks with n_i qubits and N_i = 2^{n_i} amplitudes each, coupled by jump penalties on values, derivatives, and physical fluxes, with an error analysis separating discretization, expressivity, optimization, shot noise, reblocking, and interface terms.[25] Its experiments are explicitly described as reproducible residual-based simulations and "not a gate-level simulation," with conditionally stated utility.[25]

### 6. Boundary and initial condition enforcement

Over, Bengoechea, Rung, and Clerici (arXiv:2402.18619, v2 29 May 2024) formulate second-order initial-boundary value problems as an optimal-control problem with a modular control-to-state ansatz and use a Hadamard-test-based structure with single-qubit measurements.[6] They report full-quantum implementation of arbitrary Dirichlet, Neumann, and Robin conditions via a correction technique that imposes no extra constraints on the trial functions, restricted to unsteady one-dimensional problems, with implicit backward Euler or a second-order three-time-level scheme and an explicit four-qubit boundary-circuit example.[6]

Lu, Wang, Liu, Li, Ewe, and Zhuang (arXiv:2310.11764, v2 03 Nov 2023) present the LSBT algorithm for general boundary conditions in discretized PDEs, using n = log₂N qubits for N degrees of freedom and stating that each prescribed boundary condition at a single node entails evaluation of an expectation ⟨ϕ(θ)|K_bc|ϕ(θ)⟩.[11]

## Comparison of demonstrated losses, ansätze, optimizers, and sizes

| Approach (paper) | Loss / objective | Ansatz / encoding | Classical optimizer | Demonstrated size |
|---|---|---|---|---|
| VQLS, Bravo-Prieto et al. [26] | Global/local overlap `1 − |⟨b|A|ψ⟩|²/⟨ψ|A†A|ψ⟩` | `V(α)|0⟩`, LCU `A=Σc_lA_l`, Hadamard tests | Heuristic random-direction search | 1024×1024 on Rigetti; 2^50×2^50 numerics (50 qubits) |
| VQLS block-encoding, Dsouza & Singhal [21] | Global cost `C_G`, exact Laplacian block encoding | Block encoding with `3+⌈log₂D⌉` ancillas | COBYLA, CMA-ES | N = 4 to 1024 |
| PDE VQA, García-Molina et al. [22] | Energy / potential + boundary terms | ZGR and R_Y, depth 1–2; Fourier interpolation | COBYLA, SPSA, ADAM, L-BFGS-B | 2–6 qubits; infidelity 10⁻⁵–10⁻⁴ |
| Poisson VQLS benchmark, Ali & Kabel [20] | Four cost variants (global, local, norm-based) | Linear alternating R_Y–CZ | BFGS, SPSA, Powell, NFT | 3–5 qubits; ~98% sim vs ~17% hardware fidelity |
| Empirical VQLS, Turati et al. [19] | Global and local cost | Fixed ansatz, varied layers | Tested across optimizers | 2, 4, 6, 8, 10 qubits; Ising / random Pauli / Darcy |
| DQC, Kyriienko et al. [3] | PDE residual from auto-differentiated observables | Feature map + hardware-efficient, depth 5–6 | Adam (lr 0.01–0.02, ~200 iters) | Six-qubit register examples; 20–100 collocation points |
| DQC cost reduction, Paine et al. [7] | Same DQC loss, engineered cost operators | Trainable observable; flipped model | Not the paper's focus | 1D/2D benchmarks; up to ~100× fewer evaluations |
| Nonlinear PDE ansatz, Sarma et al. [8] | Cost from Hadamard-tested expectation values | ZGR-QFT, adder + diagonal gates | BFGS, truncated Newton, ADAM | up to n=12 (2^n grid); 2-qubit IonQ demo |
| QCBM Q-prior, Wang et al. [13] | MMD + hybrid training loss | QCBM, 10 qubits, 120/180 params | Adam | 10 qubits / 1024 states; 500 epochs |
| QCPINN, Farea et al. [14] | PDE residual + boundary + initial penalties (λ=1,10,10) | DV and CV QNN + classical PINN | Hybrid training | Five benchmark PDEs; ~10% of classical params |
| HQPINN, Leong et al. [15] | Physics-informed loss | PQC parallel to classical NN | Hybrid training | Smooth and shock/transonic flow cases |
| qPINN convergence, Klement et al. [16] | PINN MSE loss | Quantum + classical layers, 150 params | Gradient-based | 2×10⁴ vs 10⁶ training epochs |
| VQCFD pressure Poisson, Song et al. [4] | Rayleigh–Ritz energy / residual `1−|⟨ψ|ψ_g⟩|²` | HEA, HTree readout | L-BFGS-B (sim), COBYLA (noise) | Re=100 Poisson subproblem; no hardware convergence |
| H-DES on hardware, Baumann et al. [9] | PDE residual + weighted BC penalty / floating shift | Stacked HEA, Chebyshev spectral | Multi-stage shot schedule | 40 and 50 qubits on ibm_fez; 10⁴ shots/eval |

## Limitations, negative results, and unverified claims

- **Barren plateaus and trainability.** The original VQLS paper states that global costs can have gradients vanishing exponentially in n and motivates local costs on that basis; the block-encoding VQLS study measures the gradient variance decay directly, and the empirical VQLS study cites barren plateaus as a well-known VQA issue.[26][21][19]
- **Shot cost and the output bottleneck.** VQLS cost estimation relies on O(L²) Hadamard-test circuits in the Pauli-LCU formulation (O(nL²) for the local cost), and the original authors state that finite sampling was excluded from their scaling heuristics and that verification shot noise scales with ε and κ.[26][19]
The block-encoding study states that reading out the full classical solution vector requires O(N) measurements, which eliminates an exponential speedup, and the Poisson benchmark finds at-best square-root convergence in shot count.[21][20]
- **Hardware noise stops convergence.** The incompressible Navier–Stokes hybrid solver did not converge on `ibm_torino`, with the maximal residual stuck near 10⁻¹, and Ali and Kabel report roughly 17% fidelity on `ibmq_ehningen` where simulation achieved above 98%.[4][20]
- **Expressibility–trainability tradeoff.** The advection–diffusion benchmark reports an expressivity wall where deeper circuits are needed as N grows while adding layers enlarges the parameter space in which barren plateaus arise; Sarma et al. state the Hadamard-test shot requirement likely rules out quantum advantage and leave suitable ansatz design as an open problem.[24][8]
- **Problem structure matters.** The empirical VQLS study finds good results on Ising instances but poor results on random Pauli and Darcy matrices, including cosine similarities below 0.5 at 10 qubits, and warns that deeper ansätze were not explored because of prohibitive computational cost.[19]
- **Negative PINN results.** Quantum-only PINNs show overshoot and oscillations on shocks and discontinuities, and the qPINN convergence advantage is in epochs at matched simulation conditions, with comparable final approximations after sufficient classical training.[15][16]
- **No proven speedup.** The VQCFD perspective explicitly states that an exponential quantum speedup has not been proven, and the 2026 CFD review states that advantage cannot be judged from asymptotic complexity alone and that substantial cost reductions are still required.[2][10]
- **Unverified / conditional.** Large-grid advantage thresholds in the VQLS framework paper (N > 750, > 2×10⁵, > 2×10¹²) are analytic comparisons, not demonstrations on hardware.[23]
The ~8.71×10⁶-physical-qubit, 42.6-day Navier–Stokes projection is a fault-tolerant resource model reported by the 2026 review, not a demonstrated variational solve.[10]
The block-wise PDE results are residual-based simulations, not gate-level executions.[25]
The qPINN epoch advantage is reported in state-vector simulation, not on hardware.[16]

## Primitives required by this literature

This section is a factual inventory of the primitives the cited papers use; it does not advocate building any of them.

- **State preparation / encoding.** Amplitude encoding for VQLS linear systems; multigrid renormalization encoding 2^N grid values into N qubits for VQCFD; Chebyshev spectral and basis-state encoding in H-DES; angle encoding of continuous features in QPINN hybrids; data feature maps (`U(x)`) in DQC.[26][2][9]
- **Parameterized circuit families.** Hardware-efficient ansätze with rotation and entangler layers, ZGR and ZGR-QFT Fourier ansätze, R_Y–CZ alternating layers, QAOA-style ansätze, tensor-product / tensor-network-inspired families, trainable-observable and "flipped" DQC models, and QCBM circuits.[26][8][7]
- **Linear-algebra subroutines.** LCU decomposition of A, exact block encoding of the discrete Laplacian, Hadamard tests for overlap and expectation estimation, Pauli-string decomposition and pre-measurement rotations, controlled adders and diagonal gates, QFT-based gate synthesis, and quantum phase-estimation-style amplitude-estimation variants discussed as accuracy improvements.[26][21][8]
- **Differentiation primitives.** Parameter-shift / automatic differentiation of quantum circuits with respect to both variational parameters and input variables, giving derivative circuits for PDE residuals in DQC.[3][7]
- **Loss construction.** Global and local overlap costs, Rayleigh–Ritz ground-state energies, and collocation-point PDE residuals appear in VQLS, VQE, and DQC formulations.[26][4][3]
- **Boundary and interface terms.** Boundary/initial-condition penalties, floating boundary shifts, jump penalties between blocks, and MMD-style Born-machine losses.[9][25][13]
- **Readout and measurement.** Single-qubit expectation readout, HTree readout for real-valued fields, and finite-shot sampling schedules with 10³–10⁴ shots per function evaluation appear in hardware work.[2][4][9]
- **Randomized measurement.** Classical shadows and randomized measurement are used in the cost-reduced DQC protocols.[7]
- **Classical optimizers.** BFGS, L-BFGS/L-BFGS-B, COBYLA, SPSA, Powell, NFT, Adam, CMA-ES, and gradient-free random-direction search are used across the benchmark and hardware papers.[20][21][3]
- **Optimization schedules.** Multi-stage shot schedules are used in hardware H-DES runs.[9][26]
- **Error mitigation and compilation.** Probabilistic error cancellation, noise-aware transpilation, dynamical decoupling, readout mitigation, transpilation with connectivity-aware qubit allocation, and shot-noise-aware cost estimation appear in the benchmark and hardware papers.[20][4][9]
- **Hybrid classical loop and output handling.** A classical optimizer iterating over circuit parameters, classical post-processing of expectation values into loss values, and (where a classical field is required) O(N) measurements to reconstruct the solution vector.[26][21]

## Sources

[1] https://arxiv.org/abs/1909.05820 — Bravo-Prieto et al., Variational Quantum Linear Solver (arXiv:1909.05820)
[2] https://arxiv.org/html/2209.04915 — Variational Quantum Algorithms for Computational Fluid Dynamics (arXiv:2209.04915)
[3] https://arxiv.org/html/2011.10395 — Solving nonlinear differential equations with differentiable quantum circuits (arXiv:2011.10395)
[4] https://arxiv.org/html/2406.00280v2 — Incompressible Navier-Stokes solve on noisy quantum hardware (arXiv:2406.00280)
[5] https://arxiv.org/html/2410.01130v2 — H-DES: a Quantum-Classical Hybrid Differential Equation Solver (arXiv:2410.01130)
[6] https://arxiv.org/html/2402.18619 — Boundary Treatment for Variational Quantum Simulations of PDEs (arXiv:2402.18619)
[7] https://arxiv.org/html/2503.22656 — Differential equation quantum solvers: engineering measurements to reduce cost (arXiv:2503.22656)
[8] https://arxiv.org/html/2311.01531 — Quantum variational solving of nonlinear and multi-dimensional PDEs (arXiv:2311.01531)
[9] https://arxiv.org/pdf/2601.04439 — Solving nonlinear differential equations on noisy 156-qubit quantum computers (arXiv:2601.04439)
[10] https://arxiv.org/html/2609.39962 — Quantum Algorithms for Computational Fluid Dynamics review (arXiv:2609.39962)
[11] https://arxiv.org/pdf/2310.11764 — General Boundary Conditions in Discretized PDEs in VQA (arXiv:2310.11764)
[12] https://arxiv.org/html/2109.09216 — Quantum variational PDE solver with machine learning (arXiv:2109.09216)
[13] https://arxiv.org/html/2507.19861v2 — Quantum-Informed Machine Learning for Chaotic Systems (QCBM) arXiv:2507.19861
[14] https://arxiv.org/html/2503.16678v5 — QCPINN: Quantum-Classical PINNs for Solving PDEs arXiv:2503.16678
[15] https://arxiv.org/html/2503.02202v2 — Hybrid Quantum PINN for High-Speed Flows arXiv:2503.02202
[16] https://arxiv.org/html/2601.15046 — Quantum-Enhanced Convergence of PINNs arXiv:2601.15046
[17] https://arxiv.org/html/2604.09374v1 — Variational Quantum PINNs for Hydrological PDEs arXiv:2604.09374
[18] https://arxiv.org/html/2606.04679 — Hybrid quantum-classical PINNs for nonlinear PDEs arXiv:2606.04679
[19] https://arxiv.org/html/2409.06339 — An Empirical Analysis on the Effectiveness of VQLS arXiv:2409.06339
[20] https://arxiv.org/html/2211.14064 — Performance Study of VQAs for Poisson Equation arXiv:2211.14064
[21] https://arxiv.org/html/2608.19655 — VQLS via Block Encoding for the Poisson Equation arXiv:2608.19655
[22] https://arxiv.org/html/2104.02668v2 — Solving partial differential equations in quantum computers arXiv:2104.02668
[23] https://arxiv.org/html/2405.16651v2 — Variational Quantum Framework for PDEs arXiv:2405.16651
[24] https://arxiv.org/html/2503.24045 — Performance evaluation of VQE on advection-diffusion arXiv:2503.24045
[25] https://arxiv.org/html/2609.36710 — Block-Wise VQAs for PDEs with Interface Penalty Constraints arXiv:2609.36710
[26] https://arxiv.org/pdf/1909.05820 — Bravo-Prieto et al. VQLS paper PDF arXiv:1909.05820
