# From a discretised PDE to quantum circuits: what the primary literature establishes

## Scope

This report covers the route in which a linear PDE is discretised by finite differences (FDM), finite elements (FEM), or a spectral method, producing either a static system $Au=b$ (elliptic problems, or implicit time stepping) or a semi-discrete evolution $\dot u = Au$ (parabolic/hyperbolic problems). It reports what primary papers state about the quantum algorithms applied after that discretisation: HHL and later quantum linear-system algorithms (QLSAs), QSVT/qubitization solvers, Hamiltonian simulation of the semi-discrete dynamics, and imaginary-time / diffusion treatments. The emphasis is on proved complexity scaling, the roles of sparsity $d$ (or $s$) and condition number $\kappa$, state preparation, and measurement/readout. Statements are attributed to the paper that makes them; very recent preprints are labelled as such. This is a research record only: it makes no recommendation about what to implement.

The output model differs from classical PDE solving in a way that drives most of the results below. A QLSA outputs a normalised quantum state $|x\rangle$ on $O(\log N)$ qubits whose amplitudes are proportional to the solution, not the vector $x$ itself [1][2]. Harrow's review states the input side explicitly: $A$ must be row-computable and $b$ must be supplied as a quantum state with preparation time $T_B$ [2]. Aaronson's "read the fine print" note enumerates the four assumptions that must all hold for the logarithmic scaling to survive: preparation of $|b\rangle$, efficient $e^{-iAt}$, small $\kappa$, and measurement of only a summary of $x$ [32].

## Findings by algorithm family

### 1. HHL and the first QLSA generation

Harrow, Hassidim and Lloyd (2009) consider a Hermitian $N\times N$ matrix $A$ that is $s$-sparse and efficiently row-computable, with eigenvalues in $[1/\kappa,1]\cup[-1,-1/\kappa]$, and derive a runtime $\tilde O(\log(N)s^2\kappa^2/\epsilon)$ to output a state within distance $\epsilon$ of $|x\rangle$ [1]. Their classical comparator is the conjugate-gradient method at $O(Ns\sqrt{\kappa}\log(1/\epsilon))$ for positive-definite $A$ and $O(Ns\kappa\log(1/\epsilon))$ otherwise; the paper states that "the greatest advantage our algorithm has over classical algorithms occurs when both $\kappa$ and $1/\kappa$ are poly log$(N)$" [1]. It also proves BQP-completeness of sparse well-conditioned matrix inversion (condition number $O(T^2)$ if positive definite, $O(T)$ otherwise) and concludes that substantially improving the $\kappa$ or $\epsilon$ dependence would collapse standard complexity separations [1].

Harrow's 2014/2015 review records two refined runtimes, $\tilde O(\kappa T_B + \log(N)s^2\kappa^2 T_A/\epsilon)$ and $\tilde O(\kappa T_B + \log(N)s^2\kappa T_A/\epsilon^3)$, with Hamiltonian simulation as the key subroutine [2]. The same review gives the first explicit PDE caveat: for a finite-difference discretisation of Poisson's equation on an $L\times L\times L$ box, $\kappa$ scales as $L^2$ while the number of points is $O(L^3)$, "so the quantum algorithm cannot achieve a substantial speedup" [2]. Preconditioning, pseudoinversion of the well-conditioned subspace, sampling from $|x\rangle$, and computing $f(A)|b\rangle$ are listed in the original HHL discussion as extensions [1].

Childs, Kothari and Somma (2015; SIAM J. Comput. 2017) replace phase estimation with Fourier- or Chebyshev-series representations of $A^{-1}$ and obtain $\mathrm{poly}(\log N,\log(1/\epsilon),d,1/\kappa)$ resources, an exponential improvement in the precision dependence [3]. Their Theorem 2 restates HHL as $O(d\kappa^2/\epsilon\,\mathrm{poly}(\log(d\kappa/\epsilon)))$ queries to the matrix oracle, their Fourier approach uses $O(d\kappa^2\log^{2.5}(\kappa/\epsilon))$ queries, and their variable-time-amplitude-amplification variant gets $O(d\kappa\,\mathrm{poly}(\log(d\kappa/\epsilon)))$, i.e. near-linear $\kappa$ [3]. They explicitly caution that "the QLSP is not the same as the traditional problem of solving a linear system of equations" [3]. In the same paper, the expected success-amplitude factor $1/\|A^{-1}|b\rangle\|$ is identified as a source of overhead that later solvers address.

### 2. Optimal and post-optimal linear-system solvers

Costa, An, Sanders, Su, Babbush and Berry (arXiv 2021; PRX Quantum 3, 040303, 2022) combine a discrete adiabatic theorem with a qubitized quantum walk to reach $O(\kappa\log(1/\epsilon))$ in terms of calls to a block encoding of $A$ and preparation of $|b\rangle$; the paper calls this strictly linear in $\kappa$, matching a known lower bound and optimal in the combined $(\kappa,\epsilon)$ scaling [36]. Tong, An, Wiebe and Lin (2020) introduce "fast inversion" to block-encode $A^{-1}$ through arithmetic on eigenvalues, with cost dependent on block-encoding subnormalisation factors rather than on $\|A\|$; they also state the generic $\Omega(\kappa(A))$ cost lower bound for QLSP [13].

Two very recent 2026 preprints extend this picture and should be read as unconfirmed-by-peer-review at the time of writing. Bravo-Prieto, Harrow and Kothari (arXiv:2609.35660, verified here on the arXiv HTML page) claim the tight bound $\Theta(\kappa\sqrt{d}\log(1/\epsilon))$ query complexity for the sparse-oracle QLSP, together with a square-root-sparsity block encoding using $O(\sqrt d\log(1/\eta))$ oracle queries; they cite prior lower bounds $\Omega(\kappa\log(1/\epsilon))$ and $\Omega(\kappa\sqrt d)$ [5]. Dalzell, Li and Su (arXiv:2607.07691) describe solvers whose cost is governed by an effective condition number $\kappa_{\mathrm{eff}}$ instead of the spectral $\kappa$, with $O(\kappa_{\mathrm{eff}}\,\mathrm{poly}\log(\kappa_{\mathrm{eff}}/\epsilon))$ queries to the block encoding and a filtering solver whose leading-order query count is $3\|A^{-1\dagger}x\|/(\|x\|\epsilon)$ when $\|x\|$ is known [9]. Both statements are reproduced as the papers give them; I did not find independent confirmation beyond those arXiv pages.

On the empirical side, Lefterovici, Perk, Ramacciotti, Rotundo, Skelton and Steinbach (2025) compared HHL, two LCU solvers and a QSVT solver under identical sparse-oracle assumptions on random, MIPLIB-derived and Poisson instances, and report that "HHL underperforms in comparison to the other methods across all data sets, often by orders of magnitude, while the QSVT-based method shows the best performance" [10]. This is an instance-counting study, not an asymptotic theorem, and it explicitly does not include an end-to-end cost analysis [10].

### 3. Elliptic PDEs: FDM, FEM, spectral methods, preconditioning

Childs, Liu and Ostrander (2021, Quantum 5, 574) improve previous quantum FDM/FEM/FVM algorithms from $\mathrm{poly}(1/\epsilon)$ to $\mathrm{poly}(d,\log(1/\epsilon))$: an adaptive-order FDM algorithm for Poisson with informal complexity $d^{6.5}\mathrm{poly}(\log d,\log(1/\epsilon))$ and a spectral algorithm for general second-order elliptic equations with $d^2\mathrm{poly}(\log(1/\epsilon))$ [15]. They note that outputting all entries of an $N$-vector from $|x\rangle$ takes $\Omega(N)$ time and space even on a quantum computer, and their stated complexities are for preparing the solution state, not for estimating a functional of it [15].

Montanaro and Pallister (arXiv 2015; Phys. Rev. A 93, 032324, 2016) analysed the FEM end-to-end against conjugate gradient, explicitly tying the discretisation error to the size of the linear system. For regular meshes they give $\kappa=O(N^{2/d})$; with linear basis functions and no preconditioning their Table 1 lists classical $\tilde O((\|u\|_2/\epsilon)^{(d+1)/2})$ against quantum $\tilde O(\|u\|\|u\|_2^2/\epsilon^3 + \|u\|_1\|u\|_2/\epsilon^2)$, and with optimal preconditioning classical $\tilde O((\|u\|_2/\epsilon)^{d/2})$ against quantum $\tilde O(\|u\|_1/\epsilon)$ [34]. Their conclusion is that the speedup is at most polynomial when $d$ is fixed and the solution smooth — approximately quadratic at $d=4$ — that the quantum runtime can be worse in low dimension, and that including the accuracy parameter makes an apparent exponential advantage disappear [34]. They also give barriers: an inverse-power dependence of runtime on the state-distinguishing accuracy, at most quadratic speedup for oracle-input instances, and evidence that the FEM-solving subroutine admits at most polynomial classical slowdown in fixed dimension for smooth solutions [34].

Deiml and Peterseim (arXiv 2024, v4 2025) revisit FEM with a BPX multilevel preconditioner adapted to the quantum setting. They state that unpreconditioned conditioning grows as $\mathrm{tol}^{-2}$ under $H^2$-regularity, so a linear-in-$\kappa$ QLSA gives total $\mathrm{tol}^{-3}$ and quantum advantage only for $d>3$; their BPX-based construction claims $O(\mathrm{tol}^{-1})$ up to logarithms for suitable functionals without regularity assumptions, with advantage already for $d>1$ [39]. Jin, Liu, Ma and Yu (arXiv 2505.06866, v2 2026) apply BPX to FDM Poisson via Schrödingerization, obtaining an exact block encoding of the symmetrised preconditioned matrix $A_S=S^\top A S$ with normalisation $O(d^2(L+1))$ for $L$ levels and an overall query complexity $O(\mathrm{poly}(d)\,\epsilon^{-1}\mathrm{polylog}(\epsilon^{-1}))$ for linear functionals [19]. That paper explicitly says its query count excludes "the additional sampling overhead required for readout" [19]. Tong et al.'s fast-inversion construction is another route to making the cost independent of the large norm of an unbounded discretised operator such as the Laplacian [13].

### 4. Semi-discrete evolution and Hamiltonian simulation

Costa, Jordan and Ostrander (arXiv 2017; Phys. Rev. A 99, 012323, 2019) give a wave-equation solver for Dirichlet and Neumann boundaries that relies on hypergraph-incidence factorisations of higher-order Laplacians. For a region of diameter $\ell$ in $D$ dimensions with lattice spacing $a$, they state state-preparation time $\tilde O(D^{5/2}\ell/a)$ and Hamiltonian-simulation time $\tilde O(TD^2/a)$, against a classical lower bound $\Omega((\ell/a)^D)$ for algorithms outputting the full field [42]. With an $s$-sparse Hamiltonian the discretisation error scales as $T a^{2(s/D)-2}$, so $a\sim(\epsilon/T)^{D/2(s-D)}$; they also treat Klein–Gordon and Maxwell equations [42].

At the circuit level, Low and Chuang (arXiv 2016; Quantum 3, 163, 2019) show qubitization simulates $e^{-iHt}$ for a block-encoded Hamiltonian with $O(t+\log(1/\epsilon))$ queries to both prep oracles, optimal in all parameters and using at most two ancilla qubits; the paper notes that the sparse-oracle formulation doubles the system qubits and that its black-box oracles can be difficult to realise [37]. Sato, Kondo, Hamamura, Onodera and Yamamoto (arXiv 2402.18398, 2024; Phys. Rev. Research 6, 033246) construct explicit finite-difference gate decompositions for hyperbolic PDEs; their abstract reports $d n$ qubits and $O(d n^3T^2/\epsilon)$ or $O(d n^{2.5}T^{1.5}/\epsilon^{0.5})$ non-local gates on a $d$-dimensional lattice with $2^n$ nodes per dimension [25].

For dynamics that are not unitary, the Schrödingerization line maps linear ODE/PDE systems to a Schrödinger-type system in one higher dimension via a warped phase transform. Jin, Liu, Li and Yu (2023) apply this to artificial boundary conditions (complex absorbing potentials, perfectly matched layers, Dirichlet-to-Neumann); they replace the eigenvector-matrix condition number of earlier time-marching analyses by $C(A)=\sup_{t\in[0,T]}\|\exp(At)\|$, which equals 1 for the dissipative ABC matrices considered and removes that condition-number factor [21]. Jin, Liu and Yu (2024) apply the same transformation to Fokker–Planck: for the conservation form they favour semi-discretisation-based Schrödingerization with non-periodic boundaries; they report that for unstable systems direct Schrödingerization acts as a stabilisation, and give an explicit time-splitting circuit for the heat-equation form with a comparison of Bell- versus Fourier-basis diagonalisation [24]. This is the literature's principal route for diffusion handled as evolution rather than as matrix inversion.

An, Liu, Wang and Zhao (arXiv 2211.05246) prove limits and fast-forwarding for quantum ODE solvers. They quote the best QSP/QSVT Hamiltonian-simulation scaling as $O(T\|A\|+\log(1/\epsilon)/\log\log(1/\epsilon))$; for homogeneous linear ODEs they show exponential overhead when the coefficient matrix has a pair of eigenvalues with different real parts, and linear overhead from non-normality [41]. In the other direction they obtain a quadratic improvement in $T$ for inhomogeneous ODEs with negative semi-definite coefficient matrices and an exponential improvement in both $T$ and $\|A\|$ when the eigensystem is efficiently implementable, explicitly including spatially discretised linear evolutionary PDEs [41]. Shang, Guo, An and Zhao (arXiv 2410.19628, v3 2025) encode general linear ODEs into non-diagonal blocks of density matrices evolved by Lindbladians and report near-optimal parameter dependence under their input model; they also restate the no-go result that generic solvers for non-dissipative (growing) ODEs face $e^{\Omega(T)}$ worst-case cost [4].

### 5. Imaginary-time and diffusion handling

Huang, Nishi, Kosugi, Kawada and Matsushita (arXiv 2409.18559, v2 2025) discretise the advection–diffusion–reaction equation and apply an approximate probabilistic imaginary-time evolution (PITE) operator, $u(t)=e^{-t\mathcal H}u_0$, block-encoded with a single ancilla. Their approximate PITE gate complexity is $O(\epsilon^{-1-\alpha}\mathrm{polylog}\,N)$ with $\alpha\in(0,1]$; they claim exponential improvement in matrix size $N$ over QLSA at the cost of worse $\epsilon$ dependence, and explicitly exclude state preparation and readout from that comparison [26]. The same paper states that for $d$-dimensional second-order linear PDEs $\kappa=O(dN^{2/d})$, so generic QLSAs give $O(N^{2/d}\mathrm{polylog}\,N\log(1/\epsilon))$ versus sparse CG at $O(N^{(d+1)/d}\log(1/\epsilon))$ [26].

Other imaginary-time papers make narrower or more qualified claims. A QSP-based QITE preprint (arXiv 2507.00908) states polynomial gate complexity in imaginary-time duration with one ancilla and success probability close to one [27]. A stochastic Riemannian gradient-flow study (arXiv 2504.06123) gives an overall runtime $O(\beta\|H\|_\infty D^2\exp(8\beta\|H\|_\infty))$, which it says cannot be efficient in general for complexity reasons, with $D=2^{2N}-1$ [29]. A 2026 preprint reports obtaining imaginary-time evolution by analytic continuation of measured real-time correlation functions, with demonstrations including the one-dimensional Fokker–Planck diffusion equation on IBM hardware [31]. Arrazola, Kalajdzievski, Weedbrook and Lloyd (2018) take a different, continuous-variable route: they invert differential operators polynomial in positions and momenta by applying $e^{-i\hat A\hat X\hat Y}$ to an input state plus resource states and post-selecting zero-momentum outcomes on the ancillas; for fixed-order operators they state runtime polynomial in the dimension [16].

### 6. The measurement/readout bottleneck

The original HHL paper is already explicit: "Clearly, to read out all the components of $\vec x$ would require one to perform the procedure at least $N$ times" [1]. Aaronson (Nature Physics 11, 291, 2015) restates this as the fourth caveat — learning any specific entry $x_i$ requires roughly $n$ repetitions, killing the exponential speedup — alongside the state-preparation, simulation and conditioning caveats [32]. Childs, Liu and Ostrander state that outputting all entries of an $N$-dimensional vector from its quantum state needs $\Omega(N)$ time and space [15].

Practical readout studies quantify the same divide. Huang, Nishi, Kawada, Zushi and Matsushita (arXiv 2511.20017, 2025) note that the cost of readout is "linear regarding the nonzero components of the state" and that this can eliminate the speedup; their real-space readout needs $O(N)$ repetitions, while Fourier-space readout applies the QFT and post-selects the most significant $n-m$ qubits to obtain the first $2^m$ coefficients, relying on smoothness and periodicity, with benchmark reconstructions still using $10^5$–$10^7$ shots [30]. A QSVT solution of time-dependent Maxwell's equations (Shaviner, Chen and Frankel, 2025) reports a 21st-degree polynomial approximation to $1/x$, 9 qubits including two ancillas, a 128-point grid and $>99.9\%$ fidelity in PennyLane simulation, while stating that "accessing the complete quantum state is infeasible" on hardware and restricting applicability to polynomially many observables [11].

An end-to-end negative result outside PDE but directly about QLSA readout: Pareek, Jayakumar, Coffrin and Misra (IEEE Trans. Power Systems, DOI 10.1109/TPWRS.2025.3613139) find HHL-based power flow at $O(N^{4.62}\log N)$ versus classical conjugate gradient at $O(N^{1.90}\log N)$ once condition-number growth and tomography readout are included; they take tomography cost as $O(N/\epsilon)$ and give a necessary condition $D\kappa^{3/2}\ll N s^{-1}$ for any potential advantage when $D$ components are read out [33]. The paper's conclusion is explicitly end-to-end, not about the solver kernel [33].

## Reported complexity and assumptions

All entries are as stated by the cited paper; $N$ is matrix size/grid count, $d$ or $s$ sparsity, $\kappa$ condition number, $\epsilon$ error.

| Algorithm / result | Reported scaling | Main assumptions and what is counted | Source |
|---|---|---|---|
| HHL (2009) | $\tilde O(\log N\,s^2\kappa^2/\epsilon)$ | Hermitian, $s$-sparse, row-computable; eigenvalues in $[1/\kappa,1]\cup[-1,-1/\kappa]$; outputs state, not vector | [1] |
| HHL refined (2009/2014) | $\tilde O(\kappa T_B+\log N\,s^2\kappa^2T_A/\epsilon)$; also $\tilde O(\kappa T_B+\log N\,s^2\kappa T_A/\epsilon^3)$ | Implicit matrix oracle, state-preparation time $T_B$ explicit | [2] |
| CKS LCU (2015/2017) | $\mathrm{poly}(\log N,\log(1/\epsilon),d,1/\kappa)$; near-linear variant $O(d\kappa\,\mathrm{poly}\log(d\kappa/\epsilon))$ | Sparse oracles; Fourier variant only needs Hamiltonian simulation of $A$ | [3] |
| Discrete adiabatic QLSP (2021/2022) | $O(\kappa\log(1/\epsilon))$ | Block encoding of $A$ plus $|b\rangle$ preparation; optimal in $\kappa,\epsilon$ | [36] |
| Optimal sparse QLSP (2026 preprint) | $\Theta(\kappa\sqrt d\log(1/\epsilon))$ | Sparse-matrix location/value oracles; query complexity only | [5] |
| Effective-condition-number solvers (2026 preprint) | $O(\kappa_{\mathrm{eff}}\,\mathrm{poly}\log(\kappa_{\mathrm{eff}}/\epsilon))$; filtering leading term $3\|A^{-1\dagger}x\|/(\|x\|\epsilon)$ | Block encoding of $A$, unitary for $|b\rangle$; known $\|x\|$ for the leading-order form | [9] |
| Generic QLSP lower bound | $\Omega(\kappa(A))$; also $\Omega(\kappa\log(1/\epsilon))$ in $\kappa,\epsilon$ | Generic well-conditioned input models | [13][36] |
| Adaptive FDM Poisson (2021) | $d^{6.5}\mathrm{poly}(\log d,\log(1/\epsilon))$ | Outputs solution state; global strict diagonal dominance; no functional estimation counted | [15] |
| Spectral elliptic PDE (2021) | $d^2\mathrm{poly}(\log(1/\epsilon))$ | Same; Dirichlet boundaries | [15] |
| FEM QLE (2016) | $\kappa=O(N^{2/d})$; quantum $\tilde O(\|u\|_1/\epsilon)$ with optimal preconditioning | Linear elements, regular meshes; functional of solution; accuracy tied to $N$ | [34] |
| BPX quantum FEM (2024/2025) | $O(\mathrm{tol}^{-1})$ up to logs (unpreconditioned route $\mathrm{tol}^{-3}$) | $d$-linear elements, Cartesian grids; functionals of solution | [39] |
| BPX + Schrödingerization FDM Poisson (2026) | $O(\mathrm{poly}(d)\,\epsilon^{-1}\mathrm{polylog}(\epsilon^{-1}))$ | Exact block encoding normalisation $O(d^2(L+1))$; readout excluded | [19] |
| Wave equation (2019) | Prep $\tilde O(D^{5/2}\ell/a)$, sim $\tilde O(TD^2/a)$; classical $\Omega((\ell/a)^D)$ | Dirichlet/Neumann; higher-order Laplacian factorisations | [42] |
| Qubitized Hamiltonian simulation (2019) | $O(t+\log(1/\epsilon))$ queries, 2 ancillas | Block-encoded Hamiltonian; oracle access to state $|G\rangle$ | [37] |
| Hyperbolic PDE circuits (2024) | $O(dn^3T^2/\epsilon)$ or $O(dn^{2.5}T^{1.5}/\epsilon^{0.5})$ gates | Finite differences, $2^n$ nodes per dimension; explicit gate construction | [25] |
| General linear ODE limits (2022) | Best simulation $O(T\|A\|+\log(1/\epsilon)/\log\log(1/\epsilon))$; exponential cost when real parts differ | Lower bounds for generic solvers; fast-forwarding needs special spectra | [41] |
| PITE advection–diffusion (2025) | $O(\epsilon^{-1-\alpha}\mathrm{polylog}\,N)$ | Gate complexity in $N$; preparation/readout excluded | [26] |
| Full-field readout | $\Omega(N)$ output; $O(N)$ repetitions for real-space sampling | General amplitude reconstruction from a quantum state | [15][30] |

## Limitations and negative results reported in the literature

- **Readout can erase the speedup.** The original HHL paper requires $\ge N$ repetitions to read all components; Aaronson calls this one of four caveats that can each be fatal; subsequent PDE-specific work repeats the $\Omega(N)$ output bound [1][32][15]. Recent readout papers distinguish full-grid reconstruction from smooth-function reconstruction in a Fourier basis, where efficiency depends on regularity and periodicity [30].
- **Condition number of the discretisation.** Finite-difference and finite-element discretisations of second-order elliptic operators have $\kappa$ growing polynomially in grid size — $O(L^2)$ in Harrow's 3D Poisson example, $O(N^{2/d})$ in the FEM and PITE analyses — so a linear-in-$\kappa$ QLSA yields only a polynomial gain in $N$ unless preconditioned [2][34][26].
- **Fixed-dimension FEM gains are polynomial.** Montanaro and Pallister find at most polynomial speedup for fixed $d$ and smooth solutions, ~quadratic at $d=4$, and possible quantum slow-down in low dimension; their lower-bound discussion constrains any improvement in this fixed-$d$ setting [34]. Deiml and Peterseim report that the same conclusion shifts with BPX preconditioning and no regularity assumption [39].
- **Non-unitary/non-normal dynamics carry provable overhead.** An, Liu, Wang and Zhao show exponential worst-case cost from a real-part spectral gap and linear cost from non-normality, and the Lindbladian-ODE paper cites an $e^{\Omega(T)}$ no-go for growing (non-dissipative) solutions [41][4].
- **Accuracy, not just dimension, must be counted.** Childs, Liu and Ostrander and Montanaro and Pallister both stress that earlier claims of exponential PDE speedup did not combine the $\epsilon$–$N$ relation with measurement cost [15][34].
- **Solver ranking is not the asymptotic ranking.** The functional-QLS comparison finds HHL worst and QSVT best on tested instances, including Poisson systems, while noting its study is a query-count benchmark rather than an end-to-end one [10].
- **End-to-end linear-systems negative result.** The power-flow study finds classical CG asymptotically faster than HHL once state preparation, readout and realistic condition-number growth are included [33].
- **Quantum-inspired classical competitors do not dequantise this PDE advantage, but that is a separate claim.** Wang, Adler and Hu derive upper $O(\mathrm{poly}(d)\epsilon^{-(d+2)}\log(1/\epsilon))$ and lower $\Omega(\mathrm{poly}(d)\epsilon^{-d}\log(1/\epsilon))$ bounds for randomised coordinate descent on FEM Poisson, plus a communication-complexity lower bound, concluding that this classical family cannot achieve exponential speedup in $d$; this is a negative result about the classical competitor, not about the quantum solvers [38].
- **Hardware-side caveats on explicit circuits.** Low and Chuang note that sparse-oracle Hamiltonian simulation doubles system qubits and that row-computable oracles are not always easy to implement; the hyperbolic-PDE circuit paper likewise contrasts oracle-based simulations with its explicit gate construction [37][25].
- **Verification limits of this report.** Two 2026 preprints [5][9] were read only on their arXiv pages and are labelled as such. The PDF fetch for the An–Liu–Wang–Zhao paper timed out; its content was read from an ar5iv mirror of the same arXiv identifier [41]. The NIST-hosted PDF of the wave-equation paper [23] also timed out; the arXiv version was used instead [42]. Several other URLs were retrieved at search-snippet level only and are not cited for claims beyond what those snippets state.

## Primitives required by this literature

This is a descriptive inventory of circuit-level primitives the cited papers assume or construct; it is not a recommendation.

- **Sparse-matrix oracles**: location map $|j,\ell\rangle\mapsto|j,\nu(j,\ell)\rangle$ and value map $|j,k,z\rangle\mapsto|j,k,z\oplus A_{jk}\rangle$, often for both $A$ and $A^\dagger$ [3][5].
- **State-preparation oracle** $P_B$ preparing $|b\rangle$ from a reference state, with its cost $T_B$ carried explicitly in runtimes [2][3].
- **Block encoding** of $A$, of $A^{-1}$ ("fast inversion" via eigenvalue arithmetic), and of preconditioned matrices $S^\top A S$; parameters $(\alpha,\eta)$ for normalisation and error [5][13][19].
- **Hamiltonian simulation** of $e^{-iAt}$ (Taylor series/LCU, qubitization, quantum walk), with query complexity stated in $t$ and $\log(1/\epsilon)$ [1][3][37].
- **Quantum phase estimation** with controlled time evolution and a non-unitary controlled-rotation step plus ancilla post-selection in HHL; QSP/QSVT phase factors and polynomial approximations to $1/x$ in later solvers [1][3][11].
- **Amplitude amplification and its variants**: fixed-point, variable-time (VTAA), and oblivious amplification for success-probability and norm-amplification overheads [3][5].
- **Linear combination of unitaries (LCU)** selector state preparation and controlled unitary application, including selector-state gate-count estimates [3][10].
- **Adiabatic discrete-time evolution / quantum walk step** constructed from a block encoding, with a spectral-gap condition [36].
- **Ancilla management and flag qubits** for post-selection, success flags, uncomputation of eigenphase registers, and block-encoding subnormalisation control [1][3][19].
- **Hadamard test / overlap estimation and quantum amplitude estimation** for estimating $\langle x|M|x\rangle$, Loschmidt echoes and solution norms; standard quantum limit $\epsilon^{-2}$ versus amplitude-estimation $\epsilon^{-1}$ [4][13][26].
- **Quantum Fourier transform and inverse QFT** for spectral discretisations, diagonalising shift operators, and Fourier-space readout [15][24][30].
- **Warped-phase / Schrödingerization ancilla register** mapping a linear ODE/PDE system to a one-higher-dimensional Hamiltonian system, sometimes with a second register for inhomogeneity [19][21][24].
- **Arithmetic and classical preprocessing** for eigenvalue inversion, polynomial/function approximation, and structured state preparation of right-hand sides [13][16].
- **Lindbladian simulation primitives** (Hamiltonian and jump-operator oracles) for the non-diagonal density-matrix encoding route [4].

## Sources

[1] https://math.berkeley.edu/~linlin/2018Spring_290/HHL09.pdf
[2] https://arxiv.org/html/1501.00008v1
[3] https://arxiv.org/pdf/1511.02306v2
[4] https://arxiv.org/html/2410.19628
[5] https://arxiv.org/html/2609.35660
[9] https://arxiv.org/html/2607.07691
[10] https://arxiv.org/html/2503.21420
[11] https://arxiv.org/html/2507.09686v1
[13] https://arxiv.org/html/2008.13295v2
[15] https://www.osti.gov/pages/servlets/purl/1981054
[16] https://arxiv.org/pdf/1809.02622
[19] https://arxiv.org/html/2505.06866v2
[21] https://arxiv.org/html/2304.00667v1
[23] https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=924787
[24] https://arxiv.org/html/2404.13585
[25] https://arxiv.org/html/2402.18398
[26] https://arxiv.org/html/2409.18559v2
[27] https://arxiv.org/html/2507.00908v1
[29] https://arxiv.org/html/2504.06123
[30] https://arxiv.org/pdf/2511.20017
[31] https://arxiv.org/html/2608.19943
[32] https://scottaaronson.com/papers/qml.pdf
[33] https://www.osti.gov/servlets/purl/3007530
[34] https://arxiv.org/html/1512.05903
[36] https://arxiv.org/abs/2111.08152
[37] https://arxiv.org/html/1610.06546
[38] https://arxiv.org/html/2607.06533
[39] https://arxiv.org/html/2403.19512v4
[41] https://ar5iv.labs.arxiv.org/html/2211.05246
[42] https://arxiv.org/html/1711.05394
