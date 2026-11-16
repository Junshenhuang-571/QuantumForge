"""VQE: minimize <Z> on one qubit with the parameter-shift gradient.

The ground state of H = -Z is |1> with energy -1. A single Ry rotation can
reach it exactly, so the variational minimum must match the exact value.

Run with the quantumforge wheel installed:
    python python/examples/vqe.py
"""

from quantumforge import ParameterizedCircuit

HALF_PI = 1.5707963267948966


def main() -> None:
    ansatz = ParameterizedCircuit(1)
    ansatz.rotation("Y", 0, 0)

    def energy(theta):
        return ansatz.expectation_z([theta], 1, 0)

    # Parameter-shift gradient: d<Z>/dt = 0.5*(<Z>(t+pi/2) - <Z>(t-pi/2)).
    def gradient(theta):
        return 0.5 * (energy(theta + HALF_PI) - energy(theta - HALF_PI))

    theta = 0.3
    initial = energy(theta)
    step = 0.4
    best = initial
    for _ in range(200):
        candidate = theta - step * gradient(theta)
        value = energy(candidate)
        if value < best - 1e-12:
            best = value
            theta = candidate
        else:
            step *= 0.5
            if step < 1e-9:
                break

    exact = -1.0  # ground-state energy of H = -Z
    print(f"initial <Z> = {initial:.6f}")
    print(f"vqe minimum  = {best:.6f}")
    print(f"exact        = {exact:.6f}")
    assert abs(best - exact) < 1e-6, "VQE must reach the exact ground energy"
    print("OK: variational minimum matches the exact ground state")


if __name__ == "__main__":
    main()
