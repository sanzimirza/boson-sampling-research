"""
Quantum Boson Sampling Simulator (Perceval)
-------------------------------------------
Fully self-contained: quantum simulator, classical simulator,
and helper functions.
"""

import perceval as pcvl
from perceval.algorithm import Sampler
import numpy as np
import time
from math import factorial


# ============================================================
# QUANTUM SIMULATOR (Perceval)
# ============================================================

def quantum_boson_sampling_all_probs(U, input_modes):
    """Compute the probability distribution for ALL output patterns."""
    m_modes = U.shape[0]
    
    circuit = pcvl.Circuit(m_modes)
    circuit.add(0, pcvl.Unitary(pcvl.Matrix(U)))
    
    processor = pcvl.Processor("SLOS", circuit)
    
    input_list = [0] * m_modes
    for mode in input_modes:
        input_list[mode] += 1
    input_state = pcvl.BasicState(input_list)
    
    processor.with_input(input_state)
    
    sampler = Sampler(processor)
    results = sampler.probs()["results"]
    
    output_probs = {}
    for state, prob in results.items():
        pattern = tuple(int(x) for x in str(state).strip("|>").split(","))
        output_probs[pattern] = prob
    
    return output_probs


# ============================================================
# CLASSICAL SIMULATOR
# ============================================================

def permanent_ryser(A):
    """Compute the permanent of a square matrix using Ryser's formula."""
    n = A.shape[0]
    if n == 0:
        return 1.0 + 0j
    total = 0.0 + 0j
    for mask in range(1, 1 << n):
        bits = bin(mask).count('1')
        prod = 1.0 + 0j
        for row in range(n):
            row_sum = 0.0 + 0j
            for col in range(n):
                if mask & (1 << col):
                    row_sum += A[row, col]
            prod *= row_sum
        if bits % 2 == 0:
            total -= prod
        else:
            total += prod
    return ((-1) ** n) * total


def build_submatrix(U, input_modes, output_pattern):
    """Build the submatrix U_S for a given input and output pattern."""
    n_photons = len(input_modes)
    cols = list(input_modes)
    rows = []
    for mode, count in enumerate(output_pattern):
        rows.extend([mode] * count)
    if len(rows) != n_photons:
        raise ValueError(f"Photon count mismatch: {len(rows)} vs {n_photons}")
    U_S = U[np.ix_(rows, cols)]
    norm = 1.0
    for count in output_pattern:
        norm *= factorial(count)
    return U_S, norm


def classical_boson_sampling_probability(U, input_modes, output_pattern):
    """Compute the probability of one output pattern classically."""
    U_S, norm = build_submatrix(U, input_modes, output_pattern)
    perm = permanent_ryser(U_S)
    return (np.abs(perm) ** 2) / norm


# ============================================================
# HELPERS
# ============================================================

def random_unitary(n, seed=None):
    """Generate a random n x n unitary matrix from the Haar measure."""
    if seed is not None:
        np.random.seed(seed)
    Z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, R = np.linalg.qr(Z)
    D = np.diagonal(R)
    ph = D / np.abs(D)
    return Q * ph


# ============================================================
# MAIN (self-test)
# ============================================================

if __name__ == "__main__":
    U = random_unitary(4, seed=42)
    input_modes = [0, 1]
    
    print("=== Quantum Boson Sampling (Perceval) ===\n")
    print("Input:", input_modes)
    
    start = time.perf_counter()
    probs = quantum_boson_sampling_all_probs(U, input_modes)
    elapsed = time.perf_counter() - start
    
    print(f"Computed in {elapsed:.4f} seconds\n")
    
    sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
    print(f"{'Pattern':<20} {'Probability':>15}")
    print("-" * 37)
    total = 0.0
    for pattern, prob in sorted_probs:
        if prob > 1e-8:
            print(f"{str(pattern):<20} {prob:>15.6f}")
            total += prob
    print("-" * 37)
    print(f"{'TOTAL':<20} {total:>15.6f}")
    
    print("\n=== Cross-validation with classical simulator ===\n")
    max_diff = 0.0
    for pattern, q_prob in probs.items():
        c_prob = classical_boson_sampling_probability(U, input_modes, pattern)
        max_diff = max(max_diff, abs(q_prob - c_prob))
    print(f"Max difference: {max_diff:.2e}")
    print(f"Validation: {'PASS' if max_diff < 1e-6 else 'FAIL'}")