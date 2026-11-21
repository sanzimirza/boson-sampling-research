"""
Classical Boson Sampling Simulator
----------------------------------
Computes the probability of output patterns for a Boson Sampling experiment
using Ryser's algorithm for the matrix permanent.
"""

import numpy as np
from math import factorial
from itertools import combinations_with_replacement
import time


# ============================================================
# 1. PERMANENT (Ryser's Algorithm)
# ============================================================

def permanent_ryser(A):
    """
    Compute the permanent of a square matrix using Ryser's formula.
    
    Args:
        A: numpy array of shape (n, n), complex or float
    
    Returns:
        The permanent as a complex scalar
    """
    n = A.shape[0]
    
    if n == 0:
        return 1.0 + 0j
    
    total = 0.0 + 0j
    
    # Iterate over all non-empty subsets of columns (bitmask)
    for mask in range(1, 1 << n):
        bits = bin(mask).count('1')
        
        # Product of row sums over selected columns
        prod = 1.0 + 0j
        for row in range(n):
            row_sum = 0.0 + 0j
            for col in range(n):
                if mask & (1 << col):
                    row_sum += A[row, col]
            prod *= row_sum
        
        # Inclusion-exclusion sign
        if bits % 2 == 0:
            total -= prod
        else:
            total += prod
    
    return ((-1) ** n) * total


# ============================================================
# 2. RANDOM UNITARY (Haar measure)
# ============================================================

def random_unitary(n, seed=None):
    """
    Generate a random n x n unitary matrix from the Haar measure.
    """
    if seed is not None:
        np.random.seed(seed)
    
    Z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, R = np.linalg.qr(Z)
    
    # Fix phases to get Haar-random unitary
    D = np.diagonal(R)
    ph = D / np.abs(D)
    Q = Q * ph
    
    return Q


# ============================================================
# 3. BUILD SUBMATRIX
# ============================================================

def build_submatrix(U, input_modes, output_pattern):
    """
    Build the submatrix U_S for a given input state and output pattern.
    
    Args:
        U: full unitary matrix (m x m)
        input_modes: list of input mode indices (one per photon)
        output_pattern: tuple of photon counts per output mode
    
    Returns:
        U_S: submatrix (n x n)
        norm: normalization factor = product of factorials of output counts
    """
    n_photons = len(input_modes)
    
    # Columns: from input modes
    cols = list(input_modes)
    
    # Rows: from output pattern
    rows = []
    for mode, count in enumerate(output_pattern):
        rows.extend([mode] * count)
    
    # Sanity check
    if len(rows) != n_photons:
        raise ValueError(
            f"Output pattern has {len(rows)} photons, "
            f"but input has {n_photons} photons."
        )
    
    # Build submatrix
    U_S = U[np.ix_(rows, cols)]
    
    # Normalization
    norm = 1.0
    for count in output_pattern:
        norm *= factorial(count)
    
    return U_S, norm


# ============================================================
# 4. PROBABILITY
# ============================================================

def boson_sampling_probability(U, input_modes, output_pattern):
    """
    Compute the probability of a given output pattern.
    """
    U_S, norm = build_submatrix(U, input_modes, output_pattern)
    perm = permanent_ryser(U_S)
    return (np.abs(perm) ** 2) / norm


# ============================================================
# 5. ENUMERATE ALL OUTPUT PATTERNS
# ============================================================

def all_output_patterns(n_photons, m_modes):
    """
    Generate all possible output patterns for n_photons distributed
    across m_modes.
    
    Yields tuples of length m_modes that sum to n_photons.
    """
    # Generate all compositions of n_photons into m_modes parts
    for combo in combinations_with_replacement(range(m_modes), n_photons):
        pattern = [0] * m_modes
        for mode in combo:
            pattern[mode] += 1
        yield tuple(pattern)


# ============================================================
# 6. VALIDATION TEST
# ============================================================

def validate_probabilities(U, input_modes, m_modes, verbose=True):
    """
    Compute probabilities for ALL output patterns and check they sum to 1.
    """
    n_photons = len(input_modes)
    total_prob = 0.0
    results = []
    
    for pattern in all_output_patterns(n_photons, m_modes):
        prob = boson_sampling_probability(U, input_modes, pattern)
        total_prob += prob
        results.append((pattern, prob))
    
    if verbose:
        print(f"Input: {input_modes}")
        print(f"Total probability sum: {total_prob:.10f}")
        print(f"Expected: 1.0000000000")
        print(f"Match: {np.isclose(total_prob, 1.0)}")
        print()
        print("All output patterns and probabilities:")
        for pattern, prob in sorted(results, key=lambda x: -x[1]):
            print(f"  {pattern}  ->  {prob:.6f}")
    
    return total_prob, results


# ============================================================
# 7. BENCHMARK
# ============================================================

def benchmark_classical(max_photons=6, num_trials=3):
    
    """Measure classical runtime by computing ALL output patterns.
    """
    print("=" * 55)
    print("Classical Boson Sampling Benchmark")
    print("=" * 55)
    print(f"{'Photons':>8} | {'Modes':>6} | {'Avg Time (s)':>14}")
    print("-" * 55)
    
    results = []
    
    for n in range(2, max_photons + 1):
        m = 2 * n
        total_time = 0.0
        
        for trial in range(num_trials):
            U = random_unitary(m, seed=trial * 100 + n)
            input_modes = list(range(n))
            
            start = time.perf_counter()
            # Compute ALL output patterns
            for pattern in all_output_patterns(n, m):
                _ = boson_sampling_probability(U, input_modes, pattern)
            end = time.perf_counter()
            
            total_time += (end - start)
        
        avg_time = total_time / num_trials
        results.append((n, m, avg_time))
        print(f"{n:>8} | {m:>6} | {avg_time:>14.6f}")
    
    print("=" * 55)
    return results


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":
    
    # --- Test 1: Our hand-worked example ---
    print("\n--- TEST 1: Hand-worked example ---\n")
    
    U = (1/2) * np.array([
        [1,  1,   1,  1],
        [1,  1j, -1, -1j],
        [1, -1,   1, -1],
        [1, -1j, -1,  1j]
    ])
    
    input_modes = [0, 1]
    output_pattern = (1, 0, 0, 1)
    
    U_S, norm = build_submatrix(U, input_modes, output_pattern)
    print("U_S =")
    print(U_S)
    print(f"Norm = {norm}")
    
    prob = boson_sampling_probability(U, input_modes, output_pattern)
    print(f"Probability of {output_pattern}: {prob:.6f}")
    print(f"Expected: 0.125000")
    print()
    
    # --- Test 2: Validate probabilities sum to 1 (random unitary) ---
    print("\n--- TEST 2: Validate sum = 1 (random unitary) ---\n")
    U_random = random_unitary(4, seed=42)
    validate_probabilities(U_random, input_modes=[0, 1], m_modes=4)
    
    # --- Test 3: Benchmark ---
    print("\n--- TEST 3: Benchmark ---\n")
    
    benchmark_classical(max_photons=6, num_trials=3)