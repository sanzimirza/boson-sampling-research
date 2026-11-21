"""
Runtime benchmark: classical vs quantum Boson Sampling.
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from qboson import (
    quantum_boson_sampling_all_probs,
    classical_boson_sampling_probability,
    random_unitary,
)


def all_output_patterns(n_photons, m_modes):
    """Generate all output patterns for n photons in m modes."""
    from itertools import combinations_with_replacement
    for combo in combinations_with_replacement(range(m_modes), n_photons):
        pattern = [0] * m_modes
        for mode in combo:
            pattern[mode] += 1
        yield tuple(pattern)


def benchmark(max_photons=6, num_trials=3):
    """Compare classical and quantum runtimes."""
    results = []
    
    print(f"{'n':>4} | {'m':>4} | {'Classical (s)':>16} | {'Quantum (s)':>16}")
    print("-" * 50)
    
    for n in range(2, max_photons + 1):
        m = 2 * n
        t_classical_total = 0.0
        t_quantum_total = 0.0
        
        for trial in range(num_trials):
            U = random_unitary(m, seed=trial * 100 + n)
            input_modes = list(range(n))
            
            # Classical: compute ALL output patterns
            start = time.perf_counter()
            for pattern in all_output_patterns(n, m):
                _ = classical_boson_sampling_probability(U, input_modes, pattern)
            t_classical_total += time.perf_counter() - start
            
            # Quantum: get full distribution in one call
            start = time.perf_counter()
            _ = quantum_boson_sampling_all_probs(U, input_modes)
            t_quantum_total += time.perf_counter() - start
        
        t_classical = t_classical_total / num_trials
        t_quantum = t_quantum_total / num_trials
        
        results.append((n, m, t_classical, t_quantum))
        print(f"{n:>4} | {m:>4} | {t_classical:>16.6f} | {t_quantum:>16.6f}")
    
    return results


def plot_results(results):
    """Plot the runtime comparison on a log scale."""
    n_vals = [r[0] for r in results]
    t_classical = [r[2] for r in results]
    t_quantum = [r[3] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_vals, t_classical, 'ro-', linewidth=2.5, markersize=9, 
             label='Classical (Ryser permanent)')
    plt.plot(n_vals, t_quantum, 'bo-', linewidth=2.5, markersize=9, 
             label='Quantum (Perceval SLOS)')
    
    # Annotations
    plt.annotate('Exponential scaling\n(classical wall)',
                 xy=(6, t_classical[-1]), xytext=(4.3, 3),
                 fontsize=11, color='darkred',
                 arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))
    
    plt.annotate('Polynomial scaling\n(tractable)',
                 xy=(6, t_quantum[-1]), xytext=(4.3, 0.0015),
                 fontsize=11, color='darkblue',
                 arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.5))
    
    plt.xlabel('Number of photons', fontsize=13)
    plt.ylabel('Runtime (seconds, log scale)', fontsize=13)
    plt.title('Classical vs Quantum Boson Sampling Runtime', fontsize=15)
    plt.yscale('log')
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('runtime_comparison.png', dpi=200)
    plt.show()
    print("\nPlot saved to runtime_comparison.png")


if __name__ == "__main__":
    results = benchmark(max_photons=6, num_trials=3)
    plot_results(results)