"""
Evaluation Utilities

Functions for running experiments and generating comparison metrics.
"""

import numpy as np
from typing import Dict, List

from .data_generator import generate_preference_matrix
from .naive_ccb import NaiveCCB
from .fourier_bandit import FourierDuelingBandit


def run_single_experiment(
    N: int, 
    z: int, 
    k: int, 
    max_samples: int = 1000,
    seed: int = None, 
    record_every: int = 10
) -> Dict:
    """
    Run single experiment comparing both algorithms.
    
    Returns metrics for both algorithms on the same problem instance.
    """
    # Generate problem instance
    P, X, w, true_winner, true_scores = generate_preference_matrix(N, z, k, seed=seed)
    
    # Run Naive CCB
    naive = NaiveCCB(N, delta=0.1)
    naive_metrics = naive.run(P, max_samples, record_every)
    
    # Run Fourier Bandit
    fourier = FourierDuelingBandit(N, z, k, X, delta=0.1)
    fourier_metrics = fourier.run(P, max_samples, record_every)
    
    return {
        'naive': naive_metrics,
        'fourier': fourier_metrics,
        'true_winner': true_winner,
        'N': N,
        'z': z,
        'k': k
    }


def run_multiple_experiments(
    N: int, 
    z: int, 
    k: int, 
    n_runs: int = 20,
    max_samples: int = 1000, 
    record_every: int = 10,
    verbose: bool = False
) -> Dict:
    """
    Run multiple experiments and aggregate results.
    
    Returns aggregated metrics with mean and std.
    """
    all_naive = []
    all_fourier = []
    
    for run in range(n_runs):
        if verbose:
            print(f"  Run {run+1}/{n_runs}")
        result = run_single_experiment(
            N, z, k, max_samples, 
            seed=run * 100, 
            record_every=record_every
        )
        all_naive.append(result['naive'])
        all_fourier.append(result['fourier'])
    
    def aggregate_metrics(all_runs):
        samples = all_runs[0]['samples']
        cum_regret = np.array([run['cumulative_regret'] for run in all_runs])
        is_correct = np.array([run['is_correct'] for run in all_runs])
        
        return {
            'samples': samples,
            'cum_regret_mean': cum_regret.mean(axis=0),
            'cum_regret_std': cum_regret.std(axis=0),
            'accuracy_mean': is_correct.mean(axis=0),
            'accuracy_std': is_correct.std(axis=0)
        }
    
    return {
        'naive': aggregate_metrics(all_naive),
        'fourier': aggregate_metrics(all_fourier),
        'N': N, 'z': z, 'k': k, 'n_runs': n_runs
    }


def samples_to_accuracy(metrics: Dict, threshold: float = 0.8) -> int:
    """Find number of samples to reach accuracy threshold."""
    for i, acc in enumerate(metrics['accuracy_mean']):
        if acc >= threshold:
            return metrics['samples'][i]
    return metrics['samples'][-1]


def print_summary(results: Dict):
    """Print summary statistics."""
    naive = results['naive']
    fourier = results['fourier']
    
    print(f"\n{'='*60}")
    print(f"RESULTS: N={results['N']}, z={results['z']}, k={results['k']}")
    print(f"Runs: {results['n_runs']}")
    print(f"{'='*60}")
    
    print(f"\n{'Metric':<30} {'Naive CCB':<15} {'Fourier Bandit':<15}")
    print("-" * 60)
    
    # Final regret
    n_reg = f"{naive['cum_regret_mean'][-1]:.0f} ± {naive['cum_regret_std'][-1]:.0f}"
    f_reg = f"{fourier['cum_regret_mean'][-1]:.0f} ± {fourier['cum_regret_std'][-1]:.0f}"
    print(f"{'Final Cumulative Regret':<30} {n_reg:<15} {f_reg:<15}")
    
    # Final accuracy
    n_acc = f"{naive['accuracy_mean'][-1]*100:.0f}%"
    f_acc = f"{fourier['accuracy_mean'][-1]*100:.0f}%"
    print(f"{'Final Accuracy':<30} {n_acc:<15} {f_acc:<15}")
    
    # Samples to 80%
    n_samp = samples_to_accuracy(naive)
    f_samp = samples_to_accuracy(fourier)
    print(f"{'Samples to 80% Accuracy':<30} {n_samp:<15} {f_samp:<15}")
    
    # Speedup
    speedup = n_samp / f_samp if f_samp > 0 else float('inf')
    print(f"\nSpeedup: {speedup:.1f}x")
