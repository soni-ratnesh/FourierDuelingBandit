"""
SUSHI Dataset Experiment

Run Fourier Copeland Bandit algorithms on the SUSHI preference dataset.
Compares Naive CCB vs Fourier Bandit on real/synthetic sushi preferences.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

from src.sushi_loader import (
    load_sushi_dataset,
    generate_synthetic_sushi_preferences,
    get_sushi_features,
    print_sushi_summary,
    SUSHI_ITEMS
)
from src.naive_ccb import NaiveCCB
from src.fourier_bandit import FourierDuelingBandit


def run_sushi_experiment(
    P: np.ndarray,
    X: np.ndarray,
    true_winner: int,
    k: int = 2,
    max_samples: int = 200,
    n_runs: int = 20,
    verbose: bool = True
):
    """
    Run experiment on SUSHI data.
    
    Parameters
    ----------
    P : np.ndarray
        Preference matrix
    X : np.ndarray
        Feature matrix
    true_winner : int
        True Copeland winner
    k : int
        Assumed sparsity level
    max_samples : int
        Maximum comparisons per run
    n_runs : int
        Number of runs
    verbose : bool
        Print progress
        
    Returns
    -------
    results : dict
        Experiment results
    """
    N, z = X.shape
    
    naive_samples_to_correct = []
    fourier_samples_to_correct = []
    naive_correct = []
    fourier_correct = []
    naive_regret_curves = []
    fourier_regret_curves = []
    
    for run in range(n_runs):
        if verbose and run % 5 == 0:
            print(f"  Run {run+1}/{n_runs}")
        
        # Naive CCB
        np.random.seed(run * 100)
        naive = NaiveCCB(N, delta=0.1)
        n_first = None
        n_regret = []
        cum_reg = 0
        
        for t in range(max_samples):
            i, j = naive.select_pair()
            i_wins = np.random.random() < P[i, j]
            naive.update(i, j, i_wins)
            
            if i != true_winner and j != true_winner:
                cum_reg += 1
            n_regret.append(cum_reg)
            
            if naive.get_current_winner() == true_winner and n_first is None:
                n_first = t + 1
        
        naive_samples_to_correct.append(n_first if n_first else max_samples)
        naive_correct.append(naive.get_current_winner() == true_winner)
        naive_regret_curves.append(n_regret)
        
        # Fourier Bandit
        np.random.seed(run * 100)
        fourier = FourierDuelingBandit(N, z, k, X, delta=0.1)
        f_first = None
        f_regret = []
        cum_reg = 0
        
        for t in range(max_samples):
            i, j = fourier.select_pair()
            i_wins = np.random.random() < P[i, j]
            fourier.update(i, j, i_wins)
            
            if i != true_winner and j != true_winner:
                cum_reg += 1
            f_regret.append(cum_reg)
            
            if fourier.get_current_winner() == true_winner and f_first is None:
                f_first = t + 1
        
        fourier_samples_to_correct.append(f_first if f_first else max_samples)
        fourier_correct.append(fourier.get_current_winner() == true_winner)
        fourier_regret_curves.append(f_regret)
    
    return {
        'naive_samples': np.mean(naive_samples_to_correct),
        'naive_samples_std': np.std(naive_samples_to_correct),
        'fourier_samples': np.mean(fourier_samples_to_correct),
        'fourier_samples_std': np.std(fourier_samples_to_correct),
        'naive_accuracy': np.mean(naive_correct) * 100,
        'fourier_accuracy': np.mean(fourier_correct) * 100,
        'naive_regret': np.mean(naive_regret_curves, axis=0),
        'fourier_regret': np.mean(fourier_regret_curves, axis=0),
        'speedup': np.mean(naive_samples_to_correct) / np.mean(fourier_samples_to_correct)
    }


def plot_sushi_results(results: dict, title: str = "SUSHI Dataset", save_path: str = None):
    """Plot experiment results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    samples = np.arange(1, len(results['naive_regret']) + 1)
    
    # Cumulative regret
    ax1 = axes[0]
    ax1.plot(samples, results['naive_regret'], 'b-', linewidth=2, label='Naive CCB')
    ax1.plot(samples, results['fourier_regret'], 'r-', linewidth=2, label='Fourier Bandit')
    ax1.set_xlabel('Number of Comparisons')
    ax1.set_ylabel('Cumulative Regret')
    ax1.set_title(f'{title}: Cumulative Regret')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bar chart comparison
    ax2 = axes[1]
    metrics = ['Samples to\nCorrect', 'Final\nAccuracy (%)']
    naive_vals = [results['naive_samples'], results['naive_accuracy']]
    fourier_vals = [results['fourier_samples'], results['fourier_accuracy']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, naive_vals, width, label='Naive CCB', color='steelblue')
    bars2 = ax2.bar(x + width/2, fourier_vals, width, label='Fourier Bandit', color='indianred')
    
    ax2.set_ylabel('Value')
    ax2.set_title(f'{title}: Performance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def main():
    print("=" * 70)
    print("SUSHI DATASET EXPERIMENT")
    print("A Fourier Approach to Sample-Efficient Copeland Bandit Identification")
    print("=" * 70)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # =========================================
    # Experiment 1: Synthetic SUSHI (known sparse)
    # =========================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Synthetic SUSHI (known k=2 sparse)")
    print("=" * 70)
    
    # Generate synthetic data where only oiliness and price matter
    P_syn, X_syn, w_syn, winner_syn = generate_synthetic_sushi_preferences(
        n_users=5000,
        sparse_features=[3, 5],  # oiliness and price
        noise_level=0.05,
        seed=42
    )
    
    print(f"\nTrue sparse features: oiliness (idx=3), price (idx=5)")
    print(f"True winner: {SUSHI_ITEMS[winner_syn]}")
    
    print("\nRunning experiment...")
    results_syn = run_sushi_experiment(
        P_syn, X_syn, winner_syn,
        k=2,
        max_samples=150,
        n_runs=20
    )
    
    print(f"\nResults (Synthetic SUSHI, k=2):")
    print(f"  Naive CCB:      {results_syn['naive_samples']:.0f} samples, {results_syn['naive_accuracy']:.0f}% accuracy")
    print(f"  Fourier Bandit: {results_syn['fourier_samples']:.0f} samples, {results_syn['fourier_accuracy']:.0f}% accuracy")
    print(f"  Speedup: {results_syn['speedup']:.1f}x")
    
    plot_sushi_results(results_syn, "Synthetic SUSHI (k=2 sparse)", 
                       save_path='results/sushi_synthetic.png')
    
    # =========================================
    # Experiment 2: Real/Simulated SUSHI
    # =========================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Real SUSHI Preferences")
    print("=" * 70)
    
    data = load_sushi_dataset()
    print_sushi_summary(data)
    
    P_real = data['P']
    X_real = data['X']
    winner_real = data['winner']
    
    print("\nRunning experiment with different k values...")
    
    for k in [2, 3, 4]:
        print(f"\n--- k = {k} ---")
        results_real = run_sushi_experiment(
            P_real, X_real, winner_real,
            k=k,
            max_samples=150,
            n_runs=20,
            verbose=False
        )
        
        print(f"  Naive CCB:      {results_real['naive_samples']:.0f} samples, {results_real['naive_accuracy']:.0f}% accuracy")
        print(f"  Fourier Bandit: {results_real['fourier_samples']:.0f} samples, {results_real['fourier_accuracy']:.0f}% accuracy")
        print(f"  Speedup: {results_real['speedup']:.1f}x")
    
    # Save results for k=3
    results_real = run_sushi_experiment(
        P_real, X_real, winner_real,
        k=3,
        max_samples=150,
        n_runs=20,
        verbose=False
    )
    plot_sushi_results(results_real, "SUSHI Preferences (k=3)", 
                       save_path='results/sushi_real.png')
    
    # =========================================
    # Summary
    # =========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The SUSHI dataset demonstrates the effectiveness of the Fourier approach:

1. Synthetic SUSHI (known sparse):
   - When preferences truly depend on only 2 features (oiliness, price),
     the Fourier Bandit significantly outperforms Naive CCB.

2. Real SUSHI Preferences:
   - Real human preferences may not be perfectly sparse.
   - The Fourier Bandit still performs competitively.
   - Performance depends on choosing appropriate k.

Key insight: The Fourier approach works best when:
   - Preferences are approximately low-dimensional
   - Features capture meaningful preference dimensions
   - k is chosen appropriately (not too small, not too large)
""")
    
    print("\nResults saved to results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
