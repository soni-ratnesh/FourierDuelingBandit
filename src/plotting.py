"""
Plotting Utilities

Functions for generating comparison plots and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_regret_comparison(
    results: Dict, 
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5)
) -> plt.Figure:
    """
    Plot cumulative regret and accuracy comparison.
    
    Parameters
    ----------
    results : dict
        Output from run_multiple_experiments()
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    naive = results['naive']
    fourier = results['fourier']
    samples = naive['samples']
    
    # Plot 1: Cumulative Regret
    ax1 = axes[0]
    
    ax1.plot(samples, naive['cum_regret_mean'], 'b-', linewidth=2, 
             label='Naive CCB (O(N²))')
    ax1.fill_between(samples, 
                     naive['cum_regret_mean'] - naive['cum_regret_std'],
                     naive['cum_regret_mean'] + naive['cum_regret_std'],
                     alpha=0.3, color='blue')
    
    ax1.plot(samples, fourier['cum_regret_mean'], 'r-', linewidth=2,
             label='Fourier Bandit (O(k log N))')
    ax1.fill_between(samples,
                     fourier['cum_regret_mean'] - fourier['cum_regret_std'],
                     fourier['cum_regret_mean'] + fourier['cum_regret_std'],
                     alpha=0.3, color='red')
    
    ax1.set_xlabel('Number of Comparisons', fontsize=12)
    ax1.set_ylabel('Cumulative Regret', fontsize=12)
    ax1.set_title(f'Cumulative Regret (N={results["N"]}, z={results["z"]}, k={results["k"]})', 
                  fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    ax2 = axes[1]
    
    ax2.plot(samples, naive['accuracy_mean'] * 100, 'b-', linewidth=2,
             label='Naive CCB')
    ax2.fill_between(samples,
                     (naive['accuracy_mean'] - naive['accuracy_std']) * 100,
                     (naive['accuracy_mean'] + naive['accuracy_std']) * 100,
                     alpha=0.3, color='blue')
    
    ax2.plot(samples, fourier['accuracy_mean'] * 100, 'r-', linewidth=2,
             label='Fourier Bandit')
    ax2.fill_between(samples,
                     (fourier['accuracy_mean'] - fourier['accuracy_std']) * 100,
                     (fourier['accuracy_mean'] + fourier['accuracy_std']) * 100,
                     alpha=0.3, color='red')
    
    ax2.set_xlabel('Number of Comparisons', fontsize=12)
    ax2.set_ylabel('Correct Identification Rate (%)', fontsize=12)
    ax2.set_title('Winner Identification Accuracy', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_scaling_comparison(
    all_results: List[Dict], 
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5)
) -> plt.Figure:
    """
    Plot how sample complexity scales with N.
    
    Parameters
    ----------
    all_results : list
        List of results dicts for different N values
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    Ns = [r['N'] for r in all_results]
    k = all_results[0]['k']
    
    def get_samples_to_80(metrics):
        for i, acc in enumerate(metrics['accuracy_mean']):
            if acc >= 0.8:
                return metrics['samples'][i]
        return metrics['samples'][-1]
    
    naive_samples = [get_samples_to_80(r['naive']) for r in all_results]
    fourier_samples = [get_samples_to_80(r['fourier']) for r in all_results]
    
    # Plot 1: Sample Complexity
    ax1 = axes[0]
    ax1.plot(Ns, naive_samples, 'bo-', linewidth=2, markersize=10, label='Naive CCB')
    ax1.plot(Ns, fourier_samples, 'ro-', linewidth=2, markersize=10, label='Fourier Bandit')
    
    # Reference lines
    ax1.plot(Ns, [n**2 * 0.3 for n in Ns], 'b--', alpha=0.4, label='~O(N²)')
    ax1.plot(Ns, [k * np.log(n+1) * 25 for n in Ns], 'r--', alpha=0.4, label='~O(k log N)')
    
    ax1.set_xlabel('Number of Items (N)', fontsize=12)
    ax1.set_ylabel('Samples to 80% Accuracy', fontsize=12)
    ax1.set_title('Sample Complexity Scaling', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final Regret
    ax2 = axes[1]
    naive_regret = [r['naive']['cum_regret_mean'][-1] for r in all_results]
    fourier_regret = [r['fourier']['cum_regret_mean'][-1] for r in all_results]
    
    x = np.arange(len(Ns))
    width = 0.35
    ax2.bar(x - width/2, naive_regret, width, label='Naive CCB', color='blue', alpha=0.7)
    ax2.bar(x + width/2, fourier_regret, width, label='Fourier Bandit', color='red', alpha=0.7)
    
    ax2.set_xlabel('Number of Items (N)', fontsize=12)
    ax2.set_ylabel('Final Cumulative Regret', fontsize=12)
    ax2.set_title('Final Regret vs Problem Size', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'N={n}' for n in Ns])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_single_run(
    naive_metrics: Dict,
    sparse_metrics: Dict,
    true_winner: int,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot detailed metrics from a single run."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    samples = naive_metrics['samples']
    
    # Cumulative regret
    ax1 = axes[0, 0]
    ax1.plot(samples, naive_metrics['cumulative_regret'], 'b-', label='Naive CCB')
    ax1.plot(samples, sparse_metrics['cumulative_regret'], 'r-', label='Sparse Bandit')
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Cumulative Regret')
    ax1.set_title('Cumulative Regret')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Instant regret
    ax2 = axes[0, 1]
    ax2.plot(samples, naive_metrics['regret'], 'b-', alpha=0.7, label='Naive CCB')
    ax2.plot(samples, sparse_metrics['regret'], 'r-', alpha=0.7, label='Sparse Bandit')
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Instant Regret')
    ax2.set_title('Instant Regret per Round')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Estimated winner
    ax3 = axes[1, 0]
    ax3.plot(samples, naive_metrics['estimated_winner'], 'b.-', alpha=0.7, label='Naive CCB')
    ax3.plot(samples, sparse_metrics['estimated_winner'], 'r.-', alpha=0.7, label='Sparse Bandit')
    ax3.axhline(y=true_winner, color='green', linestyle='--', label=f'True Winner ({true_winner})')
    ax3.set_xlabel('Samples')
    ax3.set_ylabel('Estimated Winner')
    ax3.set_title('Winner Estimate Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Correctness
    ax4 = axes[1, 1]
    ax4.plot(samples, naive_metrics['is_correct'], 'b-', label='Naive CCB')
    ax4.plot(samples, sparse_metrics['is_correct'], 'r-', label='Sparse Bandit')
    ax4.set_xlabel('Samples')
    ax4.set_ylabel('Correct (1) / Incorrect (0)')
    ax4.set_title('Correctness Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
