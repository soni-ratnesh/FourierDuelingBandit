"""
Main Experiment Runner

A Fourier Approach to Sample-Efficient Copeland Bandit Identification
"""

import argparse
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.evaluation import run_multiple_experiments, print_summary
from src.plotting import plot_regret_comparison, plot_scaling_comparison


def main():
    parser = argparse.ArgumentParser(
        description='A Fourier Approach to Sample-Efficient Copeland Bandit Identification'
    )
    parser.add_argument('-N', type=int, default=16, 
                        help='Number of items (default: 16)')
    parser.add_argument('-z', type=int, default=20,
                        help='Feature dimension (default: 20)')
    parser.add_argument('-k', type=int, default=3,
                        help='Fourier sparsity level (default: 3)')
    parser.add_argument('--runs', type=int, default=20,
                        help='Number of runs (default: 20)')
    parser.add_argument('--max-samples', type=int, default=500,
                        help='Max samples per run (default: 500)')
    parser.add_argument('--scaling', action='store_true',
                        help='Run scaling experiment')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory (default: results)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("A FOURIER APPROACH TO SAMPLE-EFFICIENT COPELAND BANDIT IDENTIFICATION")
    print("=" * 70)
    print(f"Parameters: N={args.N}, z={args.z}, k={args.k} (Fourier sparsity)")
    print(f"Runs: {args.runs}, Max samples: {args.max_samples}")
    print()
    
    # Main experiment
    print("Running main experiment...")
    results = run_multiple_experiments(
        N=args.N, 
        z=args.z, 
        k=args.k,
        n_runs=args.runs,
        max_samples=args.max_samples,
        record_every=10,
        verbose=True
    )
    
    print_summary(results)
    
    # Generate plots
    print("\nGenerating plots...")
    fig1 = plot_regret_comparison(
        results, 
        save_path=os.path.join(args.output_dir, 'regret_comparison.png')
    )
    print(f"  Saved: {args.output_dir}/regret_comparison.png")
    
    # Scaling experiment
    if args.scaling:
        print("\nRunning scaling experiment...")
        scaling_results = []
        for N_test in [8, 16, 24, 32]:
            print(f"  N={N_test}...")
            res = run_multiple_experiments(
                N_test, 
                z=args.z, 
                k=args.k,
                n_runs=min(args.runs, 10),
                max_samples=min(args.max_samples, N_test * N_test),
                record_every=10
            )
            scaling_results.append(res)
        
        fig2 = plot_scaling_comparison(
            scaling_results,
            save_path=os.path.join(args.output_dir, 'scaling_comparison.png')
        )
        print(f"  Saved: {args.output_dir}/scaling_comparison.png")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
