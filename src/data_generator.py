"""
Data Generator for Sparse Feature Preference Matrix

Generates preference matrices where only k out of z features
determine pairwise preferences (k-sparse model).
"""

import numpy as np
from typing import Tuple, Optional


def generate_preference_matrix(
    N: int, 
    z: int, 
    k: int, 
    seed: Optional[int] = None,
    feature_scale: float = 1.0,
    weight_scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Generate preference matrix from k-sparse feature model.
    
    Model:
        - Each item i has feature vector x_i ∈ R^z
        - Weight vector w ∈ R^z is k-sparse (only k non-zero entries)
        - Item score: s_i = x_i^T w
        - Preference: P_ij = σ(s_i - s_j) where σ is sigmoid
    
    Parameters
    ----------
    N : int
        Number of items (bandits)
    z : int
        Number of features per item
    k : int
        Sparsity level (number of relevant features, k ≤ z)
    seed : int, optional
        Random seed for reproducibility
    feature_scale : float
        Scale of feature values
    weight_scale : float
        Scale of weight values (higher = more decisive preferences)
    
    Returns
    -------
    P : np.ndarray (N, N)
        Preference matrix where P_ij = P(item i beats item j)
    X : np.ndarray (N, z)
        Feature matrix
    w : np.ndarray (z,)
        Sparse weight vector (k non-zero entries)
    winner : int
        Index of Copeland winner
    scores : np.ndarray (N,)
        Copeland scores for each item
    """
    if seed is not None:
        np.random.seed(seed)
    
    if k > z:
        raise ValueError(f"Sparsity k={k} cannot exceed feature dim z={z}")
    if k <= 0:
        raise ValueError(f"Sparsity k must be positive, got {k}")
    
    # Step 1: Generate feature matrix X (N items x z features)
    X = np.random.randn(N, z) * feature_scale
    
    # Step 2: Generate k-sparse weight vector w
    w = np.zeros(z)
    relevant_features = np.random.choice(z, size=k, replace=False)
    relevant_features.sort()
    w[relevant_features] = np.random.randn(k) * weight_scale
    
    # Step 3: Compute item scores
    item_scores = X @ w
    
    # Step 4: Build preference matrix (Bradley-Terry model)
    score_diff = item_scores[:, np.newaxis] - item_scores[np.newaxis, :]
    P = 1 / (1 + np.exp(-score_diff))
    
    # Step 5: Compute Copeland scores and winner
    scores = compute_copeland_scores(P)
    winner = int(np.argmax(scores))
    
    return P, X, w, winner, scores


def compute_copeland_scores(P: np.ndarray) -> np.ndarray:
    """Compute Copeland score for each item (number of pairwise wins)."""
    N = P.shape[0]
    wins = (P > 0.5).astype(int)
    np.fill_diagonal(wins, 0)
    return wins.sum(axis=1)


def get_relevant_features(w: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Return indices of non-zero (relevant) features."""
    return np.where(np.abs(w) > tol)[0]


def verify_sparsity(w: np.ndarray, k: int, tol: float = 1e-10) -> bool:
    """Verify that weight vector has exactly k non-zero entries."""
    return np.sum(np.abs(w) > tol) == k


def sample_duel(P: np.ndarray, i: int, j: int) -> int:
    """Simulate a duel. Returns 0 if i wins, 1 if j wins."""
    return 0 if np.random.random() < P[i, j] else 1


if __name__ == "__main__":
    # Demo
    print("Data Generator Demo")
    print("=" * 50)
    
    P, X, w, winner, scores = generate_preference_matrix(
        N=8, z=15, k=3, seed=42
    )
    
    print(f"Generated: N=8 items, z=15 features, k=3 sparse")
    print(f"Relevant features: {get_relevant_features(w)}")
    print(f"Copeland scores: {scores}")
    print(f"Winner: Item {winner}")
    print(f"Sparsity verified: {verify_sparsity(w, k=3)}")
