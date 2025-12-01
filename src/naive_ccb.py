"""
Naive Copeland Confidence Bound (CCB) Algorithm

Standard O(N²) approach that estimates all pairwise preferences
using Upper Confidence Bound selection strategy.
"""

import numpy as np
from typing import Tuple, Dict, List


class NaiveCCB:
    """
    Copeland Confidence Bound algorithm for dueling bandits.
    
    Estimates all N² pairwise probabilities and identifies
    the Copeland winner using confidence bounds.
    
    Complexity: O(N²) comparisons
    """
    
    def __init__(self, N: int, delta: float = 0.1):
        """
        Parameters
        ----------
        N : int
            Number of items
        delta : float
            Confidence parameter (1-delta confidence)
        """
        self.N = N
        self.delta = delta
        
        # Win counts: wins[i,j] = number of times i beat j
        self.wins = np.zeros((N, N), dtype=int)
        # Total comparisons between each pair
        self.comparisons = np.zeros((N, N), dtype=int)
        
        # Tracking
        self.total_samples = 0
        self.history: List[Tuple[int, int, int]] = []
        
    def reset(self):
        """Reset algorithm state."""
        self.wins = np.zeros((self.N, self.N), dtype=int)
        self.comparisons = np.zeros((self.N, self.N), dtype=int)
        self.total_samples = 0
        self.history = []
        
    def get_estimate(self) -> np.ndarray:
        """Get current estimate of preference matrix."""
        P_hat = np.full((self.N, self.N), 0.5)
        mask = self.comparisons > 0
        P_hat[mask] = self.wins[mask] / self.comparisons[mask]
        return P_hat
    
    def get_confidence_bound(self, n_comparisons: int) -> float:
        """Hoeffding confidence bound."""
        if n_comparisons == 0:
            return 1.0
        return np.sqrt(np.log(2 * self.N**2 / self.delta) / (2 * n_comparisons))
    
    def get_copeland_scores(self, P_hat: np.ndarray) -> np.ndarray:
        """Compute Copeland scores from estimated preferences."""
        wins = (P_hat > 0.5).astype(int)
        np.fill_diagonal(wins, 0)
        return wins.sum(axis=1)
    
    def select_pair(self) -> Tuple[int, int]:
        """
        Select next pair to compare using UCB-style selection.
        Prioritizes uncertain pairs involving potential winners.
        """
        P_hat = self.get_estimate()
        scores = self.get_copeland_scores(P_hat)
        
        max_uncertainty = -1
        best_pair = (0, 1)
        
        for i in range(self.N):
            for j in range(i + 1, self.N):
                n_ij = self.comparisons[i, j]
                uncertainty = self.get_confidence_bound(n_ij)
                
                # Weight by potential impact on winner determination
                importance = max(scores[i], scores[j])
                weighted_uncertainty = uncertainty * (1 + importance / self.N)
                
                if weighted_uncertainty > max_uncertainty:
                    max_uncertainty = weighted_uncertainty
                    best_pair = (i, j)
        
        return best_pair
    
    def update(self, i: int, j: int, i_wins: bool):
        """Update statistics after observing a duel outcome."""
        self.comparisons[i, j] += 1
        self.comparisons[j, i] += 1
        
        if i_wins:
            self.wins[i, j] += 1
        else:
            self.wins[j, i] += 1
        
        self.total_samples += 1
        self.history.append((i, j, i if i_wins else j))
    
    def get_current_winner(self) -> int:
        """Return current best estimate of Copeland winner."""
        P_hat = self.get_estimate()
        scores = self.get_copeland_scores(P_hat)
        return int(np.argmax(scores))
    
    def run(self, P_true: np.ndarray, max_samples: int = 10000, 
            record_every: int = 10) -> Dict:
        """
        Run the algorithm against true preference matrix.
        
        Parameters
        ----------
        P_true : np.ndarray
            True preference matrix
        max_samples : int
            Maximum number of comparisons
        record_every : int
            Record metrics every N steps
            
        Returns
        -------
        metrics : dict
            Dictionary with samples, regret, accuracy history
        """
        true_winner = int(np.argmax(self.get_copeland_scores(P_true)))
        
        metrics = {
            'samples': [],
            'estimated_winner': [],
            'is_correct': [],
            'regret': [],
            'cumulative_regret': []
        }
        
        cumulative_regret = 0
        
        for t in range(max_samples):
            # Select pair
            i, j = self.select_pair()
            
            # Simulate duel
            i_wins = np.random.random() < P_true[i, j]
            
            # Update
            self.update(i, j, i_wins)
            
            # Compute regret
            current_winner = self.get_current_winner()
            instant_regret = 0 if (i == true_winner or j == true_winner) else 1
            cumulative_regret += instant_regret
            
            # Record metrics
            if (t + 1) % record_every == 0:
                metrics['samples'].append(t + 1)
                metrics['estimated_winner'].append(current_winner)
                metrics['is_correct'].append(current_winner == true_winner)
                metrics['regret'].append(instant_regret)
                metrics['cumulative_regret'].append(cumulative_regret)
        
        return metrics


if __name__ == "__main__":
    from data_generator import generate_preference_matrix
    
    print("Naive CCB Demo")
    print("=" * 50)
    
    # Generate problem
    P, X, w, true_winner, _ = generate_preference_matrix(N=8, z=15, k=3, seed=42)
    
    # Run algorithm
    algo = NaiveCCB(N=8, delta=0.1)
    metrics = algo.run(P, max_samples=200, record_every=20)
    
    print(f"True winner: {true_winner}")
    print(f"Final estimate: {algo.get_current_winner()}")
    print(f"Final regret: {metrics['cumulative_regret'][-1]}")
    print(f"Correct: {metrics['is_correct'][-1]}")
