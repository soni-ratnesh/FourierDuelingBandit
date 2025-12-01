"""
Sparse Feature Dueling Bandit Algorithm

Exploits k-sparse feature structure to achieve O(k log N) sample complexity
instead of O(N²) for standard approaches.

Uses LASSO regression to recover sparse weight vector from comparisons.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


class SparseDuelingBandit:
    """
    Dueling bandit algorithm that exploits sparse feature structure.
    
    Uses LASSO (L1-regularized) regression to recover the k-sparse 
    weight vector from pairwise comparison outcomes.
    
    Complexity: O(k · poly(log N)) comparisons
    """
    
    def __init__(self, N: int, z: int, k: int, X: np.ndarray, 
                 delta: float = 0.1, lambda_reg: float = 0.1):
        """
        Parameters
        ----------
        N : int
            Number of items
        z : int
            Feature dimension
        k : int
            Sparsity level (known or estimated)
        X : np.ndarray (N, z)
            Feature matrix (known)
        delta : float
            Confidence parameter
        lambda_reg : float
            L1 regularization strength for LASSO
        """
        self.N = N
        self.z = z
        self.k = k
        self.X = X
        self.delta = delta
        self.lambda_reg = lambda_reg
        
        # Collected observations: (i, j, outcome)
        self.comparisons: List[Tuple[int, int, int]] = []
        
        # Current estimate of weight vector
        self.w_hat = np.zeros(z)
        
        # Tracking
        self.total_samples = 0
        self.history: List[Tuple[int, int, int]] = []
        
    def reset(self):
        """Reset algorithm state."""
        self.comparisons = []
        self.w_hat = np.zeros(self.z)
        self.total_samples = 0
        self.history = []
        
    def get_feature_diff(self, i: int, j: int) -> np.ndarray:
        """Get feature difference x_i - x_j."""
        return self.X[i] - self.X[j]
    
    def _soft_threshold(self, x: float, thresh: float) -> float:
        """Soft thresholding operator for LASSO."""
        if x > thresh:
            return x - thresh
        elif x < -thresh:
            return x + thresh
        return 0.0
    
    def estimate_weights_lasso(self) -> np.ndarray:
        """
        Estimate weights using LASSO (L1 regularization).
        Exploits sparsity assumption via coordinate descent.
        """
        if len(self.comparisons) < 5:
            return self.w_hat
        
        # Build design matrix and response
        A = np.array([self.get_feature_diff(i, j) for i, j, _ in self.comparisons])
        y = np.array([2 * outcome - 1 for _, _, outcome in self.comparisons])
        
        # Coordinate descent LASSO
        w = np.zeros(self.z)
        
        for _ in range(100):
            for feat in range(self.z):
                residual = y - A @ w + A[:, feat] * w[feat]
                rho = A[:, feat] @ residual
                z_norm = np.sum(A[:, feat] ** 2)
                
                if z_norm > 0:
                    w[feat] = self._soft_threshold(rho / z_norm, 
                                                    self.lambda_reg / z_norm)
        return w
    
    def get_item_scores(self, w: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute item scores from weight vector."""
        if w is None:
            w = self.w_hat
        return self.X @ w
    
    def get_estimated_P(self) -> np.ndarray:
        """Get estimated preference matrix from current weights."""
        scores = self.get_item_scores()
        score_diff = scores[:, np.newaxis] - scores[np.newaxis, :]
        return 1 / (1 + np.exp(-score_diff))
    
    def get_copeland_scores(self, P: np.ndarray) -> np.ndarray:
        """Compute Copeland scores."""
        wins = (P > 0.5).astype(int)
        np.fill_diagonal(wins, 0)
        return wins.sum(axis=1)
    
    def select_pair_random(self) -> Tuple[int, int]:
        """Random pair selection (exploration)."""
        i = np.random.randint(self.N)
        j = np.random.randint(self.N)
        while j == i:
            j = np.random.randint(self.N)
        return (min(i, j), max(i, j))
    
    def select_pair_informative(self) -> Tuple[int, int]:
        """
        Select most informative pair for sparse recovery.
        Prioritizes pairs with large feature differences in uncertain dimensions.
        """
        relevant = np.abs(self.w_hat) > 0.01
        if not np.any(relevant):
            relevant = np.ones(self.z, dtype=bool)
        
        max_info = -1
        best_pair = (0, 1)
        
        n_candidates = min(50, self.N * (self.N - 1) // 2)
        
        for _ in range(n_candidates):
            i, j = self.select_pair_random()
            diff = np.abs(self.get_feature_diff(i, j))
            info = np.sum(diff[relevant])
            
            if info > max_info:
                max_info = info
                best_pair = (i, j)
        
        return best_pair
    
    def select_pair_ucb(self) -> Tuple[int, int]:
        """UCB-style selection based on estimated scores."""
        scores = self.get_item_scores()
        
        n_comparisons = np.zeros(self.N)
        for i, j, _ in self.comparisons:
            n_comparisons[i] += 1
            n_comparisons[j] += 1
        
        exploration = np.sqrt(2 * np.log(self.total_samples + 1) / (n_comparisons + 1))
        ucb_scores = scores + exploration
        
        top_items = np.argsort(ucb_scores)[-2:]
        return (min(top_items), max(top_items))
    
    def select_pair(self) -> Tuple[int, int]:
        """
        Adaptive pair selection strategy.
        
        Phase 1 (t < 2z): Random exploration
        Phase 2 (t < 5z): Informative sampling for sparse recovery  
        Phase 3 (t >= 5z): UCB exploitation with ε-exploration
        """
        if self.total_samples < self.z * 2:
            return self.select_pair_random()
        elif self.total_samples < self.z * 5:
            return self.select_pair_informative()
        else:
            if np.random.random() < 0.1:
                return self.select_pair_random()
            return self.select_pair_ucb()
    
    def update(self, i: int, j: int, i_wins: bool):
        """Update after observing duel outcome."""
        outcome = 1 if i_wins else 0
        self.comparisons.append((i, j, outcome))
        self.total_samples += 1
        self.history.append((i, j, i if i_wins else j))
        
        # Update weight estimate periodically
        if self.total_samples % 5 == 0:
            self.w_hat = self.estimate_weights_lasso()
    
    def get_current_winner(self) -> int:
        """Return current best estimate of Copeland winner."""
        P_hat = self.get_estimated_P()
        scores = self.get_copeland_scores(P_hat)
        return int(np.argmax(scores))
    
    def get_estimated_relevant_features(self, threshold: float = 0.01) -> np.ndarray:
        """Return indices of estimated relevant features."""
        return np.where(np.abs(self.w_hat) > threshold)[0]
    
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
        true_scores = self.get_copeland_scores(P_true)
        true_winner = int(np.argmax(true_scores))
        
        metrics = {
            'samples': [],
            'estimated_winner': [],
            'is_correct': [],
            'regret': [],
            'cumulative_regret': [],
            'n_features_found': []
        }
        
        cumulative_regret = 0
        
        for t in range(max_samples):
            i, j = self.select_pair()
            i_wins = np.random.random() < P_true[i, j]
            self.update(i, j, i_wins)
            
            current_winner = self.get_current_winner()
            instant_regret = 0 if (i == true_winner or j == true_winner) else 1
            cumulative_regret += instant_regret
            
            if (t + 1) % record_every == 0:
                metrics['samples'].append(t + 1)
                metrics['estimated_winner'].append(current_winner)
                metrics['is_correct'].append(current_winner == true_winner)
                metrics['regret'].append(instant_regret)
                metrics['cumulative_regret'].append(cumulative_regret)
                metrics['n_features_found'].append(
                    len(self.get_estimated_relevant_features())
                )
        
        return metrics


if __name__ == "__main__":
    from data_generator import generate_preference_matrix, get_relevant_features
    
    print("Sparse Dueling Bandit Demo")
    print("=" * 50)
    
    # Generate problem
    P, X, w, true_winner, _ = generate_preference_matrix(N=8, z=15, k=3, seed=42)
    true_features = get_relevant_features(w)
    
    # Run algorithm
    algo = SparseDuelingBandit(N=8, z=15, k=3, X=X, delta=0.1)
    metrics = algo.run(P, max_samples=200, record_every=20)
    
    print(f"True winner: {true_winner}")
    print(f"True relevant features: {true_features}")
    print(f"Final estimate: {algo.get_current_winner()}")
    print(f"Estimated features: {algo.get_estimated_relevant_features()}")
    print(f"Final regret: {metrics['cumulative_regret'][-1]}")
    print(f"Correct: {metrics['is_correct'][-1]}")
