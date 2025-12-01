"""
Unit Tests for Fourier Copeland Bandit Identification
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import unittest

from src.data_generator import (
    generate_preference_matrix,
    compute_copeland_scores,
    get_relevant_features,
    verify_sparsity
)
from src.naive_ccb import NaiveCCB
from src.fourier_bandit import FourierDuelingBandit


class TestDataGenerator(unittest.TestCase):
    """Tests for data generation."""
    
    def test_basic_generation(self):
        """Test basic matrix generation."""
        P, X, w, winner, scores = generate_preference_matrix(N=8, z=10, k=3, seed=42)
        
        self.assertEqual(P.shape, (8, 8))
        self.assertEqual(X.shape, (8, 10))
        self.assertEqual(w.shape, (10,))
        self.assertTrue(0 <= winner < 8)
        self.assertEqual(len(scores), 8)
    
    def test_sparsity(self):
        """Test that weight vector is k-sparse."""
        _, _, w, _, _ = generate_preference_matrix(N=8, z=20, k=5, seed=42)
        
        self.assertTrue(verify_sparsity(w, k=5))
        self.assertEqual(len(get_relevant_features(w)), 5)
    
    def test_preference_constraints(self):
        """Test preference matrix constraints."""
        P, _, _, _, _ = generate_preference_matrix(N=8, z=10, k=3, seed=42)
        
        # P_ij in [0, 1]
        self.assertTrue(np.all(P >= 0))
        self.assertTrue(np.all(P <= 1))
        
        # P_ii = 0.5
        self.assertTrue(np.allclose(np.diag(P), 0.5))
        
        # P_ij + P_ji = 1
        self.assertTrue(np.allclose(P + P.T, 1))
    
    def test_copeland_scores(self):
        """Test Copeland score computation."""
        P = np.array([
            [0.5, 0.7, 0.6],
            [0.3, 0.5, 0.8],
            [0.4, 0.2, 0.5]
        ])
        scores = compute_copeland_scores(P)
        
        # Item 0 beats 1,2 (score=2), Item 1 beats 2 (score=1), Item 2 beats none (score=0)
        np.testing.assert_array_equal(scores, [2, 1, 0])
    
    def test_invalid_k(self):
        """Test error on invalid k."""
        with self.assertRaises(ValueError):
            generate_preference_matrix(N=8, z=10, k=15, seed=42)  # k > z


class TestNaiveCCB(unittest.TestCase):
    """Tests for Naive CCB algorithm."""
    
    def setUp(self):
        self.P, self.X, self.w, self.winner, _ = generate_preference_matrix(
            N=8, z=10, k=3, seed=42
        )
    
    def test_initialization(self):
        """Test algorithm initialization."""
        algo = NaiveCCB(N=8, delta=0.1)
        
        self.assertEqual(algo.N, 8)
        self.assertEqual(algo.total_samples, 0)
        self.assertEqual(algo.wins.shape, (8, 8))
    
    def test_pair_selection(self):
        """Test pair selection returns valid pairs."""
        algo = NaiveCCB(N=8, delta=0.1)
        
        for _ in range(10):
            i, j = algo.select_pair()
            self.assertTrue(0 <= i < 8)
            self.assertTrue(0 <= j < 8)
            self.assertNotEqual(i, j)
    
    def test_update(self):
        """Test update after duel."""
        algo = NaiveCCB(N=8, delta=0.1)
        
        algo.update(0, 1, i_wins=True)
        
        self.assertEqual(algo.wins[0, 1], 1)
        self.assertEqual(algo.wins[1, 0], 0)
        self.assertEqual(algo.comparisons[0, 1], 1)
        self.assertEqual(algo.total_samples, 1)
    
    def test_run(self):
        """Test full run."""
        algo = NaiveCCB(N=8, delta=0.1)
        metrics = algo.run(self.P, max_samples=100, record_every=10)
        
        self.assertEqual(len(metrics['samples']), 10)
        self.assertEqual(metrics['samples'][-1], 100)


class TestFourierBandit(unittest.TestCase):
    """Tests for Fourier Dueling Bandit algorithm."""
    
    def setUp(self):
        self.P, self.X, self.w, self.winner, _ = generate_preference_matrix(
            N=8, z=10, k=3, seed=42
        )
    
    def test_initialization(self):
        """Test algorithm initialization."""
        algo = FourierDuelingBandit(N=8, z=10, k=3, X=self.X, delta=0.1)
        
        self.assertEqual(algo.N, 8)
        self.assertEqual(algo.z, 10)
        self.assertEqual(algo.k, 3)
        self.assertEqual(algo.total_samples, 0)
    
    def test_feature_diff(self):
        """Test feature difference computation."""
        algo = FourierDuelingBandit(N=8, z=10, k=3, X=self.X, delta=0.1)
        
        diff = algo.get_feature_diff(0, 1)
        expected = self.X[0] - self.X[1]
        
        np.testing.assert_array_almost_equal(diff, expected)
    
    def test_lasso_estimation(self):
        """Test LASSO weight estimation."""
        algo = FourierDuelingBandit(N=8, z=10, k=3, X=self.X, delta=0.1)
        
        # Add some observations
        for _ in range(50):
            i, j = algo.select_pair_random()
            i_wins = np.random.random() < self.P[i, j]
            algo.update(i, j, i_wins)
        
        # Check that some weights are estimated
        w_hat = algo.estimate_weights_lasso()
        self.assertEqual(w_hat.shape, (10,))
    
    def test_run(self):
        """Test full run."""
        algo = FourierDuelingBandit(N=8, z=10, k=3, X=self.X, delta=0.1)
        metrics = algo.run(self.P, max_samples=100, record_every=10)
        
        self.assertEqual(len(metrics['samples']), 10)
        self.assertEqual(metrics['samples'][-1], 100)
        self.assertIn('n_features_found', metrics)


class TestComparison(unittest.TestCase):
    """Tests comparing both algorithms."""
    
    def test_both_find_winner(self):
        """Test that both algorithms can find the winner."""
        P, X, w, true_winner, _ = generate_preference_matrix(N=6, z=10, k=2, seed=123)
        
        # Naive
        naive = NaiveCCB(N=6, delta=0.1)
        naive.run(P, max_samples=300)
        
        # Fourier
        fourier = FourierDuelingBandit(N=6, z=10, k=2, X=X, delta=0.1)
        fourier.run(P, max_samples=300)
        
        # At least one should find the correct winner
        naive_correct = naive.get_current_winner() == true_winner
        fourier_correct = fourier.get_current_winner() == true_winner
        
        self.assertTrue(naive_correct or fourier_correct, 
                        "At least one algorithm should find the winner")


if __name__ == '__main__':
    unittest.main(verbosity=2)
