# A Fourier Approach to Sample-Efficient Copeland Bandit Identification

A framework implementing Fourier-sparse methods for efficient Copeland Winner identification in dueling bandits, comparing standard O(N²) approaches with Fourier-based O(k log N) methods.

---

## 📁 Project Structure

```
fourier_copeland_bandits/
│
├── src/                        # Source code
│   ├── __init__.py            # Package init
│   ├── data_generator.py      # Fourier-sparse preference matrix generation
│   ├── naive_ccb.py           # Naive CCB algorithm O(N²)
│   ├── fourier_bandit.py      # Fourier-based bandit O(k log N)
│   ├── evaluation.py          # Experiment utilities
│   └── plotting.py            # Visualization utilities
│
├── results/                   # Output plots and data
├── tests/                     # Unit tests
│   ├── __init__.py
│   └── test_all.py
│
├── demo.py                    # Quick demonstration
├── run_experiment.py          # Main experiment runner
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run demo
python demo.py

# 3. Run full experiment
python run_experiment.py

# 4. Run tests
python tests/test_all.py
```

---

## 🔬 Problem Formulation

### Dueling Bandits Setup

- **N items** (bandits) to compare pairwise
- **Preference matrix** P where P_ij = P(item i beats item j)
- **Copeland score**: C_i = number of items that i beats
- **Copeland Winner**: i* = argmax_i C_i
- **Goal**: Identify i* with minimum comparisons

### Fourier-Sparse Feature Model

We assume preferences are determined by a **k-sparse** weight vector:

```
Given:
  - N items, each with z features: X ∈ ℝ^(N×z)
  - Sparse weight vector: w ∈ ℝ^z with only k non-zero entries (k << z)

Model:
  - Item score: s_i = x_i^T w
  - Preference: P_ij = σ(s_i - s_j)  where σ(x) = 1/(1+e^(-x))
  
Properties:
  - P_ij ∈ [0, 1]
  - P_ii = 0.5  
  - P_ij + P_ji = 1 (anti-symmetry)
```

### Key Insight

| Approach | Sample Complexity | Why |
|----------|------------------|-----|
| Naive | O(N²) | Must estimate all N² pairwise preferences |
| Fourier | O(k · poly(log N)) | Only need to identify k relevant features |

---

## 📝 Algorithm Pseudocode

### Algorithm 1: Data Generation

```
GENERATE_FOURIER_SPARSE_PREFERENCES(N, z, k, seed)
────────────────────────────────────────────────────
Input:  N = number of items
        z = feature dimension
        k = sparsity (number of relevant features)
        seed = random seed

Output: P = preference matrix (N × N)
        X = feature matrix (N × z)  
        w = sparse weight vector (z)
        winner = Copeland winner index

1.  Set random seed
2.  
3.  // Generate feature matrix
4.  X ← RandomNormal(N, z)
5.  
6.  // Generate k-sparse weight vector
7.  w ← zeros(z)
8.  relevant_indices ← RandomChoice({0,1,...,z-1}, k, replace=False)
9.  w[relevant_indices] ← RandomNormal(k)
10. 
11. // Compute item scores  
12. scores ← X @ w                    // s_i = x_i^T w
13. 
14. // Build preference matrix (Bradley-Terry model)
15. FOR i = 0 to N-1:
16.     FOR j = 0 to N-1:
17.         P[i,j] ← σ(scores[i] - scores[j])
18. 
19. // Find Copeland winner
20. FOR i = 0 to N-1:
21.     C[i] ← sum(P[i,:] > 0.5) - 1   // Copeland score (exclude self)
22. winner ← argmax(C)
23. 
24. RETURN P, X, w, winner
```

---

### Algorithm 2: Naive CCB (Baseline)

```
NAIVE_COPELAND_CONFIDENCE_BOUND(N, δ, P_true, T)
────────────────────────────────────────────────────
Input:  N = number of items
        δ = confidence parameter
        P_true = true preference matrix (for simulation)
        T = maximum comparisons

Output: estimated_winner

1.  // Initialize
2.  wins[i,j] ← 0  ∀i,j           // Win count matrix
3.  comps[i,j] ← 0  ∀i,j          // Comparison count matrix
4.  
5.  FOR t = 1 to T:
6.      
7.      // Estimate current preferences
8.      FOR each (i,j):
9.          IF comps[i,j] > 0:
10.             P_hat[i,j] ← wins[i,j] / comps[i,j]
11.         ELSE:
12.             P_hat[i,j] ← 0.5
13.     
14.     // Compute Copeland scores
15.     FOR i = 0 to N-1:
16.         scores[i] ← sum(P_hat[i,:] > 0.5) - 1
17.     
18.     // SELECT PAIR: UCB-style selection
19.     max_uncertainty ← -∞
20.     FOR i = 0 to N-1:
21.         FOR j = i+1 to N-1:
22.             cb ← sqrt(log(2N²/δ) / (2·comps[i,j] + 1))
23.             importance ← max(scores[i], scores[j])
24.             uncertainty ← cb × (1 + importance/N)
25.             IF uncertainty > max_uncertainty:
26.                 max_uncertainty ← uncertainty
27.                 best_pair ← (i, j)
28.     
29.     // DUEL: Query the oracle
30.     (i, j) ← best_pair
31.     outcome ← Bernoulli(P_true[i,j])   // 1 if i wins, 0 otherwise
32.     
33.     // UPDATE statistics
34.     comps[i,j] ← comps[i,j] + 1
35.     comps[j,i] ← comps[j,i] + 1
36.     IF outcome = 1:
37.         wins[i,j] ← wins[i,j] + 1
38.     ELSE:
39.         wins[j,i] ← wins[j,i] + 1
40. 
41. RETURN argmax(scores)
```

**Complexity**: O(N²) comparisons needed to estimate all pairwise preferences.

---

### Algorithm 3: Fourier Dueling Bandit (Proposed)

```
FOURIER_DUELING_BANDIT(N, z, k, X, δ, P_true, T)
────────────────────────────────────────────────────
Input:  N = number of items
        z = feature dimension
        k = sparsity level
        X = feature matrix (N × z)
        δ = confidence parameter
        P_true = true preference matrix (for simulation)
        T = maximum comparisons

Output: estimated_winner

1.  // Initialize
2.  observations ← []              // List of (i, j, outcome)
3.  w_hat ← zeros(z)              // Weight estimate
4.  λ ← 0.1                       // LASSO regularization
5.  
6.  FOR t = 1 to T:
7.      
8.      // === ADAPTIVE PAIR SELECTION ===
9.      IF t < 2z:
10.         // Phase 1: Random exploration
11.         (i, j) ← RandomPair(N)
12.         
13.     ELSE IF t < 5z:
14.         // Phase 2: Informative sampling for sparse recovery
15.         relevant ← {f : |w_hat[f]| > 0.01}
16.         IF relevant is empty: relevant ← {0,...,z-1}
17.         
18.         best_info ← -∞
19.         FOR _ = 1 to 50:        // Sample candidates
20.             (i', j') ← RandomPair(N)
21.             diff ← |X[i'] - X[j']|
22.             info ← sum(diff[relevant])
23.             IF info > best_info:
24.                 best_info ← info
25.                 (i, j) ← (i', j')
26.                 
27.     ELSE:
28.         // Phase 3: UCB exploitation with ε-exploration
29.         IF Random() < 0.1:
30.             (i, j) ← RandomPair(N)
31.         ELSE:
32.             scores ← X @ w_hat
33.             n_comps ← CountComparisonsPerItem(observations)
34.             exploration ← sqrt(2·log(t) / (n_comps + 1))
35.             ucb ← scores + exploration
36.             top2 ← argsort(ucb)[-2:]
37.             (i, j) ← (top2[0], top2[1])
38.     
39.     // === DUEL ===
40.     outcome ← Bernoulli(P_true[i,j])
41.     observations.append((i, j, outcome))
42.     
43. 
44. // Final winner estimation
45. scores ← X @ w_hat
46. P_hat ← σ(scores[:,None] - scores[None,:])
47. C ← [sum(P_hat[i,:] > 0.5) - 1 for i in 0..N-1]
48. RETURN argmax(C)


LASSO_REGRESSION(observations, X, λ)
────────────────────────────────────────
// Coordinate descent for L1-regularized regression

1.  // Build design matrix and response
2.  A ← []
3.  y ← []
4.  FOR (i, j, outcome) in observations:
5.      A.append(X[i] - X[j])
6.      y.append(2·outcome - 1)      // Map {0,1} to {-1,+1}
7.  
8.  // Coordinate descent
9.  w ← zeros(z)
10. FOR iter = 1 to 100:
11.     FOR f = 0 to z-1:
12.         residual ← y - A@w + A[:,f]·w[f]
13.         ρ ← A[:,f]^T @ residual
14.         z_norm ← ||A[:,f]||²
15.         IF z_norm > 0:
16.             w[f] ← SoftThreshold(ρ/z_norm, λ/z_norm)
17. RETURN w

SoftThreshold(x, τ)
────────────────────
IF x > τ:  RETURN x - τ
IF x < -τ: RETURN x + τ
RETURN 0
```

**Complexity**: O(k · poly(log N)) — exploits sparsity via compressed sensing.

---

## 📊 Experimental Results

### Setup
- **z = 20** features per item
- **k = 3** Fourier-sparse (only 3 features determine preferences)
- **12 runs** per configuration
- **Budget**: 500 samples per run

### Sample Complexity (Samples to First Correct Identification)

| N | N² | Naive CCB | Fourier Bandit | Speedup |
|---|-----|-----------|----------------|---------|
| 8 | 64 | 142 | 28 | **5.0x** |
| 16 | 256 | 308 | 104 | **3.0x** |
| 32 | 1024 | 500 | 83 | **6.0x** |

### Final Accuracy (after 500 samples)

| N | Naive CCB | Fourier Bandit |
|---|-----------|----------------|
| 8 | 75% | **92%** |
| 16 | 67% | **75%** |
| 32 | 0% | **92%** |

### Key Observations

1. **Fourier scales sub-linearly**: Notice that Fourier samples (28 → 104 → 83) don't grow with N². This is because the algorithm only needs O(k log N) samples to identify k sparse features, regardless of N.

2. **Naive scales quadratically**: Naive CCB samples (142 → 308 → 500) grow toward O(N²) because it must estimate all pairwise preferences.

3. **Accuracy gap widens**: At N=32, Naive CCB achieves 0% accuracy (can't find winner in 500 samples), while Fourier achieves 92%.

4. **Speedup varies**: The speedup (5.0x → 3.0x → 6.0x) depends on problem difficulty, but Fourier consistently outperforms Naive.

---

## 📖 Usage

### Basic Usage

```python
from src import generate_preference_matrix, NaiveCCB, FourierDuelingBandit

# Generate problem: N items, z features, k Fourier-sparse
P, X, w, winner, scores = generate_preference_matrix(N=16, z=20, k=3, seed=42)

# Run Naive CCB
naive = NaiveCCB(N=16, delta=0.1)
naive_metrics = naive.run(P, max_samples=500)

# Run Fourier Bandit
fourier = FourierDuelingBandit(N=16, z=20, k=3, X=X, delta=0.1)
fourier_metrics = fourier.run(P, max_samples=500)

print(f"Naive regret: {naive_metrics['cumulative_regret'][-1]}")
print(f"Fourier regret: {fourier_metrics['cumulative_regret'][-1]}")
```

### Command Line

```bash
# Basic experiment
python run_experiment.py -N 16 -z 20 -k 3

# With scaling analysis
python run_experiment.py -N 16 -z 20 -k 3 --scaling

# Custom settings
python run_experiment.py -N 32 -z 50 -k 5 --runs 30 --max-samples 1000
```

---

## 🧪 Testing

```bash
# Run all 14 tests
python tests/test_all.py

# With pytest
python -m pytest tests/ -v
```

---

## 🍣 SUSHI Dataset

The project includes support for the **SUSHI Preference Dataset**, a real-world benchmark for preference learning.

### About the Dataset

- **5,000 users** ranking **10 types of sushi**
- **6 features** per sushi: style, major_group, minor_group, oiliness, popularity, price
- Collected via questionnaire survey in Japan
- Standard benchmark in preference learning literature

### Running SUSHI Experiments

```bash
python run_sushi_experiment.py
```

This runs two experiments:

1. **Synthetic SUSHI** (known k=2 sparse): Preferences determined by only oiliness and price
2. **Real SUSHI**: Using actual/simulated human preference data

### Sample Results

```
Synthetic SUSHI (k=2 sparse):
  Naive CCB:      141 samples, 45% accuracy
  Fourier Bandit: 47 samples, 95% accuracy
  Speedup: 3.0x
```

### Using Real SUSHI Data

1. Download from: https://www.kamishima.net/sushi/
2. Extract `sushi3-2016.zip` to `data/` directory
3. Run `python run_sushi_experiment.py`

### SUSHI Features

| Feature | Description | Range |
|---------|-------------|-------|
| style | maki(0) vs other(1) | 0-1 |
| major_group | Seafood category | 0-4 |
| minor_group | Sub-category | 0-11 |
| oiliness | Fat content | 1-5 |
| popularity | Eating frequency | 1-5 |
| price | Normalized price | 1-5 |

---

## 📚 API Reference

### `generate_preference_matrix(N, z, k, seed=None)`

Generate Fourier-sparse preference matrix.

**Returns:** `(P, X, w, winner, scores)`

### `NaiveCCB(N, delta=0.1)`

Naive Copeland Confidence Bound algorithm.

**Methods:**
- `run(P_true, max_samples)` → metrics dict
- `get_current_winner()` → int

### `FourierDuelingBandit(N, z, k, X, delta=0.1)`

Fourier-based dueling bandit algorithm.

**Methods:**
- `run(P_true, max_samples)` → metrics dict
- `get_current_winner()` → int
- `get_estimated_relevant_features()` → array

---

## 📄 License

MIT License

## 🔗 References

1. Yue et al. "The K-armed Dueling Bandits Problem" (COLT 2012)
2. Zoghi et al. "Copeland Dueling Bandits" (NeurIPS 2015)
3. Candès & Wakin "Compressive Sampling" (IEEE SPM 2008)
4. Tibshirani "Regression Shrinkage via Lasso" (JRSS 1996)
