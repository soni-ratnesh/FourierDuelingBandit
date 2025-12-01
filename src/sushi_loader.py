"""
SUSHI Dataset Loader

Provides utilities to load the SUSHI preference dataset and convert it
to the format required by our Fourier Copeland Bandit algorithms.

Dataset: https://www.kamishima.net/sushi/
Paper: Kamishima, T. "Nantonac Collaborative Filtering" (KDD 2003)

The SUSHI dataset contains:
- 5000 users ranking 10 types of sushi
- 6 features per sushi item
- Complete preference rankings

Since the dataset requires manual download due to licensing, this module
also provides a synthetic version that mimics the SUSHI structure.
"""

import numpy as np
import os
from typing import Tuple, Optional, Dict

# SUSHI item features (from the actual dataset documentation)
# 10 types of sushi with 6 features each
SUSHI_ITEMS = [
    "ebi (shrimp)",
    "anago (sea eel)", 
    "maguro (tuna)",
    "ika (squid)",
    "uni (sea urchin)",
    "sake (salmon)",
    "tamago (egg)",
    "toro (fatty tuna)",
    "tekka-maki (tuna roll)",
    "kappa-maki (cucumber roll)"
]

# Features: [style, major_group, minor_group, oiliness, popularity, price]
# style: 0=maki, 1=other
# major_group: 0=aomono, 1=akami, 2=shiromi, etc.
# oiliness: 1-5 scale (higher = more oily)
# popularity: eating frequency 1-5
# price: normalized 1-5

SUSHI_FEATURES_RAW = np.array([
    # style, major, minor, oily, popular, price
    [1, 2, 5, 2, 4, 3],   # ebi (shrimp)
    [1, 2, 6, 4, 3, 4],   # anago (sea eel)
    [1, 1, 0, 2, 5, 3],   # maguro (tuna)
    [1, 2, 4, 1, 4, 2],   # ika (squid)
    [1, 3, 9, 4, 2, 5],   # uni (sea urchin)
    [1, 0, 1, 3, 4, 3],   # sake (salmon)
    [1, 4, 10, 2, 3, 1],  # tamago (egg)
    [1, 1, 0, 5, 3, 5],   # toro (fatty tuna)
    [0, 1, 0, 2, 3, 2],   # tekka-maki (tuna roll)
    [0, 4, 11, 1, 2, 1],  # kappa-maki (cucumber roll)
], dtype=float)

# Feature names
SUSHI_FEATURE_NAMES = [
    "style (0=maki, 1=other)",
    "major_group",
    "minor_group", 
    "oiliness",
    "popularity",
    "price"
]


def normalize_features(X: np.ndarray) -> np.ndarray:
    """Normalize features to zero mean, unit variance."""
    X_norm = X.copy()
    for j in range(X.shape[1]):
        mean = X[:, j].mean()
        std = X[:, j].std()
        if std > 0:
            X_norm[:, j] = (X[:, j] - mean) / std
        else:
            X_norm[:, j] = X[:, j] - mean
    return X_norm


def get_sushi_features(normalize: bool = True) -> Tuple[np.ndarray, list, list]:
    """
    Get SUSHI item features.
    
    Returns
    -------
    X : np.ndarray (10, 6)
        Feature matrix
    item_names : list
        Names of sushi items
    feature_names : list
        Names of features
    """
    X = SUSHI_FEATURES_RAW.copy()
    if normalize:
        X = normalize_features(X)
    return X, SUSHI_ITEMS.copy(), SUSHI_FEATURE_NAMES.copy()


def generate_synthetic_sushi_preferences(
    n_users: int = 1000,
    sparse_features: list = [3, 5],  # oiliness and price
    noise_level: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Generate synthetic SUSHI preferences with known sparse structure.
    
    This creates preferences where only certain features (e.g., oiliness, price)
    determine the rankings, making it suitable for testing sparse algorithms.
    
    Parameters
    ----------
    n_users : int
        Number of synthetic users
    sparse_features : list
        Indices of features that determine preferences (k-sparse)
    noise_level : float
        Amount of random noise in preferences
    seed : int, optional
        Random seed
        
    Returns
    -------
    P : np.ndarray (10, 10)
        Preference matrix where P[i,j] = P(item i preferred over item j)
    X : np.ndarray (10, 6)
        Feature matrix
    w : np.ndarray (6,)
        True sparse weight vector
    winner : int
        True Copeland winner
    """
    if seed is not None:
        np.random.seed(seed)
    
    X, _, _ = get_sushi_features(normalize=True)
    N, z = X.shape
    
    # Create k-sparse weight vector
    k = len(sparse_features)
    w = np.zeros(z)
    w[sparse_features] = np.random.randn(k)
    
    # Compute item scores
    scores = X @ w
    
    # Build preference matrix using Bradley-Terry model
    score_diff = scores[:, np.newaxis] - scores[np.newaxis, :]
    P = 1 / (1 + np.exp(-score_diff))
    
    # Add noise
    if noise_level > 0:
        noise = np.random.randn(N, N) * noise_level
        P = np.clip(P + noise, 0.01, 0.99)
        # Maintain anti-symmetry
        P = (P + (1 - P.T)) / 2
        np.fill_diagonal(P, 0.5)
    
    # Find Copeland winner
    wins = (P > 0.5).astype(int)
    np.fill_diagonal(wins, 0)
    copeland_scores = wins.sum(axis=1)
    winner = int(np.argmax(copeland_scores))
    
    return P, X, w, winner


def load_sushi_rankings(filepath: str) -> np.ndarray:
    """
    Load SUSHI ranking data from file.
    
    The ranking file format (sushi3a.5000.10.order):
    - First line: "10 1" (number of items, version)
    - Each subsequent line: "0 10 [10 integers]"
      - First "0" indicates complete ranking
      - "10" indicates 10 items
      - Next 10 integers are the item IDs in preference order (most preferred first)
    
    Parameters
    ----------
    filepath : str
        Path to the .order file
        
    Returns
    -------
    rankings : np.ndarray (n_users, 10)
        Rankings where rankings[u, i] is the rank of item i for user u
        (0 = most preferred, 9 = least preferred)
    """
    rankings = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
        # Skip header line
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 12:  # "0 10" + 10 items
                # Skip first two values (0 and 10)
                order = [int(x) for x in parts[2:12]]
                # Convert order to ranks
                # order[0] is most preferred (rank 0), order[9] is least preferred (rank 9)
                rank = [0] * 10
                for position, item in enumerate(order):
                    rank[item] = position
                rankings.append(rank)
    
    return np.array(rankings)


def rankings_to_preferences(rankings: np.ndarray) -> np.ndarray:
    """
    Convert user rankings to aggregate preference matrix.
    
    Parameters
    ----------
    rankings : np.ndarray (n_users, N)
        Rankings where rankings[u, i] is the rank of item i for user u
        (lower rank = more preferred)
        
    Returns
    -------
    P : np.ndarray (N, N)
        Preference matrix where P[i,j] = proportion of users preferring i over j
    """
    n_users, N = rankings.shape
    P = np.zeros((N, N))
    
    for u in range(n_users):
        for i in range(N):
            for j in range(N):
                if rankings[u, i] < rankings[u, j]:  # i ranked higher (lower number)
                    P[i, j] += 1
    
    # Normalize
    P = P / n_users
    np.fill_diagonal(P, 0.5)
    
    return P


def load_sushi_item_features(filepath: str) -> Tuple[np.ndarray, list]:
    """
    Load SUSHI item features from sushi3.idata file.
    
    File format (tab-separated):
    ID  name  style  major_group  minor_group  heaviness  eat_freq  price  sold_freq
    
    Returns
    -------
    X : np.ndarray (100, 6)
        Feature matrix (we use 6 key features)
    names : list
        Item names
    """
    items = []
    names = []
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 9:
                item_id = int(parts[0])
                name = parts[1]
                style = int(parts[2])  # 0=maki, 1=other
                major_group = int(parts[3])
                minor_group = int(parts[4])
                heaviness = float(parts[5])  # oiliness/heaviness
                eat_freq = float(parts[6])   # popularity
                price = float(parts[7])
                
                items.append([style, major_group, minor_group, heaviness, eat_freq, price])
                names.append(name)
    
    return np.array(items), names


def load_sushi_dataset(data_dir: str = "data") -> Dict:
    """
    Load the full SUSHI dataset.
    
    Expected files in data_dir or data_dir/sushi3-2016/:
    - sushi3a.5000.10.order: rankings of 10 sushi by 5000 users
    - sushi3.idata: item features
    
    Parameters
    ----------
    data_dir : str
        Directory containing SUSHI data files
        
    Returns
    -------
    data : dict
        Dictionary containing:
        - P: preference matrix (10, 10)
        - X: feature matrix (10, 6)
        - rankings: raw rankings (5000, 10)
        - winner: Copeland winner
        - item_names: list of item names
        - feature_names: list of feature names
    """
    # Try to find the files
    possible_paths = [
        data_dir,
        os.path.join(data_dir, "sushi3-2016"),
    ]
    
    order_file = None
    idata_file = None
    
    for path in possible_paths:
        of = os.path.join(path, "sushi3a.5000.10.order")
        idf = os.path.join(path, "sushi3.idata")
        if os.path.exists(of):
            order_file = of
        if os.path.exists(idf):
            idata_file = idf
    
    if order_file and os.path.exists(order_file):
        print(f"Loading real SUSHI data from {order_file}")
        rankings = load_sushi_rankings(order_file)
        P = rankings_to_preferences(rankings)
        
        # Load real item features if available
        if idata_file and os.path.exists(idata_file):
            print(f"Loading item features from {idata_file}")
            X_full, names_full = load_sushi_item_features(idata_file)
            # Only use first 10 items (the ones in sushi3a)
            X = X_full[:10]
            item_names = names_full[:10]
            X = normalize_features(X)
        else:
            X, item_names, _ = get_sushi_features(normalize=True)
    else:
        # Use synthetic data
        print(f"Warning: SUSHI data not found in {data_dir}")
        print("Using synthetic SUSHI-like preferences.")
        print("Download from: https://www.kamishima.net/sushi/")
        P, X, w, winner = generate_synthetic_sushi_preferences(n_users=5000, seed=42)
        rankings = None
        _, item_names, _ = get_sushi_features()
    
    feature_names = ["style", "major_group", "minor_group", "heaviness/oiliness", "eat_frequency", "price"]
    
    # Compute Copeland winner
    wins = (P > 0.5).astype(int)
    np.fill_diagonal(wins, 0)
    copeland_scores = wins.sum(axis=1)
    winner = int(np.argmax(copeland_scores))
    
    return {
        'P': P,
        'X': X,
        'rankings': rankings,
        'winner': winner,
        'copeland_scores': copeland_scores,
        'item_names': item_names,
        'feature_names': feature_names,
        'n_users': len(rankings) if rankings is not None else 0
    }


def print_sushi_summary(data: Dict):
    """Print summary of SUSHI dataset."""
    P = data['P']
    X = data['X']
    winner = data['winner']
    scores = data['copeland_scores']
    items = data['item_names']
    
    print("=" * 60)
    print("SUSHI PREFERENCE DATASET SUMMARY")
    print("=" * 60)
    print(f"Items: {len(items)}")
    print(f"Features: {X.shape[1]}")
    print()
    
    print("Copeland Scores:")
    sorted_idx = np.argsort(scores)[::-1]
    for i, idx in enumerate(sorted_idx):
        marker = "**WINNER**" if idx == winner else ""
        print(f"  {i+1}. {items[idx]:<25} Score: {scores[idx]} {marker}")
    
    print()
    print("Preference Matrix (rows beat columns):")
    print("     ", end="")
    for i in range(len(items)):
        print(f"{i:5}", end="")
    print()
    for i in range(len(items)):
        print(f"{i:3}: ", end="")
        for j in range(len(items)):
            print(f"{P[i,j]:5.2f}", end="")
        print(f"  {items[i]}")


if __name__ == "__main__":
    # Demo with synthetic data
    print("SUSHI Dataset Loader Demo")
    print("=" * 60)
    
    # Generate synthetic SUSHI preferences
    print("\n1. Generating synthetic SUSHI preferences...")
    print("   (Sparse features: oiliness [3] and price [5])")
    
    P, X, w, winner = generate_synthetic_sushi_preferences(
        n_users=5000,
        sparse_features=[3, 5],  # oiliness and price
        seed=42
    )
    
    print(f"\n   True sparse weights:")
    _, _, feature_names = get_sushi_features()
    for i, (name, weight) in enumerate(zip(feature_names, w)):
        if abs(weight) > 0.01:
            print(f"     {name}: {weight:.3f}")
    
    print(f"\n   Copeland Winner: {SUSHI_ITEMS[winner]}")
    
    # Load or generate full dataset
    print("\n2. Loading full dataset...")
    data = load_sushi_dataset()
    print_sushi_summary(data)
    
    print("\n" + "=" * 60)
    print("To use real SUSHI data:")
    print("1. Download from: https://www.kamishima.net/sushi/")
    print("2. Extract sushi3-2016.zip to data/ directory")
    print("3. Run this script again")
    print("=" * 60)
