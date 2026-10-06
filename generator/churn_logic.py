import hashlib
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar

def compute_churn_probabilities(df: pd.DataFrame, columns: list, churn_contributions: dict, target_churn_rate: float, noise_level: float) -> np.ndarray:
    """
    Compute probability of churn for each row using an auto-calibrated logistic model.
    """
    n_samples = len(df)
    if n_samples == 0:
        return np.array([])
        
    logits = np.zeros(n_samples)
    
    # 1. Linear contributions from active columns
    for col_info in columns:
        col = getattr(col_info, 'name', None) or col_info.get('name')
        weight = float(getattr(col_info, 'churn_weight', 0.0) if hasattr(col_info, 'churn_weight') else col_info.get('churn_weight', 0.0))
        dtype = getattr(col_info, 'dtype', None) or col_info.get('dtype') or col_info.get('type')
        
        if weight == 0.0 or col not in df.columns:
            continue
            
        vals = df[col].values
        
        if dtype == 'numeric':
            numeric_vals = pd.to_numeric(pd.Series(vals), errors='coerce').fillna(0.0).values
            std = np.std(numeric_vals)
            std_adj = std if std > 1e-4 else 1.0
            norm_vals = (numeric_vals - np.mean(numeric_vals)) / std_adj
            logits += norm_vals * weight
            
        elif dtype == 'boolean':
            bool_vals = vals.astype(bool).astype(float)
            logits += (bool_vals - 0.5) * weight * 2.0
            
        elif dtype == 'categorical':
            # Deterministic MD5 hash-based mapping to [-1.0, 1.0] for true cross-process reproducibility
            unique_vals = np.unique(vals)
            effects = {}
            for v in unique_vals:
                h_int = int(hashlib.md5(str(v).encode('utf-8')).hexdigest()[:8], 16) % 10000
                effects[v] = (h_int / 5000.0) - 1.0  # maps uniformly to [-1, 1]
            mapped = np.array([effects.get(v, 0.0) for v in vals])
            logits += mapped * weight

    # 2. Add interaction and non-linear effects
    logits += churn_contributions.get('interactions', np.zeros(n_samples))
    logits += churn_contributions.get('nonlinears', np.zeros(n_samples))
    
    # 3. Add noise
    if noise_level > 0:
        logits += np.random.normal(0, noise_level, n_samples)
    
    # 4. Define sigmoid
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))
    
    # Pre-center logits around mean to keep Brent's search well-conditioned
    logits_mean = np.mean(logits)
    centered_logits = logits - logits_mean
    
    # Calibrate intercept so average probability matches target_churn_rate
    target = np.clip(target_churn_rate, 0.01, 0.99)
    def objective(shift):
        probs = sigmoid(centered_logits + shift)
        return np.mean(probs) - target
        
    intercept_shift = None
    # Expanding bracket search
    for b in [10.0, 30.0, 100.0, 300.0]:
        try:
            f_low = objective(-b)
            f_high = objective(b)
            if f_low * f_high < 0:
                res = root_scalar(objective, bracket=[-b, b], method='brentq')
                intercept_shift = res.root
                break
        except Exception:
            continue
            
    if intercept_shift is None:
        # Fallback to analytical log-odds approximation
        intercept_shift = np.log(target / (1.0 - target))
        
    final_probs = sigmoid(centered_logits + intercept_shift)
    return final_probs

def assign_churn_labels(probabilities: np.ndarray) -> np.ndarray:
    """
    Sample Bernoulli to assign final 0/1 labels.
    """
    if len(probabilities) == 0:
        return np.array([])
    return np.random.binomial(1, probabilities)
