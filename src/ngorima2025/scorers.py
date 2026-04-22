"""
scorers.py — KsaaiP2 Pipeline v4_updated
src/ngorima2025/scorers.py
"""

import os
import subprocess
import tempfile
import warnings
import numpy as np

FALLBACK_FLAGS = {"xi_n": False, "DC": False, "MI": False, "MIC": False}

THEORETICAL_EXPONENT = {
    "xi_n": 1.00, "DC": 2.00, "MI": 1.00,
    "MIC": 1.20, "pearson": 1.00, "spearman": 1.06,
}

SCORER_HYPERPARAMS = {
    "xi_n":    {"params": "None (rank-based)"},
    "DC":      {"params": "None (U-statistic)"},
    "MI":      {"params": "n_neighbors=3, random_state=42"},
    "MIC":     {"params": "alpha=0.6, c=15"},
    "pearson": {"params": "None"},
    "spearman":{"params": "None"},
}

_MIC_CONDA_ENV = os.environ.get("MIC_CONDA_ENV", "ngorima_mic")
_MIC_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mic_worker.py")
_MIC_MAX_NP = int(os.environ.get("MIC_MAX_NP", 50000))
USE_FALLBACK_MIC = False

def _check_mic_env():
    try:
        r = subprocess.run(["conda", "run", "-n", _MIC_CONDA_ENV, "python", "-c", "import minepy; print(chr(111)+chr(107))"], capture_output=True, text=True, timeout=30)
        return r.returncode == 0 and "ok" in r.stdout
    except Exception:
        return False

_MIC_ENV_AVAILABLE = _check_mic_env()
if not _MIC_ENV_AVAILABLE:
    FALLBACK_FLAGS["MIC"] = True
    USE_FALLBACK_MIC = True
    warnings.warn("ngorima_mic env not found — MIC returns NaN", RuntimeWarning, stacklevel=2)

def score_xi_n(X, y):
    try:
        n = len(y); order = np.argsort(X)
        rank_y = np.argsort(np.argsort(y[order])) + 1
        num = n * np.sum(np.abs(np.diff(rank_y)))
        den = 2 * np.sum(rank_y * (n - rank_y))
        return float(0.0 if den == 0 else 1.0 - num / den)
    except Exception:
        FALLBACK_FLAGS["xi_n"] = True
        from scipy.stats import spearmanr
        return float(abs(spearmanr(X, y).correlation))

def score_dc(X, y):
    try:
        import dcor
        return float(dcor.distance_correlation(X, y))
    except ImportError:
        FALLBACK_FLAGS["DC"] = True
        return float(abs(np.corrcoef(X, y)[0, 1]))

def score_mi(X, y, n_neighbors=3, random_state=42):
    try:
        from sklearn.feature_selection import mutual_info_regression
        X2d = X.reshape(-1, 1) if X.ndim == 1 else X
        return float(mutual_info_regression(X2d, y, n_neighbors=n_neighbors, random_state=random_state)[0])
    except ImportError:
        FALLBACK_FLAGS["MI"] = True
        return float("nan")

def score_mic_subprocess(X, y, alpha=0.6, c=15.0):
    if USE_FALLBACK_MIC or not _MIC_ENV_AVAILABLE:
        return float("nan")
    if len(X) > _MIC_MAX_NP:
        return float("nan")
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        np.savez(tmp_path, X=X.astype(np.float64), y=y.astype(np.float64), alpha=np.array([alpha]), c=np.array([c]))
        r = subprocess.run(["conda", "run", "--no-capture-output", "-n", _MIC_CONDA_ENV, "python", _MIC_WORKER, tmp_path], capture_output=True, text=True, timeout=300)
        return float("nan") if r.returncode != 0 else float(r.stdout.strip())
    except Exception:
        return float("nan")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def score_pearson(X, y):
    return float(np.corrcoef(X, y)[0, 1])

def score_spearman(X, y):
    from scipy.stats import spearmanr
    return float(spearmanr(X, y).correlation)

_SCORER_REGISTRY = {
    "xi_n": score_xi_n, "DC": score_dc, "MI": score_mi,
    "MIC": score_mic_subprocess, "pearson": score_pearson, "spearman": score_spearman,
}

xi_scorer = score_xi_n; dc_scorer = score_dc; mi_scorer = score_mi
mic_scorer = score_mic_subprocess; pearson_scorer = score_pearson; spearman_scorer = score_spearman

def get_all_scorers():
    return dict(_SCORER_REGISTRY)

def get_xi_scorer():
    return score_xi_n

def get_scorer(name):
    if name not in _SCORER_REGISTRY:
        raise KeyError(f"Unknown scorer '{name}'. Available: {list(_SCORER_REGISTRY)}")
    return _SCORER_REGISTRY[name]

def get_theoretical_exponent(name):
    return THEORETICAL_EXPONENT[name]
