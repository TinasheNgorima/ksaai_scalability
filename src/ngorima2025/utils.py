"""
utils.py — KsaaiP2 Pipeline v4_updated
src/ngorima2025/utils.py
"""

import os
import gc
import json
import time
import platform
import subprocess
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

N_REPS      = int(os.environ.get("NGORIMA_N_REPS",     30))
N_REPS_MIC  = int(os.environ.get("NGORIMA_N_REPS_MIC",  5))
N_WARMUP    = int(os.environ.get("NGORIMA_N_WARMUP",    2))
FAST_MODE   = os.environ.get("NGORIMA_FAST", "0") == "1"

def timed_call(fn, *args, n_reps=N_REPS, n_warmup=N_WARMUP, gc_collect=True, **kwargs):
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    times = []
    for _ in range(n_reps):
        if gc_collect:
            gc.collect()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    return (float(np.median(times)), float(np.percentile(times, 5)),
            float(np.percentile(times, 95)), times.tolist())

def cov_of_times(times):
    arr = np.array(times)
    if np.mean(arr) == 0:
        return float("nan")
    cov = np.std(arr) / np.mean(arr)
    if cov > 0.10:
        warnings.warn(f"MIC timing CoV = {cov:.3f} > 0.10", UserWarning, stacklevel=2)
    return float(cov)

def dc_ram_required_gb(n):
    return (n ** 2 * 8) / (1024 ** 3)

def available_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        return float("inf")

def dc_feasible(n, safety_factor=1.1):
    required = dc_ram_required_gb(n) * safety_factor
    available = available_ram_gb()
    if required > available:
        warnings.warn(f"DC at n={n}: requires {required:.1f} GB, available {available:.1f} GB", ResourceWarning, stacklevel=2)
        return False
    return True

def save_checkpoint(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)

def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def bootstrap_median_ratio_ci(times_a, times_b, n_bootstrap=10000, conf=0.95, seed=42):
    rng = np.random.default_rng(seed)
    a, b = np.array(times_a), np.array(times_b)
    point_ratio = np.median(a) / np.median(b)
    ratios = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        ra = rng.choice(a, size=len(a), replace=True)
        rb = rng.choice(b, size=len(b), replace=True)
        med_b = np.median(rb)
        ratios[i] = np.median(ra) / med_b if med_b > 0 else float("nan")
    alpha = 1 - conf
    ci_lo = float(np.nanpercentile(ratios, 100 * alpha / 2))
    ci_hi = float(np.nanpercentile(ratios, 100 * (1 - alpha / 2)))
    return float(point_ratio), ci_lo, ci_hi

def wilcoxon_timing_test(times_a, times_b, alternative="two-sided"):
    a, b = np.array(times_a), np.array(times_b)
    if len(a) != len(b):
        raise ValueError(f"Paired vectors must have equal length: {len(a)} vs {len(b)}")
    result = stats.wilcoxon(a, b, alternative=alternative)
    return float(result.statistic), float(result.pvalue)

def measure_mic_spawn_overhead(conda_env="ngorima_mic", n_reps=10, conf=0.95):
    times_ms = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        try:
            subprocess.run(["conda", "run", "--no-capture-output", "-n", conda_env, "python", "-c", "pass"], capture_output=True, timeout=60)
        except Exception:
            pass
        times_ms.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times_ms)
    alpha = 1 - conf
    return {"median_ms": float(np.median(arr)), "ci_lo_ms": float(np.percentile(arr, 100 * alpha / 2)), "ci_hi_ms": float(np.percentile(arr, 100 * (1 - alpha / 2))), "n_reps": n_reps, "conda_env": conda_env}

def log_hardware_fingerprint():
    info = {"python_version": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(), "machine": platform.machine()}
    try:
        import psutil
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["logical_cores"] = psutil.cpu_count(logical=True)
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
        freq = psutil.cpu_freq()
        info["cpu_freq_max_mhz"] = round(freq.max, 1) if freq else None
    except ImportError:
        info["physical_cores"] = os.cpu_count()
    try:
        gov = subprocess.check_output(["cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"], text=True, timeout=5).strip()
        info["cpu_governor"] = gov
    except Exception:
        info["cpu_governor"] = "unknown"
    try:
        import numpy as np
        cfg = np.__config__
        blas_str = str(getattr(cfg, "blas_opt_info", ""))
        info["blas_backend"] = "MKL" if "mkl" in blas_str.lower() else "OpenBLAS/other"
    except Exception:
        info["blas_backend"] = "unknown"
    for pkg in ["numpy", "scipy", "sklearn", "dcor", "minepy", "yaml", "joblib"]:
        try:
            import importlib
            mod = importlib.import_module(pkg.replace("-", "_").split(".")[0])
            info[f"pkg_{pkg}"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[f"pkg_{pkg}"] = "not installed"
    return info
