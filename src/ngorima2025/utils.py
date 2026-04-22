"""
utils.py — KsaaiP2 Pipeline v4_updated
src/ngorima2025/utils.py

Timing harness, statistical inference utilities, and hardware fingerprinting.

Key functions:
  - timed_call(): perf_counter timing with warm-up and N_REPS
  - bootstrap_median_ratio_ci(): 95% CI on ratio of median runtimes (RQ3)
  - wilcoxon_timing_test(): paired Wilcoxon signed-rank test (RQ3)
  - measure_mic_spawn_overhead(): fixed MIC subprocess cost in ms (RQ5)
  - log_hardware_fingerprint(): CPU/RAM/OS metadata for system_state.json
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


# ── Constants (overridable via config.yaml / env) ─────────────────────────────
N_REPS      = int(os.environ.get("NGORIMA_N_REPS",     30))
N_REPS_MIC  = int(os.environ.get("NGORIMA_N_REPS_MIC",  5))
N_WARMUP    = int(os.environ.get("NGORIMA_N_WARMUP",    2))
FAST_MODE   = os.environ.get("NGORIMA_FAST", "0") == "1"


# ── Timing harness ─────────────────────────────────────────────────────────────

def timed_call(
    fn: Callable,
    *args,
    n_reps: int = N_REPS,
    n_warmup: int = N_WARMUP,
    gc_collect: bool = True,
    **kwargs,
) -> Tuple[float, float, float, List[float]]:
    """
    Time fn(*args, **kwargs) with warm-up and repeated measurements.

    Uses time.perf_counter() — monotonic, sub-microsecond resolution.
    Warm-up repetitions are discarded before timing begins.

    Parameters
    ----------
    fn        : callable to time
    *args     : positional arguments to fn
    n_reps    : number of measured repetitions (default 30)
    n_warmup  : warm-up repetitions discarded (default 2)
    gc_collect: collect garbage before each rep to reduce noise

    Returns
    -------
    median_s  : median wall-clock time in seconds
    p5_s      : 5th percentile
    p95_s     : 95th percentile
    all_times : list of all n_reps measurements
    """
    # Warm-up
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
    return (
        float(np.median(times)),
        float(np.percentile(times, 5)),
        float(np.percentile(times, 95)),
        times.tolist(),
    )


def cov_of_times(times: List[float]) -> float:
    """Coefficient of variation of timing measurements."""
    arr = np.array(times)
    if np.mean(arr) == 0:
        return float("nan")
    cov = np.std(arr) / np.mean(arr)
    if cov > 0.10:
        warnings.warn(
            f"MIC timing CoV = {cov:.3f} > 0.10 — consider increasing N_REPS_MIC",
            UserWarning, stacklevel=2,
        )
    return float(cov)


# ── RAM guard ──────────────────────────────────────────────────────────────────

def dc_ram_required_gb(n: int) -> float:
    """Peak RAM for DC distance matrix: n² × 8 bytes, in GB."""
    return (n ** 2 * 8) / (1024 ** 3)


def available_ram_gb() -> float:
    """Available system RAM in GB via psutil."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        return float("inf")


def dc_feasible(n: int, safety_factor: float = 1.1) -> bool:
    """Return True if DC can run at n without exhausting RAM."""
    required = dc_ram_required_gb(n) * safety_factor
    available = available_ram_gb()
    if required > available:
        warnings.warn(
            f"DC at n={n}: requires {required:.1f} GB, "
            f"available {available:.1f} GB — skipping",
            ResourceWarning, stacklevel=2,
        )
        return False
    return True


# ── Checkpoint / resume ────────────────────────────────────────────────────────

def save_checkpoint(path: str, state: dict) -> None:
    """Atomic checkpoint write (tmp → replace) for session resilience."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> Optional[dict]:
    """Load checkpoint; return None if not found."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── Statistical inference (RQ3) ────────────────────────────────────────────────

def bootstrap_median_ratio_ci(
    times_a: List[float],
    times_b: List[float],
    n_bootstrap: int = 10_000,
    conf: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap 95% CI on the ratio of median runtimes (median_a / median_b).

    Uses 10,000 resamples of the per-rep timing vectors.
    Addresses RQ3: inferential support for claims such as 'ξₙ scales better than DC'.

    Parameters
    ----------
    times_a   : timing measurements for method A (e.g., ξₙ)
    times_b   : timing measurements for method B (e.g., DC)
    n_bootstrap: number of bootstrap resamples
    conf      : confidence level (default 0.95)
    seed      : random seed

    Returns
    -------
    ratio     : point estimate (median_a / median_b)
    ci_lo     : lower bound
    ci_hi     : upper bound
    """
    rng = np.random.default_rng(seed)
    a = np.array(times_a)
    b = np.array(times_b)
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


def wilcoxon_timing_test(
    times_a: List[float],
    times_b: List[float],
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test on paired per-rep timing vectors.
    Two-sided, α=0.05. Addresses RQ3.

    Parameters
    ----------
    times_a   : timing measurements for method A
    times_b   : timing measurements for method B (same n, same data draw)
    alternative: 'two-sided' | 'less' | 'greater'

    Returns
    -------
    statistic : Wilcoxon W statistic
    p_value   : two-sided p-value
    """
    a = np.array(times_a)
    b = np.array(times_b)
    if len(a) != len(b):
        raise ValueError(f"Paired vectors must have equal length: {len(a)} vs {len(b)}")
    result = stats.wilcoxon(a, b, alternative=alternative)
    return float(result.statistic), float(result.pvalue)


# ── MIC spawn overhead (RQ5) ───────────────────────────────────────────────────

def measure_mic_spawn_overhead(
    conda_env: str = "ngorima_mic",
    n_reps: int = 10,
    conf: float = 0.95,
) -> dict:
    """
    Measure fixed MIC subprocess spawn cost independent of n.
    Runs a no-op Python invocation inside the conda env and times it.

    Returns dict with keys: median_ms, ci_lo_ms, ci_hi_ms, n_reps.
    Saved to results/mic_spawn_overhead.json by Step 03.

    Addresses RQ5.
    """
    times_ms = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        try:
            subprocess.run(
                ["conda", "run", "--no-capture-output", "-n", conda_env,
                 "python", "-c", "pass"],
                capture_output=True, timeout=60,
            )
        except Exception:
            pass
        times_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times_ms)
    alpha = 1 - conf
    return {
        "median_ms": float(np.median(arr)),
        "ci_lo_ms":  float(np.percentile(arr, 100 * alpha / 2)),
        "ci_hi_ms":  float(np.percentile(arr, 100 * (1 - alpha / 2))),
        "n_reps":    n_reps,
        "conda_env": conda_env,
    }


# ── Hardware fingerprint ───────────────────────────────────────────────────────

def log_hardware_fingerprint() -> dict:
    """
    Collect CPU / RAM / OS / BLAS metadata for system_state.json.
    Called by 04_compile_results.py (canonical location).
    """
    info: Dict = {
        "python_version": platform.python_version(),
        "platform":       platform.platform(),
        "processor":      platform.processor(),
        "machine":        platform.machine(),
    }

    # CPU cores
    try:
        import psutil
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["logical_cores"]  = psutil.cpu_count(logical=True)
        info["ram_total_gb"]   = round(psutil.virtual_memory().total / 1024**3, 2)
        freq = psutil.cpu_freq()
        info["cpu_freq_max_mhz"] = round(freq.max, 1) if freq else None
    except ImportError:
        info["physical_cores"] = os.cpu_count()

    # CPU governor (Linux only)
    try:
        gov = subprocess.check_output(
            ["cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"],
            text=True, timeout=5,
        ).strip()
        info["cpu_governor"] = gov
    except Exception:
        info["cpu_governor"] = "unknown"

    # BLAS backend
    try:
        import numpy as np
        cfg = np.__config__
        blas_str = str(getattr(cfg, "blas_opt_info", ""))
        info["blas_backend"] = "MKL" if "mkl" in blas_str.lower() else "OpenBLAS/other"
    except Exception:
        info["blas_backend"] = "unknown"

    # Package versions
    for pkg in ["numpy", "scipy", "sklearn", "dcor", "minepy", "yaml", "joblib"]:
        try:
            import importlib
            mod = importlib.import_module(pkg.replace("-", "_").split(".")[0])
            info[f"pkg_{pkg}"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[f"pkg_{pkg}"] = "not installed"

    return info
