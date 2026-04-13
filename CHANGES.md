
## Pipeline fixes — 2026-04-13

### MIC timing (scorers.py)
- Added subprocess Path 2 via `minepy_env` conda environment when minepy
  is not available in the main venv. This ensures true MIC (Reshef 2011)
  is used throughout, not the histogram MI approximation.
- MIC reps reduced from 30 to 5 for synthetic benchmarks due to ~5s
  subprocess overhead per call. Variance is still estimable; this is
  disclosed in the manuscript.

### ξₙ timing (01_synthetic_benchmarks.py)
- Set NGORIMA_XI_NUMPY=1 for synthetic complexity benchmarks.
  The xicor Python package is ~8000x slower than the NumPy implementation
  at n=10,000 due to Python object overhead, which inflates the empirical
  complexity exponent. The NumPy path implements the identical formula
  (Chatterjee 2021, Eq. 1) and is the correct choice for wall-clock
  complexity measurement. Production feature scoring still uses xicor.
