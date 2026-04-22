"""
mic_worker.py — KsaaiP2 Pipeline v4_updated
MIC computation entry point for ngorima_mic conda env (Python 3.10, minepy 1.2.6)
"""

import sys
import time
import numpy as np

N_WARMUP = 2
N_REPS_MIC = 5

def compute_mic(X, y, alpha=0.6, c=15.0):
    from minepy import MINE
    mine = MINE(alpha=alpha, c=c)
    mine.compute_score(X, y)
    return float(mine.mic())

def main():
    if len(sys.argv) < 2:
        print("Usage: python mic_worker.py <data.npz>", file=sys.stderr)
        sys.exit(1)
    data = np.load(sys.argv[1])
    X = data["X"].ravel()
    y = data["y"].ravel()
    alpha = float(data["alpha"][0])
    c = float(data["c"][0])
    for _ in range(N_WARMUP):
        compute_mic(X, y, alpha=alpha, c=c)
    scores = []
    for _ in range(N_REPS_MIC):
        t0 = time.perf_counter()
        score = compute_mic(X, y, alpha=alpha, c=c)
        scores.append(score)
    print(float(np.median(scores)))
    sys.exit(0)

if __name__ == "__main__":
    main()
