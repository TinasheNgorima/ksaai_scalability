"""run_all.py — KsaaiP2 Pipeline v4_updated"""

import argparse
import os
import subprocess
import sys
import time
import yaml

with open("config.yaml") as _f:
    _CFG = yaml.safe_load(_f)

SCRIPTS = [
    "00_setup_and_download.py",
    "01_synthetic_benchmarks.py",
    "02_real_domain_benchmarks.py",
    "03_memory_and_parallelisation.py",
    "04_compile_results.py",
]

PHASE_MAP = {
    "download":  [0],
    "benchmark": [1, 2, 3],
    "compile":   [4],
}

def run_step(step_idx, fast=False):
    script = SCRIPTS[step_idx]
    env = dict(os.environ)
    if fast:
        env["NGORIMA_FAST"] = "1"
    print(f"\n{'=' * 60}")
    print(f"  Step {step_idx}: {script}")
    print(f"{'=' * 60}")
    t0 = time.perf_counter()
    result = subprocess.run([sys.executable, script], env=env)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] Step {step_idx} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n[OK] Step {step_idx} completed in {elapsed:.1f}s")
    return elapsed

def run_verification():
    import csv
    print("\n" + "=" * 60)
    print("  run_verification() — 6 automated post-run checks (Fix A)")
    print("=" * 60)
    failures = []
    passes = []

    try:
        exponents_file = _CFG.get("output", {}).get("complexity_exponents_csv", "results/complexity_exponents.csv")
        dc_beta = None
        if os.path.exists(exponents_file):
            with open(exponents_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("method", "").strip().upper() == "DC" and row.get("scenario", "").strip() == "A":
                        dc_beta = float(row["beta_hat"])
                        break
        if dc_beta is None:
            failures.append("1. DC beta: SKIP")
        elif 1.90 <= dc_beta <= 2.10:
            passes.append(f"1. DC beta: PASS = {dc_beta:.4f}")
        else:
            failures.append(f"1. DC beta: FAIL = {dc_beta:.4f}")
    except Exception as e:
        failures.append(f"1. DC beta: ERROR {e}")

    try:
        timing_file = _CFG.get("output", {}).get("synthetic_timing_csv", "results/synthetic_timing.csv")
        if not os.path.exists(timing_file):
            failures.append("2. xi_n < DC: SKIP")
        else:
            xi_times, dc_times = {}, {}
            with open(timing_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    n = int(float(row["n"]))
                    if n < 1000:
                        continue
                    method = row.get("method", "").strip()
                    t = float(row["median_time_s"])
                    if method in ("xi_n", "xin", "xi"):
                        xi_times[n] = t
                    elif method in ("DC", "dc"):
                        dc_times[n] = t
            common_ns = sorted(set(xi_times) & set(dc_times))
            violations = [n for n in common_ns if xi_times[n] >= dc_times[n]]
            if violations:
                failures.append(f"2. xi_n < DC: FAIL at n = {violations}")
            else:
                passes.append(f"2. xi_n < DC: PASS at all {len(common_ns)} n values")
    except Exception as e:
        failures.append(f"2. xi_n < DC: ERROR {e}")

    try:
        hk_file = _CFG.get("tcga", {}).get("hk_gene_list", "data/processed/tcga_hk_genes.txt")
        if not os.path.exists(hk_file):
            failures.append("3. HK genes: SKIP")
        else:
            with open(hk_file) as f:
                hk_genes = [line.strip() for line in f if line.strip()]
            if len(hk_genes) >= 3:
                passes.append(f"3. HK genes: PASS = {len(hk_genes)}")
            else:
                failures.append(f"3. HK genes: FAIL = {len(hk_genes)}")
    except Exception as e:
        failures.append(f"3. HK genes: ERROR {e}")

    try:
        timing_file = _CFG.get("output", {}).get("synthetic_timing_csv", "results/synthetic_timing.csv")
        if not os.path.exists(timing_file):
            failures.append("4. MIC CoV: SKIP")
        else:
            bad_ns = []
            with open(timing_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("method", "").strip().upper() == "MIC" and "cov" in row:
                        cov = float(row["cov"])
                        if cov > 0.30:
                            bad_ns.append((int(float(row["n"])), cov))
            if bad_ns:
                failures.append(f"4. MIC CoV: FAIL at {bad_ns[:5]}")
            else:
                passes.append("4. MIC CoV: PASS")
    except Exception as e:
        failures.append(f"4. MIC CoV: ERROR {e}")

    try:
        exponents_file = _CFG.get("output", {}).get("complexity_exponents_csv", "results/complexity_exponents.csv")
        if not os.path.exists(exponents_file):
            failures.append("5. SC R2: SKIP")
        else:
            bad_methods = []
            with open(exponents_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("dataset", "").strip().lower() in ("superconductivity", "sc") and "r2" in row:
                        r2 = float(row["r2"])
                        if r2 < 0.990:
                            bad_methods.append((row["method"], r2))
            if bad_methods:
                failures.append(f"5. SC R2: FAIL for {bad_methods}")
            else:
                passes.append("5. SC R2: PASS")
    except Exception as e:
        failures.append(f"5. SC R2: ERROR {e}")

    try:
        fred_target = _CFG.get("fred_md", {}).get("target_series", None)
        if fred_target and fred_target.strip().upper() == "INDPRO":
            passes.append(f"6. INDPRO: PASS = {fred_target}")
        else:
            failures.append(f"6. INDPRO: FAIL = '{fred_target}'")
    except Exception as e:
        failures.append(f"6. INDPRO: ERROR {e}")

    print()
    for p in passes:
        print(f"  + {p}")
    for f in failures:
        print(f"{prefix} {f}")

    n_fail = sum(1 for f in failures if "SKIP" not in f)
    print(f"\nResults: {len(passes)} passed / {n_fail} failed / {len(failures) - n_fail} skipped")
    return n_fail == 0

def main():
    parser = argparse.ArgumentParser(description="KsaaiP2 Pipeline v4_updated")
    parser.add_argument("--step", type=int, help="Run only step N (0-4)")
    parser.add_argument("--from-step", type=int, dest="from_step", help="Resume from step N")
    parser.add_argument("--phase", type=str, choices=list(PHASE_MAP.keys()), help="Run phase")
    parser.add_argument("--install", action="store_true", help="Install requirements")
    parser.add_argument("--fast", action="store_true", help="Fast mode")
    parser.add_argument("--verify", action="store_true", help="Run 6 verification checks")
    args = parser.parse_args()

    fast = args.fast or (os.environ.get("NGORIMA_FAST", "0") == "1")

    if args.install:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

    if args.step is not None:
        steps_to_run = [args.step]
    elif args.from_step is not None:
        steps_to_run = list(range(args.from_step, len(SCRIPTS)))
    elif args.phase is not None:
        steps_to_run = PHASE_MAP[args.phase]
    else:
        steps_to_run = list(range(len(SCRIPTS)))

    print(f"\nKsaaiP2 Pipeline v4_updated")
    print(f"Steps: {steps_to_run}")
    print(f"Fast: {'ON' if fast else 'OFF'}")
    print(f"Verify: {'ON' if args.verify else 'OFF'}")
    print()

    t_total = 0.0
    for step_idx in steps_to_run:
        t_total += run_step(step_idx, fast=fast)

    print(f"\n{'=' * 60}")
    print(f"  All steps completed in {t_total:.1f}s")
    print(f"{'=' * 60}")

    if args.verify:
        ok = run_verification()
        if not ok:
            sys.exit(2)

    print("\nDone.")

if __name__ == "__main__":
    main()