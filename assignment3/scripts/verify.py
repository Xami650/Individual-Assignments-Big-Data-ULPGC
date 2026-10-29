"""Correctness verification of every configuration through the driver (run before benchmarking).

Adapted from Assignment 2's scripts/verify.py (tag assignment2-v1.0). Each
configuration runs once through its normal command line with --out, and its
product C is compared with a reference using the declared criterion
    |C_ij - R_ij| <= 2 * gamma_n * (|A||B|)_ij      (atol = 0)

Cases, for n in correctness.sizes (config/bench.json), as in Assignments 1 and 2:
    random   the dense inputs A_n, B_n      (reference: numpy.matmul)
    identity A_n x I                        (exact answer: A_n)
    zero     A_n x 0                        (exact answer: 0)
Configurations: the sequential ikj; par_ikj with every thread count in
correctness.threads and both schedules; and the correctness.superword_off
configurations with -XX:-UseSuperWord. The script also checks that every
configuration of a case gives the same bits (H1). The processes are launched
four at a time; correctness does not depend on timing.

Writes results/correctness.csv and exits with status 1 on any failure.

Usage (from assignment3/):
    python3 scripts/verify.py [--skip-build]
"""

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import ROOT, build, command, dense_pair, error_bound, load_config, read_dense, write_dense

TMP = ROOT / "results" / "tmp"
OUT_CSV = ROOT / "results" / "correctness.csv"
FIELDS = ["case", "n", "method", "threads", "schedule", "simd", "max_abs_error", "max_error_over_bound",
          "within_tolerance", "bitwise_identical_across_configurations"]
NO_SUPERWORD = "-XX:-UseSuperWord"


def configurations(corr):
    """(method, threads, schedule, extra JVM flags) for every checked configuration."""
    confs = [("ikj", None, None, ())]
    confs += [("par_ikj", p, s, ()) for p in corr["threads"] for s in corr["schedules"]]
    confs += [(c["method"], c.get("threads"), c.get("schedule"), (NO_SUPERWORD,)) for c in corr["superword_off"]]
    return confs


def dense_cases(n):
    """Yield (case, A path, B path, A, B, reference)."""
    a_path, b_path = dense_pair(n)
    a, b = read_dense(a_path, n), read_dense(b_path, n)
    yield "random", a_path, b_path, a, b, a @ b
    eye_path = TMP / f"I_{n}.bin"
    write_dense(eye_path, np.eye(n))
    yield "identity", a_path, eye_path, a, np.eye(n), a.copy()
    zero_path = TMP / f"Z_{n}.bin"
    write_dense(zero_path, np.zeros((n, n)))
    yield "zero", a_path, zero_path, a, np.zeros((n, n)), np.zeros((n, n))


def run_driver(config, conf, n, a, b, tag):
    """Run one configuration once; return C as a dense array."""
    method, threads, schedule, jvm_extra = conf
    out = TMP / f"C_{tag}_{method}_p{threads or 1}_{schedule or 'seq'}{'_nosimd' if jvm_extra else ''}.bin"
    cmd = command(config, method, n, a, b, threads=threads, schedule=schedule, jvm_extra=jvm_extra,
                  warmup=0, reps=1, pre_warm_seconds=0, out=out)
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    return read_dense(out, n)


def compare(c, reference, bound):
    err = np.abs(c - reference)
    ratio = np.divide(err, bound, out=np.where(err > 0, np.inf, 0.0), where=bound > 0)
    return float(err.max(initial=0.0)), float(ratio.max(initial=0.0)), bool(np.all(err <= bound))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    config = load_config()
    corr = config["correctness"]
    if not args.skip_build:
        build()
    TMP.mkdir(parents=True, exist_ok=True)
    confs = configurations(corr)

    rows, failed = [], False
    with ThreadPoolExecutor(max_workers=4) as executor:
        for n in corr["sizes"]:
            for case, a_path, b_path, a, b, reference in dense_cases(n):
                bound = error_bound(a, b)
                tag = f"{case}_{n}"
                results = list(executor.map(lambda conf: run_driver(config, conf, n, a_path, b_path, tag), confs))
                identical = all(c.tobytes() == results[0].tobytes() for c in results)
                failed |= not identical
                worst = 0.0
                for (method, threads, schedule, jvm_extra), c in zip(confs, results):
                    max_err, max_ratio, ok = compare(c, reference, bound)
                    failed |= not ok
                    worst = max(worst, max_ratio)
                    rows.append({"case": case, "n": n, "method": method, "threads": threads or 1,
                                 "schedule": schedule or "", "simd": "off" if jvm_extra else "on",
                                 "max_abs_error": f"{max_err:.3e}", "max_error_over_bound": f"{max_ratio:.3e}",
                                 "within_tolerance": ok, "bitwise_identical_across_configurations": identical})
                print(f"{case:8s} n={n:<4d} {len(confs)} configurations  max err/bound={worst:.3e}  "
                      f"identical={identical}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_CSV.relative_to(ROOT)} ({len(rows)} rows): {'FAILURES' if failed else 'all checks passed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
