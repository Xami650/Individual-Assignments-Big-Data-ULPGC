"""Correctness verification of every method through the driver (run before benchmarking).

Adapted from Assignment 3's scripts/verify.py (tag assignment3-v1.0). Each
configuration runs once through its normal command line with --out, and its
product C is compared with a reference using the declared criterion
    |C_ij - R_ij| <= 2 * gamma_n * (|A||B|)_ij      (atol = 0)

Cases, for n in correctness.sizes (config/bench.json), as in Assignments 1 to 3:
    random   the dense inputs A_n, B_n      (reference: numpy.matmul)
    identity A_n x I                        (exact answer: A_n)
    zero     A_n x 0                        (exact answer: 0)
Configurations: basic and ikj; par_ikj as listed in correctness.par_ikj; and
dist with every (w, k) of correctness.dist_layouts, each on a real cluster of w
member processes (scripts/cluster.py) that serves all sizes and cases in turn,
so that bands with fewer rows than members (w > n) and uneven bands are
covered. The script also checks that every configuration of a case gives the
same bits (H1). The in-process methods run four at a time; the distributed
runs one at a time.

Writes results/correctness.csv and exits with status 1 on any failure.

Usage (from assignment4/):
    python3 scripts/verify.py [--skip-build]
"""

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cluster import Cluster
from common import ROOT, build, dense_pair, driver_command, error_bound, load_config, read_dense, write_dense

TMP = ROOT / "results" / "tmp"
OUT_CSV = ROOT / "results" / "correctness.csv"
FIELDS = ["case", "n", "method", "workers", "threads", "schedule", "max_abs_error", "max_error_over_bound",
          "within_tolerance", "bitwise_identical_across_configurations"]
BASE_PORT = 6101  # outside the benchmark's range (5901 + 20 i, i < 10)


def label(conf):
    method, workers, threads, schedule = conf
    if method == "dist":
        return f"dist_w{workers}k{threads}"
    if method == "par_ikj":
        return f"par_ikj_p{threads}_{schedule}"
    return method


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


def run_driver(config, conf, n, a, b, tag, cluster=None):
    """Run one configuration once; return C as a dense array."""
    method, workers, threads, schedule = conf
    out = TMP / f"C_{tag}_{label(conf)}.bin"
    dist = method == "dist"
    cmd = driver_command(config, method, n, a, b, warmup=0, reps=1, pre_warm_seconds=0, out=out,
                         threads=threads if method == "par_ikj" else None, schedule=schedule,
                         workers=workers if dist else None, threads_per_worker=threads if dist else None,
                         cluster=cluster.name if dist else None, base_port=cluster.base_port if dist else None)
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{label(conf)} n={n} {tag}: exit {result.returncode}: {result.stderr[-2000:]}")
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
        build(config)
    TMP.mkdir(parents=True, exist_ok=True)
    local = [(m, 0, 1, None) for m in corr["sequential"]]
    local += [("par_ikj", 0, c["threads"], c["schedule"]) for c in corr["par_ikj"]]
    layouts = [("dist", w, k, None) for w, k in corr["dist_layouts"]]
    confs = local + layouts

    cases = {}
    for n in corr["sizes"]:
        for case, a_path, b_path, a, b, reference in dense_cases(n):
            cases[(case, n)] = (a_path, b_path, a, b, reference)
    products = {key: {} for key in cases}

    with ThreadPoolExecutor(max_workers=4) as executor:
        jobs = {executor.submit(run_driver, config, conf, n, a_path, b_path, f"{case}_{n}"): (case, n, conf)
                for (case, n), (a_path, b_path, *_) in cases.items() for conf in local}
        for job, (case, n, conf) in jobs.items():
            products[(case, n)][conf] = job.result()
    print(f"in-process methods: {len(jobs)} runs done")

    for i, conf in enumerate(layouts):
        _, workers, threads, _ = conf
        cluster = Cluster(config, f"a4-verify-{i}", workers, threads, BASE_PORT + 20 * i, max(corr["sizes"]),
                          pre_warm_seconds=0)
        status, detail = cluster.start(config["cluster"]["member_ready_timeout_seconds"])
        try:
            if status != "ok":
                raise RuntimeError(f"cluster {label(conf)} did not start: {status} {detail}")
            for (case, n), (a_path, b_path, *_) in cases.items():
                products[(case, n)][conf] = run_driver(config, conf, n, a_path, b_path, f"{case}_{n}", cluster)
        finally:
            if status == "ok":
                stopped = cluster.stop()
                bad = [m["member"] for m in stopped if m["exit_status"] != 0]
                if bad:
                    raise RuntimeError(f"members {bad} of {label(conf)} exited with an error")
            else:
                cluster.kill()
        print(f"{label(conf)}: {len(cases)} runs on a cluster of {workers} member processes")

    rows, failed = [], False
    for (case, n), (a_path, b_path, a, b, reference) in cases.items():
        bound = error_bound(a, b)
        results = products[(case, n)]
        first = results[confs[0]].tobytes()
        identical = all(results[conf].tobytes() == first for conf in confs)
        failed |= not identical
        worst = 0.0
        for conf in confs:
            max_err, max_ratio, ok = compare(results[conf], reference, bound)
            failed |= not ok
            worst = max(worst, max_ratio)
            method, workers, threads, schedule = conf
            rows.append({"case": case, "n": n, "method": method, "workers": workers, "threads": threads,
                         "schedule": schedule or "", "max_abs_error": f"{max_err:.3e}",
                         "max_error_over_bound": f"{max_ratio:.3e}", "within_tolerance": ok,
                         "bitwise_identical_across_configurations": identical})
        print(f"{case:8s} n={n:<4d} {len(confs)} configurations  max err/bound={worst:.3e}  identical={identical}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_CSV.relative_to(ROOT)} ({len(rows)} rows): {'FAILURES' if failed else 'all checks passed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
