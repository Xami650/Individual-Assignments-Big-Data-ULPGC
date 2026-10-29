"""Correctness verification of every method through the driver (run before benchmarking).

Adapted from Assignment 1's scripts/verify.py. Each method runs once through its
normal command line with --out, and its product C is compared with a reference
using the declared criterion
    |C_ij - R_ij| <= 2 * gamma_n * (|A||B|)_ij      (atol = 0)

Dense cases, for n in correctness.dense_sizes (methods basic, ikj and blocked
with every tile size in correctness.tile_sizes):
    random   the dense inputs A_n, B_n      (reference: numpy.matmul)
    identity A_n x I                        (exact answer: A_n)
    zero     A_n x 0                        (exact answer: 0)
Sparse cases, for every structure, n and density in correctness.sparse (methods
csr, to_dense, basic, ikj and blocked with the smallest tile size):
    the reference is the SciPy CSR product; the structural nonzero count of C
    (entries reached by at least one product) comes from the product of the 0/1
    patterns, and csr must store exactly that many entries.
The script also checks that all methods of a case are bitwise identical (HA3).

Writes results/correctness.csv and exits with status 1 on any failure.

Usage (from assignment2/):
    python3 scripts/verify.py [--skip-build]
"""

import argparse
import csv
import subprocess
import sys

import numpy as np
import scipy.sparse as sp

from common import ROOT, build, command, dense_pair, error_bound, load_config, read_csr, read_dense, sparse_pair, \
    write_dense

TMP = ROOT / "results" / "tmp"
OUT_CSV = ROOT / "results" / "correctness.csv"
FIELDS = ["case", "n", "density_percent", "method", "block", "max_abs_error", "max_error_over_bound",
          "within_tolerance", "bitwise_identical_across_methods", "nnz_c_stored", "nnz_c_structural_reference",
          "nnz_c_nonzero", "nnz_c_nonzero_reference"]


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


def to_scipy(path):
    n, rowptr, colidx, values = read_csr(path)
    return sp.csr_matrix((values, colidx, rowptr), shape=(n, n))


def run_driver(config, method, n, a, b, block, tag):
    """Run one method once; return (C as a dense array, metadata dict)."""
    out = TMP / f"C_{tag}_{method}{block or ''}.bin"
    cmd = command(config, method, n, a, b, block=block, warmup=0, reps=1, pre_warm_seconds=0, out=out)
    stdout = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout
    meta = dict(kv.split("=", 1) for line in stdout.splitlines() if line.startswith("#")
                for kv in line[1:].strip().split(","))
    return read_dense(out, n), meta


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

    cases = []
    for n in corr["dense_sizes"]:
        for case, a_path, b_path, a, b, reference in dense_cases(n):
            methods = [("basic", None), ("ikj", None)] + [("blocked", bs) for bs in corr["tile_sizes"]]
            cases.append((case, n, 100, a_path, b_path, a, b, reference, None, methods))
    sparse = corr["sparse"]
    small_tile = min(corr["tile_sizes"])
    for structure in sparse["structures"]:
        for n in sparse["sizes"]:
            for percent in sparse["densities_percent"]:
                a_path, b_path = sparse_pair(structure, n, percent)
                sa, sb = to_scipy(a_path), to_scipy(b_path)
                pattern = (sa != 0).astype(np.float64) @ (sb != 0).astype(np.float64)
                reference = (sa @ sb).toarray()
                methods = [("csr", None), ("to_dense", None), ("basic", None), ("ikj", None), ("blocked", small_tile)]
                cases.append((structure, n, percent, a_path, b_path, sa.toarray(), sb.toarray(), reference,
                              pattern.nnz, methods))

    rows, failed = [], False
    for case, n, percent, a_path, b_path, a, b, reference, structural, methods in cases:
        bound = error_bound(a, b)
        reference_nonzero = int(np.count_nonzero(reference))
        results = {}
        for method, block in methods:
            results[(method, block)] = run_driver(config, method, n, a_path, b_path, block, f"{case}_{n}_{percent:g}")
        first = next(iter(results.values()))[0].tobytes()
        identical = all(c.tobytes() == first for c, _ in results.values())
        failed |= not identical
        for (method, block), (c, meta) in results.items():
            max_err, max_ratio, ok = compare(c, reference, bound)
            stored = meta.get("nnz_c", "")
            nonzero = meta.get("nnz_c_nonzero", "")
            if structural is not None and method == "csr":
                ok &= int(stored) == structural
            ok &= int(nonzero) == reference_nonzero
            failed |= not ok
            rows.append({"case": case, "n": n, "density_percent": percent, "method": method, "block": block or "",
                         "max_abs_error": f"{max_err:.3e}", "max_error_over_bound": f"{max_ratio:.3e}",
                         "within_tolerance": ok, "bitwise_identical_across_methods": identical,
                         "nnz_c_stored": stored, "nnz_c_structural_reference": structural if structural is not None else "",
                         "nnz_c_nonzero": nonzero, "nnz_c_nonzero_reference": reference_nonzero})
        print(f"{case:8s} n={n:<4d} d={percent:<4g}% {len(methods)} methods  max err/bound="
              f"{max(float(r['max_error_over_bound']) for r in rows[-len(methods):]):.3e}  identical={identical}"
              f"{'' if structural is None else f'  nnz(C) structural={structural}'}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_CSV.relative_to(ROOT)} ({len(rows)} rows): {'FAILURES' if failed else 'all checks passed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
