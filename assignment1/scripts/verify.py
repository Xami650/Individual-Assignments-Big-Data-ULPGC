"""Cross-language correctness verification (run before benchmarking).

For each size in config "sizes.correctness" and three cases
    random   : the shared seeded inputs A_n, B_n           (reference: numpy.matmul)
    identity : A_n x I                                      (exact answer: A_n)
    zero     : A_n x 0                                      (exact answer: 0)
every implementation is run once through its normal command line with --out,
and its result C is compared with the reference using the declared criterion
    |C_ij - R_ij| <= 2 * gamma_n * (|A||B|)_ij      (atol = 0)
For identity and zero the bound is applied to the exact answer, so any
deviation must stay within it (in practice the error is exactly 0). The script
also checks whether the three results are bitwise identical (hypothesis H4).

Writes results/correctness.csv and exits with status 1 on any failure.

Usage (from assignment1/):
    python3 scripts/verify.py [--sizes 1 2 3] [--skip-build]
"""

import argparse
import csv
import subprocess
import sys

import numpy as np

from common import LANGUAGES, ROOT, build, command, error_bound, input_pair, load_config, read_matrix, write_matrix

TMP = ROOT / "results" / "tmp"
OUT_CSV = ROOT / "results" / "correctness.csv"
FIELDS = ["case", "n", "language", "max_abs_error", "max_error_over_bound", "within_tolerance",
          "bitwise_identical_across_languages"]


def cases(n):
    """Yield (case, A path, B path, A, B, reference)."""
    a_path, b_path = input_pair(n)
    a, b = read_matrix(a_path, n), read_matrix(b_path, n)
    yield "random", a_path, b_path, a, b, a @ b

    eye_path = TMP / f"I_{n}.bin"
    write_matrix(eye_path, np.eye(n))
    yield "identity", a_path, eye_path, a, np.eye(n), a.copy()

    zero_path = TMP / f"Z_{n}.bin"
    write_matrix(zero_path, np.zeros((n, n)))
    yield "zero", a_path, zero_path, a, np.zeros((n, n)), np.zeros((n, n))


def compare(c, reference, bound):
    err = np.abs(c - reference)
    ratio = np.divide(err, bound, out=np.where(err > 0, np.inf, 0.0), where=bound > 0)
    return float(err.max()), float(ratio.max()), bool(np.all(err <= bound))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    config = load_config()
    sizes = args.sizes or config["sizes"]["correctness"]
    if not args.skip_build:
        build(config)
    TMP.mkdir(parents=True, exist_ok=True)

    rows, failed = [], False
    for n in sizes:
        for case, a_path, b_path, a, b, reference in cases(n):
            bound = error_bound(a, b)
            results = {}
            for language in LANGUAGES:
                out = TMP / f"C_{case}_{language}_{n}.bin"
                cmd = command(config, language, n, warmup=0, reps=1, a=a_path, b=b_path, out=out)
                subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
                results[language] = read_matrix(out, n)
            identical = all(results[x].tobytes() == results["c"].tobytes() for x in LANGUAGES)
            for language in LANGUAGES:
                max_err, max_ratio, ok = compare(results[language], reference, bound)
                failed |= not ok
                rows.append({"case": case, "n": n, "language": language, "max_abs_error": f"{max_err:.3e}",
                             "max_error_over_bound": f"{max_ratio:.3e}", "within_tolerance": ok,
                             "bitwise_identical_across_languages": identical})
                print(f"{case:8s} n={n:<4d} {language:6s} max|err|={max_err:.3e}  "
                      f"max err/bound={max_ratio:.3e}  {'PASS' if ok else 'FAIL'}  identical={identical}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_CSV.relative_to(ROOT)} ({len(rows)} rows): {'FAILURES' if failed else 'all within tolerance'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
