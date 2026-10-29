"""Shared helpers for the input, verification, benchmark and analysis scripts.

Adapted from Assignment 1's scripts/common.py (tag assignment1-v1.0): the same
dense file helpers and error bound, plus the CSR file format and the command
line of the Java driver.

Everything that depends on the declared protocol (sizes, densities, flags,
input locations) is read from config/bench.json so that the scripts cannot
drift from it.
"""

import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "bench.json"
INPUTS = ROOT / "data" / "inputs"
SPARSE = INPUTS / "sparse"
JAR = "target/matmul.jar"
U = 2.0 ** -53


def load_config():
    return json.loads(CONFIG.read_text())


def build():
    """Build target/matmul.jar (tests are run separately with `mvn package`)."""
    subprocess.run(["mvn", "-q", "-DskipTests", "package"], cwd=ROOT, check=True)


def density_label(percent):
    """1 -> 'd1', 0.5 -> 'd0.5' (density in percent)."""
    return "d" + format(percent, "g")


def dense_pair(n):
    return INPUTS / f"A_{n}.bin", INPUTS / f"B_{n}.bin"


def sparse_pair(structure, n, percent):
    stem = f"{structure}_{n}_{density_label(percent)}"
    return SPARSE / f"{stem}_A.csr", SPARSE / f"{stem}_B.csr"


def command(config, method, n, a, b, *, block=None, warmup, reps, pre_warm_seconds=None,
            min_sample_seconds=None, min_warmup_seconds=None, out=None):
    """Command line (run from ROOT) for one process of the Java driver."""
    cmd = ["java", *config["build_and_runtime"]["jvm_flags"], "-jar", JAR, "--method", method, "--n", str(n),
           "--a", str(Path(a).relative_to(ROOT)), "--b", str(Path(b).relative_to(ROOT)),
           "--warmup", str(warmup), "--reps", str(reps)]
    optional = (("--block", block), ("--pre-warm-seconds", pre_warm_seconds),
                ("--min-sample-seconds", min_sample_seconds), ("--min-warmup-seconds", min_warmup_seconds),
                ("--out", out))
    for flag, value in optional:
        if value is not None:
            cmd += [flag, str(value)]
    return cmd


def read_dense(path, n):
    return np.fromfile(path, dtype="<f8").reshape(n, n)


def write_dense(path, m):
    np.ascontiguousarray(m, dtype="<f8").tofile(path)


def read_csr(path):
    """Return (n, rowptr, colidx, values) from a .csr file."""
    with open(path, "rb") as f:
        n, nnz = np.fromfile(f, dtype="<i8", count=2)
        rowptr = np.fromfile(f, dtype="<i4", count=n + 1)
        colidx = np.fromfile(f, dtype="<i4", count=nnz)
        values = np.fromfile(f, dtype="<f8", count=nnz)
    return int(n), rowptr, colidx, values


def write_csr(path, n, rowptr, colidx, values):
    with open(path, "wb") as f:
        np.array([n, len(values)], dtype="<i8").tofile(f)
        np.asarray(rowptr, dtype="<i4").tofile(f)
        np.asarray(colidx, dtype="<i4").tofile(f)
        np.asarray(values, dtype="<f8").tofile(f)


def error_bound(a, b):
    """Componentwise bound 2 * gamma_n * (|A||B|) on the difference of two correct results."""
    n = a.shape[1]
    gamma = n * U / (1 - n * U)
    return 2 * gamma * (np.abs(a) @ np.abs(b))
