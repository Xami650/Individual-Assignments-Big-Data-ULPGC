"""Shared helpers for the input, verification, benchmark and analysis scripts.

Adapted from Assignment 2's scripts/common.py (tag assignment2-v1.0): the same
dense file helpers and error bound; the CSR helpers are dropped, and the
command line of the Java driver now carries the thread count, the row
schedule and extra JVM flags.

Everything that depends on the declared protocol (sizes, thread counts, flags,
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
JAR = "target/matmul.jar"
U = 2.0 ** -53


def load_config():
    return json.loads(CONFIG.read_text())


def build():
    """Build target/matmul.jar (tests are run separately with `mvn package`)."""
    subprocess.run(["mvn", "-q", "-DskipTests", "package"], cwd=ROOT, check=True)


def dense_pair(n):
    return INPUTS / f"A_{n}.bin", INPUTS / f"B_{n}.bin"


def command(config, method, n, a, b, *, threads=None, schedule=None, jvm_extra=(), warmup, reps,
            pre_warm_seconds=None, min_sample_seconds=None, min_warmup_seconds=None, out=None):
    """Command line (run from ROOT) for one process of the Java driver."""
    cmd = ["java", *config["build_and_runtime"]["jvm_flags"], *jvm_extra, "-jar", JAR, "--method", method,
           "--n", str(n), "--a", str(Path(a).relative_to(ROOT)), "--b", str(Path(b).relative_to(ROOT)),
           "--warmup", str(warmup), "--reps", str(reps)]
    optional = (("--threads", threads), ("--schedule", schedule), ("--pre-warm-seconds", pre_warm_seconds),
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


def error_bound(a, b):
    """Componentwise bound 2 * gamma_n * (|A||B|) on the difference of two correct results."""
    n = a.shape[1]
    gamma = n * U / (1 - n * U)
    return 2 * gamma * (np.abs(a) @ np.abs(b))
