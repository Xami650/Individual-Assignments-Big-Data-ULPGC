"""Shared helpers for the verification and benchmark scripts.

Everything that depends on the declared protocol (flags, interpreter options,
input locations) is read from config/bench.json so that the scripts cannot
drift from it.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "bench.json"
INPUTS = ROOT / "data" / "inputs"
LANGUAGES = ("c", "java", "python")
U = 2.0 ** -53


def load_config():
    return json.loads(CONFIG.read_text())


def build(config, languages=LANGUAGES):
    """Build the compiled implementations with the declared settings."""
    if "c" in languages:
        c = config["build_and_runtime"]["c"]
        cc = "clang" if "clang" in c["compiler"] else c["compiler"]
        subprocess.run(["make", "-B", "-C", "c", f"CC={cc}", f"CFLAGS={' '.join(c['cflags'])}", "all"],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    if "java" in languages:
        subprocess.run(["mvn", "-q", "-f", "java/pom.xml", "-DskipTests", "package"], cwd=ROOT, check=True)


def command(config, language, n, *, warmup, reps, min_sample_seconds=None, a=None, b=None, out=None):
    """Command line (run from ROOT) for one process of the given implementation."""
    runtime = config["build_and_runtime"]
    if language == "c":
        cmd = ["c/build/matmul"]
    elif language == "java":
        cmd = ["java", *runtime["java"]["jvm_flags"], "-jar", "java/target/matmul.jar"]
    elif language == "python":
        cmd = [sys.executable, *runtime["python"]["interpreter_flags"], "python/main.py"]
    else:
        raise ValueError(f"unknown language {language!r}")
    cmd += ["--n", str(n), "--warmup", str(warmup), "--reps", str(reps)]
    if min_sample_seconds is not None:
        cmd += ["--min-sample-seconds", str(min_sample_seconds)]
    for flag, path in (("--a", a), ("--b", b), ("--out", out)):
        if path is not None:
            cmd += [flag, str(path)]
    return cmd


def read_matrix(path, n):
    return np.fromfile(path, dtype="<f8").reshape(n, n)


def write_matrix(path, m):
    np.ascontiguousarray(m, dtype="<f8").tofile(path)


def input_pair(n):
    return INPUTS / f"A_{n}.bin", INPUTS / f"B_{n}.bin"


def error_bound(a, b):
    """Componentwise bound 2 * gamma_n * (|A||B|) on the difference of two correct results."""
    n = a.shape[1]
    gamma = n * U / (1 - n * U)
    return 2 * gamma * (np.abs(a) @ np.abs(b))
