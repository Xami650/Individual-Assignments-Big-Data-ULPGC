"""Shared helpers for the input, verification, benchmark and analysis scripts.

Adapted from Assignment 3's scripts/common.py (tag assignment3-v1.0): the same
dense file helpers and error bound. New here: the JDK is pinned by path
(config/bench.json, build_and_runtime.java_home), the command line of the
driver covers the distributed method, and the command line of a cluster member
and its heap limit are built here.

Everything that depends on the declared protocol (sizes, flags, input
locations, cluster settings) is read from config/bench.json so that the
scripts cannot drift from it.
"""

import json
import math
import os
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "bench.json"
INPUTS = ROOT / "data" / "inputs"
JAR = "target/matmul.jar"
U = 2.0 ** -53
MIB = 2 ** 20

TIME_PATTERNS = {
    "max_rss_bytes": re.compile(r"^\s*(\d+)\s+maximum resident set size", re.M),
    "peak_footprint_bytes": re.compile(r"^\s*(\d+)\s+peak memory footprint", re.M),
    "instructions_retired": re.compile(r"^\s*(\d+)\s+instructions retired", re.M),
    "cycles_elapsed": re.compile(r"^\s*(\d+)\s+cycles elapsed", re.M),
}
# /usr/bin/time writes these with the decimal separator of the locale (a comma under es_ES, for example)
TIMES = re.compile(r"(\d+[.,]\d+) real\s+(\d+[.,]\d+) user\s+(\d+[.,]\d+) sys")


def load_config():
    return json.loads(CONFIG.read_text())


def java_home(config):
    return Path(os.path.expanduser(config["build_and_runtime"]["java_home"]))


def java(config):
    """The pinned java executable (the java on PATH has changed between sessions before)."""
    path = java_home(config) / "bin" / "java"
    if not path.exists():
        raise SystemExit(f"JDK not found: {path} (build_and_runtime.java_home in config/bench.json)")
    return str(path)


def build(config):
    """Build target/matmul.jar and target/lib/ with the pinned JDK (tests run separately with `mvn package`)."""
    env = {**os.environ, "JAVA_HOME": str(java_home(config))}
    subprocess.run(["mvn", "-q", "-DskipTests", "package"], cwd=ROOT, check=True, env=env)


def dense_pair(n):
    return INPUTS / f"A_{n}.bin", INPUTS / f"B_{n}.bin"


def relative(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_absolute() and path.is_relative_to(ROOT) else str(path)


def driver_command(config, method, n, a, b, *, warmup, reps, threads=None, schedule=None, workers=None,
                   threads_per_worker=None, cluster=None, base_port=None, member_log=None, ping_rounds=None,
                   pre_warm_seconds=None, min_sample_seconds=None, min_warmup_seconds=None, out=None):
    """Command line (run from ROOT) for one driver process."""
    cmd = [java(config), *config["build_and_runtime"]["jvm_flags"], "-jar", JAR, "--method", method,
           "--n", str(n), "--a", relative(a), "--b", relative(b), "--warmup", str(warmup), "--reps", str(reps)]
    optional = (("--threads", threads), ("--schedule", schedule), ("--workers", workers),
                ("--threads-per-worker", threads_per_worker), ("--cluster", cluster), ("--base-port", base_port),
                ("--member-log", member_log), ("--ping-rounds", ping_rounds),
                ("--pre-warm-seconds", pre_warm_seconds), ("--min-sample-seconds", min_sample_seconds),
                ("--min-warmup-seconds", min_warmup_seconds), ("--out", out))
    for flag, value in optional:
        if value is not None:
            cmd += [flag, relative(value) if isinstance(value, Path) else str(value)]
    return cmd


def member_heap_mib(n, workers):
    """-Xmx of a member: 512 MiB + (3 + 4 / w) 8n^2 bytes, rounded up to a multiple of 64 MiB."""
    need = 512 * MIB + (3 + 4 / workers) * 8 * n * n
    return int(math.ceil(need / (64 * MIB)) * 64)


def member_command(config, cluster, index, workers, threads, base_port, n, pre_warm_seconds=None):
    """Command line (run from ROOT) for member `index` of a cluster of `workers` members."""
    cmd = [java(config), f"-Xmx{member_heap_mib(n, workers)}m", *config["cluster"]["member_jvm_flags"], "-jar", JAR,
           "member", "--cluster", cluster, "--index", str(index), "--members", str(workers), "--threads",
           str(threads), "--base-port", str(base_port)]
    if pre_warm_seconds is not None:
        cmd += ["--pre-warm-seconds", str(pre_warm_seconds)]
    return cmd


def parse_time_output(stderr):
    """Peak memory, counters and times from the output of /usr/bin/time -l (empty strings where missing)."""
    values = {key: (int(m.group(1)) if (m := pattern.search(stderr)) else "") for key, pattern in TIME_PATTERNS.items()}
    if m := TIMES.search(stderr):
        wall, user, system = (g.replace(",", ".") for g in m.groups())
        values.update(process_wall_s=wall, process_user_s=user, process_sys_s=system)
    return values


def read_dense(path, n):
    return np.fromfile(path, dtype="<f8").reshape(n, n)


def write_dense(path, m):
    np.ascontiguousarray(m, dtype="<f8").tofile(path)


def error_bound(a, b):
    """Componentwise bound 2 * gamma_n * (|A||B|) on the difference of two correct results."""
    n = a.shape[1]
    gamma = n * U / (1 - n * U)
    return 2 * gamma * (np.abs(a) @ np.abs(b))
