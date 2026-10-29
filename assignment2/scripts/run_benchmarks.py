"""Benchmark harness: runs every configuration of the declared experiments.

Adapted from Assignment 1 (scripts/run_benchmarks.py at tag assignment1-v1.0):
the process model, time budget, peak-RSS measurement and output files are the
same; configurations are now (experiment, method, input, tile size) instead of
(language, n), and the driver reports nonzero counts and an out-of-memory status.

Each configuration is one fresh JVM, launched under `/usr/bin/time -l`, that
runs the JIT pre-warm, the warm-up and the measured samples (config/bench.json).
The process gets startup_slack_seconds to load and convert its inputs and print
its CSV header, then max_seconds_per_operation for each sample line. If a
deadline passes, the process group is killed and the configuration recorded as
a timeout; an OutOfMemoryError (driver exit status 3) is recorded as
memory_budget_exceeded. In both cases larger n are skipped for the same series
(experiment, method, structure, density, tile size).

A session writes, under results/raw/<session>/:
    samples.csv  one row per sample, warm-ups included (kind = warmup | measure)
    runs.csv     one row per process: status, nonzero counts, peak RSS, process times, command
    env.txt      environment snapshot (scripts/collect_env.sh) before and after
    config.json  copy of the protocol used
Rows are appended as soon as they are known, so an interrupted session keeps
everything measured so far.

Usage (from assignment2/):
    python3 scripts/run_benchmarks.py                                   # full protocol
    python3 scripts/run_benchmarks.py --experiments A_dense --max-n 256 --reps 2 --session smoke
"""

import argparse
import csv
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

from common import ROOT, build, command, dense_pair, load_config, sparse_pair

EXPERIMENTS = ("A_dense", "A_block", "B_density", "B_size")
KEYS = ["experiment", "method", "n", "structure", "density_percent", "block"]
SAMPLE_FIELDS = ["session", "run_id", *KEYS, "numeric_type", "threads", "kind", "index", "batch", "elapsed_ns",
                 "ns_per_op", "checksum"]
META_FIELDS = ["nnz_a", "nnz_b", "nnz_c", "nnz_c_nonzero", "multiply_adds", "pre_warm_ops"]
RUN_FIELDS = ["session", "run_id", *KEYS, "status", "warmup_runs", "measured_reps", "samples_received", *META_FIELDS,
              "max_rss_bytes", "peak_footprint_bytes", "process_wall_s", "process_user_s", "process_sys_s",
              "instructions_retired", "cycles_elapsed", "started_at", "detail", "command"]
TIME_PATTERNS = {
    "max_rss_bytes": re.compile(r"^\s*(\d+)\s+maximum resident set size", re.M),
    "peak_footprint_bytes": re.compile(r"^\s*(\d+)\s+peak memory footprint", re.M),
    "instructions_retired": re.compile(r"^\s*(\d+)\s+instructions retired", re.M),
    "cycles_elapsed": re.compile(r"^\s*(\d+)\s+cycles elapsed", re.M),
}
TIMES = re.compile(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys")
OUT_OF_MEMORY_EXIT = 3


def configurations(config):
    """Every declared configuration, in run order, without duplicates."""
    ex = config["experiments"]
    tile = config["methods"]["blocked"]["tile_size"]
    runs, seen = [], set()

    def add(experiment, method, n, structure, percent, pair, block=None):
        if method == "blocked" and block is None:
            block = tile
        key = (method, n, structure, percent, block)
        if key not in seen:
            seen.add(key)
            runs.append({"experiment": experiment, "method": method, "n": n, "structure": structure,
                         "density_percent": percent, "block": block if method == "blocked" else "",
                         "a": pair[0], "b": pair[1]})

    for n in ex["A_dense"]["sizes"]:
        for method in ex["A_dense"]["methods"]:
            add("A_dense", method, n, "dense", 100, dense_pair(n))
    for n in ex["A_block"]["sizes"]:
        for bs in ex["A_block"]["tile_sizes"]:
            add("A_block", "blocked", n, "dense", 100, dense_pair(n), bs)
    bd = ex["B_density"]
    for percent in bd["densities_percent"]:
        for structure in bd["structures"]:
            for method in bd["methods"]:
                add("B_density", method, bd["n"], structure, percent, sparse_pair(structure, bd["n"], percent))
    bs_ = ex["B_size"]
    for n in bs_["sizes"]:
        for method in bs_["methods"]:
            pair = sparse_pair(bs_["structure"], n, bs_["density_percent"])
            add("B_size", method, n, bs_["structure"], bs_["density_percent"], pair)
    return runs


def series(run):
    """Configurations that differ only in n; a failure skips the larger n of its series."""
    return run["experiment"], run["method"], run["structure"], run["density_percent"], run["block"]


def run_id(run):
    parts = [run["experiment"], run["method"], str(run["n"])]
    if run["structure"] != "dense":
        parts += [run["structure"], f"d{run['density_percent']:g}"]
    if run["block"] != "":
        parts.append(f"bs{run['block']}")
    return "-".join(parts)


class CsvLog:
    """Append-only CSV with a header, flushed after every row."""

    def __init__(self, path, fields):
        self.fields = fields
        self.file = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fields, lineterminator="\n")
        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow({k: row.get(k, "") for k in self.fields})
        self.file.flush()


def pump(stream, sink):
    for line in stream:
        sink.put(line)
    sink.put(None)


def run_configuration(cmd, budget, on_sample, meta):
    """Run one process, streaming its samples; returns (status, samples, stderr text, detail)."""
    proc = subprocess.Popen(["/usr/bin/time", "-l", *cmd], cwd=ROOT, text=True, bufsize=1,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    lines, err_lines = queue.Queue(), []
    threading.Thread(target=pump, args=(proc.stdout, lines), daemon=True).start()
    err_thread = threading.Thread(target=lambda: err_lines.extend(proc.stderr), daemon=True)
    err_thread.start()

    samples, header = 0, None
    limit = budget["max_seconds_per_operation"]
    while True:
        wait = budget["startup_slack_seconds"] if header is None else limit
        try:
            line = lines.get(timeout=wait)
        except queue.Empty:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            stage = "start-up (loading, conversion, pre-warm)" if header is None else f"limit {limit:g} s per operation"
            return "timeout", samples, "", f"no output within {wait:g} s ({stage})"
        if line is None:
            break
        line = line.strip()
        if line.startswith("#"):
            meta.update(kv.split("=", 1) for kv in line[1:].strip().split(","))
        elif header is None:
            header = line.split(",")
        else:
            on_sample(dict(zip(header, line.split(","))))
            samples += 1

    proc.wait()
    err_thread.join()
    stderr = "".join(err_lines)
    errors = [l for l in stderr.splitlines() if l.startswith("error:")]
    if proc.returncode == OUT_OF_MEMORY_EXIT:
        return "memory_budget_exceeded", samples, stderr, errors[0] if errors else "out of memory"
    if proc.returncode != 0:
        return "error", samples, stderr, f"exit status {proc.returncode}: {(errors or stderr.strip().splitlines())[:3]}"
    return "ok", samples, stderr, ""


def parse_time_output(stderr):
    values = {key: (int(m.group(1)) if (m := pattern.search(stderr)) else "") for key, pattern in TIME_PATTERNS.items()}
    if m := TIMES.search(stderr):
        values.update(process_wall_s=m.group(1), process_user_s=m.group(2), process_sys_s=m.group(3))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--max-n", type=int, help="only configurations with n <= MAX_N (smoke tests)")
    parser.add_argument("--reps", type=int, help="override the declared repetitions (smoke tests)")
    parser.add_argument("--max-seconds", type=float, help="override the per-operation limit (to test the timeout path)")
    parser.add_argument("--session", help="session name (default: timestamp; smoke* sessions are not tracked)")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    config = load_config()
    measurement, budget = config["measurement"], dict(config["budgets"])
    if args.max_seconds is not None:
        budget["max_seconds_per_operation"] = args.max_seconds
    reps = args.reps if args.reps is not None else measurement["measured_repetitions"]
    runs = [r for r in configurations(config) if r["experiment"] in args.experiments
            and (args.max_n is None or r["n"] <= args.max_n)]
    missing = sorted({str(p.relative_to(ROOT)) for r in runs for p in (r["a"], r["b"]) if not p.exists()})
    if missing:
        sys.exit(f"missing inputs (run scripts/gen_inputs.py): {missing[:4]}")

    session = args.session or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "results" / "raw" / session
    out_dir.mkdir(parents=True, exist_ok=False)
    if not args.skip_build:
        build()
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = subprocess.run(["sh", "scripts/collect_env.sh", sys.executable], cwd=ROOT, capture_output=True, text=True)
    (out_dir / "env.txt").write_text(env.stdout)
    if "AC Power" not in env.stdout or "lowpowermode         1" in env.stdout:
        print("WARNING: not on AC power or Low Power Mode is on (see env.txt)", file=sys.stderr)

    samples_log = CsvLog(out_dir / "samples.csv", SAMPLE_FIELDS)
    runs_log = CsvLog(out_dir / "runs.csv", RUN_FIELDS)
    stopped = {}  # series -> reason for skipping larger n
    session_start = time.monotonic()
    session_limit = budget["max_session_minutes"] * 60
    print(f"session {session}: {len(runs)} configurations")

    for run in runs:
        rid = run_id(run)
        base = {"session": session, "run_id": rid, **{k: run[k] for k in KEYS}}
        warmup = measurement["warmup_runs"]
        if series(run) in stopped or time.monotonic() - session_start > session_limit:
            reason = stopped.get(series(run), "session time budget reached")
            runs_log.write({**base, "status": "skipped", "detail": reason})
            print(f"{rid:40s} skipped ({reason})")
            continue

        cmd = command(config, run["method"], run["n"], run["a"], run["b"], block=run["block"] or None,
                      warmup=warmup, reps=reps, pre_warm_seconds=measurement["pre_warm"]["seconds"],
                      min_sample_seconds=measurement["batching"]["min_sample_seconds"],
                      min_warmup_seconds=measurement["min_warmup_seconds"])
        started = datetime.now().isoformat(timespec="seconds")
        print(f"{rid:40s}", end="", flush=True)

        def on_sample(s, base=base):
            batch, elapsed = int(s["batch"]), int(s["elapsed_ns"])
            samples_log.write({**base, "numeric_type": "float64", "threads": 1, "kind": s["kind"],
                               "index": s["index"], "batch": batch, "elapsed_ns": elapsed,
                               "ns_per_op": f"{elapsed / batch:.1f}", "checksum": s["checksum"]})
            print(f" {s['kind'][0]}{elapsed / batch / 1e6:.4g}ms", end="", flush=True)

        meta = {}
        status, received, stderr, detail = run_configuration(cmd, budget, on_sample, meta)
        resources = parse_time_output(stderr)
        rss = resources.get("max_rss_bytes")
        if status == "ok" and rss and rss > budget["max_peak_rss_bytes"]:
            status, detail = "memory_budget_exceeded", f"peak RSS {rss} B > {budget['max_peak_rss_bytes']} B"
        if status != "ok":
            stopped[series(run)] = f"{status} at n={run['n']}"
        runs_log.write({**base, **resources, **{k: meta.get(k, "") for k in META_FIELDS}, "status": status,
                        "warmup_runs": warmup, "measured_reps": reps, "samples_received": received,
                        "started_at": started, "detail": detail, "command": " ".join(cmd)})
        rss_text = f"{rss / 2**20:.0f} MiB" if rss else "n/a"
        print(f"  -> {status}, peak RSS {rss_text}{', ' + detail if detail else ''}")
        time.sleep(measurement["pause_between_runs_seconds"])

    after = subprocess.run(["sh", "-c", "date '+%Y-%m-%dT%H:%M:%S%z'; pmset -g therm; pmset -g batt | head -2; uptime"],
                           capture_output=True, text=True)
    with open(out_dir / "env.txt", "a") as f:
        f.write("\n## After the session\n" + after.stdout)
    minutes = (time.monotonic() - session_start) / 60
    print(f"session {session} finished in {minutes:.1f} min -> {out_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
