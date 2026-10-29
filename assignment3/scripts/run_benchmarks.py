"""Benchmark harness: runs every configuration of the declared experiments.

Adapted from Assignment 2 (scripts/run_benchmarks.py at tag assignment2-v1.0):
the process model, time budget, peak-RSS measurement, failure handling and
output files are the same. Configurations are now (experiment, method, n,
threads, schedule, SIMD setting). The session runs one block per size in
ascending n: the sequential ikj first and again last (drift control), and the
other configurations of that size in a seeded random order (session_order in
config/bench.json). Every sample carries the change of the process counters
over the sample (CPU time, CPU time on performance cores, instructions,
cycles, context switches), measured by the driver itself. Around every process
the harness also reads the system-wide CPU time (psutil), so that the CPU time
used by other processes during the run is known.

Each configuration is one fresh JVM, launched under `/usr/bin/time -l`, that
creates its thread pool and runs the JIT pre-warm, the warm-up and the measured
samples. The process gets startup_slack_seconds to load its inputs and print
its CSV header, then max_seconds_per_operation for each sample line. If a
deadline passes, the process group is killed and the configuration recorded as
a timeout; an OutOfMemoryError (driver exit status 3) is recorded as
memory_budget_exceeded. In both cases larger n are skipped for the same series
(experiment, method, threads, schedule, SIMD setting, repeat).

A session writes, under results/raw/<session>/:
    samples.csv  one row per sample, warm-ups included (kind = warmup | measure)
    runs.csv     one row per process: status, settings reported by the driver, JVM activity,
                 peak RSS, process times and counters, CPU time of other processes, command
    env.txt      environment snapshot (scripts/collect_env.sh) before and after
    config.json  copy of the protocol used
Rows are appended as soon as they are known, so an interrupted session keeps
everything measured so far.

Usage (from assignment3/):
    python3 scripts/run_benchmarks.py                                   # full protocol
    python3 scripts/run_benchmarks.py --max-n 256 --reps 2 --pause 0 --session smoke
    python3 scripts/run_benchmarks.py --dry-run                         # list the configurations in run order
"""

import argparse
import csv
import json
import os
import queue
import random
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

import psutil

from common import ROOT, build, command, dense_pair, load_config

EXPERIMENTS = ("P_threads", "P_schedule", "V_simd")
KEYS = ["experiment", "method", "n", "threads", "schedule", "simd", "repeat"]
COUNTER_FIELDS = ["cpu_ns", "cpu_p_ns", "instructions", "cycles", "instructions_p", "cycles_p", "context_switches"]
SAMPLE_FIELDS = ["session", "run_id", *KEYS, "numeric_type", "kind", "index", "batch", "elapsed_ns", "ns_per_op",
                 "checksum", *COUNTER_FIELDS]
QOS_FIELDS = [f"qos_{q}_ns" for q in ("default", "maintenance", "background", "utility", "legacy", "user_initiated",
                                      "user_interactive")]
META_FIELDS = ["multiply_adds", "available_processors", "superword", "max_vector_size", "counters", "pre_warm_n",
               "pre_warm_ops", "jvm_threads", "os_threads", "gc_collections", "gc_ms", "jit_ms",
               "allocated_bytes_measured", "allocated_bytes_total", "gc_collections_total", *QOS_FIELDS]
RUN_FIELDS = ["session", "run_id", *KEYS, "oversubscribed", "status", "warmup_runs", "measured_reps",
              "samples_received", *META_FIELDS, "max_rss_bytes", "peak_footprint_bytes", "process_wall_s",
              "process_user_s", "process_sys_s", "instructions_retired", "cycles_elapsed", "window_s",
              "system_busy_cpu_s", "other_cpu_s", "started_at", "detail", "command"]
TIME_PATTERNS = {
    "max_rss_bytes": re.compile(r"^\s*(\d+)\s+maximum resident set size", re.M),
    "peak_footprint_bytes": re.compile(r"^\s*(\d+)\s+peak memory footprint", re.M),
    "instructions_retired": re.compile(r"^\s*(\d+)\s+instructions retired", re.M),
    "cycles_elapsed": re.compile(r"^\s*(\d+)\s+cycles elapsed", re.M),
}
TIMES = re.compile(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys")
OUT_OF_MEMORY_EXIT = 3


def configurations(config):
    """Every declared configuration, in run order (size blocks; see session_order)."""
    ex = config["experiments"]
    pt, ps, vs = ex["P_threads"], ex["P_schedule"], ex["V_simd"]
    seed = config["session_order"]["seed"]

    def run(experiment, method, n, threads=1, schedule="", simd="on", repeat="", jvm_extra=()):
        a, b = dense_pair(n)
        return {"experiment": experiment, "method": method, "n": n, "threads": threads, "schedule": schedule,
                "simd": simd, "repeat": repeat, "jvm_extra": list(jvm_extra), "a": a, "b": b}

    runs = []
    for n in sorted(set(pt["sizes"]) | set(ps["sizes"]) | set(vs["sizes"])):
        block = []
        if n in pt["sizes"]:
            block += [run("P_threads", pt["parallel"]["method"], n, p, pt["parallel"]["schedule"])
                      for p in pt["threads"]]
        if n in ps["sizes"]:
            block += [run("P_schedule", ps["method"], n, p, ps["schedule"]) for p in ps["threads"]]
        if n in vs["sizes"]:
            block += [run("V_simd", c["method"], n, c.get("threads", 1), c.get("schedule", ""), "off",
                          jvm_extra=vs["jvm_flags"]) for c in vs["configurations"]]
        random.Random(seed + n).shuffle(block)
        if n in pt["sizes"]:
            block = [run("P_threads", pt["sequential"], n, repeat="first"), *block,
                     run("P_threads", pt["sequential"], n, repeat="last")]
        runs += block
    return runs


def series(run):
    """Configurations that differ only in n; a failure skips the larger n of its series."""
    return run["experiment"], run["method"], run["threads"], run["schedule"], run["simd"], run["repeat"]


def run_id(run):
    parts = [run["experiment"], run["method"], str(run["n"])]
    if run["method"] != "ikj":
        parts += [f"p{run['threads']}", run["schedule"]]
    if run["simd"] == "off":
        parts.append("nosimd")
    if run["repeat"]:
        parts.append(run["repeat"])
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
            stage = "start-up (loading, pool, pre-warm)" if header is None else f"limit {limit:g} s per operation"
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


def busy_cpu_seconds():
    """System-wide busy CPU time (user + nice + system, summed over all cores) since boot."""
    t = psutil.cpu_times()
    return t.user + t.nice + t.system


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
    parser.add_argument("--pause", type=float, help="override the pause between processes (smoke tests)")
    parser.add_argument("--session", help="session name (default: timestamp; smoke* and pilot* sessions are not tracked)")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the configurations in run order and exit")
    args = parser.parse_args()

    config = load_config()
    measurement, budget = config["measurement"], dict(config["budgets"])
    if args.max_seconds is not None:
        budget["max_seconds_per_operation"] = args.max_seconds
    reps = args.reps if args.reps is not None else measurement["measured_repetitions"]
    pause = args.pause if args.pause is not None else measurement["pause_between_runs_seconds"]
    runs = [r for r in configurations(config) if r["experiment"] in args.experiments
            and (args.max_n is None or r["n"] <= args.max_n)]
    if args.dry_run:
        for r in runs:
            print(run_id(r))
        print(f"{len(runs)} configurations")
        return 0
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
    load = os.getloadavg()[0]
    if load > measurement["load_average_warning"]:
        print(f"WARNING: 1-minute load average {load:.2f} > {measurement['load_average_warning']}: "
              "other processes compete for the cores", file=sys.stderr)

    samples_log = CsvLog(out_dir / "samples.csv", SAMPLE_FIELDS)
    runs_log = CsvLog(out_dir / "runs.csv", RUN_FIELDS)
    logical = os.cpu_count()
    stopped = {}  # series -> reason for skipping larger n
    session_start = time.monotonic()
    session_limit = budget["max_session_minutes"] * 60
    print(f"session {session}: {len(runs)} configurations")

    for run in runs:
        rid = run_id(run)
        base = {"session": session, "run_id": rid, **{k: run[k] for k in KEYS}}
        warmup = measurement["warmup_runs"]
        oversubscribed = run["threads"] > logical
        if series(run) in stopped or time.monotonic() - session_start > session_limit:
            reason = stopped.get(series(run), "session time budget reached")
            runs_log.write({**base, "oversubscribed": oversubscribed, "status": "skipped", "detail": reason})
            print(f"{rid:44s} skipped ({reason})")
            continue

        parallel = run["method"] != "ikj"
        cmd = command(config, run["method"], run["n"], run["a"], run["b"],
                      threads=run["threads"] if parallel else None, schedule=run["schedule"] if parallel else None,
                      jvm_extra=run["jvm_extra"], warmup=warmup, reps=reps,
                      pre_warm_seconds=measurement["pre_warm"]["seconds"],
                      min_sample_seconds=measurement["batching"]["min_sample_seconds"],
                      min_warmup_seconds=measurement["min_warmup_seconds"])
        started = datetime.now().isoformat(timespec="seconds")
        print(f"{rid:44s}", end="", flush=True)

        def on_sample(s, base=base):
            batch, elapsed = int(s["batch"]), int(s["elapsed_ns"])
            samples_log.write({**base, "numeric_type": "float64", "kind": s["kind"], "index": s["index"],
                               "batch": batch, "elapsed_ns": elapsed, "ns_per_op": f"{elapsed / batch:.1f}",
                               "checksum": s["checksum"], **{k: s.get(k, "") for k in COUNTER_FIELDS}})
            if s["kind"] == "measure":
                print(f" {elapsed / batch / 1e6:.4g}ms", end="", flush=True)

        meta = {}
        cpu_before, window_start = busy_cpu_seconds(), time.monotonic()
        status, received, stderr, detail = run_configuration(cmd, budget, on_sample, meta)
        window = time.monotonic() - window_start
        system_busy = busy_cpu_seconds() - cpu_before
        resources = parse_time_output(stderr)
        own = (float(resources["process_user_s"]) + float(resources["process_sys_s"])
               if "process_user_s" in resources else None)
        other = f"{system_busy - own:.2f}" if own is not None else ""
        rss = resources.get("max_rss_bytes")
        if status == "ok" and rss and rss > budget["max_peak_rss_bytes"]:
            status, detail = "memory_budget_exceeded", f"peak RSS {rss} B > {budget['max_peak_rss_bytes']} B"
        if status != "ok":
            stopped[series(run)] = f"{status} at n={run['n']}"
        runs_log.write({**base, **resources, **{k: meta.get(k, "") for k in META_FIELDS},
                        "oversubscribed": oversubscribed, "status": status, "warmup_runs": warmup,
                        "measured_reps": reps, "samples_received": received, "window_s": f"{window:.2f}",
                        "system_busy_cpu_s": f"{system_busy:.2f}", "other_cpu_s": other,
                        "started_at": started, "detail": detail, "command": " ".join(cmd)})
        rss_text = f"{rss / 2**20:.0f} MiB" if rss else "n/a"
        other_text = f"{float(other) / window:.2f} cores" if other else "n/a"
        print(f"  -> {status}, RSS {rss_text}, others {other_text}{', ' + detail if detail else ''}")
        time.sleep(pause)

    after = subprocess.run(["sh", "-c", "date '+%Y-%m-%dT%H:%M:%S%z'; pmset -g therm; pmset -g batt | head -2; uptime"],
                           capture_output=True, text=True)
    with open(out_dir / "env.txt", "a") as f:
        f.write("\n## After the session\n" + after.stdout)
    minutes = (time.monotonic() - session_start) / 60
    print(f"session {session} finished in {minutes:.1f} min -> {out_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
