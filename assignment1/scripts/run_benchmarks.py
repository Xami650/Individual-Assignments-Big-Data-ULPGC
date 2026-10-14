"""Benchmark harness: runs every (language, n) configuration of the declared protocol.

Each configuration is one fresh process, launched under `/usr/bin/time -l`, that
performs the declared warm-up and measured samples (see config/bench.json).
The harness enforces the time budget while the process runs: the process gets
startup_slack_seconds to load its inputs and print the CSV header, and then
max_seconds_per_multiplication for each sample line (batched samples are short
by construction). If a deadline passes, the process group is killed, the configuration is recorded
as a timeout, and larger sizes are skipped for that language. Peak RSS is
checked against the memory budget afterwards (macOS cannot enforce it).

A session writes, under results/raw/<session>/:
    samples.csv  one row per sample, warm-ups included (kind = warmup | measure)
    runs.csv     one row per process: status, peak RSS, process times, command
    env.txt      environment snapshot (scripts/collect_env.sh) before and after
    config.json  copy of the protocol used
Rows are appended as soon as they are known, so an interrupted session keeps
everything measured so far.

Usage (from assignment1/):
    python3 scripts/run_benchmarks.py                              # full protocol
    python3 scripts/run_benchmarks.py --sizes 1 64 128 --reps 2 --session smoke
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

from common import LANGUAGES, ROOT, build, command, input_pair, load_config

SAMPLE_FIELDS = ["session", "run_id", "language", "n", "numeric_type", "threads", "kind", "index", "batch",
                 "elapsed_ns", "ns_per_multiplication", "checksum"]
RUN_FIELDS = ["session", "run_id", "language", "n", "status", "warmup_runs", "measured_reps", "samples_received",
              "max_rss_bytes", "peak_footprint_bytes", "process_wall_s", "process_user_s", "process_sys_s",
              "instructions_retired", "cycles_elapsed", "started_at", "detail", "command"]
TIME_PATTERNS = {
    "max_rss_bytes": re.compile(r"^\s*(\d+)\s+maximum resident set size", re.M),
    "peak_footprint_bytes": re.compile(r"^\s*(\d+)\s+peak memory footprint", re.M),
    "instructions_retired": re.compile(r"^\s*(\d+)\s+instructions retired", re.M),
    "cycles_elapsed": re.compile(r"^\s*(\d+)\s+cycles elapsed", re.M),
}
TIMES = re.compile(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys")


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


def run_configuration(cmd, budget, on_sample):
    """Run one process, streaming its samples; returns (status, samples, stderr text, detail)."""
    proc = subprocess.Popen(["/usr/bin/time", "-l", *cmd], cwd=ROOT, text=True, bufsize=1,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    lines, err_lines = queue.Queue(), []
    threading.Thread(target=pump, args=(proc.stdout, lines), daemon=True).start()
    err_thread = threading.Thread(target=lambda: err_lines.extend(proc.stderr), daemon=True)
    err_thread.start()

    samples, header = 0, None
    limit = budget["max_seconds_per_multiplication"]
    while True:
        wait = budget["startup_slack_seconds"] if header is None else limit
        try:
            line = lines.get(timeout=wait)
        except queue.Empty:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            stage = "start-up (loading inputs)" if header is None else f"limit {limit:g} s per multiplication"
            return "timeout", samples, "", f"no output within {wait:g} s ({stage})"
        if line is None:
            break
        if header is None:
            header = line.strip().split(",")
            continue
        on_sample(dict(zip(header, line.strip().split(","))))
        samples += 1

    proc.wait()
    err_thread.join()
    stderr = "".join(err_lines)
    if proc.returncode != 0:
        return "error", samples, stderr, f"exit status {proc.returncode}: {stderr.strip().splitlines()[:3]}"
    return "ok", samples, stderr, ""


def parse_time_output(stderr):
    values = {key: (int(m.group(1)) if (m := pattern.search(stderr)) else "") for key, pattern in TIME_PATTERNS.items()}
    if m := TIMES.search(stderr):
        values.update(process_wall_s=m.group(1), process_user_s=m.group(2), process_sys_s=m.group(3))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", help="override the declared sizes (smoke tests)")
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES, default=list(LANGUAGES))
    parser.add_argument("--reps", type=int, help="override the declared repetitions (smoke tests)")
    parser.add_argument("--max-seconds", type=float, help="override the per-multiplication limit (to test the timeout path)")
    parser.add_argument("--session", help="session name (default: timestamp; smoke* sessions are not tracked)")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    config = load_config()
    measurement, budget = config["measurement"], dict(config["budgets"])
    if args.max_seconds is not None:
        budget["max_seconds_per_multiplication"] = args.max_seconds
    sizes = args.sizes or [config["sizes"]["memory_baseline_n"], *config["sizes"]["benchmark"]]
    reps = args.reps if args.reps is not None else measurement["measured_repetitions"]
    session = args.session or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "results" / "raw" / session
    out_dir.mkdir(parents=True, exist_ok=False)

    missing = [str(p) for n in sizes for p in input_pair(n) if not p.exists()]
    if missing:
        sys.exit(f"missing inputs (run scripts/gen_inputs.py): {missing[:4]}")
    if not args.skip_build:
        build(config, args.languages)

    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = subprocess.run(["sh", "scripts/collect_env.sh", sys.executable], cwd=ROOT, capture_output=True, text=True)
    (out_dir / "env.txt").write_text(env.stdout)
    if "AC Power" not in env.stdout or "lowpowermode         1" in env.stdout:
        print("WARNING: not on AC power or Low Power Mode is on (see env.txt)", file=sys.stderr)

    samples_log = CsvLog(out_dir / "samples.csv", SAMPLE_FIELDS)
    runs_log = CsvLog(out_dir / "runs.csv", RUN_FIELDS)
    stopped = {}  # language -> reason for skipping larger sizes
    session_start = time.monotonic()
    session_limit = budget["max_session_minutes"] * 60

    for n in sizes:
        for language in args.languages:
            run_id = f"{language}-{n}"
            base = {"session": session, "run_id": run_id, "language": language, "n": n}
            warmup = measurement["warmup_runs"][language]
            if language in stopped or time.monotonic() - session_start > session_limit:
                reason = stopped.get(language, "session time budget reached")
                runs_log.write({**base, "status": "skipped", "detail": reason})
                print(f"{run_id:12s} skipped ({reason})")
                continue

            cmd = command(config, language, n, warmup=warmup, reps=reps,
                          min_sample_seconds=measurement["batching"]["min_sample_seconds"],
                          min_warmup_seconds=measurement["min_warmup_seconds"])
            started = datetime.now().isoformat(timespec="seconds")
            print(f"{run_id:12s} running ...", end="", flush=True)

            def on_sample(s, base=base):
                batch, elapsed = int(s["batch"]), int(s["elapsed_ns"])
                samples_log.write({**base, "numeric_type": "float64", "threads": 1, "kind": s["kind"],
                                   "index": s["index"], "batch": batch, "elapsed_ns": elapsed,
                                   "ns_per_multiplication": f"{elapsed / batch:.1f}", "checksum": s["checksum"]})
                print(f" {s['kind'][0]}{elapsed / batch / 1e6:.4g}ms", end="", flush=True)

            status, received, stderr, detail = run_configuration(cmd, budget, on_sample)
            resources = parse_time_output(stderr)
            rss = resources.get("max_rss_bytes")
            if status == "ok" and rss and rss > budget["max_peak_rss_bytes"]:
                status, detail = "memory_budget_exceeded", f"peak RSS {rss} B > {budget['max_peak_rss_bytes']} B"
            if status != "ok":
                stopped[language] = f"{status} at n={n}"
            runs_log.write({**base, **resources, "status": status, "warmup_runs": warmup, "measured_reps": reps,
                            "samples_received": received, "started_at": started, "detail": detail,
                            "command": " ".join(cmd)})
            rss_text = f"{rss / 2**20:.1f} MiB" if rss else "n/a"
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
