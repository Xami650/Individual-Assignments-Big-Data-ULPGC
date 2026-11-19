"""Benchmark harness: runs every configuration of the declared experiments.

Adapted from Assignment 3 (scripts/run_benchmarks.py at tag assignment3-v1.0):
the same process model, time budget, failure handling, drift controls,
cool-down, background-CPU check and output files. Configurations are now
(experiment, method, n, workers, threads, schedule, repeat). For the
distributed method each configuration also starts a fresh cluster of w member
JVMs (scripts/cluster.py), waits until every member is ready, measures the
members' idle CPU, runs the driver as a Hazelcast client, and terminates the
members afterwards; the setup time is taken from the members' and the driver's
event lines (measurement.setup in config/bench.json).

Every process (driver and members) runs under `/usr/bin/time -l`. The driver
gets startup_slack_seconds to load its inputs, connect and pre-warm, then
max_seconds_per_operation for every sample line; on a missed deadline the
configuration is killed and recorded as a timeout. An OutOfMemoryError in the
driver or a member (exit status 3) is recorded as memory_budget_exceeded, a
cluster failure (driver exit status 4) as error. In all these cases larger n
are skipped for the same series.

A session writes, under results/raw/<session>/:
    samples.csv         one row per driver sample, warm-ups included, with the phase timers and bytes of dist
    member_samples.csv  one row per member and sample of dist (task timers and counters, from the driver)
    runs.csv            one row per configuration: status, setup, driver resources and metadata, members'
                        totals, CPU time of other processes, command
    members.csv         one row per member process: events, idle CPU, peak memory, CPU times, threads
    env.txt             environment snapshot (scripts/collect_env.sh) before and after
    config.json         copy of the protocol used
Rows are appended as soon as they are known, so an interrupted session keeps
everything measured so far.

A tracked session (not smoke* or pilot*) refuses to start on battery power or
in Low Power Mode, and stops at the first configuration during which the Mac
slept: time.monotonic() (mach_absolute_time) does not advance while the system
sleeps and the wall clock does, so a gap between the two marks the run
"slept" (status), and nothing measured across a sleep is kept as a result.

Usage (from assignment4/):
    python3 scripts/run_benchmarks.py                                     # full protocol
    python3 scripts/run_benchmarks.py --max-n 256 --reps 2 --pause 0 --session smoke
    python3 scripts/run_benchmarks.py --sizes 1024 --session pilot-1
    python3 scripts/run_benchmarks.py --dry-run                           # list the configurations in run order
"""

import argparse
import csv
import json
import os
import queue
import random
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

import psutil

from cluster import OUT_OF_MEMORY_EXIT, Cluster
from common import ROOT, build, dense_pair, driver_command, java, load_config, member_heap_mib, parse_time_output

EXPERIMENTS = ("D_strong", "D_grain", "D_weak", "D_large")
KEYS = ["experiment", "method", "n", "workers", "threads", "schedule", "repeat"]
COUNTER_FIELDS = ["cpu_ns", "cpu_p_ns", "instructions", "cycles", "instructions_p", "cycles_p", "context_switches"]
DIST_FIELDS = ["serialize_ns", "scatter_ns", "broadcast_ns", "compute_ns", "gather_ns", "assemble_ns", "task_max_ns",
               "task_mean_ns", "kernel_max_ns", "kernel_mean_ns", "read_max_ns", "store_max_ns", "dispatch_max_ns",
               "tasks_done_ns", "a_payload_bytes", "b_payload_bytes", "c_payload_bytes", "scatter_socket_out",
               "broadcast_socket_out", "compute_socket_out", "compute_socket_in", "gather_socket_in", "socket_out",
               "socket_in"]
SAMPLE_FIELDS = ["session", "run_id", *KEYS, "numeric_type", "kind", "index", "batch", "elapsed_ns", "ns_per_op",
                 "checksum", *COUNTER_FIELDS, *DIST_FIELDS]
QOS_FIELDS = [f"qos_{q}_ns" for q in ("default", "maintenance", "background", "utility", "legacy", "user_initiated",
                                      "user_interactive")]
META_FIELDS = ["multiply_adds", "available_processors", "superword", "max_vector_size", "counters", "pre_warm_n",
               "pre_warm_ops", "jvm_threads", "os_threads", "gc_collections", "gc_ms", "jit_ms",
               "allocated_bytes_measured", "allocated_bytes_total", "gc_collections_total", "client_start_ns",
               "placement_ns", "partition_keys", "socket_counters", "ping_rounds", "ping_median_ns", "ping_q1_ns",
               "ping_q3_ns", "ping_min_ns", "ping_max_ns", "ping_member_median_max_ns", "map_a_entries",
               "map_b_entries", "map_c_entries", *QOS_FIELDS]
RUN_FIELDS = ["session", "run_id", *KEYS, "compute_threads", "status", "warmup_runs", "measured_reps",
              "samples_received", *META_FIELDS, "max_rss_bytes", "peak_footprint_bytes", "process_wall_s",
              "process_user_s", "process_sys_s", "instructions_retired", "cycles_elapsed", "cluster_name",
              "base_port", "member_heap_mib", "setup_cluster_s", "members_ready_s", "setup_client_s", "setup_s",
              "members_idle_cpu_s", "members_max_rss_bytes", "members_rss_sum_bytes", "members_footprint_sum_bytes",
              "members_footprint_max_bytes", "members_user_s", "members_sys_s", "total_footprint_bytes", "window_s",
              "system_busy_cpu_s", "own_cpu_s", "other_cpu_s", "started_at", "detail", "command"]
MEMBER_FIELDS = ["session", "run_id", *KEYS, "member", "pid", "exit_status", "killed", "heap_mib", "hz_start_ms",
                 "cluster_wait_ms", "joined_s", "ready_s", "partitions_owned", "pre_warm_ns", "jvm_threads_ready",
                 "os_threads_ready", "jvm_threads_stopping", "os_threads_stopping", "gc_collections", "gc_ms",
                 "member_cpu_ns", "member_cpu_p_ns", "idle_cpu_s", "max_rss_bytes", "peak_footprint_bytes",
                 "process_wall_s", "process_user_s", "process_sys_s", "instructions_retired", "cycles_elapsed"]
MEMBER_SAMPLE_PREFIX = ["session", "run_id", *KEYS]
DRIVER_OOM_EXIT, CLUSTER_FAILURE_EXIT = 3, 4


def configurations(config):
    """Every declared configuration, in run order (size blocks; see session_order)."""
    ex = config["experiments"]
    strong, grain, weak, large = ex["D_strong"], ex["D_grain"], ex["D_weak"], ex["D_large"]
    seed = config["session_order"]["seed"]

    def run(experiment, method, n, workers=0, threads=1, schedule="", repeat="", reps=None, max_seconds=None):
        a, b = dense_pair(n)
        return {"experiment": experiment, "method": method, "n": n, "workers": workers, "threads": threads,
                "schedule": schedule, "repeat": repeat, "a": a, "b": b, "reps": reps, "max_seconds": max_seconds}

    weak_points = {n: w for w, n in weak["points"]}
    sizes = sorted(set(strong["sizes"]) | set(grain["sizes"]) | set(weak_points) | set(large["sizes"]))
    w_max = max(strong["workers"])
    runs = []
    for n in sizes:
        block = []
        if n in strong["sizes"]:
            k = strong["threads_per_worker"]
            block += [run("D_strong", "dist", n, w, k) for w in strong["workers"]]
            block += [run("D_strong", "par_ikj", n, 0, p, "static") for p in strong["workers"]]
            block += [run("D_strong", "par_ikj", n, 0, p, "dynamic") for p in strong["parallel_dynamic"]]
            if n <= strong["basic_max_n"]:
                block.append(run("D_strong", "basic", n))
        if n in grain["sizes"]:
            block += [run("D_grain", "dist", n, w, k) for w, k in grain["layouts"] if (w, k) != (w_max, 1)]
        if n in weak_points and n not in strong["sizes"]:
            w = weak_points[n]
            block += [run("D_weak", "dist", n, w, 1), run("D_weak", "par_ikj", n, 0, w, "static"),
                      run("D_weak", "ikj", n)]
        if n in large["sizes"]:
            for c in large["configurations"]:
                block.append(run("D_large", c["method"], n, c.get("workers", 0),
                                 c.get("threads_per_worker", c.get("threads", 1)), c.get("schedule", ""),
                                 reps=large["measured_repetitions"], max_seconds=large["max_seconds_per_operation"]))
        random.Random(seed + n).shuffle(block)
        if n in strong["sizes"]:
            block = [run("D_strong", "ikj", n, repeat="first"), run("D_strong", "dist", n, w_max, 1, repeat="first"),
                     *block,
                     run("D_strong", "dist", n, w_max, 1, repeat="last"), run("D_strong", "ikj", n, repeat="last")]
        runs += block
    return runs


def series(run):
    """Configurations that differ only in n; a failure skips the larger n of its series."""
    return run["experiment"], run["method"], run["workers"], run["threads"], run["schedule"], run["repeat"]


def run_id(run):
    parts = [run["experiment"], run["method"], str(run["n"])]
    if run["method"] == "dist":
        parts.append(f"w{run['workers']}k{run['threads']}")
    elif run["method"] == "par_ikj":
        parts += [f"p{run['threads']}", run["schedule"]]
    if run["repeat"]:
        parts.append(run["repeat"])
    return "-".join(parts)


def compute_threads(run):
    if run["method"] == "dist":
        return run["workers"] * run["threads"]
    return run["threads"] if run["method"] == "par_ikj" else 1


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
        sink.put((line, time.time()))
    sink.put((None, time.time()))


def run_driver(cmd, startup_slack, limit, on_sample, meta, alive=lambda: True):
    """Run the driver, streaming its samples; returns (status, samples, stderr, detail, connected_epoch)."""
    proc = subprocess.Popen(["/usr/bin/time", "-l", *cmd], cwd=ROOT, text=True, bufsize=1,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    lines, err_lines = queue.Queue(), []
    threading.Thread(target=pump, args=(proc.stdout, lines), daemon=True).start()
    err_thread = threading.Thread(target=lambda: err_lines.extend(proc.stderr), daemon=True)
    err_thread.start()

    samples, header, connected = 0, None, None
    while True:
        wait = startup_slack if header is None else limit
        deadline = time.monotonic() + wait
        line = None
        while line is None:
            try:
                line, received = lines.get(timeout=1.0)
                break
            except queue.Empty:
                if time.monotonic() > deadline or not alive():
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    if not alive():
                        return "error", samples, "", "a cluster member exited during the run", connected
                    stage = "start-up (loading, connecting, pre-warm)" if header is None \
                        else f"limit {limit:g} s per operation"
                    return "timeout", samples, "", f"no output within {wait:g} s ({stage})", connected
        if line is None:
            break
        line = line.strip()
        if line.startswith("#"):
            fields = dict(kv.split("=", 1) for kv in line[1:].strip().split(",") if "=" in kv)
            if "connected" in fields:
                connected = received
            meta.update(fields)
        elif header is None:
            header = line.split(",")
        else:
            on_sample(dict(zip(header, line.split(","))))
            samples += 1

    proc.wait()
    err_thread.join()
    stderr = "".join(err_lines)
    errors = [l for l in stderr.splitlines() if l.startswith("error:")]
    if proc.returncode == DRIVER_OOM_EXIT:
        return "memory_budget_exceeded", samples, stderr, errors[0] if errors else "out of memory", connected
    if proc.returncode != 0:
        return "error", samples, stderr, f"exit status {proc.returncode}: " \
                                         f"{(errors or stderr.strip().splitlines())[:3]}", connected
    return "ok", samples, stderr, "", connected


def busy_cpu_seconds():
    """System-wide busy CPU time (user + nice + system, summed over all cores) since boot."""
    t = psutil.cpu_times()
    return t.user + t.nice + t.system


def on_ac_power():
    """True when the Mac draws from AC power and Low Power Mode is off."""
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    settings = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
    low_power = any(line.split()[:2] == ["lowpowermode", "1"] for line in settings.splitlines())
    return "AC Power" in batt and not low_power


def sleep_gap():
    """Wall-clock minus monotonic seconds; it grows by the time the system spends asleep."""
    return time.time() - time.monotonic()


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--methods", nargs="+", help="only these methods (smoke tests)")
    parser.add_argument("--max-n", type=int, help="only configurations with n <= MAX_N (smoke tests)")
    parser.add_argument("--sizes", type=int, nargs="+", help="only these sizes (pilot runs)")
    parser.add_argument("--match", help="only configurations whose run id contains this text (pilot runs)")
    parser.add_argument("--reps", type=int, help="override the declared repetitions (smoke tests)")
    parser.add_argument("--max-seconds", type=float, help="override the per-operation limit (to test the timeout path)")
    parser.add_argument("--pause", type=float, help="fixed pause between configurations instead of the cool-down")
    parser.add_argument("--session", help="session name (default: timestamp; smoke* and pilot* sessions are not tracked)")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the configurations in run order and exit")
    parser.add_argument("--allow-battery", action="store_true",
                        help="run a tracked session on battery power or in Low Power Mode (not for real sessions)")
    args = parser.parse_args()

    config = load_config()
    measurement, budget, cl = config["measurement"], dict(config["budgets"]), config["cluster"]
    cool = measurement["cool_down"]
    runs = [r for r in configurations(config) if r["experiment"] in args.experiments
            and (args.methods is None or r["method"] in args.methods)
            and (args.max_n is None or r["n"] <= args.max_n) and (args.sizes is None or r["n"] in args.sizes)
            and (args.match is None or args.match in run_id(r))]
    if args.dry_run:
        for r in runs:
            heap = f"  member -Xmx{member_heap_mib(r['n'], r['workers'])}m" if r["method"] == "dist" else ""
            print(f"{run_id(r)}{heap}")
        print(f"{len(runs)} configurations, {sum(r['method'] == 'dist' for r in runs)} of them distributed")
        return 0
    missing = sorted({str(p.relative_to(ROOT)) for r in runs for p in (r["a"], r["b"]) if not p.exists()})
    if missing:
        sys.exit(f"missing inputs (run scripts/gen_inputs.py): {missing[:4]}")

    session = args.session or datetime.now().strftime("%Y%m%d-%H%M%S")
    tracked = not session.startswith(("smoke", "pilot"))
    if tracked and not args.allow_battery and not on_ac_power():
        sys.exit("not on AC power or Low Power Mode is on: plug the Mac in (and keep the lid open) "
                 "before a tracked session")
    out_dir = ROOT / "results" / "raw" / session
    out_dir.mkdir(parents=True, exist_ok=False)
    if not args.skip_build:
        build(config)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = subprocess.run(["sh", "scripts/collect_env.sh", sys.executable, java(config)], cwd=ROOT,
                         capture_output=True, text=True)
    (out_dir / "env.txt").write_text(env.stdout)
    if "AC Power" not in env.stdout or "lowpowermode         1" in env.stdout:
        print("WARNING: not on AC power or Low Power Mode is on (see env.txt)", file=sys.stderr)
    probe_start, cpu0 = time.monotonic(), busy_cpu_seconds()
    time.sleep(5)
    background = (busy_cpu_seconds() - cpu0) / (time.monotonic() - probe_start)
    print(f"background CPU before the session: {background:.2f} cores (5 s)")

    samples_log = CsvLog(out_dir / "samples.csv", SAMPLE_FIELDS)
    member_samples_log = None
    runs_log = CsvLog(out_dir / "runs.csv", RUN_FIELDS)
    members_log = CsvLog(out_dir / "members.csv", MEMBER_FIELDS)
    stopped = {}  # series -> reason for skipping larger n
    session_start = time.monotonic()
    session_limit = budget["max_session_minutes"] * 60
    dist_index, aborted = 0, False
    print(f"session {session}: {len(runs)} configurations")

    for run in runs:
        rid = run_id(run)
        base = {"session": session, "run_id": rid, **{k: run[k] for k in KEYS}}
        warmup = measurement["warmup_runs"]
        reps = args.reps if args.reps is not None else (run["reps"] or measurement["measured_repetitions"])
        limit = args.max_seconds or run["max_seconds"] or budget["max_seconds_per_operation"]
        if series(run) in stopped or time.monotonic() - session_start > session_limit:
            reason = stopped.get(series(run), "session time budget reached")
            runs_log.write({**base, "compute_threads": compute_threads(run), "status": "skipped", "detail": reason})
            print(f"{rid:40s} skipped ({reason})")
            continue

        dist = run["method"] == "dist"
        started = datetime.now().isoformat(timespec="seconds")
        print(f"{rid:40s}", end="", flush=True)
        extra = {}
        cluster, member_log, idle = None, None, []
        cpu_before, window_start, gap_before = busy_cpu_seconds(), time.monotonic(), sleep_gap()
        status, detail = "ok", ""
        if dist:
            base_port = 5901 + 20 * (dist_index % 10)
            name = f"a4-{session}-{dist_index}"
            dist_index += 1
            cluster = Cluster(config, name, run["workers"], run["threads"], base_port, run["n"])
            extra.update(cluster_name=name, base_port=base_port, member_heap_mib=member_heap_mib(run["n"], run["workers"]))
            status, detail = cluster.start(cl["member_ready_timeout_seconds"])
            if status != "ok":
                cluster.kill()
            else:
                joined_s, ready_s = cluster.setup_times()
                idle = cluster.idle_cpu(cl["idle_window_seconds"])
                extra.update(setup_cluster_s=f"{joined_s:.3f}", members_ready_s=f"{ready_s:.3f}",
                             members_idle_cpu_s=f"{sum(x for x in idle if x is not None):.3f}")
            member_log = out_dir / f"members-{rid}.tmp.csv"

        meta, received, connected = {}, 0, None
        stderr = ""
        cmd = driver_command(config, run["method"], run["n"], run["a"], run["b"], warmup=warmup, reps=reps,
                             threads=run["threads"] if run["method"] == "par_ikj" else None,
                             schedule=run["schedule"] or None,
                             workers=run["workers"] if dist else None,
                             threads_per_worker=run["threads"] if dist else None,
                             cluster=extra.get("cluster_name"), base_port=extra.get("base_port"),
                             member_log=member_log, ping_rounds=cl["ping_rounds"] if dist else None,
                             pre_warm_seconds=measurement["pre_warm"]["seconds"],
                             min_sample_seconds=measurement["batching"]["min_sample_seconds"],
                             min_warmup_seconds=measurement["min_warmup_seconds"])
        if status == "ok":
            def on_sample(s, base=base):
                batch, elapsed = int(s["batch"]), int(s["elapsed_ns"])
                samples_log.write({**base, "numeric_type": "float64", "kind": s["kind"], "index": s["index"],
                                   "batch": batch, "elapsed_ns": elapsed, "ns_per_op": f"{elapsed / batch:.1f}",
                                   "checksum": s["checksum"], **{k: s.get(k, "") for k in COUNTER_FIELDS},
                                   **{k: s.get(k, "") for k in DIST_FIELDS}})
                if s["kind"] == "measure":
                    print(f" {elapsed / batch / 1e6:.4g}ms", end="", flush=True)

            driver_launch = time.time()
            status, received, stderr, detail, connected = run_driver(
                cmd, budget["startup_slack_seconds"], limit, on_sample, meta,
                alive=cluster.alive if cluster else (lambda: True))
            if dist and connected is not None:
                setup_client = connected - driver_launch
                extra.update(setup_client_s=f"{setup_client:.3f}",
                             setup_s=f"{float(extra['setup_cluster_s']) + setup_client:.3f}")

        resources = parse_time_output(stderr)
        own = number(resources.get("process_user_s")) or 0.0
        own += number(resources.get("process_sys_s")) or 0.0
        footprint_total = resources.get("peak_footprint_bytes") or 0
        if cluster is not None:
            members = cluster.stop()
            rss = [m["max_rss_bytes"] for m in members if m["max_rss_bytes"] != ""]
            foot = [m["peak_footprint_bytes"] for m in members if m["peak_footprint_bytes"] != ""]
            users = [number(m.get("process_user_s")) or 0.0 for m in members]
            syss = [number(m.get("process_sys_s")) or 0.0 for m in members]
            own += sum(users) + sum(syss)
            footprint_total += sum(foot)
            extra.update(members_max_rss_bytes=max(rss, default=""), members_rss_sum_bytes=sum(rss),
                         members_footprint_sum_bytes=sum(foot), members_footprint_max_bytes=max(foot, default=""),
                         members_user_s=f"{sum(users):.2f}", members_sys_s=f"{sum(syss):.2f}")
            oom = [m["member"] for m in members if m["exit_status"] == OUT_OF_MEMORY_EXIT]
            if oom and status != "memory_budget_exceeded":
                status, detail = "memory_budget_exceeded", f"members {oom} ran out of heap; {detail}".strip("; ")
            for m, mem in zip(members, cluster.members):
                joined = mem.events.get("joined", ({}, None))[0]
                ready = mem.events.get("ready", ({}, None))[0]
                stopping = mem.events.get("stopping", ({}, None))[0]
                members_log.write({**base, "member": m["member"], "pid": m["pid"], "exit_status": m["exit_status"],
                                   "killed": m["killed"], "heap_mib": extra["member_heap_mib"],
                                   "hz_start_ms": joined.get("hz_start_ms", ""),
                                   "cluster_wait_ms": joined.get("cluster_wait_ms", ""),
                                   "joined_s": f"{int(joined['epoch_ms']) / 1000 - cluster.launch_epoch:.3f}"
                                   if "epoch_ms" in joined else "",
                                   "ready_s": f"{int(ready['epoch_ms']) / 1000 - cluster.launch_epoch:.3f}"
                                   if "epoch_ms" in ready else "",
                                   "partitions_owned": joined.get("partitions_owned", ""),
                                   "pre_warm_ns": ready.get("pre_warm_ns", ""),
                                   "jvm_threads_ready": ready.get("jvm_threads", ""),
                                   "os_threads_ready": ready.get("os_threads", ""),
                                   "jvm_threads_stopping": stopping.get("jvm_threads", ""),
                                   "os_threads_stopping": stopping.get("os_threads", ""),
                                   "gc_collections": stopping.get("gc_collections", ""),
                                   "gc_ms": stopping.get("gc_ms", ""), "member_cpu_ns": stopping.get("cpu_ns", ""),
                                   "member_cpu_p_ns": stopping.get("cpu_p_ns", ""),
                                   "idle_cpu_s": f"{idle[m['member']]:.3f}" if m["member"] < len(idle)
                                   and idle[m["member"]] is not None else "",
                                   **{k: m.get(k, "") for k in ("max_rss_bytes", "peak_footprint_bytes",
                                                                "process_wall_s", "process_user_s", "process_sys_s",
                                                                "instructions_retired", "cycles_elapsed")}})
            if member_log is not None and member_log.exists():
                with open(member_log) as f:
                    reader = csv.DictReader(f)
                    if member_samples_log is None:
                        member_samples_log = CsvLog(out_dir / "member_samples.csv",
                                                    MEMBER_SAMPLE_PREFIX + reader.fieldnames)
                    for row in reader:
                        member_samples_log.write({**base, **row})
                member_log.unlink()
        window = time.monotonic() - window_start
        system_busy = busy_cpu_seconds() - cpu_before
        other = system_busy - own if resources.get("process_user_s") else None
        if status == "ok" and footprint_total > budget["max_total_footprint_bytes"]:
            status, detail = "memory_budget_exceeded", \
                f"total peak footprint {footprint_total} B > {budget['max_total_footprint_bytes']} B"
        slept = sleep_gap() - gap_before
        if slept > measurement["max_sleep_gap_seconds"]:
            status, detail = "slept", f"the system slept for about {slept:.0f} s during this configuration"
        if status != "ok":
            stopped[series(run)] = f"{status} at n={run['n']}"
        runs_log.write({**base, **resources, **{k: meta.get(k, "") for k in META_FIELDS}, **extra,
                        "compute_threads": compute_threads(run), "status": status, "warmup_runs": warmup,
                        "measured_reps": reps, "samples_received": received,
                        "total_footprint_bytes": footprint_total or "", "window_s": f"{window:.2f}",
                        "system_busy_cpu_s": f"{system_busy:.2f}", "own_cpu_s": f"{own:.2f}",
                        "other_cpu_s": f"{other:.2f}" if other is not None else "",
                        "started_at": started, "detail": detail, "command": " ".join(cmd)})
        foot_text = f"{footprint_total / 2**30:.2f} GiB" if footprint_total else "n/a"
        other_text = f"{other / window:.2f} cores" if other is not None else "n/a"
        setup_text = f", setup {extra['setup_s']} s" if "setup_s" in extra else ""
        print(f"  -> {status}, footprint {foot_text}, others {other_text}{setup_text}"
              f"{', ' + detail if detail else ''}")
        if status == "slept" and tracked:
            print(f"session stopped: the system slept during {rid}; start a new session", file=sys.stderr)
            aborted = True
            break
        busy = number(resources.get("process_wall_s")) or window
        pause = args.pause if args.pause is not None else min(cool["max_seconds"],
                                                                max(cool["min_seconds"], cool["factor"] * busy))
        time.sleep(pause)

    after = subprocess.run(["sh", "-c", "date '+%Y-%m-%dT%H:%M:%S%z'; pmset -g therm; pmset -g batt | head -2; uptime"],
                           capture_output=True, text=True)
    with open(out_dir / "env.txt", "a") as f:
        f.write("\n## After the session\n" + after.stdout)
    minutes = (time.monotonic() - session_start) / 60
    print(f"session {session} {'stopped' if aborted else 'finished'} in {minutes:.1f} min -> {out_dir.relative_to(ROOT)}")
    return 1 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
