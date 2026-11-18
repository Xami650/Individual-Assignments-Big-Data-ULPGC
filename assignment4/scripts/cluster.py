"""Start, watch and stop the w member processes of one distributed configuration.

Each member is one JVM (`java -jar target/matmul.jar member ...`, see
MemberMain.java), launched under `/usr/bin/time -l` so that its peak memory
and CPU times are known when it exits. A member reports its life cycle on
stdout as lines "@ <event> key=value,..." (started, joined, ready, stopping);
the harness records the fields and the wall-clock instant of every event. The
members run until their standard input is closed, which is how they are told
to terminate; if the harness itself dies, the members see end-of-file and
terminate too.

Used by scripts/run_benchmarks.py and scripts/verify.py.
"""

import os
import signal
import subprocess
import threading
import time

import psutil

from common import ROOT, member_command, parse_time_output

OUT_OF_MEMORY_EXIT = 3


class Member:
    def __init__(self, index, cmd):
        self.index = index
        self.cmd = cmd
        self.events = {}  # name -> (fields, epoch seconds when the harness received it)
        self.stderr = []
        self.proc = subprocess.Popen(["/usr/bin/time", "-l", *cmd], cwd=ROOT, text=True, bufsize=1,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     start_new_session=True)
        self.changed = threading.Condition()
        threading.Thread(target=self._read_events, daemon=True).start()
        self.err_thread = threading.Thread(target=lambda: self.stderr.extend(self.proc.stderr), daemon=True)
        self.err_thread.start()

    def _read_events(self):
        for line in self.proc.stdout:
            if line.startswith("@ "):
                parts = line[2:].strip().split(" ", 1)
                fields = dict(kv.split("=", 1) for kv in parts[1].split(",")) if len(parts) > 1 else {}
                with self.changed:
                    self.events[parts[0]] = (fields, time.time())
                    self.changed.notify_all()
        with self.changed:
            self.changed.notify_all()

    def has(self, event):
        return event in self.events

    def field(self, event, key, default=""):
        return self.events.get(event, ({}, None))[0].get(key, default)

    @property
    def pid(self):
        """Process id of the member's JVM (not of /usr/bin/time)."""
        value = self.field("started", "pid")
        return int(value) if value else None


class Cluster:
    """The members of one configuration: w JVMs with k compute threads each."""

    def __init__(self, config, name, workers, threads, base_port, n, pre_warm_seconds=None):
        self.config = config
        self.name = name
        self.workers = workers
        self.threads = threads
        self.base_port = base_port
        self.n = n
        self.pre_warm_seconds = pre_warm_seconds
        self.members = []
        self.launch_epoch = None

    def start(self, timeout):
        """Launch every member and wait until all are ready; returns (status, detail)."""
        self.launch_epoch = time.time()
        for m in range(self.workers):
            cmd = member_command(self.config, self.name, m, self.workers, self.threads, self.base_port, self.n,
                                 self.pre_warm_seconds)
            self.members.append(Member(m, cmd))
        deadline = time.monotonic() + timeout
        while True:
            if all(m.has("ready") for m in self.members):
                return "ok", ""
            dead = [m for m in self.members if m.proc.poll() is not None]
            if dead:
                m = dead[0]
                m.err_thread.join(timeout=5)
                errors = [l.strip() for l in m.stderr if l.startswith("error:")]
                status = "memory_budget_exceeded" if m.proc.returncode == OUT_OF_MEMORY_EXIT else "error"
                return status, f"member {m.index} exited with status {m.proc.returncode} before it was ready: " \
                               f"{errors[:2] or [l.strip() for l in m.stderr if 'Exception' in l][:2]}"
            if time.monotonic() > deadline:
                missing = [m.index for m in self.members if not m.has("ready")]
                return "timeout", f"members {missing} not ready within {timeout:g} s"
            time.sleep(0.02)

    def setup_times(self):
        """Seconds from launch until the last member joined, and until the last member was ready."""
        joined = [int(m.field("joined", "epoch_ms", 0)) / 1000 for m in self.members]
        ready = [int(m.field("ready", "epoch_ms", 0)) / 1000 for m in self.members]
        return max(joined) - self.launch_epoch, max(ready) - self.launch_epoch

    def idle_cpu(self, seconds):
        """CPU seconds used by each member's JVM over `seconds` of idleness (no driver yet)."""
        procs = []
        for m in self.members:
            try:
                procs.append(psutil.Process(m.pid))
            except (psutil.Error, TypeError):
                procs.append(None)
        before = [sum(p.cpu_times()[:2]) if p else None for p in procs]
        time.sleep(seconds)
        after = [sum(p.cpu_times()[:2]) if p else None for p in procs]
        return [a - b if a is not None and b is not None else None for a, b in zip(after, before)]

    def alive(self):
        return all(m.proc.poll() is None for m in self.members)

    def stop(self, timeout=60):
        """Close every member's stdin, wait for it to exit (kill after `timeout`); returns one dict per member."""
        for m in self.members:
            try:
                m.proc.stdin.close()
            except OSError:
                pass
        deadline = time.monotonic() + timeout
        results = []
        for m in self.members:
            killed = False
            try:
                m.proc.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(m.proc.pid, signal.SIGKILL)
                m.proc.wait()
                killed = True
            m.err_thread.join(timeout=5)
            stderr = "".join(m.stderr)
            results.append({"member": m.index, "pid": m.pid, "exit_status": m.proc.returncode, "killed": killed,
                            "stderr": stderr, **parse_time_output(stderr)})
        return results

    def kill(self):
        for m in self.members:
            if m.proc.poll() is None:
                try:
                    os.killpg(m.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for m in self.members:
            m.proc.wait()
