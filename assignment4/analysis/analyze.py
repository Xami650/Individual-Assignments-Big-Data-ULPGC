"""Analysis of one benchmark session: summary tables and figures for the report.

Adapted from Assignment 3's analysis/analyze.py (tag assignment3-v1.0): the same
statistics, table and figure helpers and palette. New here: the phases of the
distributed multiplication, the bytes moved, the setup of the cluster, the
members' compute times and counters, their memory, and the grain, weak-scaling
and large-size experiments.

Reads results/raw/<session>/{samples.csv, member_samples.csv, runs.csv,
members.csv, config.json} (default: the latest session that is not a smoke or
pilot run) and writes
    analysis/tables/summary.csv     one row per configuration, full precision
    analysis/tables/members.csv     one row per member of every distributed configuration
    analysis/tables/checksums.csv   checksum agreement across the configurations of each n
    analysis/tables/*.tex           booktabs tables included by the report
    analysis/figures/*.pdf, *.png   figures (vector PDF for the report)

Statistics: only measured samples (kind = measure) are summarised; warm-ups
are excluded. The central value is the median of the time per multiplication
(elapsed_ns / batch) and the variability is the interquartile range (pandas
quantiles with linear interpolation), also relative to the median; with three
samples (D_large) the range is reported. Every phase of dist is summarised the
same way (median over the samples of phase_ns / batch); the phases are
consecutive, so the sum of their medians is compared with the median end-to-end
time. The speedup of a configuration is S = median T_ikj / median T, with
T_ikj from the first ikj run of its size block (the only ikj run in D_weak and
D_large blocks), and its interval [Q1_ikj / Q3, Q3_ikj / Q1]; the efficiency is
E = S / c with c the compute threads (w k or p). The cost of distribution is
T_dist / T_par, against par_ikj with the static schedule and as many threads as
the cluster has compute threads, at the same n. Counter ratios are ratios of
sums over the measured samples.

Usage (from assignment4/):
    python3 analysis/analyze.py [--session 20261118-101500]
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
TABLES = ROOT / "analysis" / "tables"
FIGURES = ROOT / "analysis" / "figures"

# Ordinal blue ramp for n (validated with the palette validator, --ordinal), markers as secondary encoding.
RAMP = {256: "#86b6ef", 512: "#5598e7", 1024: "#2a78d6", 2048: "#1c5cab", 4096: "#0d366b"}
MARKER_N = {256: "o", 512: "s", 1024: "D", 2048: "^", 4096: "v"}
# Categorical slots in their fixed order (blue, orange, aqua, yellow, magenta; validated, adjacent pairs).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
PHASES = [("serialize", "serialize"), ("scatter", "scatter A"), ("broadcast", "broadcast B"),
          ("compute", "compute"), ("gather", "gather C")]
PHASE_COLOR = dict(zip([p for p, _ in PHASES], SLOTS))
DIST_STYLE = dict(ls="-")
PAR_STYLE = dict(ls="--", mfc="white")
INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
MIB, GIB = 2 ** 20, 2 ** 30
P_CORES, LOGICAL = 4, 10
PHASE_FIELDS = ["serialize_ns", "scatter_ns", "broadcast_ns", "compute_ns", "gather_ns", "assemble_ns",
                "task_max_ns", "task_mean_ns", "kernel_max_ns", "kernel_mean_ns", "read_max_ns", "store_max_ns",
                "dispatch_max_ns", "tasks_done_ns"]
BYTE_FIELDS = ["a_payload_bytes", "b_payload_bytes", "c_payload_bytes", "scatter_socket_out",
               "broadcast_socket_out", "compute_socket_out", "compute_socket_in", "gather_socket_in", "socket_out",
               "socket_in"]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.5, "lines.markersize": 4.5,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


# ---------------------------------------------------------------- data

def latest_session():
    sessions = sorted(p for p in RAW.iterdir() if p.is_dir() and not p.name.startswith(("smoke", "pilot")))
    if not sessions:
        raise SystemExit("no benchmark session found in results/raw")
    return sessions[-1]


def summarise(samples, runs, msamples):
    measured = samples[samples.kind == "measure"].copy()
    g = measured.groupby("run_id")
    t = g.ns_per_op
    stats = pd.DataFrame({
        "reps": t.count(), "median_ns": t.median(), "q1_ns": t.quantile(0.25), "q3_ns": t.quantile(0.75),
        "min_ns": t.min(), "max_ns": t.max(), "batch": g.batch.first(), "ops": g.batch.sum(),
        "elapsed_sum": g.elapsed_ns.sum(), "cpu_sum": g.cpu_ns.sum(), "cpu_p_sum": g.cpu_p_ns.sum(),
        "instr_sum": g.instructions.sum(), "cycles_sum": g.cycles.sum(),
    })
    for field in PHASE_FIELDS:
        measured[field + "_op"] = measured[field] / measured.batch
        stats[field.replace("_ns", "_med")] = measured.groupby("run_id")[field + "_op"].median()
        stats[field.replace("_ns", "_sum")] = g[field].sum()
    for field in BYTE_FIELDS:
        stats[field + "_op"] = (g[field].sum() / g.batch.sum())
    stats["warmup_samples"] = samples[samples.kind == "warmup"].groupby("run_id").size()
    s = runs.join(stats, on="run_id")
    for column in ("schedule", "repeat"):
        s[column] = s[column].fillna("")
    n = s.n.astype(float)
    n2, n3 = n ** 2, n ** 3
    s["iqr_rel"] = (s.q3_ns - s.q1_ns) / s.median_ns
    s["range_rel"] = (s.max_ns - s.min_ns) / s.median_ns
    s["mean_ns"] = s.elapsed_sum / s.ops
    s["driver_p_share"] = s.cpu_p_sum / s.cpu_sum
    s["cycles_per_madd"] = np.where(s.method == "dist", np.nan, s.cycles_sum / (s.ops * n3))
    s["p_share"] = np.where(s.method == "dist", np.nan, s.cpu_p_sum / s.cpu_sum)
    s["gflops"] = 2 * n3 / s.median_ns

    # distributed phases
    s["phase_sum_med"] = s[["serialize_med", "scatter_med", "broadcast_med", "compute_med", "gather_med",
                            "assemble_med"]].sum(axis=1, min_count=1)
    s["comm_med"] = s[["serialize_med", "scatter_med", "broadcast_med", "gather_med", "assemble_med"]].sum(
        axis=1, min_count=1)
    s["comm_share"] = s.comm_med / s.phase_sum_med
    s["phase_check"] = s.phase_sum_med / s.median_ns
    s["return_med"] = s.compute_med - s.tasks_done_med
    s["imbalance"] = s.kernel_max_sum / s.kernel_mean_sum
    s["payload_op"] = s.a_payload_bytes_op + s.b_payload_bytes_op + s.c_payload_bytes_op
    s["model_bytes"] = np.where(s.method == "dist", (s.workers + 2) * 8 * n2, np.nan)
    s["payload_over_model"] = s.payload_op / s.model_bytes
    s["socket_total_op"] = s.socket_out_op + s.socket_in_op
    s["socket_overhead"] = s.socket_total_op / s.payload_op - 1
    # bytes per nanosecond, which is GB/s
    s["rate_serialize"] = (s.a_payload_bytes_op + s.b_payload_bytes_op / s.workers.clip(lower=1)) / s.serialize_med
    s["rate_scatter"] = s.a_payload_bytes_op / s.scatter_med
    s["rate_broadcast"] = s.b_payload_bytes_op / s.broadcast_med
    s["rate_gather"] = s.c_payload_bytes_op / s.gather_med
    s["compute_threads"] = np.where(s.method == "dist", s.workers * s.threads,
                                    np.where(s.method == "par_ikj", s.threads, 1))

    # the members' kernels (member log): counters summed over members and samples
    ms = msamples[msamples.kind == "measure"].copy()
    if not ms.empty:
        ms["madds"] = ms.ops * ms.rows * ms.n.astype(float) ** 2
        mg = ms.groupby("run_id")
        agg = pd.DataFrame({"m_cycles": mg.cycles.sum(), "m_madds": mg.madds.sum(), "m_cpu": mg.cpu_ns.sum(),
                            "m_cpu_p": mg.cpu_p_ns.sum(), "m_kernel": mg.kernel_ns.sum(),
                            "m_local": mg.local.all()})
        s = s.join(agg, on="run_id")
        dist = s.method == "dist"
        s.loc[dist, "cycles_per_madd"] = s.m_cycles / s.m_madds
        s.loc[dist, "p_share"] = s.m_cpu_p / s.m_cpu
        s["members_busy"] = s.m_cpu / s.m_kernel
    # setup and memory
    s["members_idle_cores"] = s.members_idle_cpu_s / 2.0
    s["driver_footprint"] = s.peak_footprint_bytes
    s["member_model_bytes"] = np.where(s.method == "dist", 8 * n2 + 16 * n2 / s.workers.clip(lower=1), np.nan)

    # baselines: ikj of the size block (first run where there are two), par_ikj static with the same threads
    ikj = s[(s.method == "ikj") & (s.repeat.isin(["first", ""]))].sort_values("repeat", ascending=False)
    ikj = ikj.drop_duplicates("n").set_index("n")
    base = ikj.reindex(s.n)
    base.index = s.index
    s["speedup"] = base.median_ns / s.median_ns
    s["speedup_lo"] = base.q1_ns / s.q3_ns
    s["speedup_hi"] = base.q3_ns / s.q1_ns
    s["efficiency"] = s.speedup / s.compute_threads
    par = s[(s.method == "par_ikj") & (s.schedule == "static") & (s.repeat == "")]
    par = par.drop_duplicates(["n", "threads"]).set_index(["n", "threads"])
    key = pd.MultiIndex.from_arrays([s.n, s.compute_threads])
    match = par.reindex(key)
    match.index = s.index
    s["par_median_ns"] = match.median_ns
    s["dist_cost"] = np.where(s.method == "dist", s.median_ns / match.median_ns, np.nan)
    s["compute_over_par"] = np.where(s.method == "dist", s.compute_med / match.median_ns, np.nan)
    s["kernel_over_par"] = np.where(s.method == "dist", s.kernel_max_med / match.median_ns, np.nan)
    s["par_cycles_per_madd"] = match.cycles_per_madd
    return s


def member_table(msamples, members, runs):
    """One row per member of each distributed configuration: kernel time per op, P-core share, memory."""
    ms = msamples[msamples.kind == "measure"].copy()
    if ms.empty:
        return pd.DataFrame()
    ms["kernel_op"] = ms.kernel_ns / ms.ops
    g = ms.groupby(["run_id", "member_index"])
    m = pd.DataFrame({"rows": g.rows.first(), "kernel_med_ns": g.kernel_op.median(),
                      "cpu": g.cpu_ns.sum(), "cpu_p": g.cpu_p_ns.sum(), "cycles": g.cycles.sum(),
                      "ops": g.ops.sum(), "n": g.n.first(), "workers": g.workers.first(),
                      "threads": g.threads.first(), "experiment": g.experiment.first()}).reset_index()
    m["p_share"] = m.cpu_p / m.cpu
    m["cycles_per_madd"] = m.cycles / (m.ops * m.rows * m.n.astype(float) ** 2)
    mem = members.rename(columns={"member": "member_index"})[
        ["run_id", "member_index", "max_rss_bytes", "peak_footprint_bytes", "os_threads_ready", "jvm_threads_ready",
         "idle_cpu_s", "hz_start_ms", "joined_s", "gc_collections", "heap_mib"]]
    return m.merge(mem, on=["run_id", "member_index"], how="left")


def checksum_agreement(samples):
    """Per n: are the measured checksums of every configuration identical (exact float())?"""
    measured = samples[samples.kind == "measure"].copy()
    measured["value"] = measured.checksum.map(float)  # round-trip exact, unlike pandas' fast parser
    rows = []
    for n, g in measured.groupby("n"):
        rows.append({"n": n, "configurations": g.run_id.nunique(), "samples": len(g),
                     "distinct_values": g.value.nunique(), "identical": g.value.nunique() == 1})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- formatting

def missing(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def sig(x, digits=3):
    """Format with about `digits` significant digits, never in scientific notation."""
    if missing(x):
        return "--"
    if x == 0:
        return "0"
    decimals = max(0, digits - 1 - math.floor(math.log10(abs(x))))
    return f"{x:,.{decimals}f}".replace(",", "\\,")


def pct(x, decimals=1):
    if missing(x):
        return "--"
    value = 100 * x
    return f"{0.0 if abs(value) < 0.5 * 10 ** -decimals else value:.{decimals}f}"


def ms(x):
    return "--" if missing(x) else sig(x / 1e6)


def write_table(name, header, rows, colspec, notes=None):
    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [(r"\midrule" if r == "midrule" else " & ".join(r) + r" \\") for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines.append(r"\par\smallskip{\footnotesize " + notes + "}")
    (TABLES / name).write_text("\n".join(lines) + "\n")


def pick(s, **keys):
    hit = s
    for k, v in keys.items():
        hit = hit[hit[k] == v]
    return hit.iloc[0] if len(hit) else None


def get(row, column):
    return None if row is None else row[column]


def dist_row(s, n, w, k=1, experiment=None, repeat=""):
    keys = dict(method="dist", n=n, workers=w, threads=k, repeat=repeat)
    if experiment:
        keys["experiment"] = experiment
    return pick(s, **keys)


def par_row(s, n, p, schedule="static"):
    return pick(s, method="par_ikj", n=n, threads=p, schedule=schedule, repeat="")


def ikj_row(s, n, repeat=None):
    if repeat is None:
        row = pick(s, method="ikj", n=n, repeat="first")
        return row if row is not None else pick(s, method="ikj", n=n, repeat="")
    return pick(s, method="ikj", n=n, repeat=repeat)


def ok(row):
    return row is not None and row.status == "ok"


def time_cell(row):
    if row is None:
        return "--"
    return ms(row.median_ns) if ok(row) else r"\textit{" + row.status.replace("_", " ") + "}"


# ---------------------------------------------------------------- tables

def tables(s, config, members):
    ex = config["experiments"]
    strong = ex["D_strong"]
    sizes, workers = strong["sizes"], strong["workers"]
    head_n = [f"$n={n}$" for n in sizes]

    rows = [["basic"] + [time_cell(pick(s, method="basic", n=n)) for n in sizes],
            ["ikj"] + [time_cell(ikj_row(s, n)) for n in sizes]]
    for p in strong["parallel_dynamic"]:
        rows.append([f"par\\_ikj dyn., $p={p}$"] + [time_cell(par_row(s, n, p, "dynamic")) for n in sizes])
    rows.append("midrule")
    for p in workers:
        rows.append([f"par\\_ikj static, $p={p}$"] + [time_cell(par_row(s, n, p)) for n in sizes])
    rows.append("midrule")
    for w in workers:
        rows.append([f"dist, $w={w}$"] + [time_cell(dist_row(s, n, w, experiment="D_strong")) for n in sizes])
    shown = s[(s.experiment == "D_strong") & (s.status == "ok") & (s.repeat.isin(["", "first"]))]
    rows += ["midrule", ["largest IQR"] + [f"{100 * shown[shown.n == n].iqr_rel.max():.1f}\\%" for n in sizes]]
    write_table("tab_time.tex", ["configuration"] + head_n, rows, "l" + "r" * len(sizes),
                "Median of 5 measured samples, in ms per multiplication, end to end (Sec.~\\ref{sec:timing}). "
                "basic was run up to $n=2048$ only.")

    rows = []
    for w in workers:
        cells = [str(w)]
        for n in sizes:
            d, p = dist_row(s, n, w, experiment="D_strong"), par_row(s, n, w)
            cells.append(f"{sig(get(d, 'speedup'))} / {sig(get(p, 'speedup'))} / {sig(get(d, 'dist_cost'))}"
                         if ok(d) and ok(p) else "--")
        rows.append(cells)
    write_table("tab_speedup.tex", ["$w=p$"] + head_n, rows, "r" + "c" * len(sizes),
                r"Each cell: speedup of dist over ikj / speedup of par\_ikj (static, $p=w$) / "
                r"$T_\mathrm{dist}/T_\mathrm{par}$, the cost of distributing the same bands.")

    for n in (512, 4096):
        rows = []
        for w in workers:
            d = dist_row(s, n, w, experiment="D_strong")
            if not ok(d):
                continue
            rows.append([str(w), ms(d.serialize_med), ms(d.scatter_med), ms(d.broadcast_med), ms(d.compute_med),
                         ms(d.gather_med + d.assemble_med), ms(d.phase_sum_med), ms(d.median_ns),
                         pct(d.comm_share), ms(get(par_row(s, n, w), "median_ns"))])
        write_table(f"tab_phases_{n}.tex", ["$w$", "serialize", "scatter A", "broadcast B", "compute", "gather C",
                                            "sum", "end to end", "comm.\\ (\\%)", "par\\_ikj"],
                    rows, "rrrrrrrrrr",
                    "Medians over 5 samples, ms per multiplication. comm.: share of serialize, scatter, broadcast "
                    "and gather in the sum of the phases. par\\_ikj: static schedule, $p=w$.")

    rows = []
    for n in (1024, 4096):
        for w in workers:
            d, p = dist_row(s, n, w, experiment="D_strong"), par_row(s, n, w)
            if not ok(d):
                continue
            rows.append([str(n) if w == workers[0] else "", str(w), ms(get(p, "median_ns")), ms(d.compute_med),
                         ms(d.kernel_max_med), ms(d.kernel_mean_med), sig(d.imbalance),
                         ms(d.dispatch_max_med), ms(d.return_med), ms(d.read_max_med), ms(d.store_max_med),
                         pct(d.p_share, 0), sig(d.cycles_per_madd), sig(get(p, "cycles_per_madd"))])
        if n == 1024:
            rows.append("midrule")
    group = (r" & & & \multicolumn{8}{c}{dist: compute phase and its parts (ms)} & & \multicolumn{2}{c}{cyc./m.-add}"
             r" \\ \cmidrule(lr){4-11} \cmidrule(lr){13-14} $n$")
    write_table("tab_compute.tex", [group, "$w$", "par (ms)", "phase", "kernel max", "kernel mean", "max/mean",
                                    "dispatch", "return", "read", "store", "P (\\%)", "dist", "par"],
                rows, "rrrrrrrrrrrrrr",
                "Medians over 5 samples. kernel max/mean: slowest and mean member; dispatch: from the start of the "
                "phase to the start of the last task; return: from the end of the last task to the end of the phase; "
                "read and store: slowest member. P: share of the members' kernel CPU time on performance cores.")

    rows = []
    for n in (1024, 4096):
        for w in workers:
            d = dist_row(s, n, w, experiment="D_strong")
            if not ok(d):
                continue
            rows.append([str(n) if w == workers[0] else "", str(w), sig(d.a_payload_bytes_op / MIB),
                         sig(d.b_payload_bytes_op / MIB), sig(d.c_payload_bytes_op / MIB),
                         f"{d.payload_over_model:.5f}", pct(d.socket_overhead, 3),
                         sig(d.rate_serialize), sig(d.rate_scatter), sig(d.rate_broadcast),
                         sig(d.rate_gather)])
        if n == 1024:
            rows.append("midrule")
    group = (r" & & \multicolumn{3}{c}{payload (MiB)} & & & \multicolumn{4}{c}{effective rate (GB/s)} \\ "
             r"\cmidrule(lr){3-5} \cmidrule(lr){8-11} $n$")
    write_table("tab_transfer.tex", [group, "$w$", "A", "B", "C", "/ model", "socket +\\%", "serialize", "scatter",
                                     "broadcast", "gather"], rows, "rrrrrrrrrrr",
                r"Bytes per multiplication. model: $(w+2)\,8n^2$. socket: bytes counted on the client's connections "
                r"over the payload, in \%. Rates: payload / phase time; serialize counts A and one copy of B.")

    rows = []
    for w in workers:
        d = s[(s.method == "dist") & (s.workers == w) & (s.threads == 1) & (s.status == "ok")]
        if d.empty:
            continue
        mem = members[(members.workers == w) & (members.threads == 1)] if not members.empty else members
        rows.append([str(w), sig(d.setup_cluster_s.median()),
                     f"{sig(d.setup_cluster_s.min())}--{sig(d.setup_cluster_s.max())}",
                     sig(d.setup_client_s.median()), sig(d.setup_s.median()),
                     sig(mem.hz_start_ms.median() / 1000) if len(mem) else "--",
                     sig(mem.os_threads_ready.median()) if len(mem) else "--",
                     sig(1000 * (d.members_idle_cores / w).median()),
                     sig(d.ping_median_ns.median() / 1000), str(len(d))])
    write_table("tab_setup.tex", ["$w$", "cluster (s)", "range (s)", "client (s)", "setup (s)",
                                  "member start (s)", "OS threads", "idle (mcores)", "ping ($\\mu$s)", "runs"],
                rows, "rrrrrrrrrr",
                "Medians over all distributed runs with $k=1$ (all sizes). cluster: from launching the members to the "
                "last member joined; client: from launching the driver to connected; member start: "
                "Hazelcast.newHazelcastInstance per member; OS threads and idle CPU per member; ping: median "
                "round trip of an empty task.")

    rows = []
    for n in (1024, 4096):
        for w in workers:
            d = dist_row(s, n, w, experiment="D_strong")
            if not ok(d):
                continue
            mem = members[members.run_id == d.run_id]
            rows.append([str(n) if w == workers[0] else "", str(w), str(int(d.member_heap_mib)),
                         sig(mem.peak_footprint_bytes.median() / MIB), sig(mem.peak_footprint_bytes.max() / MIB),
                         sig(mem.max_rss_bytes.median() / MIB), sig(d.member_model_bytes / MIB),
                         sig(d.members_footprint_sum_bytes / GIB), sig(d.driver_footprint / GIB),
                         sig(d.total_footprint_bytes / GIB)])
        if n == 1024:
            rows.append("midrule")
    for c in config["experiments"]["D_large"]["configurations"]:
        if c["method"] != "dist":
            continue
        d = dist_row(s, 8192, c["workers"], experiment="D_large")
        if d is None:
            continue
        mem = members[members.run_id == d.run_id]
        rows += ["midrule", ["8192", str(c["workers"]), str(int(d.member_heap_mib)),
                             sig(mem.peak_footprint_bytes.median() / MIB), sig(mem.peak_footprint_bytes.max() / MIB),
                             sig(mem.max_rss_bytes.median() / MIB), sig(d.member_model_bytes / MIB),
                             sig(d.members_footprint_sum_bytes / GIB), sig(d.driver_footprint / GIB),
                             sig(d.total_footprint_bytes / GIB)]]
    group = (r" & & & \multicolumn{4}{c}{per member (MiB)} & \multicolumn{3}{c}{totals (GiB)} \\ "
             r"\cmidrule(lr){4-7} \cmidrule(lr){8-10} $n$")
    write_table("tab_memory.tex", [group, "$w$", "-Xmx", "footpr.", "max", "RSS", "model", "members", "driver",
                                   "all"], rows, "rrrrrrrrrr",
                r"Peak memory footprint and resident set size from \texttt{/usr/bin/time -l} (median and maximum "
                r"over the members); model: matrix elements stored per member, $8n^2 + 16n^2/w$; all: members "
                r"plus driver.")

    grain = ex["D_grain"]
    rows = []
    for n in grain["sizes"]:
        for w, k in grain["layouts"]:
            d = dist_row(s, n, w, k)
            if d is None:
                continue
            rows.append([str(n) if (w, k) == tuple(grain["layouts"][0]) else "", f"{w} $\\times$ {k}",
                         time_cell(d), sig(get(d, "speedup")), ms(get(d, "comm_med")), ms(get(d, "compute_med")),
                         ms(get(d, "kernel_max_med")), sig(get(d, "imbalance")), pct(get(d, "p_share"), 0),
                         sig(get(d, "members_footprint_sum_bytes") / GIB if ok(d) else None)])
        p = par_row(s, n, LOGICAL)
        rows.append(["", r"par\_ikj, $p=10$", time_cell(p), sig(get(p, "speedup")), "", "", "", "",
                     pct(get(p, "p_share"), 0), ""])
        if n != grain["sizes"][-1]:
            rows.append("midrule")
    write_table("tab_grain.tex", ["$n$", "$w \\times k$", "time (ms)", "$S$", "comm.\\ (ms)", "compute (ms)",
                                  "kernel max", "max/mean", "P (\\%)", "members (GiB)"], rows, "rlrrrrrrrr",
                "Ten compute threads grouped into $w$ members of $k$ threads. comm.: serialize, scatter, broadcast "
                "and gather; members: sum of the members' peak footprints.")

    weak = ex["D_weak"]
    rows = []
    d1, p1 = dist_row(s, 2048, 1), par_row(s, 2048, 1)
    for w, n in weak["points"]:
        d, p = dist_row(s, n, w), par_row(s, n, w)
        work = n ** 3 / w / 2048 ** 3 - 1
        rows.append([str(w), str(n), f"{100 * work:+.3f}", time_cell(d), time_cell(p), time_cell(ikj_row(s, n)),
                     sig(d1.median_ns / d.median_ns if ok(d) and ok(d1) else None),
                     sig(p1.median_ns / p.median_ns if ok(p) and ok(p1) else None),
                     pct(get(d, "comm_share")), sig(get(d, "payload_op") / MIB if ok(d) else None)])
    write_table("tab_weak.tex", ["$w$", "$n_w$", "work (\\%)", "dist (ms)", "par (ms)", "ikj (ms)",
                                 "$E_w$ dist", "$E_w$ par", "comm.\\ (\\%)", "payload (MiB)"], rows, "rrrrrrrrrr",
                r"Work per worker $n_w^3/w$ relative to $2048^3$. $E_w = T(1, 2048)/T(w, n_w)$; par: par\_ikj "
                r"static with $p=w$.")

    large = ex["D_large"]
    rows = []
    for c in large["configurations"]:
        if c["method"] == "dist":
            r = dist_row(s, 8192, c["workers"], c.get("threads_per_worker", 1), experiment="D_large")
            label = f"dist, $w={c['workers']}$"
        elif c["method"] == "par_ikj":
            r = pick(s, method="par_ikj", n=8192, threads=c["threads"])
            label = f"par\\_ikj static, $p={c['threads']}$"
        else:
            r = pick(s, method=c["method"], n=8192)
            label = c["method"]
        if r is None:
            continue
        rng = f"{ms(r.min_ns)}--{ms(r.max_ns)}" if ok(r) else "--"
        rows.append([label, r.status.replace("_", " "), time_cell(r), rng, sig(get(r, "speedup")) if ok(r) else "--",
                     sig(get(r, "dist_cost")) if ok(r) else "--", sig(r.total_footprint_bytes / GIB),
                     str(r.detail).replace("_", r"\_") if isinstance(r.detail, str) and r.status != "ok" else ""])
    header = ["configuration", "status", "median (ms)", "range (ms)", "$S$", "dist/par", "footprint (GiB)", "detail"]
    spec = "llrrrrrp{4.2cm}"
    if not any(row[-1] for row in rows):  # no failure to explain: drop the empty detail column
        header, spec, rows = header[:-1], spec[:-len("p{4.2cm}")], [row[:-1] for row in rows]
    write_table("tab_large.tex", header, rows, spec,
                "Three measured samples (range instead of IQR). footprint: peak memory footprint of the driver plus "
                "all members.")


# ---------------------------------------------------------------- figures

def save(fig, name):
    # No creation timestamp, so re-running the analysis reproduces the files byte for byte.
    fig.savefig(FIGURES / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)


def workers_axis(ax, workers, label=True):
    ax.set_xlim(0.4, max(workers) + 0.6)
    ax.set_xticks(workers)
    ax.axvline(P_CORES, color=INK_3, lw=0.7, ls=":", zorder=0.5)
    if label:
        ax.set_xlabel("workers $w$ (dist) = threads $p$ (par\\_ikj)")


def strong_series(s, n, method):
    if method == "dist":
        rows = s[(s.experiment == "D_strong") & (s.method == "dist") & (s.n == n) & (s.repeat == "")
                 & (s.status == "ok")]
        return rows.sort_values("workers"), "workers"
    rows = s[(s.experiment == "D_strong") & (s.method == "par_ikj") & (s.schedule == "static") & (s.n == n)
             & (s.status == "ok")]
    return rows.sort_values("threads"), "threads"


def plain_log_ticks(ax, ticks):
    """Major ticks of a logarithmic axis at the given values, written as plain numbers."""
    lo, hi = ax.get_ylim()
    ax.yaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def fig_strong(s, config):
    strong = config["experiments"]["D_strong"]
    sizes, workers = strong["sizes"], strong["workers"]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.8), gridspec_kw={"wspace": 0.3, "hspace": 0.45})
    ax_t, ax_s, ax_c, ax_m = axes.flat
    for n in sizes:
        style = dict(color=RAMP[n], marker=MARKER_N[n])
        d, xd = strong_series(s, n, "dist")
        p, xp = strong_series(s, n, "par")
        ax_t.plot(d[xd], d.median_ns / 1e6, label=f"$n={n}$", **style, **DIST_STYLE)
        ax_t.plot(p[xp], p.median_ns / 1e6, **style, **PAR_STYLE)
        if n >= 1024:
            for rows, x, st in ((d, xd, DIST_STYLE), (p, xp, PAR_STYLE)):
                y = rows.speedup.to_numpy()
                err = np.vstack([y - rows.speedup_lo.to_numpy(), rows.speedup_hi.to_numpy() - y])
                ax_s.errorbar(rows[x], y, yerr=err, capsize=1.5, elinewidth=0.7, **style, **st)
        ax_c.plot(d.workers, d.dist_cost, **style, **DIST_STYLE)
        ax_m.plot(d.workers, 100 * d.comm_share, **style, **DIST_STYLE)
    ax_t.set_yscale("log")
    ax_t.set_ylabel("time per multiplication (ms)")
    ax_t.set_title("(a) end-to-end time", loc="left", color=INK)
    w = np.linspace(1, max(workers), 50)
    ax_s.plot(w, w, color=INK_3, lw=0.9, ls=":", zorder=1)
    ax_s.annotate("ideal", (max(workers), max(workers)), xytext=(-2, -2), textcoords="offset points",
                  ha="right", va="top", fontsize=7, color=INK_2)
    top = s[(s.experiment == "D_strong") & (s.n >= 1024)].speedup.max()
    ax_s.set_ylim(0, max(2.0, top * 1.25))
    ax_s.axhline(1, color=INK_3, lw=0.7, ls="--", zorder=1)
    ax_s.set_ylabel("speedup over ikj")
    ax_s.set_title("(b) speedup, $n \\geq 1024$", loc="left", color=INK)
    ax_c.axhline(1, color=INK_3, lw=0.7, ls="--", zorder=1)
    ax_c.set_yscale("log")
    plain_log_ticks(ax_c, [1, 1.5, 2, 3, 4, 6, 8])
    ax_c.set_ylabel("$T_\\mathrm{dist} / T_\\mathrm{par}$ (same bands)")
    ax_c.set_title("(c) cost of distribution", loc="left", color=INK)
    ax_m.set_ylim(0, 100)
    ax_m.set_ylabel("communication share (%)")
    ax_m.set_title("(d) serialize + scatter + broadcast + gather", loc="left", color=INK)
    for ax in axes.flat:
        workers_axis(ax, workers)
    handles, labels = ax_t.get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=INK_2, marker="o", **DIST_STYLE),
                plt.Line2D([], [], color=INK_2, marker="o", **PAR_STYLE)]
    labels += ["dist ($w$ members)", "par\\_ikj static ($p$ threads)"]
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=7)
    save(fig, "fig_strong")


def stacked_phases(ax, rows, x, width=0.7, labels=True):
    bottom = np.zeros(len(rows))
    for key, label in PHASES:
        values = rows[f"{key}_med"].to_numpy() / 1e6
        if key == "gather":
            values = values + rows.assemble_med.to_numpy() / 1e6
        ax.bar(x, values, width, bottom=bottom, color=PHASE_COLOR[key], label=label if labels else None,
               edgecolor="white", linewidth=0.6, zorder=2)
        bottom += values
    return bottom


def fig_phases(s, config):
    workers = config["experiments"]["D_strong"]["workers"]
    shown = [n for n in (512, 1024, 4096) if n in config["experiments"]["D_strong"]["sizes"]]
    fig, axes = plt.subplots(1, len(shown), figsize=(6.8, 2.7), gridspec_kw={"wspace": 0.32})
    for ax, n in zip(np.atleast_1d(axes), shown):
        d, _ = strong_series(s, n, "dist")
        p, _ = strong_series(s, n, "par")
        x = np.arange(len(d))
        stacked_phases(ax, d, x, labels=n == shown[0])
        par = p.set_index("threads").reindex(d.workers).median_ns.to_numpy() / 1e6
        ax.plot(x, par, color=INK, marker="_", ms=11, mew=1.6, ls="none", zorder=3,
                label="par\\_ikj, $p=w$" if n == shown[0] else None)
        ax.set_xticks(x)
        ax.set_xticklabels([str(w) for w in d.workers])
        ax.set_xlabel("workers $w$")
        ax.set_title(f"$n={n}$", loc="left", color=INK)
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("ms per multiplication")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.07), fontsize=7)
    save(fig, "fig_phases")


def fig_compute(s, config, members):
    workers = config["experiments"]["D_strong"]["workers"]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.6), gridspec_kw={"wspace": 0.4})
    ax_m, ax_c, ax_i = axes
    n = 4096
    for w in workers:
        d = dist_row(s, n, w, experiment="D_strong")
        if not ok(d) or members.empty:
            continue
        m = members[members.run_id == d.run_id]
        mean = m.kernel_med_ns.mean()
        on_p = m.p_share >= 0.5
        ax_m.scatter(np.full(on_p.sum(), w), m.kernel_med_ns[on_p] / mean, s=16, color=SLOTS[0], zorder=3,
                     label="mostly on P cores" if w == workers[-1] else None)
        ax_m.scatter(np.full((~on_p).sum(), w), m.kernel_med_ns[~on_p] / mean, s=16, facecolor="white",
                     edgecolor=SLOTS[1], zorder=3, label="mostly on E cores" if w == workers[-1] else None)
    ax_m.axhline(1, color=INK_3, lw=0.7, ls="--", zorder=1)
    ax_m.set_ylabel("member kernel time / mean")
    ax_m.set_title(f"(a) members, $n={n}$", loc="left", color=INK)
    ax_m.legend(loc="upper left", fontsize=6.5)
    for n, color, marker in ((1024, SLOTS[0], "D"), (4096, SLOTS[1], "v")):
        d, _ = strong_series(s, n, "dist")
        p, _ = strong_series(s, n, "par")
        ax_c.plot(d.workers, d.cycles_per_madd, color=color, marker=marker, label=f"dist, $n={n}$", **DIST_STYLE)
        ax_c.plot(p.threads, p.cycles_per_madd, color=color, marker=marker, label=f"par, $n={n}$", **PAR_STYLE)
        ax_i.plot(d.workers, d.imbalance, color=color, marker=marker, label=f"$n={n}$")
    ax_c.set_ylim(bottom=0)
    ax_c.set_ylabel("cycles per multiply-add")
    ax_c.set_title("(b) kernel cycles", loc="left", color=INK)
    ax_c.legend(loc="upper left", fontsize=6.5)
    ax_i.axhline(1, color=INK_3, lw=0.7, ls="--", zorder=1)
    ax_i.set_ylabel("slowest / mean member kernel")
    ax_i.set_title("(c) imbalance", loc="left", color=INK)
    ax_i.legend(loc="upper left", fontsize=6.5)
    for ax in axes:
        workers_axis(ax, workers, label=False)
        ax.set_xlabel("$w$ (or $p$)")
    save(fig, "fig_compute")


def fig_transfer(s, config):
    workers = config["experiments"]["D_strong"]["workers"]
    fig, (ax_b, ax_r) = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw={"wspace": 0.3})
    n = 4096
    d, _ = strong_series(s, n, "dist")
    x = np.arange(len(d))
    bottom = np.zeros(len(d))
    for column, label, color in (("a_payload_bytes_op", "A (scatter)", PHASE_COLOR["scatter"]),
                                 ("b_payload_bytes_op", "B (broadcast)", PHASE_COLOR["broadcast"]),
                                 ("c_payload_bytes_op", "C (gather)", PHASE_COLOR["gather"])):
        values = d[column].to_numpy() / MIB
        ax_b.bar(x, values, 0.7, bottom=bottom, color=color, label=label, edgecolor="white", linewidth=0.6,
                 zorder=2)
        bottom += values
    ax_b.plot(x, d.socket_total_op / MIB, color=INK, marker="_", ms=11, mew=1.6, ls="none", zorder=3,
              label="client sockets")
    ax_b.plot(x, d.model_bytes / MIB, color=INK_2, lw=0.9, ls="--", zorder=3, label="$(w+2)\\,8n^2$")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([str(w) for w in d.workers])
    ax_b.set_xlabel("workers $w$")
    ax_b.set_ylabel("MiB per multiplication")
    ax_b.set_title(f"(a) data moved, $n={n}$", loc="left", color=INK)
    ax_b.grid(axis="x", visible=False)
    ax_b.legend(loc="upper left", fontsize=6.5)
    for (column, label, color, marker) in (("rate_serialize", "serialize", PHASE_COLOR["serialize"], "o"),
                                           ("rate_scatter", "scatter", PHASE_COLOR["scatter"], "s"),
                                           ("rate_broadcast", "broadcast", PHASE_COLOR["broadcast"], "D"),
                                           ("rate_gather", "gather", PHASE_COLOR["gather"], "^")):
        for nn, st in ((4096, DIST_STYLE), (1024, PAR_STYLE)):
            rows, _ = strong_series(s, nn, "dist")
            ax_r.plot(rows.workers, rows[column], color=color, marker=marker,
                      label=label if nn == 4096 else None, **st)
    workers_axis(ax_r, workers, label=False)
    ax_r.set_xlabel("workers $w$")
    ax_r.set_ylim(bottom=0)
    ax_r.set_ylabel("payload / phase time (GB/s)")
    ax_r.set_title("(b) effective rates", loc="left", color=INK)
    handles, labels = ax_r.get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=INK_2, **DIST_STYLE), plt.Line2D([], [], color=INK_2, **PAR_STYLE)]
    labels += ["$n=4096$", "$n=1024$"]
    ax_r.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.62), fontsize=6.5, ncol=2)
    save(fig, "fig_transfer")


def fig_setup_memory(s, config, members):
    strong = config["experiments"]["D_strong"]
    sizes, workers = strong["sizes"], strong["workers"]
    fig, (ax_s, ax_m, ax_t) = plt.subplots(1, 3, figsize=(6.8, 2.6), gridspec_kw={"wspace": 0.42})
    d = s[(s.method == "dist") & (s.threads == 1) & (s.status == "ok")]
    g = d.groupby("workers")
    cluster, client = g.setup_cluster_s.median(), g.setup_client_s.median()
    x = np.arange(len(cluster))
    ax_s.bar(x, cluster, 0.7, color=SLOTS[0], label="cluster formation", zorder=2, edgecolor="white", linewidth=0.6)
    ax_s.bar(x, client, 0.7, bottom=cluster, color=SLOTS[1], label="client", zorder=2, edgecolor="white",
             linewidth=0.6)
    ax_s.scatter(np.searchsorted(cluster.index, d.workers) + 0.0, d.setup_s, s=6, color=INK_2, zorder=3,
                 label="each run")
    ax_s.set_xticks(x)
    ax_s.set_xticklabels([str(w) for w in cluster.index])
    ax_s.set_xlabel("workers $w$")
    ax_s.set_ylabel("setup (s)")
    ax_s.set_title("(a) setup", loc="left", color=INK)
    ax_s.grid(axis="x", visible=False)
    ax_s.set_ylim(0, 1.55 * max(d.setup_s.max(), (cluster + client).max()))
    ax_s.legend(loc="upper left", fontsize=6.5)
    for n in sizes:
        rows, _ = strong_series(s, n, "dist")
        if rows.empty or members.empty:
            continue
        foot = members[members.run_id.isin(rows.run_id)].groupby("workers").peak_footprint_bytes.median()
        ax_m.plot(foot.index, foot / MIB, color=RAMP[n], marker=MARKER_N[n], label=f"$n={n}$")
        ax_t.plot(rows.workers, rows.total_footprint_bytes / GIB, color=RAMP[n], marker=MARKER_N[n])
    ax_m.set_yscale("log")
    plain_log_ticks(ax_m, [125, 250, 500, 1000, 2000, 4000])
    ax_m.set_ylabel("peak footprint per member (MiB)")
    ax_m.set_title("(b) per member", loc="left", color=INK)
    handles, labels = ax_m.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.62, 1.08), fontsize=7)
    budget = config["budgets"]["max_total_footprint_bytes"] / GIB
    ax_t.axhline(budget, color=INK_3, lw=0.9, ls="--", zorder=1)
    ax_t.annotate("budget", (0.5, budget), xytext=(2, 2), textcoords="offset points", fontsize=7, color=INK_2)
    ax_t.set_ylabel("driver + members (GiB)")
    ax_t.set_title("(c) all processes", loc="left", color=INK)
    ax_t.set_ylim(bottom=0)
    for ax in (ax_m, ax_t):
        workers_axis(ax, workers, label=False)
        ax.set_xlabel("workers $w$")
    save(fig, "fig_setup_memory")


def fig_grain_weak(s, config):
    ex = config["experiments"]
    grain, weak = ex["D_grain"], ex["D_weak"]
    fig, (ax_g, ax_w) = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw={"wspace": 0.3, "width_ratios": [1.3, 1]})
    labels, xs, rows = [], [], []
    pos = 0
    for n in grain["sizes"]:
        for w, k in grain["layouts"]:
            d = dist_row(s, n, w, k)
            if ok(d):
                rows.append(d)
                xs.append(pos)
                labels.append(f"{w}$\\times${k}")
            pos += 1
        pos += 0.6
    if rows:
        frame = pd.DataFrame(rows)
        frame["norm"] = 1.0
        for i, n in enumerate(grain["sizes"]):
            frame.loc[frame.n == n, "norm"] = frame[frame.n == n].median_ns.min()
        scaled = frame.copy()
        for key, _ in PHASES:
            scaled[f"{key}_med"] = frame[f"{key}_med"] / frame.norm * 1e6
        scaled["assemble_med"] = frame.assemble_med / frame.norm * 1e6
        stacked_phases(ax_g, scaled, np.array(xs))
        ax_g.set_xticks(xs)
        ax_g.set_xticklabels(labels, fontsize=7)
        for n in grain["sizes"]:
            sel = [x for x, r in zip(xs, rows) if r.n == n]
            if sel:
                ax_g.annotate(f"$n={n}$", (np.mean(sel), 0), xytext=(0, -24), textcoords="offset points",
                              ha="center", fontsize=8, color=INK, annotation_clip=False)
        ax_g.set_ylabel("time / fastest layout")
        ax_g.set_title("(a) 10 cores as $w$ members $\\times$ $k$ threads", loc="left", color=INK)
        ax_g.grid(axis="x", visible=False)
        ax_g.legend(loc="upper left", fontsize=6.5, ncol=2)
    d1, p1 = dist_row(s, 2048, 1), par_row(s, 2048, 1)
    pts = []
    for w, n in weak["points"]:
        d, p = dist_row(s, n, w), par_row(s, n, w)
        pts.append((w, d1.median_ns / d.median_ns if ok(d) and ok(d1) else np.nan,
                    p1.median_ns / p.median_ns if ok(p) and ok(p1) else np.nan))
    pts = np.array(pts, dtype=float)
    if len(pts):
        ax_w.plot(pts[:, 0], pts[:, 1], color=SLOTS[0], marker="o", label="dist", **DIST_STYLE)
        ax_w.plot(pts[:, 0], pts[:, 2], color=SLOTS[1], marker="s", label="par\\_ikj static", **PAR_STYLE)
    ax_w.axhline(1, color=INK_3, lw=0.7, ls="--", zorder=1)
    ax_w.set_xscale("log", base=2)
    ax_w.minorticks_off()
    ax_w.set_xticks([w for w, _ in weak["points"]])
    ax_w.set_xticklabels([f"{w}\n$n={n}$" for w, n in weak["points"]], fontsize=7)
    ax_w.set_ylim(0, 1.15)
    ax_w.set_xlabel("workers $w$ (threads $p$)")
    ax_w.set_ylabel("$E_w = T(1, 2048) / T(w, n_w)$")
    ax_w.set_title("(b) weak scaling, $n_w^3/w = 2048^3$", loc="left", color=INK)
    ax_w.legend(loc="upper right", fontsize=7)
    save(fig, "fig_grain_weak")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", help="session directory name under results/raw (default: latest)")
    args = parser.parse_args()

    session = RAW / args.session if args.session else latest_session()
    config = json.loads((session / "config.json").read_text())
    samples = pd.read_csv(session / "samples.csv", dtype={"checksum": str})
    runs = pd.read_csv(session / "runs.csv")
    members_raw = pd.read_csv(session / "members.csv")
    msamples = pd.read_csv(session / "member_samples.csv")
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    summary = summarise(samples, runs, msamples)
    members = member_table(msamples, members_raw, runs)
    checks = checksum_agreement(samples)
    summary.to_csv(TABLES / "summary.csv", index=False, lineterminator="\n")
    members.to_csv(TABLES / "members.csv", index=False, lineterminator="\n")
    checks.to_csv(TABLES / "checksums.csv", index=False, lineterminator="\n")
    tables(summary, config, members)
    fig_strong(summary, config)
    fig_phases(summary, config)
    fig_compute(summary, config, members)
    fig_transfer(summary, config)
    fig_setup_memory(summary, config, members)
    fig_grain_weak(summary, config)

    good = summary[summary.status == "ok"]
    dist = good[good.method == "dist"]
    print(f"session {session.name}: {len(samples)} samples, {len(runs)} runs; statuses "
          f"{runs.status.value_counts().to_dict()}")
    print("checksums identical across configurations for every n:", bool(checks.identical.all()))
    print(f"relative IQR: max {good.iqr_rel.max():.4f} ({good.loc[good.iqr_rel.idxmax(), 'run_id']}), "
          f"median {good.iqr_rel.median():.4f}")
    print(f"phase sum / end to end: {dist.phase_check.min():.4f}-{dist.phase_check.max():.4f}; "
          f"all tasks local: {bool(dist.m_local.all())}")
    print(f"payload / model: {dist.payload_over_model.min():.5f}-{dist.payload_over_model.max():.5f}; "
          f"socket overhead: {100 * dist.socket_overhead.min():.4f}-{100 * dist.socket_overhead.max():.4f} %")
    background = (good.system_busy_cpu_s - good.own_cpu_s) / good.window_s
    print(f"background cores (other processes): median {background.median():.2f}, max {background.max():.2f} "
          f"({good.loc[background.idxmax(), 'run_id']})")
    print(f"GC collections in the driver during measured samples: {int(good.gc_collections.sum())}")
    for n in sorted(summary[summary.experiment == "D_strong"].n.unique()):
        first, last = ikj_row(summary, n, "first"), ikj_row(summary, n, "last")
        df, dl = dist_row(summary, n, 10, repeat="first"), dist_row(summary, n, 10, repeat="last")
        drift_d = dl.median_ns / df.median_ns if ok(df) and ok(dl) else float("nan")
        best = dist[(dist.n == n) & (dist.experiment == "D_strong")].sort_values("median_ns")
        print(f"n={n:5d}: ikj {first.median_ns / 1e6:10.4f} ms, drift ikj {last.median_ns / first.median_ns:.3f}, "
              f"dist w=10 {drift_d:.3f}; best dist w={int(best.iloc[0].workers) if len(best) else -1} "
              f"S={best.iloc[0].speedup if len(best) else float('nan'):.2f}")
    print(f"wrote {TABLES.relative_to(ROOT)}/ and {FIGURES.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
