"""Analysis of one benchmark session: summary tables and figures for the report.

Adapted from Assignment 2's analysis/analyze.py (tag assignment2-v1.0): the same
statistics and table/figure helpers. New here: speedup, efficiency and the
overhead of the parallel method against the sequential baseline of the same
size block, the per-sample process counters (busy cores, share of CPU time on
efficiency cores, clock rate, cycles and instructions per multiply-add), the
decomposition of the speedup, the drift control, the SuperWord comparison and
the disassembly counts.

Reads results/raw/<session>/{samples.csv, runs.csv, config.json} (default: the
latest session that is not a smoke or pilot run) and results/simd/summary.csv,
and writes
    analysis/tables/summary.csv     one row per configuration, full precision
    analysis/tables/checksums.csv   checksum agreement across the configurations of each n
    analysis/tables/*.tex           booktabs tables included by the report
    analysis/figures/*.pdf, *.png   figures (vector PDF for the report)

Statistics: only measured samples (kind = measure) are summarised; warm-ups
are excluded. The central value is the median of the time per multiplication
(elapsed_ns / batch) and the variability is the interquartile range (pandas
quantiles with linear interpolation), also relative to the median. The speedup
of a configuration is S = median T_seq / median T_p, with T_seq from the first
ikj run of its size block, and its interval is [Q1_seq / Q3_p, Q3_seq / Q1_p];
the efficiency is E = S / p. Counter-based quantities are ratios of sums over the
measured samples, so that T = C / (f U) holds exactly for the mean time per
multiplication (C cycles per multiplication summed over all cores, f the mean
clock rate of the busy cores, U the mean number of busy cores), and the
speedup of the means splits into (C_seq / C_p) (f_p / f_seq) (U_p / U_seq).

Usage (from assignment3/):
    python3 analysis/analyze.py [--session 20261030-101500]
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
SIMD = ROOT / "results" / "simd" / "summary.csv"
TABLES = ROOT / "analysis" / "tables"
FIGURES = ROOT / "analysis" / "figures"

SMALL, LARGE = [64, 128, 256], [512, 1024, 2048, 4096]
# Ordinal blue ramps for n (validated with the palette validator, --ordinal): one per size group,
# because seven steps of one hue are too close to tell apart.
RAMP = {64: "#6da7ec", 128: "#256abf", 256: "#0d366b",
        512: "#86b6ef", 1024: "#3987e5", 2048: "#1c5cab", 4096: "#0d366b"}
MARKER_N = {64: "o", 128: "s", 256: "D", 512: "o", 1024: "s", 2048: "D", 4096: "^"}
# Categorical slots in their fixed order (blue, orange, aqua, yellow), markers as secondary encoding.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SCHEDULE_STYLE = {"dynamic": dict(color=SLOTS[0], ls="-", marker="o"),
                  "static": dict(color=SLOTS[1], ls="--", marker="s", mfc="white")}
INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
MIB = 2 ** 20
P_CORES, LOGICAL = 4, 10

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


def summarise(samples, runs):
    measured = samples[samples.kind == "measure"]
    g = measured.groupby("run_id")
    t = g.ns_per_op
    stats = pd.DataFrame({
        "reps": t.count(), "median_ns": t.median(), "q1_ns": t.quantile(0.25), "q3_ns": t.quantile(0.75),
        "min_ns": t.min(), "max_ns": t.max(), "batch": g.batch.first(), "ops": g.batch.sum(),
        "elapsed_sum": g.elapsed_ns.sum(), "cpu_sum": g.cpu_ns.sum(), "cpu_p_sum": g.cpu_p_ns.sum(),
        "instr_sum": g.instructions.sum(), "cycles_sum": g.cycles.sum(), "instr_p_sum": g.instructions_p.sum(),
        "cycles_p_sum": g.cycles_p.sum(), "csw_sum": g.context_switches.sum(),
    })
    stats["warmup_samples"] = samples[samples.kind == "warmup"].groupby("run_id").size()
    s = runs.join(stats, on="run_id")
    for column in ("schedule", "repeat", "simd"):
        s[column] = s[column].fillna("")
    n = s.n.astype(float)
    n3 = n ** 3
    s["iqr_rel"] = (s.q3_ns - s.q1_ns) / s.median_ns
    s["mean_ns"] = s.elapsed_sum / s.ops
    s["busy_cores"] = s.cpu_sum / s.elapsed_sum
    s["e_cpu_share"] = 1 - s.cpu_p_sum / s.cpu_sum
    s["e_work_share"] = 1 - s.instr_p_sum / s.instr_sum
    s["clock_ghz"] = s.cycles_sum / s.cpu_sum
    s["clock_p_ghz"] = s.cycles_p_sum / s.cpu_p_sum
    e_cpu = s.cpu_sum - s.cpu_p_sum
    s["clock_e_ghz"] = np.where(e_cpu > 0.01 * s.cpu_sum, (s.cycles_sum - s.cycles_p_sum) / e_cpu, np.nan)
    # instructions per CPU-second on each core type, and their ratio (work rate of an E core relative to a P core)
    rate_p = s.instr_p_sum / s.cpu_p_sum
    rate_e = (s.instr_sum - s.instr_p_sum) / e_cpu
    s["e_to_p_rate"] = np.where(e_cpu > 0.01 * s.cpu_sum, rate_e / rate_p, np.nan)
    s["cycles_per_op"] = s.cycles_sum / s.ops
    s["cycles_per_madd"] = s.cycles_sum / (s.ops * n3)
    s["instr_per_madd"] = s.instr_sum / (s.ops * n3)
    s["ipc"] = s.instr_sum / s.cycles_sum
    s["csw_per_s"] = s.csw_sum / (s.elapsed_sum / 1e9)
    s["gflops"] = 2 * n3 / s.median_ns
    # CPU time of other processes = system-wide busy CPU time over the run minus the process's own CPU time.
    # The own time comes from /usr/bin/time; where the harness could not parse it (session 20261104-030623
    # ran under a decimal-comma locale), from the per-QoS CPU times the driver reads at exit, which cover
    # the whole process and matched /usr/bin/time within 0.02 s per process in the pilot and smoke sessions.
    qos = s[[c for c in s.columns if c.startswith("qos_") and c.endswith("_ns")]].sum(axis=1) / 1e9
    timed = s.process_user_s + s.process_sys_s
    s["own_cpu_s"] = timed.where(timed.notna(), qos)
    s["own_cpu_source"] = np.where(timed.notna(), "time", "qos")
    s["background_cores"] = (s.system_busy_cpu_s - s.own_cpu_s) / s.window_s
    s["element_bytes"] = 24 * n * n

    seq = s[(s.experiment == "P_threads") & (s.method == "ikj") & (s.repeat == "first")].set_index("n")
    base = seq.reindex(s.n)
    base.index = s.index
    s["speedup"] = base.median_ns / s.median_ns
    s["speedup_lo"] = base.q1_ns / s.q3_ns
    s["speedup_hi"] = base.q3_ns / s.q1_ns
    s["efficiency"] = s.speedup / s.threads
    s["speedup_mean"] = base.mean_ns / s.mean_ns
    s["factor_cycles"] = base.cycles_per_op / s.cycles_per_op
    s["factor_clock"] = s.clock_ghz / base.clock_ghz
    s["factor_busy"] = s.busy_cores / base.busy_cores
    # SuperWord off: time ratio to the matching SuperWord-on configuration of P_threads (first ikj, shuffled par_ikj)
    on = s[(s.experiment == "P_threads") & (((s.method == "ikj") & (s.repeat == "first"))
                                            | ((s.method == "par_ikj") & (s.repeat == "")))]
    on = on.set_index(["method", "n", "threads"])
    key = pd.MultiIndex.from_frame(s[["method", "n", "threads"]])
    match = on.reindex(key)
    match.index = s.index
    s["simd_gain"] = np.where(s.simd == "off", s.median_ns / match.median_ns, np.nan)
    s["simd_on_median_ns"] = np.where(s.simd == "off", match.median_ns, np.nan)
    s["simd_on_instr_per_madd"] = np.where(s.simd == "off", match.instr_per_madd, np.nan)
    s["simd_on_cycles_per_madd"] = np.where(s.simd == "off", match.cycles_per_madd, np.nan)
    return s


def checksum_agreement(samples):
    """Per n: are the measured checksums of every configuration identical (exact float())?"""
    measured = samples[samples.kind == "measure"].copy()
    measured["value"] = measured.checksum.map(float)  # round-trip exact, unlike pandas' fast parser
    rows = []
    for n, g in measured.groupby("n"):
        rows.append({"n": n, "configurations": g.run_id.nunique(), "samples": len(g),
                     "distinct_values": g.value.nunique(), "identical": g.value.nunique() == 1})
    return pd.DataFrame(rows)


def capacity_ratio(s):
    """Work rate of an efficiency core relative to a performance core, from par_ikj at p = 10, n = 512."""
    row = pick(s, experiment="P_threads", method="par_ikj", n=512, threads=LOGICAL, repeat="")
    return float(row.e_to_p_rate)


def capacity(p, r):
    """Speedup bound if p threads fill the 4 performance cores first, then the 6 efficiency cores at rate r."""
    p = np.asarray(p, float)
    return np.minimum(p, P_CORES) + r * np.clip(p - P_CORES, 0, LOGICAL - P_CORES)


# ---------------------------------------------------------------- formatting

def sig(x, digits=3):
    """Format with about `digits` significant digits, never in scientific notation."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if x == 0:
        return "0"
    decimals = max(0, digits - 1 - math.floor(math.log10(abs(x))))
    return f"{x:,.{decimals}f}".replace(",", "\\,")


def pct(x, decimals=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    value = 100 * x
    return f"{0.0 if abs(value) < 0.5 * 10 ** -decimals else value:.{decimals}f}"  # no "-0" from rounding noise


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


def thread_label(p):
    return f"{p}\\textsuperscript{{\\dag}}" if p > LOGICAL else str(p)


OVERSUBSCRIBED_NOTE = (r"\textsuperscript{\dag} oversubscribed: more worker threads than the "
                       rf"{LOGICAL} logical processors.")


def par(s, experiment, n, p, schedule="dynamic", simd="on", repeat=""):
    return pick(s, experiment=experiment, method="par_ikj", n=n, threads=p, schedule=schedule, simd=simd,
                repeat=repeat)


def seq(s, n, repeat="first", simd="on"):
    experiment = "P_threads" if simd == "on" else "V_simd"
    return pick(s, experiment=experiment, method="ikj", n=n, repeat=repeat if simd == "on" else "", simd=simd)


# ---------------------------------------------------------------- tables

def tables(s, config):
    pt = config["experiments"]["P_threads"]
    sizes, threads = pt["sizes"], pt["threads"]

    rows = [["ikj (seq.)"] + [sig(seq(s, n).median_ns / 1e6) for n in sizes], "midrule"]
    for p in threads:
        rows.append([thread_label(p)] + [sig(par(s, "P_threads", n, p).median_ns / 1e6) for n in sizes])
    shown = s[(s.experiment == "P_threads") & (((s.method == "ikj") & (s.repeat == "first"))
                                              | ((s.method == "par_ikj") & (s.repeat == "")))]
    worst = [shown[shown.n == n].iqr_rel.max() for n in sizes]
    rows += ["midrule", ["largest IQR"] + [f"{100 * w:.1f}\\%" for w in worst]]
    write_table("tab_time.tex", ["$p$"] + [f"$n={n}$" for n in sizes], rows, "r" * (len(sizes) + 1),
                OVERSUBSCRIBED_NOTE)

    rows = []
    for p in threads:
        cells = [thread_label(p)]
        for n in sizes:
            r = par(s, "P_threads", n, p)
            cells.append(f"{sig(r.speedup)} ({r.efficiency:.2f})")
        rows.append(cells)
    write_table("tab_speedup.tex", ["$p$"] + [f"$n={n}$" for n in sizes], rows, "r" * (len(sizes) + 1),
                OVERSUBSCRIBED_NOTE)

    static_sizes = config["experiments"]["P_schedule"]["sizes"]
    rows = []
    for n in sizes:
        first, last, one = seq(s, n), seq(s, n, "last"), par(s, "P_threads", n, 1)
        best = s[(s.experiment == "P_threads") & (s.method == "par_ikj") & (s.n == n)
                 & (s.repeat == "")].sort_values("median_ns").iloc[0]
        p_first, p_last = par(s, "P_threads", n, LOGICAL, repeat="first"), par(s, "P_threads", n, LOGICAL, repeat="last")
        stat = par(s, "P_schedule", n, 1, "static") if n in static_sizes else None
        rows.append([str(n), sig(first.median_ns / 1e6), sig(one.median_ns / 1e6),
                     f"{100 * (one.median_ns / first.median_ns - 1):+.1f}",
                     f"{100 * (stat.median_ns / first.median_ns - 1):+.1f}" if stat is not None else "--",
                     thread_label(int(best.threads)), sig(best.speedup), f"{best.efficiency:.2f}",
                     f"{last.median_ns / first.median_ns:.3f}", f"{p_last.median_ns / p_first.median_ns:.3f}"])
    write_table("tab_overhead.tex", ["$n$", "ikj (ms)", "par, $p=1$ (ms)", "overhead (\\%)", "static (\\%)",
                                     "best $p$", "$S$", "$E$", "drift, ikj", "drift, $p=10$"], rows, "rrrrrrrrrr",
                OVERSUBSCRIBED_NOTE)

    rows = []
    for p in threads:
        cells = [thread_label(p)]
        for n in static_sizes:
            d, st = par(s, "P_threads", n, p), par(s, "P_schedule", n, p, "static")
            cells += [sig(d.speedup), sig(st.speedup), f"{st.busy_cores:.1f}"]
        rows.append(cells)
    group = " & ".join(rf"\multicolumn{{3}}{{c}}{{$n={n}$}}" for n in static_sizes)
    rules = " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(static_sizes)))
    write_table("tab_schedule.tex", [rf" & {group} \\ {rules} $p$"] + ["dyn.", "static", "busy"] * len(static_sizes),
                rows, "r" * (1 + 3 * len(static_sizes)), OVERSUBSCRIBED_NOTE)

    rows = []
    for n in LARGE:
        for p in (1, 2, 4, 6, 8, 10, 16):
            r = par(s, "P_threads", n, p)
            rows.append([str(n) if p == 1 else "", thread_label(p), sig(r.speedup_mean), f"{r.busy_cores:.2f}",
                         pct(r.e_cpu_share, 0), sig(r.clock_ghz), sig(r.cycles_per_madd), sig(r.instr_per_madd),
                         sig(r.factor_cycles), sig(r.factor_clock), sig(r.factor_busy)])
        base = seq(s, n)
        rows.append(["", "seq.", "1", f"{base.busy_cores:.2f}", pct(base.e_cpu_share, 0), sig(base.clock_ghz),
                     sig(base.cycles_per_madd), sig(base.instr_per_madd), "1", "1", "1"])
        if n != LARGE[-1]:
            rows.append("midrule")
    write_table("tab_decomp.tex", ["$n$", "$p$", r"$\bar S$", "busy $U$", "E-core (\\%)", "clock (GHz)",
                                   "cyc./m.-add", "instr./m.-add", r"$C_s/C_p$", r"$f_p/f_s$", r"$U_p/U_s$"],
                rows, "rrrrrrrrrrr", OVERSUBSCRIBED_NOTE)

    rows = []
    for n in sizes:
        cells = [str(n), sig(seq(s, n).max_rss_bytes / MIB)]
        cells += [sig(par(s, "P_threads", n, p).max_rss_bytes / MIB) for p in (1, 4, 10, 20)]
        cells += [sig(par(s, "P_threads", n, p).allocated_bytes_total / MIB) for p in (1, 20)]
        cells.append(sig(24 * n * n / MIB))
        rows.append(cells)
    group = (r" & & \multicolumn{4}{c}{par\_ikj, peak RSS} & \multicolumn{2}{c}{allocated} & \\ "
             r"\cmidrule(lr){3-6} \cmidrule(lr){7-8} $n$")
    write_table("tab_memory.tex", [group, "ikj", "$p=1$", "$p=4$", "$p=10$", "$p=20$", "$p=1$", "$p=20$",
                                   "$24n^2$"], rows, "rrrrrrrrr")

    rows = []
    off = s[s.simd == "off"].sort_values(["n", "threads"])
    for r in off.itertuples():
        label = "ikj (seq.)" if r.method == "ikj" else f"par\\_ikj, $p={r.threads}$"
        rows.append([str(r.n), label, sig(r.simd_on_median_ns / 1e6), sig(r.median_ns / 1e6), sig(r.simd_gain),
                     sig(r.simd_on_instr_per_madd), sig(r.instr_per_madd), sig(r.simd_on_cycles_per_madd),
                     sig(r.cycles_per_madd)])
    group = (r" & & \multicolumn{3}{c}{time per multiplication} & \multicolumn{2}{c}{instr./mult.-add}"
             r" & \multicolumn{2}{c}{cycles/mult.-add} \\ \cmidrule(lr){3-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} $n$")
    write_table("tab_simd.tex", [group, "configuration", "on (ms)", "off (ms)", "off/on", "on", "off", "on", "off"],
                rows, "rlrrrrrrr")

    if SIMD.exists():
        code = pd.read_csv(SIMD)
        rows = []
        for r in code.itertuples():
            method = r.method.replace("org.ulpgc.bd.matmul.", "").replace("$", r"\$").replace("_", r"\_")
            rows.append([r.run.replace("_", r"\_"), r"\texttt{" + method + "}", r.superword, str(r.vector_fmul),
                         str(r.vector_fadd), str(r.scalar_fmul), str(r.scalar_fadd), str(r.fused),
                         f"{r.q_loads}/{r.q_stores}"])
        write_table("tab_simd_code.tex", ["run", "compiled method", "SuperWord", "fmul.2d", "fadd.2d", "fmul",
                                          "fadd", "fused", "q ld/st"], rows, "llcrrrrrr")


# ---------------------------------------------------------------- figures

def save(fig, name):
    # No creation timestamp, so re-running the analysis reproduces the files byte for byte.
    fig.savefig(FIGURES / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)


def thread_axis(ax, threads, label=True, ticks=(1, 2, 4, 6, 8, 10, 12, 16, 20)):
    ax.set_xlim(0.3, max(threads) + 0.7)
    ax.set_xticks(list(ticks))
    ax.axvspan(LOGICAL + 0.5, max(threads) + 0.7, color="#f0efec", zorder=0, lw=0)
    for x in (P_CORES, LOGICAL):
        ax.axvline(x, color=INK_3, lw=0.7, ls=":", zorder=0.5)
    if label:
        ax.set_xlabel("worker threads $p$")


def mark_regions(ax, y):
    """Label the dotted line at p = 4 and the shaded oversubscribed region (p > 10) at height y."""
    ax.annotate("4 P cores", (P_CORES, y), xytext=(3, 0), textcoords="offset points", fontsize=7, color=INK_2,
                va="top")
    ax.annotate("oversubscribed", (LOGICAL + 0.5, y), xytext=(3, 0), textcoords="offset points", fontsize=7,
                color=INK_2, va="top")


def series_rows(s, experiment, n, schedule="dynamic"):
    rows = s[(s.experiment == experiment) & (s.method == "par_ikj") & (s.n == n) & (s.schedule == schedule)
             & (s.simd == "on") & (s.repeat == "") & (s.status == "ok")]
    return rows.sort_values("threads")


def errorbar_speedup(ax, rows, **style):
    y = rows.speedup.to_numpy()
    err = np.vstack([y - rows.speedup_lo.to_numpy(), rows.speedup_hi.to_numpy() - y])
    ax.errorbar(rows.threads, y, yerr=err, capsize=1.5, elinewidth=0.7, **style)


def fig_speedup(s, config, r):
    threads = config["experiments"]["P_threads"]["threads"]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.6), gridspec_kw={"wspace": 0.28, "hspace": 0.42})
    p = np.linspace(1, max(threads), 200)
    for col, group in enumerate((LARGE, SMALL)):
        ax_s, ax_e = axes[0, col], axes[1, col]
        for n in group:
            rows = series_rows(s, "P_threads", n)
            style = dict(color=RAMP[n], marker=MARKER_N[n], label=f"$n={n}$")
            errorbar_speedup(ax_s, rows, **style)
            ax_e.plot(rows.threads, rows.efficiency, **style)
        thread_axis(ax_s, threads)
        thread_axis(ax_e, threads)
        if col == 0:
            # reference lines labelled directly at their right end, so the legend only lists the sizes
            ax_s.plot(p, np.minimum(p, LOGICAL), color=INK_3, lw=0.9, ls="--", zorder=1)
            ax_s.plot(p, capacity(p, r), color=INK_3, lw=0.9, ls="-.", zorder=1)
            ax_s.annotate("ideal, $S=p$", (max(threads), LOGICAL), xytext=(-2, -3), textcoords="offset points",
                          ha="right", va="top", fontsize=7, color=INK_2)
            ax_s.annotate(f"4P+6E capacity, $r={r:.2f}$", (max(threads), capacity(LOGICAL, r)), xytext=(-2, 3),
                          textcoords="offset points", ha="right", va="bottom", fontsize=7, color=INK_2)
            ax_s.set_ylim(0, LOGICAL + 0.8)
            ax_s.set_title("(a) speedup, $n \\geq 512$", loc="left", color=INK)
            ax_e.set_title("(c) efficiency, $n \\geq 512$", loc="left", color=INK)
        else:
            ax_s.axhline(1, color=INK_3, lw=0.9, ls="--", zorder=1, label="as fast as ikj")
            ax_s.set_ylim(0, max(2.0, ax_s.get_ylim()[1]) * 1.12)
            ax_s.set_title("(b) speedup, $n \\leq 256$", loc="left", color=INK)
            ax_e.set_title("(d) efficiency, $n \\leq 256$", loc="left", color=INK)
        mark_regions(ax_s, ax_s.get_ylim()[1])
        mark_regions(ax_e, 1.15)
        ax_s.set_ylabel("speedup $S = T_{seq}/T_p$")
        ax_e.set_ylabel("efficiency $E = S/p$")
        ax_e.set_ylim(0, 1.15)
        ax_e.axhline(1, color=INK_3, lw=0.7, ls="--", zorder=1)
        if col == 0:
            ax_s.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=7)
        else:  # the empty band between n = 64 (S near 1) and n = 128 (S near 2.5)
            ax_s.legend(loc="center right", bbox_to_anchor=(1.0, 0.45), fontsize=7)
    save(fig, "fig_speedup")


def fig_cores(s, config):
    threads = config["experiments"]["P_threads"]["threads"]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.4), gridspec_kw={"wspace": 0.3, "hspace": 0.42})
    panels = [("busy_cores", "busy cores $U$ = CPU time / elapsed", "(a) busy cores", 1),
              ("e_cpu_share", "share of CPU time on E cores (%)", "(b) efficiency-core share", 100),
              ("clock_ghz", "cycles / CPU time (GHz)", "(c) mean clock of the busy cores", 1),
              ("cycles_per_madd", "cycles per multiply-add (all cores)", "(d) cycles per multiply-add", 1)]
    for ax, (column, ylabel, title, scale) in zip(axes.flat, panels):
        for n in LARGE:
            rows = series_rows(s, "P_threads", n)
            ax.plot(rows.threads, rows[column] * scale, color=RAMP[n], marker=MARKER_N[n], label=f"$n={n}$")
            base = seq(s, n)
            ax.plot([0.55], [base[column] * scale], color=RAMP[n], marker=MARKER_N[n], mfc="white", ls="none")
        thread_axis(ax, threads)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", color=INK)
    p = np.linspace(1, max(threads), 200)
    axes[0, 0].plot(p, np.minimum(p, LOGICAL), color=INK_3, lw=0.9, ls="--", zorder=1, label="$U=p$ (at most 10)")
    axes[0, 0].set_ylim(0, LOGICAL + 0.8)
    busy = np.minimum(p, LOGICAL)
    (ref,) = axes[0, 1].plot(p, 100 * np.clip(busy - P_CORES, 0, None) / busy, color=INK_3, lw=0.9, ls="--",
                             zorder=1)
    axes[0, 1].legend([ref], ["4 busy threads on P cores,\nthe rest on E cores"], loc="lower right", fontsize=7)
    axes[0, 1].set_ylim(0, 100)
    axes[1, 0].ticklabel_format(axis="y", useOffset=False)
    axes[1, 0].set_ylim(0, 4.6)
    axes[1, 1].set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK_2, marker="o", mfc="white", ls="none"))
    labels.append("ikj (sequential)")
    axes[0, 0].legend(handles, labels, loc="lower right", fontsize=7)
    save(fig, "fig_cores")


def fig_schedule(s, config):
    static_sizes = config["experiments"]["P_schedule"]["sizes"]
    threads = config["experiments"]["P_schedule"]["threads"]
    fig, axes = plt.subplots(1, len(static_sizes), figsize=(6.8, 2.6), gridspec_kw={"wspace": 0.3})
    p = np.linspace(1, max(threads), 200)
    for ax, n in zip(axes, static_sizes):
        for schedule, experiment in (("dynamic", "P_threads"), ("static", "P_schedule")):
            rows = series_rows(s, experiment, n, schedule)
            errorbar_speedup(ax, rows, label=schedule, **SCHEDULE_STYLE[schedule])
        ax.plot(p, np.minimum(p, LOGICAL), color=INK_3, lw=0.9, ls="--", zorder=1)
        thread_axis(ax, threads, ticks=(1, 4, 10, 16, 20))
        top = max(s[(s.n == n) & (s.method == "par_ikj") & (s.simd == "on")].speedup.max() * 1.15, 2)
        ax.set_ylim(0, top)
        ax.set_title(f"$n={n}$", loc="left", color=INK)
    axes[0].set_ylabel("speedup $S$")
    axes[0].legend(loc="lower right", fontsize=7)
    save(fig, "fig_schedule")


def fig_size(s, config):
    sizes = config["experiments"]["P_threads"]["sizes"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for color, (p, marker) in zip(SLOTS, ((1, "o"), (4, "s"), (10, "D"), (20, "^"))):
        rows = s[(s.experiment == "P_threads") & (s.method == "par_ikj") & (s.threads == p)
                 & (s.repeat == "")].sort_values("n")
        y = rows.speedup.to_numpy()
        err = np.vstack([y - rows.speedup_lo.to_numpy(), rows.speedup_hi.to_numpy() - y])
        ax.errorbar(rows.n, y, yerr=err, capsize=1.5, elinewidth=0.7, color=color, marker=marker,
                    label=f"$p={p}$" + (" (oversubscribed)" if p > LOGICAL else ""))
    ax.axhline(1, color=INK_3, lw=0.9, ls="--", zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.minorticks_off()
    ax.set_xlabel("matrix size $n$")
    ax.set_ylabel("speedup $S$ over ikj")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=7)
    save(fig, "fig_size")


def fig_simd(s):
    off = s[s.simd == "off"]
    if off.empty:
        return
    sizes = sorted(off.n.unique())
    confs = [("ikj", 1, "ikj (seq.)"), ("par_ikj", 4, "par_ikj, $p=4$"), ("par_ikj", 10, "par_ikj, $p=10$")]
    markers = ["o", "s", "D"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.8), gridspec_kw={"wspace": 0.32})
    width = 0.26
    x = np.arange(len(sizes))
    for i, ((method, p, label), color) in enumerate(zip(confs, SLOTS)):
        gains, on_i, off_i = [], [], []
        for n in sizes:
            r = pick(off, method=method, n=n, threads=p)
            gains.append(r.simd_gain)
            on_i.append(r.simd_on_instr_per_madd)
            off_i.append(r.instr_per_madd)
        xs = x + (i - 1) * width
        ax1.bar(xs, gains, width * 0.88, color=color, label=label, zorder=2)
        ax2.plot(sizes, off_i, color=color, marker=markers[i], ls="--", mfc="white", ms=5.5 - i)
        ax2.plot(sizes, on_i, color=color, marker=markers[i], ms=5.5 - i)
    ax1.axhline(1, color=INK_3, lw=0.9, zorder=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"$n={n}$" for n in sizes])
    ax1.set_ylabel("time without / with SuperWord")
    ax1.set_title("(a) gain from vectorisation", loc="left", color=INK)
    ax1.grid(axis="x", visible=False)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), fontsize=7)
    ax2.legend([plt.Line2D([], [], color=INK_2, marker="o"),
                plt.Line2D([], [], color=INK_2, marker="o", ls="--", mfc="white")],
               ["SuperWord on", "SuperWord off"], loc="lower right", fontsize=7)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(sizes)
    ax2.set_xticklabels([str(n) for n in sizes])
    ax2.minorticks_off()
    ax2.set_xlabel("matrix size $n$")
    ax2.set_ylabel("instructions per multiply-add")
    ax2.set_ylim(bottom=0)
    ax2.set_title("(b) instructions per multiply-add", loc="left", color=INK)
    save(fig, "fig_simd")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", help="session directory name under results/raw (default: latest)")
    args = parser.parse_args()

    session = RAW / args.session if args.session else latest_session()
    config = json.loads((session / "config.json").read_text())
    samples = pd.read_csv(session / "samples.csv", dtype={"checksum": str})
    runs = pd.read_csv(session / "runs.csv")
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    summary = summarise(samples, runs)
    checks = checksum_agreement(samples)
    r = capacity_ratio(summary)
    summary.to_csv(TABLES / "summary.csv", index=False, lineterminator="\n")
    checks.to_csv(TABLES / "checksums.csv", index=False, lineterminator="\n")
    tables(summary, config)
    fig_speedup(summary, config, r)
    fig_cores(summary, config)
    fig_schedule(summary, config)
    fig_size(summary, config)
    fig_simd(summary)

    ok = summary[summary.status == "ok"]
    print(f"session {session.name}: {len(samples)} samples, {len(runs)} runs; statuses "
          f"{runs.status.value_counts().to_dict()}")
    print("checksums identical across configurations for every n:", bool(checks.identical.all()))
    print(f"relative IQR: max {ok.iqr_rel.max():.4f} ({ok.loc[ok.iqr_rel.idxmax(), 'run_id']}), "
          f"median {ok.iqr_rel.median():.4f}")
    print(f"GC collections during measured samples: {int(ok.gc_collections.sum())}; "
          f"JIT ms during measured samples: max {ok.jit_ms.max():.0f}")
    print(f"background cores (other processes): median {ok.background_cores.median():.2f}, "
          f"max {ok.background_cores.max():.2f} ({ok.loc[ok.background_cores.idxmax(), 'run_id']})")
    print(f"E-core work rate relative to P core (p=10, n=512): r = {r:.3f}; by n at p=10: "
          + ", ".join(f"{n}: {par(summary, 'P_threads', n, 10).e_to_p_rate:.2f}" for n in LARGE))
    for n in config["experiments"]["P_threads"]["sizes"]:
        best = ok[(ok.experiment == "P_threads") & (ok.method == "par_ikj") & (ok.n == n)
                  & (ok.repeat == "")].sort_values("median_ns")
        p10 = [par(summary, "P_threads", n, LOGICAL, repeat=rp) for rp in ("first", "", "last")]
        print(f"n={n:5d}: ikj {seq(summary, n).median_ns / 1e6:10.4f} ms, drift {seq(summary, n, 'last').median_ns / seq(summary, n).median_ns:.3f}; "
              f"p=10 first/shuffled/last " + "/".join(f"{r.median_ns / 1e6:.4g}ms@{r.clock_ghz:.2f}GHz" for r in p10) + "; "
              f"S(1)={par(summary, 'P_threads', n, 1).speedup:.3f} S(4)={par(summary, 'P_threads', n, 4).speedup:.2f} "
              f"S(10)={par(summary, 'P_threads', n, 10).speedup:.2f} best p={int(best.iloc[0].threads)} "
              f"S={best.iloc[0].speedup:.2f}")
    threads = ok[ok.method == "par_ikj"].groupby("threads")[["jvm_threads", "os_threads"]].median()
    print("median Java / OS threads by p:", {int(p): (int(t.jvm_threads), int(t.os_threads))
                                             for p, t in threads.iterrows()},
          "| ikj:", (int(ok[ok.method == "ikj"].jvm_threads.median()), int(ok[ok.method == "ikj"].os_threads.median())))
    product = ok.factor_cycles * ok.factor_clock * ok.factor_busy / ok.speedup_mean
    print(f"decomposition check (product / mean speedup): {product.min():.6f}-{product.max():.6f}")
    print(f"wrote {TABLES.relative_to(ROOT)}/ and {FIGURES.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
