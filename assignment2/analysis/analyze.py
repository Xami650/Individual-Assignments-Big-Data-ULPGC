"""Analysis of one benchmark session: summary tables and figures for the report.

Adapted from Assignment 1's analysis/analyze.py. Reads
results/raw/<session>/{samples.csv, runs.csv, config.json} (default: the latest
non-smoke session) and writes
    analysis/tables/summary.csv     one row per configuration, full precision
    analysis/tables/checksums.csv   checksum agreement between methods on the same inputs
    analysis/tables/*.tex           booktabs tables included by the report
    analysis/figures/*.pdf, *.png   figures (vector PDF for the report)

Statistics: only measured samples (kind = measure) are summarised; warm-ups
are excluded. The central value is the median of the time per operation
(elapsed_ns / batch) and the variability is the interquartile range (pandas
quantiles with linear interpolation), also reported relative to the median.
Process-wide counters (instructions, cycles) cover the pre-warm and warm-ups
too, so they are normalised by every multiply-add the process ran.

Usage (from assignment2/):
    python3 analysis/analyze.py [--session 20261029-104200]
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
TABLES = ROOT / "analysis" / "tables"
FIGURES = ROOT / "analysis" / "figures"

DENSE = ["basic", "ikj", "blocked"]
STRUCTURES = ["uniform", "banded"]
# Slots of the validated categorical palette: blue, orange and aqua (all-pairs
# CVD-safe) for the dense methods, red for csr (checked against blue with the
# palette validator). Markers and line styles are the secondary encoding.
COLOR = {"blocked": "#2a78d6", "basic": "#eb6834", "ikj": "#1baf7a", "csr": "#e34948"}
MARKER = {"blocked": "o", "basic": "s", "ikj": "^", "csr": "D"}
LINESTYLE = {"uniform": "-", "banded": "--"}
INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
MIB, GIB = 2 ** 20, 2 ** 30
PRE_WARM_N = 256

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": INK_3, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.5, "lines.markersize": 5,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


# ---------------------------------------------------------------- data

def latest_session():
    sessions = sorted(p for p in RAW.iterdir() if p.is_dir() and not p.name.startswith("smoke"))
    if not sessions:
        raise SystemExit("no benchmark session found in results/raw")
    return sessions[-1]


def summarise(samples, runs):
    measured = samples[samples.kind == "measure"]
    t = measured.groupby("run_id").ns_per_op
    stats = pd.DataFrame({
        "reps": t.count(), "median_ns": t.median(), "q1_ns": t.quantile(0.25), "q3_ns": t.quantile(0.75),
        "min_ns": t.min(), "max_ns": t.max(), "batch": measured.groupby("run_id").batch.first(),
    })
    stats["warmup_samples"] = samples[samples.kind == "warmup"].groupby("run_id").size()
    stats["operations"] = samples.groupby("run_id").batch.sum()
    stats["timed_seconds"] = samples.groupby("run_id").elapsed_ns.sum() / 1e9
    s = runs.join(stats, on="run_id")
    s["iqr_rel"] = (s.q3_ns - s.q1_ns) / s.median_ns
    n = s.n.astype(float)
    s["n2"] = n * n
    s["gflops"] = np.where(s.method.isin(DENSE), 2 * n ** 3 / s.median_ns, 2 * s.multiply_adds / s.median_ns)
    s["ns_per_multiply_add"] = s.median_ns / s.multiply_adds
    # Every multiply-add of the process: timed operations plus the dense pre-warm (m = min(n, 256), m^3 each).
    m = np.minimum(n, PRE_WARM_N)
    total = s.operations * s.multiply_adds + s.pre_warm_ops.fillna(0) * m ** 3
    s["instructions_per_multiply_add"] = np.where(s.method.isin(DENSE), s.instructions_retired / total, np.nan)
    s["cycles_per_multiply_add"] = np.where(s.method.isin(DENSE), s.cycles_elapsed / total, np.nan)
    s["timed_share_of_user_time"] = s.timed_seconds / s.process_user_s
    s["density_c_percent"] = 100 * s.nnz_c / s.n2
    s["dense_bytes"] = 8 * s.n2
    s["csr_bytes_a"] = 12 * s.nnz_a + 4 * (n + 1)
    s["csr_bytes_b"] = 12 * s.nnz_b + 4 * (n + 1)
    s["csr_bytes_c"] = 12 * s.nnz_c + 4 * (n + 1)
    return s


def checksum_agreement(samples):
    """Per input: are the measured checksums of all methods that compute C identical (exact float())?"""
    measured = samples[(samples.kind == "measure") & (samples.method != "to_csr")].copy()
    measured["value"] = measured.checksum.map(float)  # round-trip exact, unlike pandas' fast parser
    rows = []
    for (n, structure, density), g in measured.groupby(["n", "structure", "density_percent"]):
        rows.append({"n": n, "structure": structure, "density_percent": density,
                     "methods": " ".join(sorted(g.method.unique())), "configurations": g.run_id.nunique(),
                     "distinct_values": g.value.nunique(), "identical": g.value.nunique() == 1})
    return pd.DataFrame(rows)


def growth_exponents(dense):
    """Global log-log slope of the median time against n, per dense method (completed sizes, n >= 256)."""
    fits = {}
    for method in DENSE:
        ok = dense[(dense.method == method) & (dense.status == "ok") & (dense.n >= 256)]
        slope, _ = np.polyfit(np.log(ok.n.astype(float)), np.log(ok.median_ns), 1)
        fits[method] = (slope, int(ok.n.min()), int(ok.n.max()))
    return fits


# ---------------------------------------------------------------- formatting

def sig(x, digits=3):
    """Format with about `digits` significant digits, never in scientific notation."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if x == 0:
        return "0"
    decimals = max(0, digits - 1 - math.floor(math.log10(abs(x))))
    return f"{x:,.{decimals}f}".replace(",", "\\,")


def count(x):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{int(x):,}".replace(",", "\\,")


def iqr_text(rel):
    return r"$<$0.1\%" if rel < 0.001 else f"{rel * 100:.1f}\\%"


def time_text(row, unit=1e6, iqr=True):
    if row is None:
        return "--"
    if row.status == "ok":
        return f"{sig(row.median_ns / unit)} ({iqr_text(row.iqr_rel)})" if iqr else sig(row.median_ns / unit)
    return {"timeout": r"timeout\textsuperscript{a}", "skipped": r"skipped\textsuperscript{b}",
            "memory_budget_exceeded": r"out of memory\textsuperscript{c}"}.get(row.status, row.status)


def write_table(name, header, rows, colspec, notes=None):
    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines.append(r"\par\smallskip{\footnotesize " + notes + "}")
    (TABLES / name).write_text("\n".join(lines) + "\n")


def pick(s, **keys):
    hit = s
    for k, v in keys.items():
        hit = hit[hit[k] == v]
    return hit.iloc[0] if len(hit) else None


def status_notes(budget_s, heap):
    return (rf"\textsuperscript{{a}} the first operation exceeded the {budget_s:g}\,s budget and was stopped. "
            r"\textsuperscript{b} not attempted after a failure at a smaller size. "
            rf"\textsuperscript{{c}} the JVM ran out of its {heap} heap.")


# ---------------------------------------------------------------- tables

def tables(s, budget_s, heap):
    dense = s[s.experiment == "A_dense"]
    sizes = sorted(dense.n.unique())
    notes = status_notes(budget_s, heap)

    rows = []
    for n in sizes:
        r = {m: pick(dense, method=m, n=n) for m in DENSE}
        cells = [str(n)] + [time_text(r[m]) for m in DENSE]
        for m in ("ikj", "blocked"):
            ok = r[m].status == "ok" and r["basic"].status == "ok"
            cells.append(sig(r["basic"].median_ns / r[m].median_ns) if ok else "--")
        cells.append(sig(r["ikj"].median_ns / r["blocked"].median_ns))
        rows.append(cells)
    write_table("tab_dense_time.tex", ["$n$", "basic (ms)", "ikj (ms)", "blocked (ms)", "basic/ikj",
                                       "basic/blocked", "ikj/blocked"], rows, "rrrrrrr", notes)

    rows = []
    for n in sizes:
        cells = [str(n)]
        for column in ("instructions_per_multiply_add", "cycles_per_multiply_add"):
            for m in DENSE:
                r = pick(dense, method=m, n=n)
                cells.append(sig(r[column]) if r.status == "ok" else "--")
        for m in DENSE:
            r = pick(dense, method=m, n=n)
            cells.append(sig(r.gflops) if r.status == "ok" else "--")
        rows.append(cells)
    group = (r" & \multicolumn{3}{c}{instructions / mult.-add} & \multicolumn{3}{c}{cycles / mult.-add}"
             r" & \multicolumn{3}{c}{GFLOP/s} \\ \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} $n$")
    write_table("tab_dense_counters.tex", [group, *["basic", "ikj", "blocked"] * 3], rows, "r" * 10)

    rows = []
    for n in sizes:
        cells = [str(n)]
        for m in DENSE:
            r = pick(dense, method=m, n=n)
            cells.append(sig(r.max_rss_bytes / MIB) if r.status == "ok" else "--")
        cells.append(sig(24 * n * n / MIB))
        rows.append(cells)
    write_table("tab_dense_memory.tex", ["$n$", "basic", "ikj", "blocked", r"$24n^2$"], rows, "rrrrr")

    tiles = s[(s.method == "blocked") & s.experiment.isin(["A_dense", "A_block"]) & s.n.isin([2048, 4096])]
    rows = []
    for bs in sorted(tiles.block.dropna().unique()):
        cells = [str(int(bs))]
        for n in (2048, 4096):
            r = pick(tiles, n=n, block=bs)
            cells.append(time_text(r, 1e9))
            ikj = pick(dense, method="ikj", n=n)
            cells.append(sig(ikj.median_ns / r.median_ns) if r is not None and r.status == "ok" else "--")
        rows.append(cells)
    write_table("tab_block.tex", ["tile size $b$", r"$n=2048$ (s)", "ikj/blocked", r"$n=4096$ (s)", "ikj/blocked"],
                rows, "rrrrr")

    bd = s[s.experiment == "B_density"]
    rows = []
    for density in sorted(bd.density_percent.unique()):
        for structure in STRUCTURES:
            g = {m: pick(bd, method=m, density_percent=density, structure=structure)
                 for m in ("csr", "blocked", "to_csr", "to_dense")}
            csr, dense_ = g["csr"], g["blocked"]
            total = csr.median_ns + g["to_csr"].median_ns + g["to_dense"].median_ns
            rows.append([f"{density:g}" if structure == "uniform" else "", structure, count(csr.nnz_a),
                         f"{sig(csr.nnz_c / 1e3)} ({sig(csr.density_c_percent)}\\%)", sig(csr.multiply_adds / 1e6),
                         sig(csr.median_ns / 1e6), sig(g["to_csr"].median_ns / 1e6), sig(g["to_dense"].median_ns / 1e6),
                         sig(dense_.median_ns / 1e6), sig(dense_.median_ns / csr.median_ns),
                         sig(dense_.median_ns / total)])
    write_table("tab_density.tex", ["$d$ (\\%)", "structure", "nnz($A$)", "nnz($C$), $10^3$ (density)",
                                    "mult.-adds, $10^6$",
                                    "csr", "to\\_csr", "to\\_dense", "blocked", "speed-up", "incl. conv."],
                rows, "rlrrrrrrrrr")

    rows = []
    for density in sorted(bd.density_percent.unique()):
        for structure in STRUCTURES:
            csr = pick(bd, method="csr", density_percent=density, structure=structure)
            dense_ = pick(bd, method="blocked", density_percent=density, structure=structure)
            rows.append([f"{density:g}" if structure == "uniform" else "", structure,
                         sig((csr.csr_bytes_a + csr.csr_bytes_b) / MIB), sig(csr.csr_bytes_c / MIB),
                         sig(2 * csr.dense_bytes / MIB), sig(csr.dense_bytes / MIB),
                         sig(csr.max_rss_bytes / MIB), sig(dense_.max_rss_bytes / MIB)])
    write_table("tab_density_memory.tex", ["$d$ (\\%)", "structure", "CSR $A{+}B$", "CSR $C$", "dense $A{+}B$",
                                           "dense $C$", "RSS csr", "RSS blocked"], rows, "rlrrrrrr")

    size_rows = b_size(s)
    rows = []
    for n in sorted(size_rows.n.unique()):
        csr, dense_ = pick(size_rows, n=n, method="csr"), pick(size_rows, n=n, method="blocked")
        nnz_c = csr.nnz_c if csr.status == "ok" else math.nan
        rows.append([str(n), sig(csr.nnz_a / 1e6), sig(nnz_c / 1e6), sig(csr.multiply_adds / 1e6),
                     time_text(csr, 1e9, iqr=False), time_text(dense_, 1e9, iqr=False),
                     sig((csr.csr_bytes_a + csr.csr_bytes_b + (csr.csr_bytes_c if csr.status == "ok" else 0)) / GIB)
                     if csr.status == "ok" else "--",
                     sig(24 * n * n / GIB),
                     sig(csr.peak_footprint_bytes / GIB) if csr.status == "ok" else "--"])
    write_table("tab_size.tex", ["$n$", "nnz($A$)", "nnz($C$)", "mult.-adds", "csr", "blocked",
                                 "CSR", "dense", "footprint"],
                rows, "rrrrrrrrr", notes)


def b_size(s):
    """B_size rows, with the shared n = 2048 point taken from B_density."""
    config_density = s[s.experiment == "B_size"].density_percent.iloc[0]
    shared = s[(s.experiment == "B_density") & (s.structure == "uniform") & (s.density_percent == config_density)
               & s.method.isin(["csr", "blocked"])]
    return pd.concat([s[s.experiment == "B_size"], shared]).sort_values(["n", "method"])


# ---------------------------------------------------------------- figures

def size_axis(ax, sizes, labels=None):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels(labels or [str(x) for x in sizes])
    ax.minorticks_off()
    ax.set_xlabel("matrix size $n$")


def end_label(ax, x, y, text, dy=0, color=INK):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", color=color, fontsize=8)


def plain_log_ticks(ax, ticks, axis="y"):
    getattr(ax, f"set_{axis}ticks")(ticks)
    getattr(ax, f"set_{axis}ticklabels")([f"{t:g}" for t in ticks])
    ax.minorticks_off()


def save(fig, name):
    # No creation timestamp, so re-running the analysis reproduces the files byte for byte.
    fig.savefig(FIGURES / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)


def errorbars(ax, ok, unit, x=None, **style):
    """Median with an IQR error bar; x defaults to n."""
    med = ok.median_ns / unit
    err = np.vstack([med - ok.q1_ns / unit, ok.q3_ns / unit - med])
    ax.errorbar(ok.n if x is None else x, med, yerr=err, capsize=2, elinewidth=0.8, **style)
    return med


def mark_failures(ax, rows, budget, color, marker):
    for row in rows.itertuples():
        if row.status not in ("timeout", "memory_budget_exceeded"):
            continue
        ax.plot(row.n, budget, marker=marker, mfc="white", mec=color, ls="none")
        # timeouts are labelled above the marker and memory failures below it, so neighbours never collide
        text, dy, va = ("timeout", 7, "bottom") if row.status == "timeout" else ("out of\nmemory", -7, "top")
        ax.annotate(text, (row.n, budget), xytext=(0, dy), textcoords="offset points", ha="center", va=va,
                    color=INK, fontsize=7)


def fig_dense(s, budget_s):
    dense = s[s.experiment == "A_dense"]
    sizes = sorted(dense.n.unique())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.3), gridspec_kw={"wspace": 0.32})
    for m in DENSE:
        rows = dense[dense.method == m]
        ok = rows[rows.status == "ok"]
        med = errorbars(ax1, ok, 1e9, color=COLOR[m], marker=MARKER[m], label=m)
        mark_failures(ax1, rows, budget_s, COLOR[m], MARKER[m])
        ax2.plot(ok.n, ok.gflops, color=COLOR[m], marker=MARKER[m], label=m)
    ref = pick(dense, method="basic", n=256).median_ns / 1e9
    guide = np.array([64, 4096], float)
    ax1.plot(guide, ref * (guide / 256) ** 3, ls="--", lw=0.9, color=INK_3, zorder=0)
    ax1.annotate(r"$\propto n^3$", (900, ref * (900 / 256) ** 3), xytext=(-26, 4), textcoords="offset points",
                 color=INK_2, fontsize=8)
    ax1.axhline(budget_s, ls=":", lw=0.9, color=INK_3, zorder=0)
    ax1.annotate(f"budget {budget_s:g} s", (64, budget_s), xytext=(0, -9), textcoords="offset points",
                 color=INK_2, fontsize=7)
    size_axis(ax1, sizes[::2])
    ax1.set_yscale("log")
    ax1.set_ylim(top=budget_s * 5)
    ax1.set_ylabel("time per multiplication (s, log scale)")
    ax1.set_title("(a) runtime", loc="left", color=INK)
    size_axis(ax2, sizes[::2])
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel(r"$2n^3$ / time (GFLOP/s)")
    ax2.set_title("(b) arithmetic rate", loc="left", color=INK)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    save(fig, "fig_dense")


def fig_block(s):
    tiles = s[(s.method == "blocked") & s.experiment.isin(["A_dense", "A_block"]) & s.n.isin([2048, 4096])]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for n, marker in ((2048, "o"), (4096, "s")):
        ikj = pick(s[s.experiment == "A_dense"], method="ikj", n=n).median_ns
        ok = tiles[(tiles.n == n) & (tiles.status == "ok")].sort_values("block")
        ratio = ikj / ok.median_ns
        ax.plot(ok.block, ratio, color=COLOR["blocked"], marker=marker, ls="-" if n == 2048 else "--",
                label=f"$n = {n}$")
        end_label(ax, ok.block.iloc[-1], ratio.iloc[-1], f"$n={n}$")
    ax.axhline(1, ls=":", lw=0.9, color=INK_3, zorder=0)
    ax.annotate("as fast as ikj", (32, 1), xytext=(0, 4), textcoords="offset points", color=INK_2, fontsize=7)
    blocks = sorted(int(b) for b in tiles.block.dropna().unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(blocks)
    ax.set_xticklabels([str(b) for b in blocks])
    ax.minorticks_off()
    ax.set_xlabel("tile size $b$")
    ax.set_ylabel("speed-up of blocked over ikj")
    ax.legend(loc="lower right")
    save(fig, "fig_block")


def fig_density(s):
    bd = s[s.experiment == "B_density"]
    n = int(bd.n.iloc[0])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.3), gridspec_kw={"wspace": 0.34})
    for structure in STRUCTURES:
        hollow = {"mfc": "white"} if structure == "banded" else {}
        csr = bd[(bd.method == "csr") & (bd.structure == structure)].sort_values("density_percent")
        errorbars(ax1, csr, 1e6, x=csr.density_percent, color=COLOR["csr"], marker=MARKER["csr"],
                  ls=LINESTYLE[structure], label=f"csr, {structure}", **hollow)
        dense = bd[(bd.method == "blocked") & (bd.structure == structure)].sort_values("density_percent")
        errorbars(ax1, dense, 1e6, x=dense.density_percent, color=COLOR["blocked"], marker=MARKER["blocked"],
                  ls=LINESTYLE[structure], label=f"blocked, {structure}", **hollow)
        conv = bd[bd.method.isin(["to_csr", "to_dense"]) & (bd.structure == structure)]
        total = csr.set_index("density_percent").median_ns + conv.groupby("density_percent").median_ns.sum()
        if structure == "uniform":
            ax1.plot(total.index, total / 1e6, color=COLOR["csr"], ls=":", lw=1.2,
                     label="csr + conversions, uniform")
        ax2.plot(csr.density_percent, csr.density_c_percent, color=COLOR["csr"], marker=MARKER["csr"],
                 ls=LINESTYLE[structure], label=structure, **hollow)
    d = np.logspace(np.log10(0.1), np.log10(50), 100) / 100
    ax2.plot(100 * d, 100 * (1 - (1 - d * d) ** n), color=INK_3, lw=0.9, ls="-", zorder=0,
             label=r"$1-(1-d^2)^n$ (uniform model)")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        plain_log_ticks(ax, [0.1, 0.5, 1, 5, 10, 50], "x")
        ax.set_xlabel("density of $A$ and $B$ (%, log scale)")
    ax1.set_yscale("log")
    ax1.set_ylabel("time per multiplication (ms, log scale)")
    ax1.set_title(f"(a) runtime, $n = {n}$", loc="left", color=INK)
    ax1.legend(loc="lower right", fontsize=7)
    ax2.set_yscale("log")
    ax2.set_ylabel("density of $C$ (%, log scale)")
    ax2.set_title("(b) fill-in of the product", loc="left", color=INK)
    plain_log_ticks(ax2, [0.01, 0.1, 1, 10, 100])
    ax2.legend(loc="lower right", fontsize=7)
    save(fig, "fig_density")


def fig_size(s, budget_s, heap_bytes):
    rows = b_size(s)
    sizes = sorted(rows.n.unique())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.3), gridspec_kw={"wspace": 0.34})
    for m in ("csr", "blocked"):
        g = rows[rows.method == m]
        ok = g[g.status == "ok"]
        med = errorbars(ax1, ok, 1e9, color=COLOR[m], marker=MARKER[m], label=m)
        ax1.annotate(m, (ok.n.iloc[0], med.iloc[0]), xytext=(0, 8), textcoords="offset points", ha="center",
                     color=INK, fontsize=8)
        mark_failures(ax1, g, budget_s, COLOR[m], MARKER[m])
    ax1.axhline(budget_s, ls=":", lw=0.9, color=INK_3, zorder=0)
    size_axis(ax1, sizes, [f"{x // 1024}k" for x in sizes])
    ax1.set_yscale("log")
    ax1.set_ylim(top=budget_s * 5)
    ax1.set_ylabel("time per multiplication (s, log scale)")
    ax1.set_title("(a) runtime", loc="left", color=INK)

    csr = rows[(rows.method == "csr") & (rows.status == "ok")]
    storage = (csr.csr_bytes_a + csr.csr_bytes_b + csr.csr_bytes_c) / GIB
    ax2.plot(csr.n, storage, color=COLOR["csr"], marker=MARKER["csr"], label="CSR $A+B+C$ (elements)")
    ax2.plot(csr.n, csr.peak_footprint_bytes / GIB, color=COLOR["csr"], marker=MARKER["csr"], ls="--",
             mfc="white", label="csr process (peak footprint)")
    n = np.array(sizes, float)
    ax2.plot(n, 24 * n * n / GIB, color=COLOR["blocked"], ls="-", label="dense $A+B+C$ ($24n^2$)")
    ax2.axhline(heap_bytes / GIB, ls=":", lw=0.9, color=INK_3, zorder=0)
    ax2.annotate(f"heap limit {heap_bytes / GIB:g} GiB", (sizes[0], heap_bytes / GIB), xytext=(0, 4),
                 textcoords="offset points", color=INK_2, fontsize=7)
    size_axis(ax2, sizes, [f"{x // 1024}k" for x in sizes])
    ax2.set_yscale("log")
    ax2.set_ylim(5e-4, heap_bytes / GIB * 12)
    ax2.set_ylabel("memory (GiB, log scale)")
    ax2.set_title("(b) memory", loc="left", color=INK)
    ax2.legend(loc="lower right", fontsize=7)
    save(fig, "fig_size")


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

    budget_s = config["budgets"]["max_seconds_per_operation"]
    heap_flag = next(f for f in config["build_and_runtime"]["jvm_flags"] if f.startswith("-Xmx"))
    heap_bytes = int(heap_flag[4:-1]) * GIB
    summary = summarise(samples, runs)
    checks = checksum_agreement(samples)
    fits = growth_exponents(summary[summary.experiment == "A_dense"])

    summary.to_csv(TABLES / "summary.csv", index=False, lineterminator="\n")
    checks.to_csv(TABLES / "checksums.csv", index=False, lineterminator="\n")
    tables(summary, budget_s, f"{heap_bytes // GIB}\\,GiB")
    fig_dense(summary, budget_s)
    fig_block(summary)
    fig_density(summary)
    fig_size(summary, budget_s, heap_bytes)

    print(f"session {session.name}: {len(samples)} samples, {len(runs)} runs; statuses "
          f"{runs.status.value_counts().to_dict()}")
    print("checksums identical across methods for every input:", bool(checks.identical.all()))
    print("global log-log slopes (n >= 256):", {k: f"{v[0]:.2f} ({v[1]}-{v[2]})" for k, v in fits.items()})
    ok = summary[(summary.status == "ok") & summary.method.isin(DENSE)]
    print(f"timed share of user CPU time (dense): {ok.timed_share_of_user_time.min():.3f}-"
          f"{ok.timed_share_of_user_time.max():.3f}; max relative IQR: {summary.iqr_rel.max():.4f}")
    print(f"wrote {TABLES.relative_to(ROOT)}/ and {FIGURES.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
