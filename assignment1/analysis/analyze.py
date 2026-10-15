"""Analysis of one benchmark session: summary tables and figures for the report.

Reads results/raw/<session>/{samples.csv, runs.csv} (default: the latest
non-smoke session) and writes
    analysis/tables/summary.csv     one row per (language, n), full precision
    analysis/tables/checksums.csv   cross-language checksum agreement per n
    analysis/tables/*.tex           booktabs tables included by the report
    analysis/figures/*.pdf, *.png   figures (vector PDF for the report)

Statistics: only measured samples (kind = measure) are summarised; warm-ups
are excluded. The central value is the median of the time per multiplication
(elapsed_ns / batch) and the variability is the interquartile range (pandas
quantiles with linear interpolation), also reported relative to the median.

Usage (from assignment1/):
    python3 analysis/analyze.py [--session 20261014-225910]
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

LANGS = ["c", "java", "python"]
LABEL = {"c": "C", "java": "Java", "python": "Python"}
# First three slots of the validated categorical palette (all-pairs CVD-safe in
# light mode); markers are the secondary encoding.
COLOR = {"c": "#2a78d6", "java": "#eb6834", "python": "#1baf7a"}
MARKER = {"c": "o", "java": "s", "python": "^"}
INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
MIB = 2 ** 20

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


def summarise(samples, runs, benchmark_sizes):
    measured = samples[samples.kind == "measure"]
    t = measured.groupby(["language", "n"]).ns_per_multiplication
    summary = pd.DataFrame({
        "reps": t.count(),
        "median_ns": t.median(),
        "q1_ns": t.quantile(0.25),
        "q3_ns": t.quantile(0.75),
        "min_ns": t.min(),
        "max_ns": t.max(),
        "batch": measured.groupby(["language", "n"]).batch.first(),
    })
    summary["warmup_samples"] = samples[samples.kind == "warmup"].groupby(["language", "n"]).size()
    # Process-wide counters cover warm-ups too, so normalise them by every multiplication the process ran.
    summary["multiplications"] = samples.groupby(["language", "n"]).batch.sum()
    summary["kernel_seconds"] = samples.groupby(["language", "n"]).elapsed_ns.sum() / 1e9
    summary["iqr_ns"] = summary.q3_ns - summary.q1_ns
    summary["iqr_rel"] = summary.iqr_ns / summary.median_ns

    runs = runs.set_index(["language", "n"])
    summary = runs[["status", "detail", "max_rss_bytes", "peak_footprint_bytes", "instructions_retired",
                    "cycles_elapsed", "process_user_s"]].join(summary, how="left").reset_index()
    summary["ns_per_inner_iteration"] = summary.median_ns / summary.n.astype(float) ** 3
    summary["theoretical_bytes"] = 24 * summary.n.astype(float) ** 2
    baseline = summary[summary.n == 1].set_index("language").max_rss_bytes
    summary["rss_above_baseline"] = summary.max_rss_bytes - summary.language.map(baseline)
    summary["bytes_per_element_above_baseline"] = summary.rss_above_baseline / (3 * summary.n.astype(float) ** 2)
    summary["ipc_process"] = summary.instructions_retired / summary.cycles_elapsed
    iterations = summary.multiplications * summary.n.astype(float) ** 3
    summary["instructions_per_inner_iteration"] = summary.instructions_retired / iterations
    summary["cycles_per_inner_iteration"] = summary.cycles_elapsed / iterations
    summary["kernel_share_of_user_time"] = summary.kernel_seconds / summary.process_user_s
    summary["cycles_per_user_second"] = summary.cycles_elapsed / summary.process_user_s
    summary["benchmark_size"] = summary.n.isin(benchmark_sizes)
    return summary.sort_values(["language", "n"]).reset_index(drop=True)


def checksum_agreement(samples):
    """Per n: are all measured checksums (parsed exactly with float()) identical across languages?"""
    measured = samples[samples.kind == "measure"].copy()
    measured["value"] = measured.checksum.map(float)  # round-trip exact, unlike pandas' fast parser
    rows = []
    for n, g in measured.groupby("n"):
        rows.append({"n": n, "languages": " ".join(sorted(g.language.unique())),
                     "distinct_values": g.value.nunique(), "identical": g.value.nunique() == 1,
                     "checksum_hex": float(g.value.iloc[0]).hex()})
    return pd.DataFrame(rows)


def growth_exponents(summary):
    """Local exponents log(t2/t1)/log(n2/n1) between consecutive completed sizes, and a global fit."""
    local, fits = [], {}
    for lang in LANGS:
        ok = summary[(summary.language == lang) & (summary.status == "ok") & summary.benchmark_size]
        n, t = ok.n.to_numpy(float), ok.median_ns.to_numpy(float)
        for i in range(1, len(n)):
            local.append({"language": lang, "n_from": int(n[i - 1]), "n_to": int(n[i]),
                          "exponent": math.log(t[i] / t[i - 1]) / math.log(n[i] / n[i - 1])})
        slope, intercept = np.polyfit(np.log(n), np.log(t), 1)
        fits[lang] = {"slope": slope, "sizes": f"{int(n[0])}-{int(n[-1])}"}
    return pd.DataFrame(local), fits


# ---------------------------------------------------------------- formatting

def sig(x, digits=3):
    """Format with about `digits` significant digits, never in scientific notation."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if x == 0:
        return "0"
    decimals = max(0, digits - 1 - math.floor(math.log10(abs(x))))
    return f"{x:.{decimals}f}"


def iqr_text(rel):
    return r"$<$0.1\%" if rel < 0.001 else f"{rel * 100:.1f}\\%"


def status_text(row):
    if row.status == "timeout":
        return r"timeout\textsuperscript{a}"
    if row.status == "skipped":
        return r"skipped\textsuperscript{b}"
    return row.status.replace("_", " ")


def write_table(name, header, rows, colspec, notes=None):
    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if notes:
        lines.append(r"\par\smallskip{\footnotesize " + notes + "}")
    (TABLES / name).write_text("\n".join(lines) + "\n")


def by(summary, lang, n):
    return summary[(summary.language == lang) & (summary.n == n)].iloc[0]


def tables(summary, local, fits, budget_s):
    sizes = sorted(summary[summary.benchmark_size].n.unique())
    notes = (rf"\textsuperscript{{a}} first multiplication exceeded the {budget_s:g}\,s budget and was stopped. "
             r"\textsuperscript{b} not attempted after the timeout at a smaller size.")

    rows = []
    for n in sizes:
        cells = [str(n)]
        for lang in LANGS:
            r = by(summary, lang, n)
            cells.append(f"{sig(r.median_ns / 1e6)} ({iqr_text(r.iqr_rel)})" if r.status == "ok" else status_text(r))
        rows.append(cells)
    write_table("tab_time.tex", ["$n$", "C", "Java", "Python"], rows, "rrrr", notes)

    rows = []
    for n in sizes:
        cells = [str(n)]
        for lang in LANGS:
            r = by(summary, lang, n)
            cells.append(sig(r.ns_per_inner_iteration) if r.status == "ok" else "--")
        c, j, p = (by(summary, lang, n) for lang in LANGS)
        cells.append(sig(j.median_ns / c.median_ns) if j.status == c.status == "ok" else "--")
        cells.append(sig(p.median_ns / c.median_ns) if p.status == c.status == "ok" else "--")
        rows.append(cells)
    write_table("tab_cost.tex", ["$n$", "C", "Java", "Python", "Java/C", "Python/C"], rows, "rrrrrr")

    rows = []
    for n in [1, *sizes]:
        cells = [str(n) + (" (baseline)" if n == 1 else "")]
        for lang in LANGS:
            r = by(summary, lang, n)
            cells.append(sig(r.max_rss_bytes / MIB) if r.status == "ok" else "--")
        cells.append(sig(24 * n * n / MIB) if n > 1 else "--")
        rows.append(cells)
    write_table("tab_memory.tex", ["$n$", "C", "Java", "Python", r"$24n^2$"], rows, "lrrrr")

    intervals = local[local.language == "c"][["n_from", "n_to"]].to_records(index=False)
    rows = []
    for n_from, n_to in intervals:
        cells = [f"{n_from}$\\to${n_to}"]
        for lang in LANGS:
            hit = local[(local.language == lang) & (local.n_from == n_from) & (local.n_to == n_to)]
            cells.append(f"{hit.exponent.iloc[0]:.2f}" if len(hit) else "--")
        rows.append(cells)
    rows.append([r"\midrule global fit"] + [f"{fits[lang]['slope']:.2f} ({fits[lang]['sizes'].replace('-', '--')})" for lang in LANGS])
    write_table("tab_growth.tex", ["sizes", "C", "Java", "Python"], rows, "lrrr")

    rows = []
    for n in sizes:
        cells = [str(n)]
        for column in ("instructions_per_inner_iteration", "cycles_per_inner_iteration"):
            for lang in LANGS:
                r = by(summary, lang, n)
                cells.append(sig(r[column]) if r.status == "ok" else "--")
        rows.append(cells)
    group = (r" & \multicolumn{3}{c}{instructions per iteration} & \multicolumn{3}{c}{cycles per iteration} \\"
             r" \cmidrule(lr){2-4} \cmidrule(lr){5-7} $n$")
    write_table("tab_counters.tex", [group, *["C", "Java", "Python"] * 2], rows, "rrrrrrr")


# ---------------------------------------------------------------- figures

def size_axis(ax, sizes):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("matrix size $n$")


def end_label(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", color=INK, fontsize=8)


def plain_log_ticks(ax, ticks):
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()


def save(fig, name):
    # No creation timestamp, so re-running the analysis reproduces the files byte for byte.
    fig.savefig(FIGURES / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)


def fig_time(summary, budget_s):
    sizes = sorted(summary[summary.benchmark_size].n.unique())
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    for lang in LANGS:
        ok = summary[(summary.language == lang) & (summary.status == "ok") & summary.benchmark_size]
        med = ok.median_ns / 1e9
        err = np.vstack([med - ok.q1_ns / 1e9, ok.q3_ns / 1e9 - med])
        ax.errorbar(ok.n, med, yerr=err, color=COLOR[lang], marker=MARKER[lang], capsize=2, elinewidth=0.8,
                    label=LABEL[lang])
        end_label(ax, ok.n.iloc[-1], med.iloc[-1], LABEL[lang], dy={"c": 6, "java": -6}.get(lang, 0))
    # n^3 guide through the C median at n = 256
    ref = by(summary, "c", 256).median_ns / 1e9
    guide = np.array([64, 2048], float)
    ax.plot(guide, ref * (guide / 256) ** 3, ls="--", lw=0.9, color=INK_3, zorder=0)
    ax.annotate(r"$\propto n^3$", (1100, ref * (1100 / 256) ** 3), xytext=(-24, 4), textcoords="offset points",
                color=INK_2, fontsize=8)
    ax.axhline(budget_s, ls=":", lw=0.9, color=INK_3, zorder=0)
    ax.annotate(f"budget: {budget_s:g} s per multiplication", (64, budget_s), xytext=(0, 4),
                textcoords="offset points", color=INK_2, fontsize=8)
    for row in summary[(summary.status == "timeout")].itertuples():
        # hollow marker on the budget line; the arrow means "more than the budget"
        ax.plot(row.n, budget_s, marker=MARKER[row.language], mfc="white", mec=COLOR[row.language], ls="none")
        ax.annotate("", xy=(row.n, budget_s * 2.6), xytext=(row.n, budget_s * 1.15),
                    arrowprops={"arrowstyle": "->", "color": INK_2, "lw": 0.8})
        ax.annotate(f"{LABEL[row.language]} timeout ($>${budget_s:g} s)", (row.n, budget_s), xytext=(-8, 6),
                    textcoords="offset points", ha="right", color=INK, fontsize=8)
    size_axis(ax, sizes)
    ax.set_yscale("log")
    ax.set_ylim(top=budget_s * 4)
    ax.set_ylabel("time per multiplication (s, log scale)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.86))
    save(fig, "fig_time")


def fig_cost(summary):
    sizes = sorted(summary[summary.benchmark_size].n.unique())
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    for lang in LANGS:
        ok = summary[(summary.language == lang) & (summary.status == "ok") & summary.benchmark_size]
        ax.plot(ok.n, ok.ns_per_inner_iteration, color=COLOR[lang], marker=MARKER[lang], label=LABEL[lang])
        end_label(ax, ok.n.iloc[-1], ok.ns_per_inner_iteration.iloc[-1], LABEL[lang])
    size_axis(ax, sizes)
    ax.set_yscale("log")
    ax.set_ylabel("median time / $n^3$ (ns, log scale)")
    plain_log_ticks(ax, [0.3, 0.5, 1, 2, 5, 10, 20, 40])
    ax.legend(loc="center left", bbox_to_anchor=(0, 0.62))
    save(fig, "fig_cost")


def fig_memory(summary):
    sizes = sorted(summary[summary.benchmark_size].n.unique())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.2), gridspec_kw={"wspace": 0.38})
    for lang in LANGS:
        ok = summary[(summary.language == lang) & (summary.status == "ok") & summary.benchmark_size]
        ax1.plot(ok.n, ok.max_rss_bytes / MIB, color=COLOR[lang], marker=MARKER[lang], label=LABEL[lang])
        big = ok[ok.n >= 256]
        ax2.plot(big.n, big.bytes_per_element_above_baseline, color=COLOR[lang], marker=MARKER[lang],
                 label=LABEL[lang])
    n = np.array(sizes, float)
    ax1.plot(n, 24 * n ** 2 / MIB, ls="--", lw=0.9, color=INK_3, label="$24n^2$ bytes (3 arrays)")
    size_axis(ax1, sizes)
    ax1.set_xticks(sizes[::2])
    ax1.set_xticklabels([str(s) for s in sizes[::2]])
    ax1.set_yscale("log")
    ax1.set_ylabel("peak RSS (MiB, log scale)")
    ax1.set_title("(a) peak resident set size", loc="left", color=INK)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07))
    ax2.axhline(8, ls="--", lw=0.9, color=INK_3)
    ax2.annotate("8 B = one double", (2048, 8), xytext=(0, -10), textcoords="offset points", ha="right",
                 color=INK_2, fontsize=8)
    size_axis(ax2, [s for s in sizes if s >= 256])
    ax2.set_xticks([256, 512, 1024, 2048])
    ax2.set_xticklabels(["256", "512", "1024", "2048"])
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel("bytes per matrix element\nabove the $n=1$ baseline")
    ax2.set_title("(b) memory cost per element", loc="left", color=INK)
    save(fig, "fig_memory")


def fig_ratio(summary):
    sizes = sorted(summary[summary.benchmark_size].n.unique())
    c = summary[summary.language == "c"].set_index("n")
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    for lang in ["java", "python"]:
        other = summary[summary.language == lang].set_index("n")
        both = [n for n in sizes if other.loc[n, "status"] == "ok" and c.loc[n, "status"] == "ok"]
        ratio = other.loc[both, "median_ns"] / c.loc[both, "median_ns"]
        ax.plot(both, ratio, color=COLOR[lang], marker=MARKER[lang], label=f"{LABEL[lang]} / C")
        end_label(ax, both[-1], ratio.iloc[-1], f"{LABEL[lang]} / C")
    ax.axhline(1, ls="--", lw=0.9, color=INK_3, zorder=0)
    size_axis(ax, sizes)
    ax.set_yscale("log")
    ax.set_ylabel("median time ratio to C (log scale)")
    plain_log_ticks(ax, [0.5, 1, 2, 5, 10, 20, 50])
    ax.legend(loc="center left", bbox_to_anchor=(0, 0.55))
    save(fig, "fig_ratio")


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

    budget_s = config["budgets"]["max_seconds_per_multiplication"]
    summary = summarise(samples, runs, config["sizes"]["benchmark"])
    checks = checksum_agreement(samples)
    local, fits = growth_exponents(summary)

    summary.to_csv(TABLES / "summary.csv", index=False, lineterminator="\n")
    checks.to_csv(TABLES / "checksums.csv", index=False, lineterminator="\n")
    local.to_csv(TABLES / "growth_exponents.csv", index=False, lineterminator="\n", float_format="%.4f")
    tables(summary, local, fits, budget_s)
    fig_time(summary, budget_s)
    fig_cost(summary)
    fig_memory(summary)
    fig_ratio(summary)

    print(f"session {session.name}: {len(samples)} samples, {len(runs)} runs")
    print("checksums identical across languages at every n:", bool(checks.identical.all()))
    print("global log-log slopes:", {k: round(v["slope"], 3) for k, v in fits.items()})
    ok = summary[summary.status == "ok"]
    print(f"kernel share of user CPU time: {ok.kernel_share_of_user_time.min():.3f}-"
          f"{ok.kernel_share_of_user_time.max():.3f}; cycles per user second: "
          f"{ok.cycles_per_user_second.min() / 1e9:.2f}-{ok.cycles_per_user_second.max() / 1e9:.2f} GHz")
    print(f"wrote {TABLES.relative_to(ROOT)}/ and {FIGURES.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
