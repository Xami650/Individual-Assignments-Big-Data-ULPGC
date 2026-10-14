# Assignment 1 — Basic Matrix Multiplication in Python, Java and C

BIG DATA · ULPGC · 2026-2027 — Máximo Prosperi Reynoso

Empirical comparison of the straightforward triple-loop dense matrix multiplication
implemented in Python, Java and C: how matrix size affects runtime and memory.

> Work in progress. Build, run and reproduction instructions are completed once the
> implementations and benchmark harness are in place.

## Repository layout

```
assignment1/
├── c/          C implementation (Makefile, sources, tests)
├── java/       Java implementation (standalone Maven project)
├── python/     Python implementation and tests
├── config/     Declared experimental protocol (sizes, repetitions, budgets)
├── scripts/    Input generation, correctness verification, benchmark harness
├── data/       Shared input matrices (generated, not tracked) and checksums
├── results/    Raw measurements (CSV) and environment snapshots
├── analysis/   Scripts producing the tables and figures
└── report/     LaTeX source of the research report and final PDF
```

## Experimental protocol

The protocol was declared before running any benchmark in
[`config/bench.json`](config/bench.json): research question and hypotheses,
algorithm and storage layouts, seeded inputs, matrix sizes, timer boundary,
warm-up and repetition counts, time/memory budgets, correctness tolerance, and
the fixed compiler, JVM and Python settings. The scripts read that file, so it is
the single source of truth for every run.

## Note on the project structure

The three language implementations live side by side in one project, inside a
repository that also holds the other three course assignments. This is **not**
the recommended practice for multi-language software — each implementation would
normally be its own project with its own build and dependency management. It is
done here only to keep the whole class submission (code, data, analysis and report)
together in one place.
