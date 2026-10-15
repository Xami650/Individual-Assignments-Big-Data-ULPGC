# Submission checklist — Assignment 1

This file maps each requirement of the course guide (Assignment 1 page and the
common pages) to where it is met. "§" refers to sections of the report
([`report/main.pdf`](report/main.pdf)).

## Required implementations and comparisons

| Requirement | Where it is met |
|---|---|
| Triple-loop algorithm in Python, Java and C, O(n³) work | `python/matmul.py`, `java/.../MatMul.java`, `c/src/matmul.c`; report Fig. 1 |
| Comparable loop order and numeric type; storage layouts and unavoidable differences described | i-j-k order and float64 in all three; report §3.1, Fig. 1; `config/bench.json` (`algorithm`) |
| No optimised library call in place of the loops | kernels use plain loops only; NumPy only generates inputs and checks results (§3.2, §3.4) |
| Feasible increasing sizes under a common protocol; all three compared wherever they complete | n = 64 … 2048 (§3.5); Tables 2–6, Figs. 2–5 |
| C compiler and flags, JVM and warm-up settings, Python implementation and version; fixed settings explained | report Table 1 and §3.5; `config/bench.json` (`build_and_runtime`, `measurement`); `results/raw/20261014-225910/env.txt` |

## Required evidence

| Measure | Where it is reported |
|---|---|
| Time: repeated measurements, median and variability | 5 measured samples per configuration, median and IQR: Table 2, Fig. 2; every sample in `results/raw/20261014-225910/samples.csv` |
| Memory: named metric and tool | peak resident set size from `/usr/bin/time -l` (§3.5); Table 6, Fig. 5; `runs.csv` |
| Correctness: test cases, references, tolerance | unit tests in each language and `scripts/verify.py` (§3.4); results in §4.1 and `results/correctness.csv` |
| Limits: largest feasible size, timeouts, memory failures | §4.6: largest size within budget 2048 (C, Java) and 1024 (Python); Python timeout at 1536, 2048 skipped; no memory failures |

## Analysis

| Requirement | Where it is met |
|---|---|
| Observed growth compared with the cubic model, with constants and system effects | §4.3 (Tables 3–4, Fig. 3) and §5 (H1) |
| Interpreter, runtime and memory-layout effects where the evidence supports them | §4.4 (Table 5) and §5 (H2, H3) |
| Conclusions scoped to the tested implementations, settings, inputs and hardware; no claim that one language is universally faster | abstract, §5, §6 |

## Common protocol

| Requirement | Where it is met |
|---|---|
| Question, factor and outcome stated; hypotheses kept apart from results | §1; hypotheses fixed in `config/bench.json` before the benchmarks ran |
| Same logical inputs; seed, generator, range, dimensions and type recorded | §3.2; `scripts/gen_inputs.py`, `data/checksums.sha256` |
| Hardware, OS, versions, flags, runtime settings and thread counts documented | Table 1; `env.txt` |
| Timer boundary stated; monotonic clocks; result consumed | §3.5; checksum printed after every sample |
| Warm-up used, excluded from summaries and its count recorded | §3.5; `kind = warmup` rows in `samples.csv` |
| At least five repetitions; very short tasks batched with a reported time per multiplication | §3.5; `batch` column in `samples.csv` |
| Every raw measurement kept; median and IQR with units and repetitions | `results/raw/`; Table 2 |
| Small known cases (identity, zero), a trusted reference and a justified tolerance | §3.4, §4.1 |
| Time and memory budgets declared before the large tests; timeouts reported without invented times | `config/bench.json` (`budgets`); §3.5, §4.6 |
| Memory metric and tool named; limitations reported | §3.5; §5 (Limitations) |
| Configurations labelled with size, method, numeric type, repetition and threads | columns of `samples.csv` |

## Report and submission package

| Requirement | Where it is met |
|---|---|
| Report structure: title and abstract, introduction, background, methodology, results, discussion, conclusions, references and materials | `report/main.tex`, §1–§7 |
| English; labelled figures and tables; consistent IEEE references, each one checked | report; references [1]–[14] |
| PDF with name, assignment number and research title | `Assignment01_ProsperiReynoso_Report.pdf` (copy of `report/main.pdf`) |
| Source code, README, dependencies, environment details, exact build and run commands | [`README.md`](README.md), `requirements.txt`, `java/pom.xml`, `c/Makefile` |
| Raw results, input generation instructions, scripts reproducing the analysis and figures | `results/`, `scripts/gen_inputs.py`, `analysis/analyze.py` |
| GitHub repository | https://github.com/Xami650/Individual-Assignments-Big-Data-ULPGC (tag `assignment1-v1.0`) |
| Use of generative AI disclosed (tool, purpose, affected parts) | report §7, "Use of generative AI" |
