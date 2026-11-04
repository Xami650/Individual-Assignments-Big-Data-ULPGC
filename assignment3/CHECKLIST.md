# Submission checklist — Assignment 3

This file maps each requirement of the course guide (Assignment 3 page and the
common pages) to where it is met. "§" refers to sections of the report
([`report/main.pdf`](report/main.pdf)).

## Implementations and comparisons

| Requirement | Where it is met |
|---|---|
| Equivalent sequential baseline and a parallel implementation with threads; the comparison is mandatory | `ikj` (Assignment 2 kernel, copied unchanged) and `par_ikj` (fixed `ThreadPoolExecutor` of p workers), `DenseMatMul.java`, `ParallelMatMul.java`; §3.1, Fig. 1; every speedup is against `ikj` (§3.7) |
| Same algorithm, numeric type, storage layout and inputs; differences documented | same i-k-j loop per row (`ikjRow`), double, `double[][]`, same input files (§3.1, §3.2); differences: the hand-off to the pool is inside the timer (§3.7), `ikjRow` is compiled separately from `ikj` (§5, H2) |
| Several thread counts, including one | p = 1 … 10, 12, 16, 20 for every n (Table 1); p = 1 in Tables 3–5 |
| Physical cores, logical processors and affinity recorded | Table 2 (4 P + 6 E cores, 10 logical, no SMT; affinity not settable: `KERN_NOT_SUPPORTED`); `env.txt`; the P-core share of CPU time is measured instead (§3.5, §4.5, Fig. 4b) |
| Oversubscription identified | p = 12, 16, 20 marked † in every table and shaded in every figure; §4.7 |
| Common timing and repetition protocol | §3.7 (timer boundary, pre-warm, warm-up, 5 samples, batching, median and IQR, budgets) |
| Correctness and safe access to shared data | §2 (Java memory model, `java.util.concurrent` happens-before), §3.1 "Shared data", §3.6 (32 JUnit tests incl. 200 repeated calls on one pool, `verify.py`), §4.2 (all configurations bitwise identical, 835 checksums) |
| Internal library threads controlled so that p describes the configuration | no library in the measured code; the main thread only waits, so p = compute threads (§3.1); Java and OS thread counts of every process (Table 2, `runs.csv`); GC and JIT activity during the measured samples (§4.1) |

## Metrics

| Requirement | Where it is met |
|---|---|
| S(p) = T_seq / T_p and E(p) = S / p, with the summary statistic stated | medians of the five measured samples, T_seq from the first `ikj` run of the size block, interval [Q1/Q3, Q3/Q1] (§3.7); Table 4, Figs. 2–3 |
| Runtime and variability | Table 3 (with the largest relative IQR per size), §4.1; every sample in `samples.csv` |
| Memory, with a named metric and tool | peak RSS and peak footprint (`/usr/bin/time -l`), Java heap allocation and element storage 24n² (§3.5; Table 8; §4.8) |
| Parallel implementation with one thread (overhead) | Table 5 and §4.4 (dynamic and static, p = 1) |
| Unusual or superlinear speedups explained with evidence | no superlinear speedup (largest efficiency 0.86 for p ≥ 2); unusual results examined with counters: the n = 128 baseline (§4.1), the speedup decomposition and the memory-bandwidth limit (§4.5, Table 6), static faster than dynamic (§4.6, §5 H4), oversubscription at n = 1024 (§4.7), memory at small n (§4.8) |

## Optional vectorisation extension

| Requirement | Where it is met |
|---|---|
| Compiler options or explicit instructions described | HotSpot C2 SuperWord, on by default, off with `-XX:-UseSuperWord`; 128-bit NEON, no FMA contraction in Java (§2, §3.8) |
| Evidence that vectorisation occurs | decoded C2 machine code (`scripts/simd_evidence.py`, `results/simd/`): 8 `fmul.2d` + 8 `fadd.2d` with SuperWord, scalar only without (Table 9, §4.9) |
| Vectorised and non-vectorised versions compared under controlled conditions | experiment V_simd: same jar, inputs, n and p, one flag changed (Table 1, Table 10, Fig. 6) |
| Vectorisation effects separated from thread count | comparisons at fixed p = 1, 4 and 10 for n = 512, 1024, 2048 (Table 10, §5 H8) |

## Common protocol

| Requirement | Where it is met |
|---|---|
| Question, factor and outcome stated; hypotheses kept apart from results | §1; hypotheses H1–H8 fixed in `config/bench.json` before the session, tested in §5 |
| Same logical inputs; seed, generator, range, dimensions and type recorded | §3.2; `data/checksums.sha256` |
| Hardware, OS, versions, flags, runtime settings and thread counts documented | Table 2; `env.txt` |
| Timer boundary stated; monotonic clock; result consumed | §3.7; checksum printed after every sample |
| Warm-up used, excluded from summaries and its count recorded | §3.7 (pre-warm and warm-up); `kind = warmup` rows in `samples.csv` |
| At least five repetitions; short tasks batched with a time per operation | §3.7; `batch` column in `samples.csv` |
| Every raw measurement kept; median and IQR with units and repetitions | `results/raw/20261104-030623/`; §4 |
| Small known cases (identity, zero), a trusted reference and a justified tolerance | §3.6 |
| Time and memory budgets declared before the large tests; failures reported without invented times | `config/bench.json` (`budgets`); §3.7 (no budget was reached; the largest size attempted, 4096, completed for every configuration) |
| Configurations labelled with size, method, numeric type, repetition and thread count | columns of `samples.csv` and `runs.csv` (experiment, method, n, threads, schedule, simd, repeat) |

## Report and submission package

| Requirement | Where it is met |
|---|---|
| Report structure: title and abstract, introduction, background, methodology, results, discussion, conclusions, references and materials | `report/main.tex`, §1–§7 |
| English; labelled figures and tables; consistent IEEE references, each one checked | report |
| PDF with name, assignment number and research title | `Assignment03_MaximoProsperiReynoso_Report.pdf` (copy of `report/main.pdf`) |
| Source code, README, dependencies, environment details, exact build and run commands | [`README.md`](README.md), `requirements.txt`, `pom.xml` |
| Thread configurations and measurements | `config/bench.json`, `results/raw/20261104-030623/` (`runs.csv` has the exact command of every process) |
| Raw results, input generation instructions, scripts reproducing the analysis and figures | `results/`, `scripts/gen_inputs.py`, `analysis/analyze.py` |
| Reused materials identified with their exact version | Assignment 2 at tag `assignment2-v1.0` (§7, README "Reuse of Assignments 1 and 2") |
| GitHub repository | https://github.com/Xami650/Individual-Assignments-Big-Data-ULPGC (tag `assignment3-v1.0`) |
| Use of generative AI disclosed (tool, purpose, affected parts) | report §7, "Use of generative AI" |
