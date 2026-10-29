# Submission checklist — Assignment 2

This file maps each requirement of the course guide (Assignment 2 page and the
common pages) to where it is met. "§" refers to sections of the report
([`report/main.pdf`](report/main.pdf)).

## Part A: dense optimisation

| Requirement | Where it is met |
|---|---|
| One validated basic implementation as the baseline | `basic` = the Assignment 1 Java kernel, copied unchanged (tag `assignment1-v1.0`); validated there and again here (§3.1, §3.5) |
| At least two distinct optimised versions (≥ 3 dense methods) | `ikj` (loop reordering) and `blocked` (cache blocking), `DenseMatMul.java`; §3.1, Fig. 1 |
| Identify the actual change and explain its expected effect | §2, §3.1, Fig. 1; `config/bench.json` (`methods.*.change`, `expected_effect`); README "Methods" |
| Same input matrices, numeric type and timing boundary | same dense files, double, kernel-only timer for all three (§3.2, §3.6) |
| Sequential comparison (no parallel variant) | one thread for every method (§3.1, Table 2) |

## Part B: sparse matrices

| Requirement | Where it is met |
|---|---|
| Implement a sparse multiplication; state the storage format; own code vs library | CSR, Gustavson's algorithm, all own code (`CsrMatrix.java`); SciPy only as a correctness reference (§3.1, §3.5) |
| Multiple declared sparsity levels; sparsity = % zeros, density = % nonzeros | densities 0.1, 0.5, 1, 5, 10, 50 % (sparsity 99.9 … 50 %), definitions in §1 |
| Input generation process, nonzero distribution and seeds recorded | §3.2; `scripts/gen_inputs.py`; `config/bench.json` (`inputs.sparse`); `data/checksums.sha256` |
| Compare the sparse approach with a dense method on matched logical inputs | `csr` and `blocked` read the same CSR files (§3.3, §4) |
| Conversion costs: excluded, included or measured separately | excluded from the csr kernel and measured separately (`to_csr`, `to_dense`), §3.6; both totals reported (§4.3, Table 7) |
| Input and output nonzero counts | nnz(A), nnz(B), nnz(C): Tables 7 and 9, Fig. 4b; every configuration in `runs.csv` |

## Required evidence and analysis

| Requirement | Where it is met |
|---|---|
| Time with repeated measurements | 5 measured samples, median and IQR: Tables 3, 6, 7, 9 and Figs. 2–5; `samples.csv` |
| Memory with a named metric and tool | peak RSS and peak footprint (`/usr/bin/time -l`) and element storage (§3.6; Tables 5, 8, 9; Fig. 5b) |
| Largest feasible tested size under the declared budget | §4.2 (basic 3072; ikj and blocked ≥ 4096, the largest attempted) and §4.4 (blocked 8192, csr 16384), with timeouts and the memory failure as results |
| Observed bottlenecks | §4–§5: stall cycles of basic (Table 4), memory stream of ikj at n ≥ 3072, fill-in and output size of csr, conversion cost, heap limit |
| Effects of optimisation discussed separately from sparse storage and input structure | §5 (Part A, Part B and "Optimisation versus storage") |
| Where sparse helps and where indexing, conversion or output costs reduce the advantage | §4.3–§4.4, §5 (HB1–HB6) |
| No claim that the percentage of zeros alone determines performance | uniform vs banded at equal density and equal multiply-adds (§4.3, §5 HB2) |

## Common protocol

| Requirement | Where it is met |
|---|---|
| Question, factor and outcome stated; hypotheses kept apart from results | §1; hypotheses fixed in `config/bench.json` before the session |
| Same logical inputs; seed, generator, range, dimensions and type recorded | §3.2 |
| Hardware, OS, versions, flags, runtime settings and thread counts documented | Table 2; `env.txt` |
| Timer boundary stated; monotonic clock; result consumed | §3.6; checksum printed after every sample |
| Warm-up used, excluded from summaries and its count recorded | §3.6 (pre-warm and warm-up); `kind = warmup` rows in `samples.csv` |
| At least five repetitions; short tasks batched with a time per operation | §3.6; `batch` column in `samples.csv` |
| Every raw measurement kept; median and IQR with units and repetitions | `results/raw/`; §4 |
| Small known cases (identity, zero), a trusted reference and a justified tolerance | §3.5 |
| Time and memory budgets declared before the large tests; failures reported without invented times | `config/bench.json` (`budgets`); §3.6, §4 |
| Configurations labelled with size, method, numeric type, repetition, threads, density and representation | columns of `samples.csv` and `runs.csv` |

## Report and submission package

| Requirement | Where it is met |
|---|---|
| Report structure: title and abstract, introduction, background, methodology, results, discussion, conclusions, references and materials | `report/main.tex`, §1–§7 |
| English; labelled figures and tables; consistent IEEE references, each one checked | report |
| PDF with name, assignment number and research title | `Assignment02_MaximoProsperiReynoso_Report.pdf` (copy of `report/main.pdf`) |
| Source code, README, dependencies, environment details, exact build and run commands | [`README.md`](README.md), `requirements.txt`, `pom.xml` |
| Raw results, input generation instructions, scripts reproducing the analysis and figures | `results/`, `scripts/gen_inputs.py`, `analysis/analyze.py` |
| Reused materials identified with their exact version | Assignment 1 at tag `assignment1-v1.0` (§7, README "Reuse of Assignment 1") |
| GitHub repository | https://github.com/Xami650/Individual-Assignments-Big-Data-ULPGC (tag `assignment2-v1.0`) |
| Use of generative AI disclosed (tool, purpose, affected parts) | report §7, "Use of generative AI" |
