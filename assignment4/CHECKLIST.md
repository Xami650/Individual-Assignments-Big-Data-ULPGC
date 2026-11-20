# Submission checklist — Assignment 4

This file maps each requirement of the course guide (Assignment 4 page and the
common pages) to where it is met. "§" refers to sections of the report
([`report/main.pdf`](report/main.pdf)).

## Implementation and comparisons

| Requirement | Where it is met |
|---|---|
| Matrix multiplication with a distributed computing framework | Hazelcast Platform 5.7.0: w member JVMs and a client driver (`DistributedMatMul`, `BandTask`, `MemberMain`, `HazelcastSetup`); §3.1 |
| Partitioning of the matrices explained | 1D row bands of A, B replicated, C in the same bands (§2, §3.1 "Partitioning") |
| Placement of computation explained | `PartitionAware` keys whose partitions member t owns, `submitToKeyOwner`; every task checks it ran where its band is stored (§3.1 "Placement", §4.2) |
| Collection or storage of the result explained | C bands stored in an `IMap` on their members, gathered with `getAll` and assembled in the driver (§3.1 "Storage", "One multiplication") |
| Comparison with a validated basic sequential version and the parallel version, for feasible common sizes | basic (n ≤ 2048), ikj and par_ikj (static p = w, dynamic p = 4, 10) in the same session, same inputs, same boundary (Table "time", §4.3) |
| Hardware, algorithm or timing differences that prevent an isolated comparison documented | same kernel, layout, type and inputs; differences: separate processes, serialization, loopback TCP, per-member copies of B, heterogeneous cores without affinity (§3.1, §3.4, §5 limitations) |
| Multiple worker or node configurations | w = 1, 2, 3, 4, 5, 6, 8, 10 (D_strong); 1×10, 2×5, 5×2, 10×1 (D_grain); weak-scaling points (D_weak) |
| Workers, processes and physical machines distinguished | terminology table (§3.4); `config/bench.json` (`terminology`) |
| Sufficiently large feasible inputs | n up to 4096 for every configuration, 8192 for the largest feasible configurations (D_large, §4.11) |

## Scaling and overhead

| Requirement | Where it is met |
|---|---|
| Strong scaling with a fixed problem size | D_strong (§4.3) |
| Weak scaling with the workload increase explained; not n per worker fixed | D_weak: n_w³/w = 2048³ within 0.1 %, n_w = 2048 w^(1/3) rounded to a multiple of w (§2, §3.3, §4.10) |
| End-to-end time with an explicit boundary | from A and B in the driver's heap to C complete there (§3.5) |
| Computation, setup, transfer and result-collection costs | phases serialize, scatter, broadcast, compute, gather (§4.4), compute split into dispatch, kernel and return (§4.5), setup measured separately (§4.7) |
| Overlapping activities explained, not added | consecutive phases separated by barriers; inside the compute phase only the slowest and mean member are reported (§3.5) |
| Resource counts | w, k, processes and threads per configuration (`runs.csv`, `members.csv`, §3.4, §4.7) |
| Memory per worker | peak RSS and footprint of every member and of the driver (`/usr/bin/time -l`), against a model (§4.8) |
| Data-transfer measurements | exact serialized payload and bytes counted on the client's connections, per phase (§4.6) |
| Tools used | Hazelcast's serialization service and networking counters, `/usr/bin/time -l`, `proc_pid_rusage`, psutil (§3.6) |
| Network costs that cannot be isolated stated as a limitation | loopback only: transfer = serialization + copies + local TCP; no network figure is inferred (§1, §3.6, §5) |

## Local execution limitation

| Requirement | Where it is met |
|---|---|
| Local execution only with the instructor's permission | permission requested by the author for the only available machine (one laptop); the report labels the environment |
| Environment clearly labelled | title page, §1 "Execution environment", Table "environment", README banner |
| What local execution cannot show | no inter-machine network performance, no multi-machine memory capacity (§1, §5) |

## Common protocol

| Requirement | Where it is met |
|---|---|
| Question, factor and outcome stated; hypotheses kept apart from results | §1; hypotheses H1–H9 fixed in `config/bench.json` before the session, tested in §5 |
| Same logical inputs; seed, generator, range, dimensions and type recorded | §3.2; `data/checksums.sha256` |
| Hardware, OS, versions, flags, runtime settings and thread counts documented | Table "environment", cluster configuration, `env.txt` |
| Timer boundary stated; monotonic clock; result consumed | §3.5; checksum printed after every sample |
| Warm-up used, excluded from summaries and its count recorded | §3.8; `kind = warmup` rows in `samples.csv` |
| At least five repetitions; fewer justified; short tasks batched | §3.8 (5 samples; 3 at n = 8192 with the range reported); `batch` column |
| Every raw measurement kept; median and IQR with units and repetitions | `results/raw/20261120-031850/`; §4 |
| Small known cases (identity, zero), a trusted reference and a justified tolerance | §3.7, `results/correctness.csv` |
| Time and memory budgets declared before the large tests; failures reported without invented times | `config/bench.json` (`budgets`); §3.8; the n = 8192, w = 10 attempt of the pilot (§3.8, §4.11) |
| Configurations labelled with size, method, numeric type, repetition, threads and workers | columns of `samples.csv`, `runs.csv`, `members.csv` |

## Report and submission package

| Requirement | Where it is met |
|---|---|
| Report structure: title and abstract, introduction, background, methodology, results, discussion, conclusions, references and materials | `report/main.tex`, §1–§7 |
| English; labelled figures and tables; consistent IEEE references, each one checked | report |
| PDF with name, assignment number and research title | `Assignment04_MaximoProsperiReynoso_Report.pdf` (copy of `report/main.pdf`) |
| Source code, README, dependencies, environment details, exact build and run commands | [`README.md`](README.md), `requirements.txt`, `pom.xml` |
| Framework configuration | `HazelcastSetup.java`, `config/bench.json` (`cluster`), README "How the distributed multiplication works" |
| Raw results, input generation instructions, scripts reproducing the analysis and figures | `results/`, `scripts/gen_inputs.py`, `analysis/analyze.py` |
| Reused materials identified with their exact version | Assignment 3 at tag `assignment3-v1.0` (§7, README "Reuse") |
| GitHub repository | https://github.com/Xami650/Individual-Assignments-Big-Data-ULPGC (tag `assignment4-v1.0`) |
| Use of generative AI disclosed (tool, purpose, affected parts) | report §7, "Use of generative AI" |
