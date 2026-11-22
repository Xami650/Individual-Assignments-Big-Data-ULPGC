# Assignment 4 — Distributed Matrix Multiplication with Hazelcast

BIG DATA · GCID · ULPGC · 2026-2027 — Máximo Prosperi Reynoso

Empirical study of how a distributed matrix multiplication scales with the
number of workers, and of what the distribution costs. The rows of A are
split into w bands; each band, a copy of B and the band of C live on one
member of a [Hazelcast](https://hazelcast.com) 5.7 cluster, which computes its
band with the validated i-k-j row kernel of Assignment 3. The driver, a
Hazelcast client, times every phase of a multiplication (serialize, scatter A,
broadcast B, compute, gather C), counts the bytes it moves, and is compared
with the sequential `basic` and `ikj` and the shared-memory parallel `par_ikj`
on the same row bands. Every configuration computes in IEEE-754 double and
produces bitwise-identical results.

> **Local execution.** All processes run on **one machine** (an Apple M4
> MacBook Air): w member JVMs and one driver JVM, talking TCP over the
> loopback interface. This measures the framework, serialization, copies and
> separate processes; it does **not** measure a physical network, and the
> memory of all members is the memory of one machine. The course guide allows
> local execution only with the instructor's permission.

## Results at a glance

Session `20261120-031850` (130 configurations, all ok; full analysis in
[`report/main.pdf`](report/main.pdf)):

| | n = 256 | n = 512 | n = 1024 | n = 2048 | n = 4096 |
|---|---|---|---|---|---|
| ikj (ms) | 2.21 | 17.4 | 140 | 1 168 | 15 264 |
| best par_ikj static (ms) | 0.566 (p = 8) | 3.95 (p = 10) | 28.8 (p = 10) | 326 (p = 8) | 3 849 (p = 8) |
| best dist (ms) | 2.57 (w = 3) | 11.7 (w = 4) | 90.4 (w = 2) | 862 (w = 10) | 6 196 (w = 10) |
| best dist speedup over ikj | 0.86 | 1.49 | 1.54 | 1.35 | 2.46 |
| communication share, w = 10 | 57 % | 52 % | 27 % | 11 % | 4.9 % |

- Every configuration is **bitwise identical** to `ikj`, and the bytes moved
  are the model (w + 2)·8n² within 0.05 %.
- dist is never faster than `par_ikj` on the same bands (1.03 to 7× its time).
  Communication dominates only for n ≤ 512; from n = 1024 on, the members
  compute more slowly than threads because each of the w processes reads its own
  copy of B (up to 2.6× the cycles per multiply-add).
- Ten cores as one member with ten threads beat ten members of one thread
  (395 vs 862 ms at n = 2048). In weak scaling dist stays 1.7 to 2.3× slower
  than `par_ikj` with under 5 % of its time in communication; the efficiency of
  both falls to about 0.3 at eight workers on one chip.
- Forming a cluster takes about 10 s (0.6 s with one member); the memory of the
  cluster grows by about 0.9 GiB per member at n = 4096.
- Two earlier sessions were discarded (sleep on battery; macOS indexing during
  the run); see [`results/discarded/`](results/discarded/README.md).

## Repository layout

```
assignment4/
├── pom.xml                      Maven build (standalone, Java 25, Hazelcast 5.7.0, JUnit 5)
├── src/main/java/org/ulpgc/bd/matmul/
│   ├── Main.java                benchmark driver (one configuration per process; `member` starts a member)
│   ├── MemberMain.java          one cluster member (a separate JVM)
│   ├── kernel/                  the computation inside one process
│   │   ├── DenseMatMul.java     basic and ikj, the sequential methods (Assignments 1-3, unchanged)
│   │   ├── ParallelMatMul.java  par_ikj, row-parallel ikj on a thread pool (Assignment 3)
│   │   └── BandKernel.java      a member's kernel: ikj rows of its band on k threads
│   ├── dist/                    the distributed method
│   │   ├── DistributedMatMul.java  dist, driver side: placement, the five phases, bytes
│   │   ├── HazelcastSetup.java     the whole Hazelcast configuration of members and client
│   │   └── protocol/            what travels between processes (IdentifiedDataSerializable)
│   │       ├── BandTask.java    the task that computes one band on its member
│   │       └── RowBand.java, BandKey.java, TaskReport.java, PingTask.java, MatMulSerialization.java
│   ├── io/MatrixIO.java         dense file format of Assignments 1-3
│   └── metrics/ProcessCounters.java  CPU time, P-core time, instructions, cycles (FFM, macOS; Assignment 3)
├── src/test/java/...            JUnit 5 tests (68) in the same packages, including embedded multi-member clusters
├── config/bench.json            declared experimental protocol and cluster settings (single source of truth)
├── scripts/
│   ├── gen_inputs.py            seeded inputs (Assignment 1 generator) + SHA-256 check
│   ├── verify.py                correctness of every method, dist on real multi-process clusters
│   ├── cluster.py               starts, watches and stops the member processes
│   ├── run_benchmarks.py        benchmark harness (size blocks, setup, drift controls, cool-down, budgets)
│   ├── collect_env.sh           environment snapshot
│   └── common.py                shared helpers (paths, command lines, heap rules, error bound)
├── data/
│   ├── checksums.sha256         digests of the generated inputs (shared sizes identical to Assignment 3's)
│   └── inputs/                  generated inputs (not tracked)
├── results/
│   ├── correctness.csv          output of verify.py
│   ├── raw/20261120-031850/     raw benchmark data of the reported session (console.log included)
│   └── discarded/               two earlier sessions that were not used, with the reasons
├── analysis/
│   ├── analyze.py               summary statistics, tables and figures
│   ├── tables/                  CSV summaries and LaTeX tables used by the report
│   └── figures/                 figures (PDF for the report, PNG for viewing)
├── report/                      main.tex (LaTeX source of the research report) and main.pdf
├── CHECKLIST.md                 each course requirement mapped to where it is met
└── requirements.txt             Python packages for the scripts and the analysis
```

## Requirements

The results were produced with exactly this software, on this machine:

| Component | Version used |
|---|---|
| Machine | MacBook Air 15-inch, M4, 2025 (Mac16,13, fanless): 4 performance + 6 efficiency cores, 10 logical processors, 24 GB RAM, 120 GB/s memory bandwidth |
| Operating system | macOS 27.2.1 (build 26B131) |
| Java | OpenJDK 25.0.1 (build 25.0.1+8-27, HotSpot), pinned by path in `config/bench.json`; Apache Maven 3.9.12 |
| Framework | Hazelcast Platform 5.7.0, open-source edition (`com.hazelcast:hazelcast:5.7.0`, Apache License 2.0), downloaded by Maven |
| Python | CPython 3.13.9 (Anaconda build) |
| Python packages | numpy 2.3.5, pandas 2.3.3, matplotlib 3.10.6, psutil 7.0.0 (`requirements.txt`) |

```sh
python3 -m pip install -r requirements.txt
```

The JDK path is set in `config/bench.json` (`build_and_runtime.java_home`);
change it there if your JDK 25 lives elsewhere. The process counters, the
harness and the environment snapshot rely on macOS (`proc_pid_rusage`,
`/usr/bin/time -l`, `sysctl`, `pmset`); on another system the driver still
runs, with empty counter fields. Hazelcast is configured not to send usage
statistics ("phone home" off) and binds only to 127.0.0.1.

## Reproducing the study

Run every command from `assignment4/`, with `JAVA_HOME` pointing to a JDK 25.

```sh
cd assignment4
export JAVA_HOME=~/Library/Java/JavaVirtualMachines/openjdk-25.0.1/Contents/Home

# 1. Inputs: a few seconds, about 1.6 GB in data/inputs/, checked against data/checksums.sha256
python3 scripts/gen_inputs.py

# 2. Unit tests (68, about 1 min: they start embedded clusters) and build of
#    target/matmul.jar and target/lib/hazelcast-5.7.0.jar
mvn package

# 3. Correctness on real multi-process clusters: about 1.5 min, writes results/correctness.csv
python3 scripts/verify.py

# 4. Benchmarks: about 2 h, writes results/raw/<YYYYMMDD-HHMMSS>/
#    Run from Terminal with other applications closed, the Mac plugged in and the lid open (up to
#    11 JVMs use every core and up to 17 GB of memory). The harness refuses to start on battery,
#    stops if the Mac sleeps, and waits before every configuration until the machine is quiet
#    (< 0.3 cores); `pmset displaysleepnow` avoids the CPU of an animated lock screen.
caffeinate -ims python3 scripts/run_benchmarks.py

# 5. Tables and figures from the most recent session
python3 analysis/analyze.py

# 6. Report (needs tectonic)
cd report && tectonic -X compile main.tex
```

What each step does:

1. **`gen_inputs.py`** writes the inputs with the Assignment 1 generator,
   unchanged (`numpy.random.Generator(PCG64(SeedSequence([2026, n])))`, uniform
   on [-1, 1), raw little-endian float64). The 22 files of sizes used before
   have the same SHA-256 as Assignment 3's; 2580, 3252 and 8192 are new.
2. **The unit tests** check the reused kernels (hand-computed products,
   identity and zero, an exact BigDecimal reference, basic = ikj = par_ikj bit
   for bit), the band kernel, the serialized formats, and dist on embedded
   clusters of 1 to 4 members (bitwise against ikj for n up to 100, w > n
   included; placement; bytes; repeated calls; errors of a member), plus the
   driver contract with a real client.
3. **`verify.py`** runs every method through its command line, and dist on
   five real clusters of separate member processes (w × k = 1×1, 2×1, 3×1,
   4×2, 10×1), for n = 1 … 256 with random inputs (reference `numpy.matmul`),
   A·I and A·0. The criterion is |C − R| ≤ 2γₙ(|A||B|) elementwise, as in
   Assignments 1-3, and all configurations of a case must agree bit for bit.
4. **`run_benchmarks.py`** builds the jar and runs every configuration in
   blocks of ascending n. For dist it starts a fresh cluster of w member JVMs
   (new cluster name and ports), waits until every member is ready, measures
   2 s of idle CPU, runs the driver as a Hazelcast client and terminates the
   members. Every process runs under `/usr/bin/time -l`. Each D_strong block
   starts and ends with `ikj` and with dist on 10 members (drift controls); the
   rest runs in a seeded random order, with a cool-down pause after each
   configuration and a wait for a quiet machine before the next. `--sizes`, `--match`, `--methods`, `--reps`, `--pause`,
   `--dry-run` and `--session smoke…`/`pilot…` give quick checks (smoke and
   pilot sessions are not tracked).
5. **`analyze.py`** summarises only the measured samples (warm-ups excluded)
   and writes `analysis/tables/` and `analysis/figures/`.

## How the distributed multiplication works

| Step | What happens | Hazelcast mechanism |
|---|---|---|
| Placement (once) | band t is assigned to member t; the driver finds an integer whose partition member t owns and uses it as band t's partition key | `PartitionAware` key, client partition table |
| serialize | the driver serializes the w bands of A and, once, B | `IdentifiedDataSerializable` |
| scatter | band t of A is stored on member t | `IMap.setAll`, backup count 0 |
| broadcast | the same serialized B is stored once on every member | `IMap.setAll` with w keys |
| compute | each member takes its band and its B, runs the ikj row kernel on k threads, stores its band of C locally | `IExecutorService.submitToKeyOwner` |
| gather | the w bands of C are fetched and placed into C | `IMap.getAll` |

The phases run one after the other, so their times add up to the end-to-end
time (checked in the analysis). Setup (starting the members and the client) is
measured separately and is not part of a multiplication. The configuration of
the cluster is in `HazelcastSetup.java` and summarised in `config/bench.json`
(`cluster`): TCP/IP join on 127.0.0.1 with a fixed address list, a new cluster
name per configuration, 271 partitions, no backups, one executor thread per
member, metrics and phone-home off.

## Running one configuration by hand

```sh
J=$JAVA_HOME/bin/java
# a cluster of two members; a member runs until its standard input closes, so give each one
# a pipe that stays open (here for 10 minutes) or run it in its own terminal
sleep 600 | $J -Xmx1g -jar target/matmul.jar member --cluster demo --index 0 --members 2 --threads 1 &
sleep 600 | $J -Xmx1g -jar target/matmul.jar member --cluster demo --index 1 --members 2 --threads 1 &
# the driver, as a client of that cluster
$J -Xmx2g -jar target/matmul.jar --method dist --workers 2 --cluster demo --n 1024 \
   --a data/inputs/A_1024.bin --b data/inputs/B_1024.bin --member-log members.csv --ping-rounds 50
# the in-process methods
$J -Xmx2g -jar target/matmul.jar --method ikj --n 1024 --a data/inputs/A_1024.bin --b data/inputs/B_1024.bin
$J -Xmx2g -jar target/matmul.jar --method par_ikj --threads 4 --schedule static --n 1024 \
   --a data/inputs/A_1024.bin --b data/inputs/B_1024.bin
```

| Option | Meaning (default) |
|---|---|
| `--method M` | `basic`, `ikj`, `par_ikj` or `dist` (required) |
| `--n N`, `--a PATH`, `--b PATH` | matrix side length and input files (required) |
| `--threads P`, `--schedule S` | par_ikj: worker threads (1) and `dynamic` or `static` rows (`dynamic`) |
| `--workers W`, `--cluster NAME` | dist: number of members to wait for and the cluster name (required for dist) |
| `--threads-per-worker K` | dist: compute threads per member, as the members were started (1; informational) |
| `--base-port PORT` | dist: member m listens on PORT + m (5901) |
| `--member-log PATH` | dist: one row per member and sample with the task's timers and counters |
| `--ping-rounds R` | dist: after the samples, time R empty tasks per member (0) |
| `--pre-warm-seconds`, `--warmup`, `--min-warmup-seconds`, `--reps`, `--min-sample-seconds`, `--out` | as in Assignment 3: pre-warm (1.0 s), warm-up (≥ 1 sample, ≥ 1 s), measured samples (5), batching (0.1 s), write C |

`member` options: `--cluster`, `--index m`, `--members w`, `--threads k`,
`--base-port`, `--pre-warm-seconds`. A member prints its life cycle as
`@ started|joined|ready|stopping key=value,...` lines and terminates when its
standard input closes.

The driver prints CSV on stdout: the time of each sample, a checksum of C, the
driver's process counters, and for dist the phase timers (summed over the
batch), the slowest and mean task and kernel, dispatch and completion times,
the exact payload of A, B and C and the bytes counted on the client's
connections. Lines starting with `#` carry metadata (connection and placement
time, partition keys, thread counts, GC/JIT activity, ping statistics, map
sizes). Exit status 3 means the driver ran out of heap; 4 means the cluster
failed.

## Experimental protocol

The full protocol is in [`config/bench.json`](config/bench.json): research
question, hypotheses H1–H9, terminology, methods, cluster settings, inputs,
hardware, experiments, session order, statistics, timer boundary, setup,
transfer, budgets, memory metrics and correctness tolerance. In short:

| Item | Setting |
|---|---|
| Experiments | D_strong (n = 256 … 4096; dist w = 1 … 10, par_ikj static p = w, dynamic p = 4 and 10, ikj, basic ≤ 2048), D_grain (10 cores as 1×10, 2×5, 5×2, 10×1; n = 2048, 4096), D_weak (n³/w = 2048³: (1, 2048), (2, 2580), (4, 3252), (8, 4096)), D_large (n = 8192: ikj, par_ikj p = 4 and 10, dist w = 4; 3 samples) |
| Terminology | worker = Hazelcast member = 1 JVM process; physical machine = 1; driver = a client process that stores no cluster data |
| Timer boundary | from A and B in the driver's heap to C complete there; setup separately |
| Session order | size blocks in ascending n; `ikj` and dist w = 10 first and last in each D_strong block; the rest in a seeded random order |
| JVMs | JDK 25.0.1; driver `-Xmx` = 1 GiB + 64n² B; member `-Xmx` = 512 MiB + (3 + 4/w)·8n² B, `-XX:+ExitOnOutOfMemoryError` |
| Measurement | pre-warm 1 s (driver and every member), warm-up ≥ 1 s, 5 samples of ≥ 0.1 s; median and IQR |
| Cool-down | pause after each configuration as long as the driver ran, 5–90 s; then wait until the machine uses < 0.3 cores over 5 s (at most 15 min) |
| Machine state | tracked sessions refuse battery power and Low Power Mode, and stop if the system sleeps |
| Budgets | 120 s per operation (300 s at n = 8192), 18 GB total peak footprint, 150 min per session; failures are results |

## Data and outputs

| Path | Content |
|---|---|
| `data/checksums.sha256` | SHA-256 of all input files |
| `results/correctness.csv` | one row per (case, configuration): max error, error/bound ratio, pass, bitwise agreement |
| `results/raw/20261120-031850/samples.csv` | every driver sample of the reported session, warm-ups included, with phases and bytes |
| `results/raw/20261120-031850/member_samples.csv` | every member's task timers and kernel counters per sample |
| `results/raw/20261120-031850/runs.csv` | one row per configuration: status, setup, driver metadata and resources, members' totals, CPU of other processes, exact command |
| `results/raw/20261120-031850/members.csv` | one row per member process: events, partitions, threads, idle CPU, peak RSS and footprint |
| `results/raw/20261120-031850/env.txt`, `config.json` | environment before and after the session; the protocol used |
| `analysis/tables/summary.csv`, `members.csv` | all summary statistics per configuration and per member |
| `analysis/tables/*.tex`, `analysis/figures/*.pdf` | the tables and figures of the report |

When reading `samples.csv` yourself, parse the `checksum` column with Python's
`float()` or pass `float_precision="round_trip"` to pandas.

## Reuse of Assignments 1 to 3

`basic` (Assignment 1, kept unchanged in Assignment 2), `ikj`, `par_ikj`, the
process counters, the dense file format, the input generator and its files,
the correctness tolerance, and the structure of the driver, the harness, the
verification script, the environment snapshot and the analysis come from
Assignment 3 at the tag `assignment3-v1.0`. Each reused file says so in its
header. The kernels' code is unchanged; Assignment 4 only puts the classes in
sub-packages (`kernel`, `io`, `metrics`) and makes public what another package
calls.
