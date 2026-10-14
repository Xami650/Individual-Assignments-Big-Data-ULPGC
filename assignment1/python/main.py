"""Benchmark driver for the pure-Python basic matrix multiplication.

Common command-line contract shared with the Java and C drivers:

    python3 -u python/main.py --n N [--a PATH] [--b PATH] [--warmup W] [--reps R]
                              [--min-sample-seconds S] [--out PATH]

Inputs default to data/inputs/A_<n>.bin and data/inputs/B_<n>.bin. Output on
stdout is CSV with the header

    kind,index,batch,elapsed_ns,checksum

and one row per sample. kind is "warmup" or "measure". Each sample times `batch`
back-to-back multiplications with a monotonic clock (kernel only: loading and
allocation are outside the timed region). Warm-up samples use batch = 1; the
measured batch is k = max(1, ceil(S / t_last_warmup)). checksum is the sum of
all entries of C, printed with enough digits to round-trip the double.
With --out, C is written as raw little-endian float64 after the last sample.
"""

import argparse
import math
import sys
import time
from pathlib import Path

from matmul import checksum, load_matrix, multiply, write_matrix, zeros

INPUTS = Path(__file__).resolve().parent.parent / "data" / "inputs"


def parse_args():
    p = argparse.ArgumentParser(description="Basic triple-loop matrix multiplication (Python)")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--a", type=Path)
    p.add_argument("--b", type=Path)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--min-sample-seconds", type=float, default=0.1)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.n < 1 or args.warmup < 0 or args.reps < 0 or args.warmup + args.reps < 1:
        p.error("need n >= 1 and at least one warm-up or measured sample")
    args.a = args.a or INPUTS / f"A_{args.n}.bin"
    args.b = args.b or INPUTS / f"B_{args.n}.bin"
    return args


def run_sample(a, b, c, n, batch):
    t0 = time.perf_counter_ns()
    for _ in range(batch):
        multiply(a, b, c, n)
    return time.perf_counter_ns() - t0


def main():
    args = parse_args()
    n = args.n
    a = load_matrix(args.a, n)
    b = load_matrix(args.b, n)
    c = zeros(n)

    print("kind,index,batch,elapsed_ns,checksum", flush=True)
    last_warmup_ns = None
    for i in range(args.warmup):
        last_warmup_ns = run_sample(a, b, c, n, 1)
        print(f"warmup,{i},1,{last_warmup_ns},{checksum(c)!r}", flush=True)

    batch = 1
    if last_warmup_ns:
        batch = max(1, math.ceil(args.min_sample_seconds * 1e9 / last_warmup_ns))
    for i in range(args.reps):
        elapsed = run_sample(a, b, c, n, batch)
        print(f"measure,{i},{batch},{elapsed},{checksum(c)!r}", flush=True)

    if args.out:
        write_matrix(args.out, c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
