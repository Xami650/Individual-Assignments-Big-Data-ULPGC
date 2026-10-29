"""Generate the seeded dense and sparse input matrices declared in config/bench.json.

Dense inputs (Part A) use the Assignment 1 generator unchanged, so the files for
n <= 2048 are byte-identical to Assignment 1's (same SHA-256):
    numpy.random.Generator(PCG64(SeedSequence([seed, n]))), uniform on [-1, 1),
    first A (n*n values, row-major), then B; raw little-endian float64,
    data/inputs/A_<n>.bin and B_<n>.bin.

Sparse inputs (Part B) are written in CSR (format in common.read_csr), one
generator per matrix:
    numpy.random.Generator(PCG64(SeedSequence([seed, n, structure, ppm, matrix])))
with structure 0 = uniform, 1 = banded, ppm = density in parts per million and
matrix 0 = A, 1 = B. A matrix of density d% holds exactly round(d/100 * n^2)
nonzeros at distinct positions, drawn without replacement:
    uniform  every position of the n x n matrix is equally likely;
    banded   only positions with |i - j| <= w, w = ceil(d/100 * n), are possible,
             and they are equally likely (about half of the band is filled).
The values are then drawn uniform on [-1, 1), redrawing exact zeros, in
row-major order of the positions. Files: data/inputs/sparse/<structure>_<n>_d<percent>_<A|B>.csr.

NumPy does not promise identical Generator streams across versions, so the
SHA-256 of every file is tracked in data/checksums.sha256 and checked on each
regeneration.

Usage (from assignment2/):
    python3 scripts/gen_inputs.py                     # every declared input
    python3 scripts/gen_inputs.py --only sparse       # dense | sparse
    python3 scripts/gen_inputs.py --check             # verify existing files only
    python3 scripts/gen_inputs.py --write-checksums   # (re)create the checksum file
"""

import argparse
import hashlib
import math
import sys
from pathlib import Path

import numpy as np

from common import INPUTS, ROOT, SPARSE, dense_pair, load_config, sparse_pair, write_csr

DATA = ROOT / "data"
CHECKSUMS = DATA / "checksums.sha256"
STRUCTURES = {"uniform": 0, "banded": 1}
MATRICES = {"A": 0, "B": 1}


def dense_sizes(config):
    return sorted(set(config["correctness"]["dense_sizes"]) | set(config["experiments"]["A_dense"]["sizes"]))


def sparse_specs(config):
    """Every declared (structure, n, density percent), deduplicated and sorted."""
    ex, corr = config["experiments"], config["correctness"]["sparse"]
    specs = {(s, n, p) for s in corr["structures"] for n in corr["sizes"] for p in corr["densities_percent"]}
    specs |= {(s, ex["B_density"]["n"], p)
              for s in ex["B_density"]["structures"] for p in ex["B_density"]["densities_percent"]}
    specs |= {(ex["B_size"]["structure"], n, ex["B_size"]["density_percent"]) for n in ex["B_size"]["sizes"]}
    return sorted(specs, key=lambda s: (s[1], s[2], s[0]))


def dense_inputs(n, seed):
    """The (A, B) pair for size n, exactly as in Assignment 1."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, n])))
    a = rng.uniform(-1.0, 1.0, size=(n, n))
    b = rng.uniform(-1.0, 1.0, size=(n, n))
    return a, b


def sparse_input(structure, n, percent, matrix, seed):
    """Return (rowptr, colidx, values) of one sparse matrix in canonical CSR."""
    fraction = percent / 100
    nnz = round(fraction * n * n)
    ppm = round(percent * 10_000)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [seed, n, STRUCTURES[structure], ppm, MATRICES[matrix]])))
    if structure == "uniform":
        idx = np.sort(rng.choice(n * n, size=nnz, replace=False, shuffle=False))
        rows, cols = np.divmod(idx, n)
    else:
        w = math.ceil(fraction * n)
        i = np.arange(n)
        lo, hi = np.maximum(0, i - w), np.minimum(n - 1, i + w)
        offsets = np.concatenate([[0], np.cumsum(hi - lo + 1)])
        idx = np.sort(rng.choice(int(offsets[-1]), size=nnz, replace=False, shuffle=False))
        rows = np.searchsorted(offsets, idx, side="right") - 1
        cols = lo[rows] + idx - offsets[rows]
    values = rng.uniform(-1.0, 1.0, size=nnz)
    while (zeros := values == 0.0).any():
        values[zeros] = rng.uniform(-1.0, 1.0, size=int(zeros.sum()))
    rowptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n))])
    return rowptr, cols, values


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_checksums():
    """Map 'inputs/<file>' -> hex digest from the shasum-compatible file."""
    if not CHECKSUMS.exists():
        return {}
    sums = {}
    for line in CHECKSUMS.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.strip()] = digest
    return sums


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["dense", "sparse"], help="process only one kind of input")
    parser.add_argument("--check", action="store_true", help="only verify existing files against the checksums")
    parser.add_argument("--write-checksums", action="store_true", help="record the digests of the processed files")
    args = parser.parse_args()

    config = load_config()
    dense_seed, sparse_seed = config["inputs"]["dense"]["seed"], config["inputs"]["sparse"]["seed"]
    SPARSE.mkdir(parents=True, exist_ok=True)
    paths = []
    if args.only != "sparse":
        for n in dense_sizes(config):
            pair = dense_pair(n)
            if not args.check:
                for path, matrix in zip(pair, dense_inputs(n, dense_seed)):
                    matrix.astype("<f8", copy=False).tofile(path)
            paths += pair
    if args.only != "dense":
        for structure, n, percent in sparse_specs(config):
            pair = sparse_pair(structure, n, percent)
            if not args.check:
                for path, matrix in zip(pair, MATRICES):
                    write_csr(path, n, *sparse_input(structure, n, percent, matrix, sparse_seed))
            paths += pair

    known = read_checksums()
    computed = {str(p.relative_to(DATA)): sha256(p) for p in paths}
    if args.write_checksums:
        merged = {**known, **computed}
        CHECKSUMS.write_text("".join(f"{merged[k]}  {k}\n" for k in sorted(merged)))
        print(f"wrote {len(computed)} digests to {CHECKSUMS.relative_to(ROOT)}")
        return 0

    mismatches = [k for k, v in computed.items() if known.get(k) not in (None, v)]
    missing = [k for k in computed if k not in known]
    for k in mismatches:
        print(f"MISMATCH {k}", file=sys.stderr)
    for k in missing:
        print(f"no recorded checksum for {k}", file=sys.stderr)
    action = "checked" if args.check else "generated"
    print(f"{action} {len(computed)} files (numpy {np.__version__}); "
          f"{len(mismatches)} mismatches, {len(missing)} without checksum")
    return 1 if mismatches or (args.check and missing) else 0


if __name__ == "__main__":
    sys.exit(main())
