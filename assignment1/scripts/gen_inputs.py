"""Generate the shared, seeded input matrices used by all three implementations.

For every size n the matrices are drawn from
    numpy.random.Generator(PCG64(SeedSequence([seed, n])))
as uniform values on [-1, 1): first A (n*n values, row-major), then B.
They are written as raw little-endian float64, row-major, without header,
to data/inputs/A_<n>.bin and data/inputs/B_<n>.bin.

NumPy does not promise identical Generator streams across versions, so the
SHA-256 of every file is tracked in data/checksums.sha256 and checked on each
regeneration. The file can also be checked with:
    cd data && shasum -a 256 -c checksums.sha256

Usage (from assignment1/):
    python3 scripts/gen_inputs.py                     # all sizes in config/bench.json
    python3 scripts/gen_inputs.py --sizes 64 128      # a subset
    python3 scripts/gen_inputs.py --check             # verify existing files only
    python3 scripts/gen_inputs.py --write-checksums   # (re)create the checksum file
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "bench.json"
DATA = ROOT / "data"
INPUTS = DATA / "inputs"
CHECKSUMS = DATA / "checksums.sha256"


def generate(n, seed):
    """Return the (A, B) pair for size n as C-ordered float64 arrays."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, n])))
    a = rng.uniform(-1.0, 1.0, size=(n, n))
    b = rng.uniform(-1.0, 1.0, size=(n, n))
    return a, b


def input_paths(n):
    return INPUTS / f"A_{n}.bin", INPUTS / f"B_{n}.bin"


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


def write_checksums(sums):
    lines = [f"{sums[name]}  {name}" for name in sorted(sums, key=_sort_key)]
    CHECKSUMS.write_text("\n".join(lines) + "\n")


def _sort_key(name):
    # inputs/A_64.bin -> (64, "A")
    stem = Path(name).stem
    matrix, n = stem.split("_")
    return int(n), matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", help="sizes to process (default: all sizes in the config)")
    parser.add_argument("--check", action="store_true", help="only verify existing files against the checksums")
    parser.add_argument("--write-checksums", action="store_true", help="record the digests of the processed files")
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text())
    seed = config["inputs"]["seed"]
    sizes = args.sizes or sorted(set(config["sizes"]["correctness"]) | set(config["sizes"]["benchmark"]))

    INPUTS.mkdir(parents=True, exist_ok=True)
    known = read_checksums()
    computed = {}
    for n in sizes:
        if not args.check:
            for path, matrix in zip(input_paths(n), generate(n, seed)):
                matrix.astype("<f8", copy=False).tofile(path)
        for path in input_paths(n):
            computed[f"inputs/{path.name}"] = sha256(path)

    if args.write_checksums:
        write_checksums({**known, **computed})
        print(f"wrote {len(computed)} digests to {CHECKSUMS.relative_to(ROOT)}")
        return 0

    mismatches = [name for name, digest in computed.items() if known.get(name) not in (None, digest)]
    missing = [name for name in computed if name not in known]
    for name in mismatches:
        print(f"MISMATCH {name}", file=sys.stderr)
    for name in missing:
        print(f"no recorded checksum for {name}", file=sys.stderr)
    action = "checked" if args.check else "generated"
    print(f"{action} {len(computed)} files for n in {sizes} (seed {seed}, numpy {np.__version__}); "
          f"{len(mismatches)} mismatches, {len(missing)} without checksum")
    return 1 if mismatches or (args.check and missing) else 0


if __name__ == "__main__":
    sys.exit(main())
