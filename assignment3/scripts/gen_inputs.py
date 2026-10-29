"""Generate the seeded dense input matrices declared in config/bench.json.

The dense part of Assignment 2's scripts/gen_inputs.py (tag assignment2-v1.0),
which itself uses the Assignment 1 generator unchanged:
    numpy.random.Generator(PCG64(SeedSequence([seed, n]))), uniform on [-1, 1),
    first A (n*n values, row-major), then B; raw little-endian float64,
    data/inputs/A_<n>.bin and B_<n>.bin.
Every file is therefore byte-identical to the Assignment 2 file of the same
size (same SHA-256).

NumPy does not promise identical Generator streams across versions, so the
SHA-256 of every file is tracked in data/checksums.sha256 and checked on each
regeneration.

Usage (from assignment3/):
    python3 scripts/gen_inputs.py                     # every declared input
    python3 scripts/gen_inputs.py --check             # verify existing files only
    python3 scripts/gen_inputs.py --write-checksums   # (re)create the checksum file
"""

import argparse
import hashlib
import sys

import numpy as np

from common import INPUTS, ROOT, dense_pair, load_config

DATA = ROOT / "data"
CHECKSUMS = DATA / "checksums.sha256"


def sizes(config):
    """Every n used by the correctness checks or an experiment."""
    used = set(config["correctness"]["sizes"])
    for experiment in config["experiments"].values():
        used |= set(experiment["sizes"])
    return sorted(used)


def dense_inputs(n, seed):
    """The (A, B) pair for size n, exactly as in Assignments 1 and 2."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, n])))
    a = rng.uniform(-1.0, 1.0, size=(n, n))
    b = rng.uniform(-1.0, 1.0, size=(n, n))
    return a, b


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
    parser.add_argument("--check", action="store_true", help="only verify existing files against the checksums")
    parser.add_argument("--write-checksums", action="store_true", help="record the digests of the processed files")
    args = parser.parse_args()

    config = load_config()
    seed = config["inputs"]["seed"]
    INPUTS.mkdir(parents=True, exist_ok=True)
    paths = []
    for n in sizes(config):
        pair = dense_pair(n)
        if not args.check:
            for path, matrix in zip(pair, dense_inputs(n, seed)):
                matrix.astype("<f8", copy=False).tofile(path)
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
