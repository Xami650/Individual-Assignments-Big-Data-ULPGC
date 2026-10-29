"""Evidence of SIMD vectorisation: the C2-compiled kernel code with SuperWord on and off.

For each setting (SuperWord on, the JVM default, and off, -XX:-UseSuperWord),
the script runs the driver once for ikj and once for par_ikj (4 threads,
dynamic schedule) on n = 512 with the normal pre-warm and warm-up, and asks
HotSpot to print the compiled code of every method of the package
(-XX:+UnlockDiagnosticVMOptions -XX:CompileCommand=print,...). The kernel loop
can run in the method itself or inlined into a caller (a lambda), so every
compiled method is examined. Without the hsdis plugin HotSpot prints the
instructions as hexadecimal words ("abstract disassembly"). For each method,
the script takes the last C2 compilation that is not an on-stack replacement,
assembles its main code (up to the exception handler) with clang from .inst
directives, disassembles it with objdump and counts
    vector_fmul, vector_fadd   two-lane double-precision vector multiply/add (fmul/fadd .2d)
    scalar_fmul, scalar_fadd   scalar double-precision multiply/add (fmul/fadd on d registers)
    fused                      fused multiply-add instructions (Java forbids contracting a*b + c)
    q_loads, q_stores          128-bit (q register) loads and stores
Methods without any floating-point multiply are left out of the summary.

These diagnostic flags are used only here, never in a timed run.

Writes results/simd/<method>_superword_<on|off>.txt (disassembly of each kept
method) and results/simd/summary.csv.

Usage (from assignment3/):
    python3 scripts/simd_evidence.py [--skip-build]
"""

import argparse
import csv
import re
import subprocess
import sys

from common import ROOT, build, command, dense_pair, load_config

OUT = ROOT / "results" / "simd"
TMP = ROOT / "results" / "tmp" / "simd"
N = 512
PRINT_FLAGS = ["-XX:+UnlockDiagnosticVMOptions", "-XX:CompileCommand=quiet",
               "-XX:CompileCommand=print,org.ulpgc.bd.matmul.*::*"]
RUNS = [("ikj", None, None), ("par_ikj", 4, "dynamic")]
HEADER = re.compile(r"^Compiled method \((c[12])\)")
METHOD = re.compile(r"(\S+::\S+?)(?: @ \d+)? \((\d+) bytes\)")
WORDS = re.compile(r"^\s*0x[0-9a-f]+: ((?:[0-9a-f]{4} [0-9a-f]{4}(?: \| )?)+)\s*$")
END_OF_MAIN_CODE = ("[Exception Handler]", "[Stub Code]", "[/MachCode]")
COUNTS = {
    "vector_fmul": re.compile(r"\bfmul(?:\.2d\s|\s+v\d+\.2d)"),
    "vector_fadd": re.compile(r"\bfadd(?:\.2d\s|\s+v\d+\.2d)"),
    "scalar_fmul": re.compile(r"\bfmul\s+d\d+"),
    "scalar_fadd": re.compile(r"\bfadd\s+d\d+"),
    "fused": re.compile(r"\bf(?:n?madd|n?msub|mla|mls)\b"),
    "q_loads": re.compile(r"\b(?:ldr|ldur|ldp)\s+q\d+"),
    "q_stores": re.compile(r"\b(?:str|stur|stp)\s+q\d+"),
}
FIELDS = ["run", "superword", "method", "compile_id", "bytecode_bytes", "code_words", *COUNTS]


def nmethods(text):
    """Yield (tier, method, is_osr, compile_id, bytecode bytes, instruction words) for each printed nmethod."""
    block = None
    for line in text.splitlines():
        if HEADER.match(line):
            if block:
                yield block
            m = METHOD.search(line)
            fields = line.split()
            block = {"tier": HEADER.match(line).group(1), "method": m.group(1) if m else "?",
                     "osr": " @ " in line, "compile_id": fields[4] if len(fields) > 4 else "",
                     "bytes": m.group(2) if m else "", "words": [], "state": "header"}
        elif block is not None:
            stripped = line.strip()
            if stripped == "[MachCode]":
                block["state"] = "code"
            elif block["state"] == "code" and stripped in END_OF_MAIN_CODE:
                block["state"] = "done"
            elif block["state"] == "code" and (w := WORDS.match(line)):
                for word in w.group(1).split("|"):
                    lo, hi = word.split()
                    block["words"].append(int.from_bytes(bytes.fromhex(lo + hi), "little"))
    if block:
        yield block


def disassemble(words, name):
    source, obj = TMP / f"{name}.s", TMP / f"{name}.o"
    source.write_text(".text\n" + "".join(f".inst 0x{w:08x}\n" for w in words))
    subprocess.run(["clang", "-c", "-arch", "arm64", str(source), "-o", str(obj)], check=True)
    text = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(obj)], check=True, capture_output=True,
                          text=True).stdout
    return "\n".join(l for l in text.splitlines() if re.match(r"^\s*[0-9a-f]+:", l))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    config = load_config()
    if not args.skip_build:
        build()
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    a, b = dense_pair(N)
    rows = []
    for run, threads, schedule in RUNS:
        for superword in ("on", "off"):
            jvm_extra = PRINT_FLAGS + ([] if superword == "on" else ["-XX:-UseSuperWord"])
            cmd = command(config, run, N, a, b, threads=threads, schedule=schedule, jvm_extra=jvm_extra, warmup=1,
                          reps=1, pre_warm_seconds=1.0, min_sample_seconds=0.1, min_warmup_seconds=1.0)
            text = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout
            final = {}
            for nm in nmethods(text):
                if nm["tier"] == "c2" and not nm["osr"] and nm["words"]:
                    final[nm["method"]] = nm
            listing = []
            for method, nm in sorted(final.items()):
                asm = disassemble(nm["words"], f"{run}_{superword}_{len(listing)}")
                counts = {k: len(p.findall(asm)) for k, p in COUNTS.items()}
                if counts["vector_fmul"] + counts["scalar_fmul"] == 0:
                    continue
                rows.append({"run": run, "superword": superword, "method": method, "compile_id": nm["compile_id"],
                             "bytecode_bytes": nm["bytes"], "code_words": len(nm["words"]), **counts})
                listing.append(f"### {method} (C2, compile id {nm['compile_id']}, {len(nm['words'])} words)\n{asm}\n")
                print(f"{run:8s} SuperWord {superword:3s} {method:60s} "
                      + " ".join(f"{k}={v}" for k, v in counts.items()))
            (OUT / f"{run}_superword_{superword}.txt").write_text(
                f"# {' '.join(cmd)}\n# final C2 code of every method with a floating-point multiply\n\n"
                + "\n".join(listing))
    with open(OUT / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    vector_on = all(r["vector_fmul"] > 0 for r in rows if r["superword"] == "on")
    vector_off = any(r["vector_fmul"] + r["vector_fadd"] > 0 for r in rows if r["superword"] == "off")
    print(f"wrote {OUT.relative_to(ROOT)}/summary.csv ({len(rows)} methods); vector instructions with SuperWord: "
          f"{vector_on}; without SuperWord: {vector_off}")
    return 0 if vector_on and not vector_off and rows else 1


if __name__ == "__main__":
    sys.exit(main())
