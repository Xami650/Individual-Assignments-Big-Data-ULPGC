#!/bin/sh
# Print a snapshot of the execution environment (macOS). The benchmark harness
# stores this output as env.txt next to the raw results of every session.
# Usage (from assignment1/): sh scripts/collect_env.sh [python-executable]

PY="${1:-python3}"
section() { printf '\n## %s\n' "$1"; }

section "Timestamp and source"
date '+%Y-%m-%dT%H:%M:%S%z'
echo "git commit: $(git rev-parse HEAD 2>/dev/null)"
echo "git dirty files: $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"

section "Hardware"
sysctl -n hw.model | sed 's/^/model: /'
sysctl -n machdep.cpu.brand_string | sed 's/^/cpu: /'
echo "physical cores: $(sysctl -n hw.physicalcpu), logical processors: $(sysctl -n hw.logicalcpu)"
for level in 0 1; do
  name=$(sysctl -n hw.perflevel$level.name 2>/dev/null) || continue
  echo "perflevel$level ($name): $(sysctl -n hw.perflevel$level.physicalcpu) cores," \
       "L1d $(sysctl -n hw.perflevel$level.l1dcachesize) B, L2 $(sysctl -n hw.perflevel$level.l2cachesize) B" \
       "shared by $(sysctl -n hw.perflevel$level.cpusperl2) cores"
done
echo "cache line: $(sysctl -n hw.cachelinesize) B, page size: $(sysctl -n hw.pagesize) B"
echo "memory: $(sysctl -n hw.memsize) B"

section "Operating system"
sw_vers
uname -a

section "Power and thermal state"
pmset -g batt | head -2
pmset -g | grep -i lowpowermode || echo "lowpowermode: not reported"
pmset -g therm
uptime

section "C toolchain"
clang --version | head -2

section "Java"
java -version 2>&1
mvn -v 2>/dev/null | head -1
echo "JVM flags in effect for -Xmx4g (threads and heap):"
java -Xmx4g -XX:+PrintFlagsFinal -version 2>/dev/null \
  | grep -E ' (UseG1GC|UseSerialGC|UseParallelGC|MaxHeapSize|InitialHeapSize|TieredCompilation|CICompilerCount|ParallelGCThreads|ConcGCThreads|UseCompressedOops) ' \
  | awk '{print "  " $2 " = " $4}'

section "Python"
"$PY" -c 'import platform, sys; print(platform.python_implementation(), platform.python_version(), "|", sys.version.replace("\n", " ")); print("executable:", sys.executable); print("JIT enabled:", getattr(getattr(sys, "_jit", None), "is_enabled", lambda: False)()); print("GIL enabled:", getattr(sys, "_is_gil_enabled", lambda: True)())'
"$PY" -c 'import numpy, pandas, matplotlib; print("numpy", numpy.__version__, "| pandas", pandas.__version__, "| matplotlib", matplotlib.__version__)'
