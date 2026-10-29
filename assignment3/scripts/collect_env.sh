#!/bin/sh
# Print a snapshot of the execution environment (macOS). The benchmark harness
# stores this output as env.txt next to the raw results of every session.
# Adapted from Assignment 2 (SciPy removed; core topology, thread affinity
# support and the JVM's thread and vectorisation flags added).
# Usage (from assignment3/): sh scripts/collect_env.sh [python-executable]

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
       "$(sysctl -n hw.perflevel$level.logicalcpu) logical," \
       "L1d $(sysctl -n hw.perflevel$level.l1dcachesize) B, L2 $(sysctl -n hw.perflevel$level.l2cachesize) B" \
       "shared by $(sysctl -n hw.perflevel$level.cpusperl2) cores"
done
echo "cache line: $(sysctl -n hw.cachelinesize) B, page size: $(sysctl -n hw.pagesize) B"
echo "memory: $(sysctl -n hw.memsize) B"
"$PY" - <<'EOF'
import ctypes
libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
libc.mach_thread_self.restype = ctypes.c_uint
kr = libc.thread_policy_set(libc.mach_thread_self(), 4, (ctypes.c_int * 1)(1), 1)  # THREAD_AFFINITY_POLICY
print(f"thread affinity: thread_policy_set(THREAD_AFFINITY_POLICY) returns {kr}"
      f"{' (KERN_NOT_SUPPORTED): threads cannot be bound to cores' if kr == 46 else ''}")
EOF

section "Operating system"
sw_vers
uname -a

section "Power and thermal state"
pmset -g batt | head -2
pmset -g | grep -i lowpowermode || echo "lowpowermode: not reported"
pmset -g therm
uptime
vm_stat | head -8

section "Java"
java -version 2>&1
mvn -v 2>/dev/null | head -1
echo "JVM flags in effect for the benchmark flags (threads, heap, JIT, vectorisation):"
java -Xmx8g -XX:+PrintFlagsFinal -version 2>/dev/null \
  | grep -E ' (UseG1GC|MaxHeapSize|InitialHeapSize|G1HeapRegionSize|TieredCompilation|CICompilerCount|UseDynamicNumberOfCompilerThreads|ParallelGCThreads|ConcGCThreads|UseDynamicNumberOfGCThreads|ActiveProcessorCount|UseCompressedOops|UseSuperWord|MaxVectorSize|UseFMA|UseOnStackReplacement|ThreadStackSize) ' \
  | awk '{print "  " $2 " = " $4}'

section "Python"
"$PY" -c 'import platform, sys; print(platform.python_implementation(), platform.python_version(), "|", sys.version.replace("\n", " ")); print("executable:", sys.executable)'
"$PY" -c 'import numpy, pandas, matplotlib; print("numpy", numpy.__version__, "| pandas", pandas.__version__, "| matplotlib", matplotlib.__version__)'

section "Other processes using CPU"
ps -A -o %cpu=,comm= | sort -rn | head -6
