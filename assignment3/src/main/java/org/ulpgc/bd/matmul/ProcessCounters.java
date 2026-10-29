package org.ulpgc.bd.matmul;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Process-wide CPU time, instructions and cycles, in total and on the
 * performance cores, read from macOS {@code proc_pid_rusage(pid, RUSAGE_INFO_V6)},
 * and the context switches and thread count of the process, read from
 * {@code proc_pidinfo(pid, PROC_PIDTASKINFO)}, both through the FFM API. Every
 * counter is a sum over all threads of the process since it started; the
 * driver takes the difference around each sample.
 *
 * <p>The byte offsets follow {@code struct rusage_info_v6} in
 * {@code <sys/resource.h>} (a 16-byte UUID followed by 64-bit fields) and
 * {@code struct proc_taskinfo} in {@code <sys/proc_info.h>} of the macOS SDK.
 * Times are in Mach absolute time units and are converted to nanoseconds with
 * {@code mach_timebase_info}. On other systems the counters are unavailable
 * and {@link #read()} returns null.
 */
final class ProcessCounters {

    /** Differences or totals of the counters; times in nanoseconds. */
    record Snapshot(long cpuNs, long cpuPNs, long instructions, long cycles, long instructionsP, long cyclesP,
                    long contextSwitches) {

        Snapshot minus(Snapshot o) {
            return new Snapshot(cpuNs - o.cpuNs, cpuPNs - o.cpuPNs, instructions - o.instructions, cycles - o.cycles,
                    instructionsP - o.instructionsP, cyclesP - o.cyclesP, contextSwitches - o.contextSwitches);
        }
    }

    private static final int RUSAGE_INFO_V6 = 6;
    private static final long SIZE = 16 + 56 * 8;
    private static final long USER_TIME = field(0);
    private static final long SYSTEM_TIME = field(1);
    private static final long QOS_FIRST = field(18);
    private static final long INSTRUCTIONS = field(29);
    private static final long CYCLES = field(30);
    private static final long USER_PTIME = field(36);
    private static final long SYSTEM_PTIME = field(37);
    private static final long PINSTRUCTIONS = field(38);
    private static final long PCYCLES = field(39);
    private static final int PROC_PIDTASKINFO = 4;
    private static final int TASKINFO_SIZE = 6 * 8 + 12 * 4;
    private static final long CONTEXT_SWITCHES = 6 * 8 + 8 * 4;
    private static final long THREAD_COUNT = 6 * 8 + 9 * 4;
    /** The ri_cpu_time_qos_* fields, in the order of the struct. */
    private static final String[] QOS_CLASSES = {"default", "maintenance", "background", "utility", "legacy",
            "user_initiated", "user_interactive"};

    private static final MethodHandle RUSAGE;
    private static final MethodHandle PIDINFO;
    private static final double NS_PER_TICK;
    private static final MemorySegment BUFFER;
    private static final MemorySegment TASKINFO;
    private static final int PID = (int) ProcessHandle.current().pid();

    static {
        MethodHandle rusage = null, pidinfo = null;
        double nsPerTick = Double.NaN;
        MemorySegment buffer = null, taskinfo = null;
        try {
            Linker linker = Linker.nativeLinker();
            SymbolLookup lookup = linker.defaultLookup();
            MethodHandle timebase = linker.downcallHandle(lookup.find("mach_timebase_info").orElseThrow(),
                    FunctionDescriptor.of(JAVA_INT, ADDRESS));
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment info = arena.allocate(8, 4);
                if ((int) timebase.invokeExact(info) == 0) {
                    nsPerTick = (double) info.get(JAVA_INT, 0) / info.get(JAVA_INT, 4);
                }
            }
            rusage = linker.downcallHandle(lookup.find("proc_pid_rusage").orElseThrow(),
                    FunctionDescriptor.of(JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS));
            pidinfo = linker.downcallHandle(lookup.find("proc_pidinfo").orElseThrow(),
                    FunctionDescriptor.of(JAVA_INT, JAVA_INT, JAVA_INT, JAVA_LONG, ADDRESS, JAVA_INT));
            buffer = Arena.ofAuto().allocate(SIZE, 8);
            taskinfo = Arena.ofAuto().allocate(TASKINFO_SIZE, 8);
        } catch (Throwable e) {
            rusage = null;
        }
        RUSAGE = Double.isNaN(nsPerTick) || pidinfo == null ? null : rusage;
        PIDINFO = pidinfo;
        NS_PER_TICK = nsPerTick;
        BUFFER = buffer;
        TASKINFO = taskinfo;
    }

    private ProcessCounters() {
    }

    private static long field(int index) {
        return 16 + 8L * index;
    }

    static synchronized boolean available() {
        return RUSAGE != null && fill();
    }

    /** Current totals of the process, or null if the counters are unavailable. */
    static synchronized Snapshot read() {
        if (RUSAGE == null || !fill() || !fillTaskInfo()) {
            return null;
        }
        return new Snapshot(ns(BUFFER.get(JAVA_LONG, USER_TIME) + BUFFER.get(JAVA_LONG, SYSTEM_TIME)),
                ns(BUFFER.get(JAVA_LONG, USER_PTIME) + BUFFER.get(JAVA_LONG, SYSTEM_PTIME)),
                BUFFER.get(JAVA_LONG, INSTRUCTIONS), BUFFER.get(JAVA_LONG, CYCLES),
                BUFFER.get(JAVA_LONG, PINSTRUCTIONS), BUFFER.get(JAVA_LONG, PCYCLES),
                Integer.toUnsignedLong(TASKINFO.get(JAVA_INT, CONTEXT_SWITCHES)));
    }

    /** Number of threads of the process as the operating system sees them (JVM-internal threads included), or -1. */
    static synchronized int osThreads() {
        return RUSAGE != null && fillTaskInfo() ? TASKINFO.get(JAVA_INT, THREAD_COUNT) : -1;
    }

    /** CPU time of the process per QoS class in nanoseconds (empty if unavailable). */
    static synchronized Map<String, Long> qosCpuNs() {
        Map<String, Long> qos = new LinkedHashMap<>();
        if (RUSAGE != null && fill()) {
            for (int q = 0; q < QOS_CLASSES.length; q++) {
                qos.put(QOS_CLASSES[q], ns(BUFFER.get(JAVA_LONG, QOS_FIRST + 8L * q)));
            }
        }
        return qos;
    }

    private static boolean fill() {
        try {
            return (int) RUSAGE.invokeExact(PID, RUSAGE_INFO_V6, BUFFER) == 0;
        } catch (Throwable e) {
            return false;
        }
    }

    private static boolean fillTaskInfo() {
        try {
            return (int) PIDINFO.invokeExact(PID, PROC_PIDTASKINFO, 0L, TASKINFO, TASKINFO_SIZE) == TASKINFO_SIZE;
        } catch (Throwable e) {
            return false;
        }
    }

    private static long ns(long ticks) {
        return Math.round(ticks * NS_PER_TICK);
    }
}
