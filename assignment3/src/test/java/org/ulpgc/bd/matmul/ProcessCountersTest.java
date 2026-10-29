package org.ulpgc.bd.matmul;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.management.ManagementFactory;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** The proc_pid_rusage counters (macOS only; skipped elsewhere). */
class ProcessCountersTest {

    @Test
    void busyThreadAdvancesTheCounters() {
        assumeTrue(ProcessCounters.available(), "proc_pid_rusage is not available on this system");
        ProcessCounters.Snapshot before = ProcessCounters.read();
        long t0 = System.nanoTime();
        double x = 1.0;
        while (System.nanoTime() - t0 < 200_000_000L) {
            x = x * 1.0000001 + 1e-9;
        }
        long elapsed = System.nanoTime() - t0;
        ProcessCounters.Snapshot d = ProcessCounters.read().minus(before);
        assertTrue(x > 0);
        // One busy thread: about as much CPU time as wall time (other JVM threads add a little).
        assertTrue(d.cpuNs() > 0.5 * elapsed && d.cpuNs() < 2.0 * elapsed, "cpu " + d.cpuNs() + " vs " + elapsed);
        assertTrue(d.cpuPNs() >= 0 && d.cpuPNs() <= d.cpuNs(), "performance-core time within the total");
        assertTrue(d.instructions() > 0 && d.cycles() > 0);
        assertTrue(d.instructionsP() <= d.instructions() && d.cyclesP() <= d.cycles());
    }

    @Test
    void qosClassesAreReported() {
        assumeTrue(ProcessCounters.available(), "proc_pid_rusage is not available on this system");
        Map<String, Long> qos = ProcessCounters.qosCpuNs();
        assertFalse(qos.isEmpty());
        assertTrue(qos.containsKey("default") && qos.values().stream().allMatch(v -> v >= 0));
    }

    /** The operating system sees every Java thread plus the JVM's internal ones (GC, JIT, ...). */
    @Test
    void osThreadCountIncludesJavaThreadsAndContextSwitchesAreCounted() throws InterruptedException {
        assumeTrue(ProcessCounters.available(), "proc_pid_rusage is not available on this system");
        assertTrue(ProcessCounters.osThreads() >= ManagementFactory.getThreadMXBean().getThreadCount());
        ProcessCounters.Snapshot before = ProcessCounters.read();
        for (int i = 0; i < 20; i++) {
            Thread.sleep(1);
        }
        assertTrue(ProcessCounters.read().minus(before).contextSwitches() >= 20);
    }
}
