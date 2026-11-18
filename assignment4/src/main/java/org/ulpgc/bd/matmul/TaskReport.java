package org.ulpgc.bd.matmul;

import com.hazelcast.nio.ObjectDataInput;
import com.hazelcast.nio.ObjectDataOutput;
import com.hazelcast.nio.serialization.IdentifiedDataSerializable;
import java.io.IOException;

/**
 * What a {@link BandTask} reports back to the driver: where it ran, its
 * internal timers, the size of the C band it stored, and the change of its
 * member's process counters ({@link ProcessCounters}) over the kernel.
 *
 * <p>Times are in nanoseconds. {@code startNs} and {@code endNs} are
 * {@link System#nanoTime()} values of the member's process; all processes run
 * on the same machine, where HotSpot's nanoTime reads one system-wide
 * monotonic clock, so they can be compared with the driver's own readings.
 * The counter fields are -1 where the counters are unavailable.
 */
public final class TaskReport implements IdentifiedDataSerializable {

    int band;
    int memberIndex;
    int kernelThreads;
    int rows;
    long pid;
    /** True if the partition of the band's key is owned by the member that ran the task. */
    boolean local;
    long readANs;
    long readBNs;
    long allocNs;
    long kernelNs;
    long storeNs;
    long taskNs;
    long startNs;
    long endNs;
    /** Serialized size of the C band stored by the task. */
    long cBytes;
    long taskCpuNs;
    long cpuNs;
    long cpuPNs;
    long instructions;
    long cycles;
    long instructionsP;
    long cyclesP;
    long contextSwitches;

    /** For deserialization only. */
    public TaskReport() {
    }

    void counters(ProcessCounters.Snapshot d) {
        cpuNs = d == null ? -1 : d.cpuNs();
        cpuPNs = d == null ? -1 : d.cpuPNs();
        instructions = d == null ? -1 : d.instructions();
        cycles = d == null ? -1 : d.cycles();
        instructionsP = d == null ? -1 : d.instructionsP();
        cyclesP = d == null ? -1 : d.cyclesP();
        contextSwitches = d == null ? -1 : d.contextSwitches();
    }

    @Override
    public void writeData(ObjectDataOutput out) throws IOException {
        out.writeInt(band);
        out.writeInt(memberIndex);
        out.writeInt(kernelThreads);
        out.writeInt(rows);
        out.writeLong(pid);
        out.writeBoolean(local);
        for (long v : longs()) {
            out.writeLong(v);
        }
    }

    @Override
    public void readData(ObjectDataInput in) throws IOException {
        band = in.readInt();
        memberIndex = in.readInt();
        kernelThreads = in.readInt();
        rows = in.readInt();
        pid = in.readLong();
        local = in.readBoolean();
        readANs = in.readLong();
        readBNs = in.readLong();
        allocNs = in.readLong();
        kernelNs = in.readLong();
        storeNs = in.readLong();
        taskNs = in.readLong();
        startNs = in.readLong();
        endNs = in.readLong();
        cBytes = in.readLong();
        taskCpuNs = in.readLong();
        cpuNs = in.readLong();
        cpuPNs = in.readLong();
        instructions = in.readLong();
        cycles = in.readLong();
        instructionsP = in.readLong();
        cyclesP = in.readLong();
        contextSwitches = in.readLong();
    }

    /** The long fields in serialization order. */
    private long[] longs() {
        return new long[]{readANs, readBNs, allocNs, kernelNs, storeNs, taskNs, startNs, endNs, cBytes, taskCpuNs,
                cpuNs, cpuPNs, instructions, cycles, instructionsP, cyclesP, contextSwitches};
    }

    @Override
    public int getFactoryId() {
        return MatMulSerialization.FACTORY_ID;
    }

    @Override
    public int getClassId() {
        return MatMulSerialization.TASK_REPORT;
    }
}
