package org.ulpgc.bd.matmul;

import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.core.HazelcastInstanceAware;
import com.hazelcast.internal.serialization.Data;
import com.hazelcast.internal.serialization.SerializationService;
import com.hazelcast.map.IMap;
import com.hazelcast.nio.ObjectDataInput;
import com.hazelcast.nio.ObjectDataOutput;
import com.hazelcast.nio.serialization.IdentifiedDataSerializable;
import com.hazelcast.partition.Partition;
import com.hazelcast.spi.impl.SerializationServiceSupport;
import java.io.IOException;
import java.util.concurrent.Callable;

/**
 * The computation of band t, sent by the driver to the member that owns the
 * band's key ({@code IExecutorService.submitToKeyOwner}) and run there on the
 * member's executor thread:
 * <ol>
 *   <li>take band t of A and the copy of B for band t from their maps
 *       ({@code IMap.remove}: a local operation, because the member owns the key's
 *       partition; the value is deserialized into new arrays and the stored copy
 *       is released);</li>
 *   <li>allocate the C band and compute it with the member's {@link BandKernel}
 *       (k compute threads, created when the member started);</li>
 *   <li>serialize the C band and store it in the map of C under the same key
 *       (again local).</li>
 * </ol>
 * The task checks that the bands belong to the current multiplication and that
 * the key's partition is owned by this member, and reports its timers and the
 * change of the member's process counters over the kernel ({@link TaskReport}).
 */
public final class BandTask implements Callable<TaskReport>, IdentifiedDataSerializable, HazelcastInstanceAware {

    private long job;
    private BandKey key;
    private transient HazelcastInstance hz;

    /** For deserialization only. */
    public BandTask() {
    }

    public BandTask(long job, BandKey key) {
        this.job = job;
        this.key = key;
    }

    @Override
    public void setHazelcastInstance(HazelcastInstance hazelcastInstance) {
        this.hz = hazelcastInstance;
    }

    @Override
    public TaskReport call() {
        TaskReport r = new TaskReport();
        r.startNs = System.nanoTime();
        ProcessCounters.Snapshot taskBefore = ProcessCounters.read();
        BandKernel kernel = (BandKernel) hz.getUserContext().get(HazelcastSetup.CONTEXT_KERNEL);
        if (kernel == null) {
            throw new IllegalStateException("member has no compute kernel in its user context");
        }
        r.band = key.band();
        r.memberIndex = (Integer) hz.getUserContext().getOrDefault(HazelcastSetup.CONTEXT_INDEX, -1);
        r.kernelThreads = kernel.threads();
        r.pid = ProcessHandle.current().pid();
        Partition partition = hz.getPartitionService().getPartition(key);
        r.local = partition.getOwner() != null && partition.getOwner().localMember();
        IMap<Object, Object> mapA = hz.getMap(HazelcastSetup.MAP_A);
        IMap<Object, Object> mapB = hz.getMap(HazelcastSetup.MAP_B);
        IMap<Object, Object> mapC = hz.getMap(HazelcastSetup.MAP_C);

        RowBand a = (RowBand) mapA.remove(key);
        long t1 = System.nanoTime();
        RowBand b = (RowBand) mapB.remove(key);
        long t2 = System.nanoTime();
        check(a, "A");
        check(b, "B");
        int n = b.columns();
        if (a.columns() != n || b.rows().length != n) {
            throw new IllegalStateException(key + ": A band has " + a.columns() + " columns, B is "
                    + b.rows().length + " x " + n);
        }
        double[][] c = new double[a.rows().length][n];
        long t3 = System.nanoTime();
        ProcessCounters.Snapshot before = ProcessCounters.read();
        kernel.multiply(a.rows(), b.rows(), c, n);
        ProcessCounters.Snapshot after = ProcessCounters.read();
        long t4 = System.nanoTime();
        SerializationService ss = ((SerializationServiceSupport) hz).getSerializationService();
        Data stored = ss.toData(new RowBand(job, a.first(), n, c));
        mapC.set(key, stored);
        r.endNs = System.nanoTime();
        ProcessCounters.Snapshot taskAfter = ProcessCounters.read();

        r.rows = c.length;
        r.readANs = t1 - r.startNs;
        r.readBNs = t2 - t1;
        r.allocNs = t3 - t2;
        r.kernelNs = t4 - t3;
        r.storeNs = r.endNs - t4;
        r.taskNs = r.endNs - r.startNs;
        r.cBytes = stored.totalSize();
        r.taskCpuNs = taskBefore == null || taskAfter == null ? -1 : taskAfter.cpuNs() - taskBefore.cpuNs();
        r.counters(before == null || after == null ? null : after.minus(before));
        return r;
    }

    private void check(RowBand band, String matrix) {
        if (band == null) {
            throw new IllegalStateException(matrix + " " + key + " is not stored on member "
                    + hz.getCluster().getLocalMember().getAddress());
        }
        if (band.job() != job) {
            throw new IllegalStateException(matrix + " " + key + " belongs to multiplication " + band.job()
                    + ", expected " + job);
        }
    }

    @Override
    public void writeData(ObjectDataOutput out) throws IOException {
        out.writeLong(job);
        out.writeObject(key);
    }

    @Override
    public void readData(ObjectDataInput in) throws IOException {
        job = in.readLong();
        key = in.readObject();
    }

    @Override
    public int getFactoryId() {
        return MatMulSerialization.FACTORY_ID;
    }

    @Override
    public int getClassId() {
        return MatMulSerialization.BAND_TASK;
    }
}
