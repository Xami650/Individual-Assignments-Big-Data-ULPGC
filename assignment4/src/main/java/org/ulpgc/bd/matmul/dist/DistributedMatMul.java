package org.ulpgc.bd.matmul.dist;

import com.hazelcast.client.impl.clientside.HazelcastClientInstanceImpl;
import com.hazelcast.client.impl.clientside.HazelcastClientProxy;
import com.hazelcast.client.impl.connection.ClientConnection;
import com.hazelcast.cluster.Member;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.core.IExecutorService;
import com.hazelcast.internal.networking.Channel;
import com.hazelcast.internal.serialization.Data;
import com.hazelcast.internal.serialization.SerializationService;
import com.hazelcast.map.IMap;
import com.hazelcast.partition.Partition;
import com.hazelcast.partition.PartitionService;
import java.lang.reflect.Field;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import org.ulpgc.bd.matmul.dist.protocol.BandKey;
import org.ulpgc.bd.matmul.dist.protocol.BandTask;
import org.ulpgc.bd.matmul.dist.protocol.PingTask;
import org.ulpgc.bd.matmul.dist.protocol.RowBand;
import org.ulpgc.bd.matmul.dist.protocol.TaskReport;
import org.ulpgc.bd.matmul.kernel.ParallelMatMul;

/**
 * Distributed C = A * B on a Hazelcast cluster of w members, driven by a
 * Hazelcast client: 1D row decomposition with B replicated (scatter A,
 * broadcast B, gather C).
 *
 * <p>Partitioning and placement. A is split into w bands of consecutive rows,
 * band t = rows [t n / w, (t + 1) n / w) (sizes differ by at most one, the
 * static split of {@code par_ikj}). Band t is assigned to the member started
 * with index t. For every member the constructor finds a partition key whose
 * partition that member owns; the key of band t ({@link BandKey}) carries it,
 * so band t of A, the copy of B for band t and band t of C are stored on member
 * t, and the task of band t runs there ({@code submitToKeyOwner}).
 *
 * <p>One multiplication ({@link #multiply}) runs five phases, one after the
 * other; each ends when its last operation has completed, so their times add
 * up to the end-to-end time:
 * <ol>
 *   <li><b>serialize</b>: the driver serializes the w bands of A and B once
 *       (Hazelcast's serialization service);</li>
 *   <li><b>scatter</b>: {@code IMap.setAll} stores band t of A on member t;</li>
 *   <li><b>broadcast</b>: {@code IMap.setAll} stores the same serialized B once on
 *       every member (w copies);</li>
 *   <li><b>compute</b>: one {@link BandTask} per member; the phase ends when the
 *       slowest task has returned;</li>
 *   <li><b>gather</b>: {@code IMap.getAll} brings the w bands of C to the driver,
 *       which puts their rows into C (assemble).</li>
 * </ol>
 * The end-to-end boundary starts with A and B in the driver's heap and ends
 * with C assembled there. Starting the members, connecting the client and
 * finding the placement (the setup) happen once, before, and are not part of
 * it.
 *
 * <p>Transfer is measured twice: the exact size of every serialized value, and
 * the bytes written to and read from the client's connections (the counters
 * of Hazelcast's networking layer, which include the protocol's framing).
 */
public final class DistributedMatMul {

    /** Timers (ns), sizes (bytes) and task reports of one multiplication. */
    public record Result(long serializeNs, long scatterNs, long broadcastNs, long computeNs, long gatherNs,
                         long assembleNs, long computeStartNs, long aPayloadBytes, long bPayloadBytes,
                         long cPayloadBytes, long scatterSocketOut, long broadcastSocketOut, long computeSocketOut,
                         long computeSocketIn, long gatherSocketIn, long socketOut, long socketIn,
                         TaskReport[] reports) {

        /** Sum of the phases (the end-to-end time of the multiplication, up to the timer calls). */
        public long totalNs() {
            return serializeNs + scatterNs + broadcastNs + computeNs + gatherNs + assembleNs;
        }
    }

    /** Thrown when a band task or a cluster operation fails. */
    public static final class ClusterFailure extends RuntimeException {
        ClusterFailure(String message, Throwable cause) {
            super(message, cause);
        }
    }

    private static final Field CHANNEL_FIELD = channelField();

    private final HazelcastInstance client;
    private final HazelcastClientInstanceImpl clientImpl;
    private final SerializationService serialization;
    private final IMap<Object, Object> mapA;
    private final IMap<Object, Object> mapB;
    private final IMap<Object, Object> mapC;
    private final IExecutorService executor;
    private final BandKey[] keys;
    private final Member[] members;
    private long job;

    /**
     * Waits (up to {@code timeout}) until the client sees {@code workers}
     * members, has a connection to each of them and knows the owner of every
     * partition; then fixes the placement of the w bands.
     */
    public DistributedMatMul(HazelcastInstance client, int workers, Duration timeout) {
        if (workers < 1) {
            throw new IllegalArgumentException("workers must be >= 1, got " + workers);
        }
        this.client = client;
        this.clientImpl = ((HazelcastClientProxy) client).client;
        this.serialization = clientImpl.getSerializationService();
        long deadline = System.nanoTime() + timeout.toNanos();
        await(() -> client.getCluster().getMembers().size() >= workers, deadline, "the cluster to have " + workers
                + " members");
        if (client.getCluster().getMembers().size() != workers) {
            throw new IllegalStateException("cluster has " + client.getCluster().getMembers().size()
                    + " members, expected " + workers);
        }
        await(() -> clientImpl.getConnectionManager().getActiveConnections().size() >= workers, deadline,
                "a client connection to every member");
        this.members = membersByIndex(workers);
        PartitionService partitions = client.getPartitionService();
        await(() -> partitions.getPartitions().stream().allMatch(p -> p.getOwner() != null), deadline,
                "the owner of every partition");
        this.keys = placementKeys(partitions, members);
        this.mapA = client.getMap(HazelcastSetup.MAP_A);
        this.mapB = client.getMap(HazelcastSetup.MAP_B);
        this.mapC = client.getMap(HazelcastSetup.MAP_C);
        this.executor = client.getExecutorService(HazelcastSetup.EXECUTOR);
    }

    /** Number of members (and bands). */
    public int workers() {
        return keys.length;
    }

    /** The key of every band. */
    public List<BandKey> keys() {
        return List.of(keys);
    }

    /** True if the byte counters of the client's connections can be read. */
    public static boolean socketCountersAvailable() {
        return CHANNEL_FIELD != null;
    }

    /**
     * C = A * B on the cluster. The rows of c are replaced by the rows of the
     * gathered C bands (c's row arrays are not reused).
     */
    public Result multiply(double[][] a, double[][] b, double[][] c, int n) {
        if (a.length != n || b.length != n || c.length != n) {
            throw new IllegalArgumentException("A, B and C must have n = " + n + " rows");
        }
        long id = ++job;
        int w = keys.length;
        long[] s0 = socketBytes();
        long t0 = System.nanoTime();
        Data[] aData = new Data[w];
        long aBytes = 0;
        for (int t = 0; t < w; t++) {
            aData[t] = serialization.toData(RowBand.slice(id, a, bandStart(t, n, w), bandStart(t + 1, n, w), n));
            aBytes += aData[t].totalSize();
        }
        Data bData = serialization.toData(new RowBand(id, 0, n, b));
        long t1 = System.nanoTime();

        Map<Object, Object> entries = new HashMap<>();
        for (int t = 0; t < w; t++) {
            entries.put(keys[t], aData[t]);
        }
        mapA.setAll(entries);
        long t2 = System.nanoTime();
        long[] s1 = socketBytes();

        entries.clear();
        for (int t = 0; t < w; t++) {
            entries.put(keys[t], bData);
        }
        mapB.setAll(entries);
        long t3 = System.nanoTime();
        long[] s2 = socketBytes();

        List<Future<TaskReport>> futures = new ArrayList<>(w);
        for (int t = 0; t < w; t++) {
            futures.add(executor.submitToKeyOwner(new BandTask(id, keys[t]), keys[t]));
        }
        TaskReport[] reports = new TaskReport[w];
        long cBytes = 0;
        for (int t = 0; t < w; t++) {
            reports[t] = await(futures.get(t), t);
            cBytes += reports[t].cBytes;
        }
        long t4 = System.nanoTime();
        long[] s3 = socketBytes();

        Map<Object, Object> gathered = mapC.getAll(new LinkedHashSet<>(Arrays.asList(keys)));
        long t5 = System.nanoTime();
        long[] s4 = socketBytes();
        for (int t = 0; t < w; t++) {
            RowBand band = (RowBand) gathered.get(keys[t]);
            int lo = bandStart(t, n, w);
            int rows = bandStart(t + 1, n, w) - lo;
            if (band == null || band.job() != id || band.first() != lo || band.rows().length != rows
                    || band.columns() != n) {
                throw new IllegalStateException("C " + keys[t] + " is missing or does not belong to multiplication "
                        + id);
            }
            System.arraycopy(band.rows(), 0, c, lo, rows);
        }
        long t6 = System.nanoTime();
        return new Result(t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t3, aBytes,
                (long) bData.totalSize() * w, cBytes, diff(s1, s0, 0), diff(s2, s1, 0), diff(s3, s2, 0),
                diff(s3, s2, 1), diff(s4, s3, 1), diff(s4, s0, 0), diff(s4, s0, 1), reports);
    }

    /**
     * Round trip of {@code rounds} empty tasks ({@link PingTask}) to every
     * member in turn; returns the time of each, in ns, member by member.
     */
    public long[][] ping(int rounds) {
        long[][] times = new long[keys.length][rounds];
        for (int r = 0; r < rounds; r++) {
            for (int t = 0; t < keys.length; t++) {
                long t0 = System.nanoTime();
                await(executor.submitToKeyOwner(new PingTask(), keys[t]), t);
                times[t][r] = System.nanoTime() - t0;
            }
        }
        return times;
    }

    /** Median, quartiles and extremes of the empty-task round trips, pooled over all members. */
    public static String pingSummary(long[][] times) {
        long[] all = Arrays.stream(times).flatMapToLong(Arrays::stream).sorted().toArray();
        long[] medians = Arrays.stream(times).mapToLong(t -> {
            long[] s = t.clone();
            Arrays.sort(s);
            return s[s.length / 2];
        }).toArray();
        return "ping_rounds=" + times[0].length + ",ping_median_ns=" + quantile(all, 0.5) + ",ping_q1_ns="
                + quantile(all, 0.25) + ",ping_q3_ns=" + quantile(all, 0.75) + ",ping_min_ns=" + all[0]
                + ",ping_max_ns=" + all[all.length - 1] + ",ping_member_median_max_ns="
                + Arrays.stream(medians).max().orElse(-1);
    }

    private static long quantile(long[] sorted, double q) {
        return sorted[(int) Math.min(sorted.length - 1, Math.round(q * (sorted.length - 1)))];
    }

    /** Number of entries in the maps of A, B and C. */
    public int[] mapSizes() {
        return new int[]{mapA.size(), mapB.size(), mapC.size()};
    }

    /** Removes every entry from the maps of A, B and C. */
    public void clear() {
        mapA.clear();
        mapB.clear();
        mapC.clear();
    }

    /** First row of band t when n rows are split into w bands (the static split of par_ikj). */
    static int bandStart(int t, int n, int w) {
        return ParallelMatMul.bandStart(t, n, w);
    }

    /** Members ordered by the index they were started with (member attribute). */
    private Member[] membersByIndex(int workers) {
        Member[] byIndex = new Member[workers];
        for (Member m : client.getCluster().getMembers()) {
            String attribute = m.getAttribute(HazelcastSetup.ATTRIBUTE_INDEX);
            int index = attribute == null ? -1 : Integer.parseInt(attribute);
            if (index < 0 || index >= workers || byIndex[index] != null) {
                throw new IllegalStateException("member " + m + " has index " + attribute + ", expected a distinct "
                        + "index in 0.." + (workers - 1));
            }
            byIndex[index] = m;
        }
        return byIndex;
    }

    /** For band t, the smallest integer partition key whose partition is owned by member t. */
    private static BandKey[] placementKeys(PartitionService partitions, Member[] members) {
        BandKey[] keys = new BandKey[members.length];
        Set<UUID> owners = new HashSet<>();
        for (Partition p : partitions.getPartitions()) {
            owners.add(p.getOwner().getUuid());
        }
        for (int t = 0; t < members.length; t++) {
            UUID target = members[t].getUuid();
            if (!owners.contains(target)) {
                throw new IllegalStateException("member " + members[t] + " owns no partition");
            }
            int candidate = 0;
            while (!partitions.getPartition(candidate).getOwner().getUuid().equals(target)) {
                candidate++;
            }
            keys[t] = new BandKey(t, candidate);
        }
        return keys;
    }

    private static <T> T await(Future<T> future, int band) {
        try {
            return future.get();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ClusterFailure("interrupted while waiting for band " + band, e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            throw new ClusterFailure("band " + band + " failed: " + cause, cause);
        }
    }

    private interface Condition {
        boolean holds();
    }

    private static void await(Condition condition, long deadline, String what) {
        while (!condition.holds()) {
            if (System.nanoTime() > deadline) {
                throw new IllegalStateException("timed out waiting for " + what);
            }
            try {
                Thread.sleep(5);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted while waiting for " + what, e);
            }
        }
    }

    /**
     * Bytes written to and read from all of the client's connections so far,
     * from the counters of Hazelcast's networking layer ({@link Channel}), or
     * {-1, -1} if they cannot be read. The client connection keeps its channel
     * in a private field, read here by reflection.
     */
    private long[] socketBytes() {
        if (CHANNEL_FIELD == null) {
            return new long[]{-1, -1};
        }
        long out = 0, in = 0;
        try {
            for (ClientConnection connection : clientImpl.getConnectionManager().getActiveConnections()) {
                Channel channel = (Channel) CHANNEL_FIELD.get(connection);
                out += channel.bytesWritten();
                in += channel.bytesRead();
            }
        } catch (IllegalAccessException | RuntimeException e) {
            return new long[]{-1, -1};
        }
        return new long[]{out, in};
    }

    private static long diff(long[] after, long[] before, int i) {
        return after[i] < 0 || before[i] < 0 ? -1 : after[i] - before[i];
    }

    private static Field channelField() {
        try {
            Field f = Class.forName("com.hazelcast.client.impl.connection.tcp.TcpClientConnection")
                    .getDeclaredField("channel");
            f.setAccessible(true);
            return f;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }
}
