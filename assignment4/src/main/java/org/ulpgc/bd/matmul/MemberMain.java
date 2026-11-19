package org.ulpgc.bd.matmul;

import com.hazelcast.core.Hazelcast;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.partition.Partition;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.util.Random;
import org.ulpgc.bd.matmul.dist.HazelcastSetup;
import org.ulpgc.bd.matmul.kernel.BandKernel;
import org.ulpgc.bd.matmul.metrics.ProcessCounters;

/**
 * One cluster member: a separate JVM process that starts a Hazelcast member,
 * stores bands and runs band tasks until the harness closes its standard
 * input. Started through the driver's jar:
 *
 * <pre>
 *   java -Xmx... -jar target/matmul.jar member --cluster NAME --index M --members W
 *        [--threads K] [--base-port P] [--pre-warm-seconds PW]
 * </pre>
 *
 * Life cycle, reported on stdout as lines "@ event key=value,...":
 * <ol>
 *   <li>{@code started}: the JVM is up (JVM start time, epoch milliseconds);</li>
 *   <li>{@code joined}: the Hazelcast member has started, all W members have
 *       joined, the partitions have been assigned and the cluster is safe (no
 *       migration running);</li>
 *   <li>{@code ready}: the member's kernel has been pre-warmed for PW seconds on
 *       a small random band, so that the JIT compiles it before any timing;</li>
 *   <li>{@code stopping}: standard input was closed; the member reports its
 *       threads and garbage collections and terminates.</li>
 * </ol>
 */
public final class MemberMain {

    private static final int PRE_WARM_N = 256;
    private static final long PRE_WARM_SEED = 2026;
    private static final long JOIN_TIMEOUT_MS = 120_000;

    private MemberMain() {
    }

    /** Command-line options of a member. */
    record Options(String cluster, int index, int members, int threads, int basePort, double preWarmSeconds) {
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        Options opt;
        try {
            opt = parse(args);
        } catch (IllegalArgumentException e) {
            System.err.println("error: " + e.getMessage());
            System.err.println("usage: member --cluster NAME --index M --members W [--threads K] [--base-port P] "
                    + "[--pre-warm-seconds PW]");
            System.exit(2);
            return;
        }
        PrintStream out = System.out;
        HazelcastSetup.quietLogging();
        long pid = ProcessHandle.current().pid();
        event(out, "started", "index=" + opt.index() + ",pid=" + pid + ",jvm_start_epoch_ms="
                + ManagementFactory.getRuntimeMXBean().getStartTime() + ",epoch_ms=" + System.currentTimeMillis());

        BandKernel kernel = new BandKernel(opt.threads());
        long t0 = System.currentTimeMillis();
        HazelcastInstance hz;
        try {
            hz = Hazelcast.newHazelcastInstance(
                    HazelcastSetup.memberConfig(opt.cluster(), opt.basePort(), opt.members(), opt.index(), kernel));
        } catch (OutOfMemoryError e) {
            System.err.println("error: out of memory while starting the member");
            System.exit(3);
            return;
        }
        long t1 = System.currentTimeMillis();
        long deadline = t1 + JOIN_TIMEOUT_MS;
        while (hz.getCluster().getMembers().size() < opt.members()) {
            waitUntil(deadline, "all members to join");
        }
        // Asking for a partition owner makes the master assign the partitions (if not done yet),
        // now that every member is present; then wait until no migration is running.
        hz.getPartitionService().getPartition(0).getOwner();
        while (!hz.getPartitionService().isClusterSafe()) {
            waitUntil(deadline, "the cluster to be safe");
        }
        long t2 = System.currentTimeMillis();
        int owned = 0;
        for (Partition p : hz.getPartitionService().getPartitions()) {
            owned += p.getOwner() != null && p.getOwner().localMember() ? 1 : 0;
        }
        event(out, "joined", "index=" + opt.index() + ",members=" + hz.getCluster().getMembers().size()
                + ",partitions_owned=" + owned + ",hz_start_ms=" + (t1 - t0) + ",cluster_wait_ms=" + (t2 - t1)
                + ",epoch_ms=" + t2);

        long preWarmNs = preWarm(kernel, opt.preWarmSeconds());
        event(out, "ready", "index=" + opt.index() + ",pid=" + pid + ",kernel_threads=" + kernel.threads()
                + ",pre_warm_ns=" + preWarmNs + ",jvm_threads=" + ManagementFactory.getThreadMXBean().getThreadCount()
                + ",os_threads=" + ProcessCounters.osThreads() + ",epoch_ms=" + System.currentTimeMillis());

        while (System.in.read() != -1) {
            // the harness closes stdin when the configuration is finished
        }
        long[] gc = gcTotals();
        ProcessCounters.Snapshot total = ProcessCounters.read();
        event(out, "stopping", "index=" + opt.index() + ",jvm_threads=" + ManagementFactory.getThreadMXBean()
                .getThreadCount() + ",os_threads=" + ProcessCounters.osThreads() + ",gc_collections=" + gc[0]
                + ",gc_ms=" + gc[1] + ",cpu_ns=" + (total == null ? "" : total.cpuNs()) + ",cpu_p_ns="
                + (total == null ? "" : total.cpuPNs()) + ",epoch_ms=" + System.currentTimeMillis());
        hz.getLifecycleService().terminate();
        kernel.close();
        System.exit(0);
    }

    static Options parse(String[] args) {
        String cluster = null;
        Integer index = null, members = null;
        int threads = 1, basePort = HazelcastSetup.DEFAULT_BASE_PORT;
        double preWarm = 1.0;
        for (int i = 0; i < args.length; i += 2) {
            if (i + 1 >= args.length) {
                throw new IllegalArgumentException("missing value for " + args[i]);
            }
            String v = args[i + 1];
            switch (args[i]) {
                case "--cluster" -> cluster = v;
                case "--index" -> index = Integer.parseInt(v);
                case "--members" -> members = Integer.parseInt(v);
                case "--threads" -> threads = Integer.parseInt(v);
                case "--base-port" -> basePort = Integer.parseInt(v);
                case "--pre-warm-seconds" -> preWarm = Double.parseDouble(v);
                default -> throw new IllegalArgumentException("unknown option " + args[i]);
            }
        }
        if (cluster == null || index == null || members == null || members < 1 || index < 0 || index >= members
                || threads < 1 || preWarm < 0) {
            throw new IllegalArgumentException("need --cluster, --members >= 1, 0 <= --index < members, --threads >= 1");
        }
        return new Options(cluster, index, members, threads, basePort, preWarm);
    }

    /** Runs the kernel on a random band for the given time; returns the time spent. */
    private static long preWarm(BandKernel kernel, double seconds) {
        long t0 = System.nanoTime(), limit = (long) (seconds * 1e9);
        if (limit <= 0) {
            return 0;
        }
        Random rng = new Random(PRE_WARM_SEED);
        double[][] a = new double[PRE_WARM_N][PRE_WARM_N], b = new double[PRE_WARM_N][PRE_WARM_N];
        for (double[][] m : new double[][][]{a, b}) {
            for (double[] row : m) {
                for (int j = 0; j < row.length; j++) {
                    row[j] = rng.nextDouble(-1.0, 1.0);
                }
            }
        }
        double[][] c = new double[PRE_WARM_N][PRE_WARM_N];
        while (System.nanoTime() - t0 < limit) {
            kernel.multiply(a, b, c, PRE_WARM_N);
        }
        return System.nanoTime() - t0;
    }

    private static void waitUntil(long deadlineMs, String what) throws InterruptedException {
        if (System.currentTimeMillis() > deadlineMs) {
            System.err.println("error: timed out waiting for " + what);
            System.exit(4);
        }
        Thread.sleep(5);
    }

    private static void event(PrintStream out, String name, String fields) {
        out.println("@ " + name + " " + fields);
        out.flush();
    }

    private static long[] gcTotals() {
        long count = 0, millis = 0;
        for (GarbageCollectorMXBean gc : ManagementFactory.getGarbageCollectorMXBeans()) {
            count += Math.max(0, gc.getCollectionCount());
            millis += Math.max(0, gc.getCollectionTime());
        }
        return new long[]{count, millis};
    }
}
