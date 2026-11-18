package org.ulpgc.bd.matmul;

import com.hazelcast.client.HazelcastClient;
import com.hazelcast.core.HazelcastInstance;
import com.sun.management.HotSpotDiagnosticMXBean;
import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.ulpgc.bd.matmul.ParallelMatMul.Schedule;

/**
 * Benchmark driver for Assignment 4, adapted from the Assignment 3 driver (tag
 * assignment3-v1.0): the same pre-warm, warm-up, batching, per-sample process
 * counters and output format, for the sequential {@code basic} and
 * {@code ikj}, the shared-memory parallel {@code par_ikj} and the distributed
 * {@code dist}.
 *
 * <pre>
 *   java -Xmx8g -jar target/matmul.jar --method basic|ikj|par_ikj|dist --n N --a PATH --b PATH
 *        [--threads P] [--schedule dynamic|static]                          (par_ikj)
 *        [--workers W] [--threads-per-worker K] [--cluster NAME] [--base-port PORT]
 *        [--member-log PATH] [--ping-rounds R]                              (dist)
 *        [--pre-warm-seconds PW] [--warmup W] [--reps R] [--min-sample-seconds S]
 *        [--min-warmup-seconds SW] [--out PATH]
 *   java -jar target/matmul.jar member ...                                  (a cluster member, see MemberMain)
 * </pre>
 *
 * <p>Timing boundary, the same for every method: a sample starts with A and B
 * in this process's heap and ends with C there. For {@code basic},
 * {@code ikj} and {@code par_ikj} this is the multiplication itself (as in
 * Assignment 3). For {@code dist} it is the whole distributed multiplication
 * ({@link DistributedMatMul#multiply}: serialize, scatter A, broadcast B,
 * compute on the members, gather and assemble C). The driver is then a
 * Hazelcast client of a cluster of W members started beforehand (each with K
 * compute threads); connecting to it and finding the placement are timed
 * separately, before any sample.
 *
 * <p>Before any sample the method runs for PW seconds (default 1) on small
 * random inputs (n = min(N, 256)), so that the JIT compiles it with a complete
 * profile (in {@code dist}, the whole path through the cluster). Each sample
 * times {@code batch} back-to-back multiplications with {@link System#nanoTime()}:
 * the first warm-up uses batch = 1 and every later sample batch = max(1,
 * ceil(S / t)), t being the time per multiplication of the latest warm-up;
 * warm-up continues beyond W samples until SW seconds of warm-up work have run.
 *
 * <p>Output on stdout is CSV with the header {@link #HEADER} and one row per
 * sample: the time, a checksum of C and the change of this process's counters
 * ({@link ProcessCounters}); for {@code dist} also the phase timers, the
 * slowest and mean task, and the bytes moved, all summed over the batch (the
 * other methods leave these fields empty). With --member-log, {@code dist}
 * writes one row per member and sample to that file ({@link #MEMBER_HEADER}),
 * also summed over the batch. Lines starting with '#' carry key=value
 * metadata. With --out, C is written as raw dense float64 after the last
 * sample. Exit status 3 means this JVM ran out of heap (the memory budget, set
 * by -Xmx); 4 means the cluster failed (a member or a task).
 */
public final class Main {

    static final List<String> METHODS = List.of("basic", "ikj", "par_ikj", "dist");
    static final List<String> DIST_FIELDS = List.of("serialize_ns", "scatter_ns", "broadcast_ns", "compute_ns",
            "gather_ns", "assemble_ns", "task_max_ns", "task_mean_ns", "kernel_max_ns", "kernel_mean_ns",
            "read_max_ns", "store_max_ns", "dispatch_max_ns", "tasks_done_ns", "a_payload_bytes", "b_payload_bytes",
            "c_payload_bytes", "scatter_socket_out", "broadcast_socket_out", "compute_socket_out",
            "compute_socket_in", "gather_socket_in", "socket_out", "socket_in");
    static final String HEADER = "kind,index,batch,elapsed_ns,checksum,cpu_ns,cpu_p_ns,instructions,cycles,"
            + "instructions_p,cycles_p,context_switches," + String.join(",", DIST_FIELDS);
    static final String MEMBER_HEADER = "kind,index,band,member_index,pid,rows,kernel_threads,ops,local,read_a_ns,"
            + "read_b_ns,alloc_ns,kernel_ns,store_ns,task_ns,dispatch_ns,c_bytes,task_cpu_ns,cpu_ns,cpu_p_ns,"
            + "instructions,cycles,instructions_p,cycles_p,context_switches";
    static final int OUT_OF_MEMORY_EXIT = 3;
    static final int CLUSTER_FAILURE_EXIT = 4;

    private static final int PRE_WARM_N = 256;
    private static final long PRE_WARM_SEED = 2026;
    private static final Duration CLUSTER_TIMEOUT = Duration.ofSeconds(60);

    private static volatile String stage = "start-up";

    private Main() {
    }

    /** Command-line options; schedule is null unless the method is par_ikj. */
    record Options(String method, int n, Path a, Path b, int threads, Schedule schedule, int workers,
                   int threadsPerWorker, String cluster, int basePort, Path memberLog, int pingRounds,
                   double preWarmSeconds, int warmup, int reps, double minSampleSeconds, double minWarmupSeconds,
                   Path out) {
    }

    /** One multiplication; for dist it also returns the timers of the phases. */
    private interface Operation {
        DistributedMatMul.Result run();
    }

    /** Elapsed time, process counters and (dist) phase sums of one sample. */
    private record Sample(long elapsedNs, ProcessCounters.Snapshot counters, Phases phases) {
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        if (args.length > 0 && args[0].equals("member")) {
            MemberMain.main(Arrays.copyOfRange(args, 1, args.length));
            return;
        }
        Options opt;
        try {
            opt = parse(args);
        } catch (IllegalArgumentException e) {
            System.err.println("error: " + e.getMessage());
            System.err.println("usage: --method " + String.join("|", METHODS) + " --n N --a PATH --b PATH "
                    + "[--threads P] [--schedule dynamic|static] [--workers W] [--threads-per-worker K] "
                    + "[--cluster NAME] [--base-port PORT] [--member-log PATH] [--ping-rounds R] "
                    + "[--pre-warm-seconds PW] [--warmup W] [--reps R] [--min-sample-seconds S] "
                    + "[--min-warmup-seconds SW] [--out PATH]");
            System.exit(2);
            return;
        }
        try {
            run(opt, System.out);
        } catch (OutOfMemoryError e) {
            System.out.flush();
            System.err.println("error: out of memory during " + stage + " (" + e.getMessage() + ")");
            System.exit(OUT_OF_MEMORY_EXIT);
        } catch (DistributedMatMul.ClusterFailure | IllegalStateException e) {
            System.out.flush();
            System.err.println("error: cluster failure during " + stage + ": " + e.getMessage());
            System.exit(CLUSTER_FAILURE_EXIT);
        }
        // Hazelcast's client threads are not daemon threads; leave explicitly.
        System.exit(0);
    }

    static Options parse(String[] args) {
        String method = null, schedule = null, cluster = null;
        Integer n = null;
        Path a = null, b = null, out = null, memberLog = null;
        Integer threads = null, workers = null, threadsPerWorker = null;
        int warmup = 1, reps = 5, basePort = HazelcastSetup.DEFAULT_BASE_PORT, pingRounds = 0;
        double preWarm = 1.0, minSample = 0.1, minWarmup = 1.0;
        for (int i = 0; i < args.length; i += 2) {
            if (i + 1 >= args.length) {
                throw new IllegalArgumentException("missing value for " + args[i]);
            }
            String v = args[i + 1];
            switch (args[i]) {
                case "--method" -> method = v;
                case "--n" -> n = Integer.parseInt(v);
                case "--a" -> a = Path.of(v);
                case "--b" -> b = Path.of(v);
                case "--threads" -> threads = Integer.parseInt(v);
                case "--schedule" -> schedule = v;
                case "--workers" -> workers = Integer.parseInt(v);
                case "--threads-per-worker" -> threadsPerWorker = Integer.parseInt(v);
                case "--cluster" -> cluster = v;
                case "--base-port" -> basePort = Integer.parseInt(v);
                case "--member-log" -> memberLog = Path.of(v);
                case "--ping-rounds" -> pingRounds = Integer.parseInt(v);
                case "--pre-warm-seconds" -> preWarm = Double.parseDouble(v);
                case "--warmup" -> warmup = Integer.parseInt(v);
                case "--reps" -> reps = Integer.parseInt(v);
                case "--min-sample-seconds" -> minSample = Double.parseDouble(v);
                case "--min-warmup-seconds" -> minWarmup = Double.parseDouble(v);
                case "--out" -> out = Path.of(v);
                default -> throw new IllegalArgumentException("unknown option " + args[i]);
            }
        }
        if (method == null || !METHODS.contains(method)) {
            throw new IllegalArgumentException("need --method, one of " + METHODS);
        }
        if (n == null || n < 1 || a == null || b == null || warmup < 0 || reps < 0 || warmup + reps < 1
                || minWarmup < 0 || preWarm < 0 || pingRounds < 0) {
            throw new IllegalArgumentException("need --n >= 1, --a, --b and at least one sample");
        }
        boolean parallel = method.equals("par_ikj"), dist = method.equals("dist");
        if (!parallel && (threads != null || schedule != null)) {
            throw new IllegalArgumentException("--threads and --schedule apply only to par_ikj");
        }
        if (!dist && (workers != null || threadsPerWorker != null || cluster != null || memberLog != null
                || pingRounds > 0)) {
            throw new IllegalArgumentException("--workers, --threads-per-worker, --cluster, --member-log and "
                    + "--ping-rounds apply only to dist");
        }
        if (dist && (workers == null || workers < 1 || cluster == null)) {
            throw new IllegalArgumentException("dist needs --workers >= 1 and --cluster");
        }
        int p = threads == null ? 1 : threads;
        int k = threadsPerWorker == null ? 1 : threadsPerWorker;
        if (p < 1 || k < 1) {
            throw new IllegalArgumentException("--threads and --threads-per-worker must be >= 1");
        }
        Schedule parsed = parallel ? (schedule == null ? Schedule.DYNAMIC : Schedule.parse(schedule)) : null;
        return new Options(method, n, a, b, p, parsed, dist ? workers : 0, dist ? k : 0, cluster, basePort,
                memberLog, pingRounds, preWarm, warmup, reps, minSample, minWarmup, out);
    }

    static void run(Options opt, PrintStream out) throws IOException {
        stage = "loading inputs";
        int n = opt.n();
        double[][] a = MatrixIO.loadDense(opt.a(), n);
        double[][] b = MatrixIO.loadDense(opt.b(), n);
        double[][] c = DenseMatMul.zeros(n);
        boolean dist = opt.method().equals("dist");

        stage = dist ? "connecting to the cluster" : "creating the thread pool";
        HazelcastInstance client = null;
        DistributedMatMul cluster = null;
        PrintWriter memberLog = null;
        try (ParallelMatMul pool = opt.method().equals("par_ikj") ? new ParallelMatMul(opt.threads()) : null) {
            if (dist) {
                HazelcastSetup.quietLogging();
                long t0 = System.nanoTime();
                client = HazelcastClient.newHazelcastClient(
                        HazelcastSetup.clientConfig(opt.cluster(), opt.basePort(), opt.workers()));
                long t1 = System.nanoTime();
                cluster = new DistributedMatMul(client, opt.workers(), CLUSTER_TIMEOUT);
                long t2 = System.nanoTime();
                StringBuilder keys = new StringBuilder();
                for (BandKey key : cluster.keys()) {
                    keys.append(keys.isEmpty() ? "" : ";").append(key.getPartitionKey());
                }
                out.println("# connected=1,client_start_ns=" + (t1 - t0) + ",placement_ns=" + (t2 - t1)
                        + ",partition_keys=" + keys + ",socket_counters="
                        + (DistributedMatMul.socketCountersAvailable() ? "available" : "unavailable"));
                out.flush();
                if (opt.memberLog() != null) {
                    memberLog = new PrintWriter(Files.newBufferedWriter(opt.memberLog()));
                    memberLog.println(MEMBER_HEADER);
                }
            }
            out.println("# threads=" + opt.threads() + ",schedule=" + (opt.schedule() == null ? "" : opt.schedule().label())
                    + ",workers=" + opt.workers() + ",threads_per_worker=" + opt.threadsPerWorker()
                    + ",available_processors=" + Runtime.getRuntime().availableProcessors()
                    + ",superword=" + vmOption("UseSuperWord") + ",max_vector_size=" + vmOption("MaxVectorSize")
                    + ",counters=" + (ProcessCounters.available() ? "available" : "unavailable"));
            out.println("# multiply_adds=" + (long) n * n * n);

            stage = "JIT pre-warm";
            preWarm(opt, pool, cluster, out);

            Operation op = operation(opt, pool, cluster, a, b, c);
            out.println(HEADER);
            out.flush();
            stage = "warm-up";
            int batch = 1;
            long warmupNs = 0;
            for (int i = 0; i < opt.warmup() || (opt.warmup() > 0 && warmupNs < opt.minWarmupSeconds() * 1e9); i++) {
                Sample s = runSample(op, batch);
                out.println(row("warmup", i, batch, s, DenseMatMul.checksum(c)));
                out.flush();
                memberRows(memberLog, "warmup", i, s);
                warmupNs += s.elapsedNs();
                batch = (int) Math.max(1, Math.ceil(opt.minSampleSeconds() * 1e9 * batch / Math.max(s.elapsedNs(), 1)));
            }

            stage = "measured samples";
            int jvmThreads = ManagementFactory.getThreadMXBean().getThreadCount();
            int osThreads = ProcessCounters.osThreads();
            long[] gcBefore = gcTotals();
            long jitBefore = jitMillis();
            long allocatedBefore = allocatedBytes();
            for (int i = 0; i < opt.reps(); i++) {
                Sample s = runSample(op, batch);
                out.println(row("measure", i, batch, s, DenseMatMul.checksum(c)));
                out.flush();
                memberRows(memberLog, "measure", i, s);
            }
            long[] gcAfter = gcTotals();
            long allocatedAfter = allocatedBytes();
            out.println("# jvm_threads=" + jvmThreads + ",os_threads=" + osThreads
                    + ",gc_collections=" + (gcAfter[0] - gcBefore[0])
                    + ",gc_ms=" + (gcAfter[1] - gcBefore[1]) + ",jit_ms=" + (jitMillis() - jitBefore)
                    + ",allocated_bytes_measured=" + (allocatedAfter - allocatedBefore)
                    + ",allocated_bytes_total=" + allocatedAfter + ",gc_collections_total=" + gcAfter[0]);

            if (cluster != null) {
                stage = "ping and clean-up";
                if (opt.pingRounds() > 0) {
                    out.println("# " + DistributedMatMul.pingSummary(cluster.ping(opt.pingRounds())));
                }
                int[] sizes = cluster.mapSizes();
                out.println("# map_a_entries=" + sizes[0] + ",map_b_entries=" + sizes[1] + ",map_c_entries=" + sizes[2]);
                cluster.clear();
            }

            stage = "writing the result";
            StringBuilder qos = new StringBuilder();
            for (Map.Entry<String, Long> e : ProcessCounters.qosCpuNs().entrySet()) {
                qos.append(qos.isEmpty() ? "# " : ",").append("qos_").append(e.getKey()).append("_ns=").append(e.getValue());
            }
            if (!qos.isEmpty()) {
                out.println(qos);
            }
            if (opt.out() != null) {
                MatrixIO.writeDense(opt.out(), c);
            }
            out.flush();
        } finally {
            if (memberLog != null) {
                memberLog.close();
            }
            if (client != null) {
                client.shutdown();
            }
        }
    }

    /** One multiplication C = A * B with the method of the options. */
    private static Operation operation(Options opt, ParallelMatMul pool, DistributedMatMul cluster, double[][] a,
                                       double[][] b, double[][] c) {
        int n = c.length;
        return switch (opt.method()) {
            case "basic" -> () -> {
                DenseMatMul.basic(a, b, c, n);
                return null;
            };
            case "ikj" -> () -> {
                DenseMatMul.ikj(a, b, c, n);
                return null;
            };
            case "par_ikj" -> {
                Schedule schedule = opt.schedule();
                yield () -> {
                    pool.ikj(a, b, c, n, schedule);
                    return null;
                };
            }
            default -> () -> cluster.multiply(a, b, c, n);
        };
    }

    /** Runs the method on small random inputs for the requested time (not recorded as samples). */
    private static void preWarm(Options opt, ParallelMatMul pool, DistributedMatMul cluster, PrintStream out) {
        if (opt.preWarmSeconds() <= 0) {
            return;
        }
        int m = Math.min(opt.n(), PRE_WARM_N);
        Random rng = new Random(PRE_WARM_SEED);
        double[][] x = randomMatrix(m, rng), y = randomMatrix(m, rng), z = DenseMatMul.zeros(m);
        Operation op = operation(opt, pool, cluster, x, y, z);
        long ops = 0, t0 = System.nanoTime(), limit = (long) (opt.preWarmSeconds() * 1e9);
        double consumed = 0.0;
        while (System.nanoTime() - t0 < limit) {
            op.run();
            op.run();
            consumed += DenseMatMul.checksum(z);
            ops += 2;
        }
        out.println("# pre_warm_n=" + m + ",pre_warm_ops=" + ops + ",pre_warm_ns=" + (System.nanoTime() - t0)
                + ",pre_warm_checksum=" + consumed);
    }

    private static double[][] randomMatrix(int m, Random rng) {
        double[][] x = new double[m][m];
        for (double[] row : x) {
            for (int j = 0; j < m; j++) {
                row[j] = rng.nextDouble(-1.0, 1.0);
            }
        }
        return x;
    }

    private static Sample runSample(Operation op, int batch) {
        Phases phases = new Phases();
        ProcessCounters.Snapshot before = ProcessCounters.read();
        long t0 = System.nanoTime();
        for (int r = 0; r < batch; r++) {
            DistributedMatMul.Result result = op.run();
            if (result != null) {
                phases.add(result);
            }
        }
        long elapsed = System.nanoTime() - t0;
        ProcessCounters.Snapshot after = ProcessCounters.read();
        return new Sample(elapsed, before == null || after == null ? null : after.minus(before),
                phases.ops == 0 ? null : phases);
    }

    private static String row(String kind, int index, int batch, Sample s, double checksum) {
        StringBuilder sb = new StringBuilder();
        sb.append(kind).append(',').append(index).append(',').append(batch).append(',').append(s.elapsedNs())
                .append(',').append(checksum);
        ProcessCounters.Snapshot d = s.counters();
        if (d == null) {
            sb.append(",,,,,,,");
        } else {
            sb.append(',').append(d.cpuNs()).append(',').append(d.cpuPNs()).append(',').append(d.instructions())
                    .append(',').append(d.cycles()).append(',').append(d.instructionsP()).append(',').append(d.cyclesP())
                    .append(',').append(d.contextSwitches());
        }
        long[] values = s.phases() == null ? null : s.phases().values();
        for (int i = 0; i < DIST_FIELDS.size(); i++) {
            sb.append(',');
            if (values != null) {
                sb.append(values[i]);
            }
        }
        return sb.toString();
    }

    private static void memberRows(PrintWriter log, String kind, int index, Sample s) {
        if (log == null || s.phases() == null) {
            return;
        }
        for (Phases.Member m : s.phases().members) {
            log.println(kind + "," + index + "," + m.band + "," + m.memberIndex + "," + m.pid + "," + m.rows + ","
                    + m.kernelThreads + "," + m.ops + "," + m.local + "," + m.readANs + "," + m.readBNs + ","
                    + m.allocNs + "," + m.kernelNs + "," + m.storeNs + "," + m.taskNs + "," + m.dispatchNs + ","
                    + m.cBytes + "," + m.taskCpuNs + "," + m.cpuNs + "," + m.cpuPNs + "," + m.instructions + ","
                    + m.cycles + "," + m.instructionsP + "," + m.cyclesP + "," + m.contextSwitches);
        }
        log.flush();
    }

    /** Phase timers, bytes and per-member task figures, summed over the multiplications of one sample. */
    static final class Phases {
        long ops;
        long serializeNs, scatterNs, broadcastNs, computeNs, gatherNs, assembleNs;
        long taskMaxNs, taskMeanNs, kernelMaxNs, kernelMeanNs, readMaxNs, storeMaxNs, dispatchMaxNs, tasksDoneNs;
        long aBytes, bBytes, cBytes;
        long scatterOut, broadcastOut, computeOut, computeIn, gatherIn, socketOut, socketIn;
        Member[] members;

        /** Per-band sums over the multiplications of the sample. */
        static final class Member {
            int band, memberIndex, rows, kernelThreads;
            long pid, ops;
            boolean local = true;
            long readANs, readBNs, allocNs, kernelNs, storeNs, taskNs, dispatchNs, cBytes, taskCpuNs;
            long cpuNs, cpuPNs, instructions, cycles, instructionsP, cyclesP, contextSwitches;
        }

        void add(DistributedMatMul.Result r) {
            ops++;
            serializeNs += r.serializeNs();
            scatterNs += r.scatterNs();
            broadcastNs += r.broadcastNs();
            computeNs += r.computeNs();
            gatherNs += r.gatherNs();
            assembleNs += r.assembleNs();
            aBytes += r.aPayloadBytes();
            bBytes += r.bPayloadBytes();
            cBytes += r.cPayloadBytes();
            scatterOut = sum(scatterOut, r.scatterSocketOut());
            broadcastOut = sum(broadcastOut, r.broadcastSocketOut());
            computeOut = sum(computeOut, r.computeSocketOut());
            computeIn = sum(computeIn, r.computeSocketIn());
            gatherIn = sum(gatherIn, r.gatherSocketIn());
            socketOut = sum(socketOut, r.socketOut());
            socketIn = sum(socketIn, r.socketIn());
            TaskReport[] reports = r.reports();
            if (members == null) {
                members = new Member[reports.length];
                for (int t = 0; t < reports.length; t++) {
                    members[t] = new Member();
                }
            }
            long taskMax = 0, taskSum = 0, kernelMax = 0, kernelSum = 0, readMax = 0, storeMax = 0, dispatchMax = 0;
            long doneMax = 0;
            for (int t = 0; t < reports.length; t++) {
                TaskReport x = reports[t];
                Member m = members[t];
                long dispatch = x.startNs - r.computeStartNs();
                m.band = x.band;
                m.memberIndex = x.memberIndex;
                m.rows = x.rows;
                m.kernelThreads = x.kernelThreads;
                m.pid = x.pid;
                m.ops++;
                m.local &= x.local;
                m.readANs += x.readANs;
                m.readBNs += x.readBNs;
                m.allocNs += x.allocNs;
                m.kernelNs += x.kernelNs;
                m.storeNs += x.storeNs;
                m.taskNs += x.taskNs;
                m.dispatchNs += dispatch;
                m.cBytes += x.cBytes;
                m.taskCpuNs = sum(m.taskCpuNs, x.taskCpuNs);
                m.cpuNs = sum(m.cpuNs, x.cpuNs);
                m.cpuPNs = sum(m.cpuPNs, x.cpuPNs);
                m.instructions = sum(m.instructions, x.instructions);
                m.cycles = sum(m.cycles, x.cycles);
                m.instructionsP = sum(m.instructionsP, x.instructionsP);
                m.cyclesP = sum(m.cyclesP, x.cyclesP);
                m.contextSwitches = sum(m.contextSwitches, x.contextSwitches);
                taskMax = Math.max(taskMax, x.taskNs);
                taskSum += x.taskNs;
                kernelMax = Math.max(kernelMax, x.kernelNs);
                kernelSum += x.kernelNs;
                readMax = Math.max(readMax, x.readANs + x.readBNs);
                storeMax = Math.max(storeMax, x.storeNs);
                dispatchMax = Math.max(dispatchMax, dispatch);
                doneMax = Math.max(doneMax, x.endNs - r.computeStartNs());
            }
            taskMaxNs += taskMax;
            taskMeanNs += taskSum / reports.length;
            kernelMaxNs += kernelMax;
            kernelMeanNs += kernelSum / reports.length;
            readMaxNs += readMax;
            storeMaxNs += storeMax;
            dispatchMaxNs += dispatchMax;
            tasksDoneNs += doneMax;
        }

        /** Values in the order of {@link #DIST_FIELDS}. */
        long[] values() {
            return new long[]{serializeNs, scatterNs, broadcastNs, computeNs, gatherNs, assembleNs, taskMaxNs,
                    taskMeanNs, kernelMaxNs, kernelMeanNs, readMaxNs, storeMaxNs, dispatchMaxNs, tasksDoneNs, aBytes,
                    bBytes, cBytes, scatterOut, broadcastOut, computeOut, computeIn, gatherIn, socketOut, socketIn};
        }

        /** Sum of counters where -1 marks an unavailable value. */
        private static long sum(long total, long value) {
            return total < 0 || value < 0 ? -1 : total + value;
        }
    }

    /** Total collections and collection time (ms) over all garbage collectors. */
    private static long[] gcTotals() {
        long count = 0, millis = 0;
        for (GarbageCollectorMXBean gc : ManagementFactory.getGarbageCollectorMXBeans()) {
            count += Math.max(0, gc.getCollectionCount());
            millis += Math.max(0, gc.getCollectionTime());
        }
        return new long[]{count, millis};
    }

    /** Bytes allocated on the Java heap by all threads since start-up (an approximation kept by the JVM). */
    private static long allocatedBytes() {
        var threads = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
        return threads.isThreadAllocatedMemorySupported() ? threads.getTotalThreadAllocatedBytes() : -1;
    }

    private static long jitMillis() {
        var jit = ManagementFactory.getCompilationMXBean();
        return jit != null && jit.isCompilationTimeMonitoringSupported() ? jit.getTotalCompilationTime() : -1;
    }

    private static String vmOption(String name) {
        try {
            return ManagementFactory.getPlatformMXBean(HotSpotDiagnosticMXBean.class).getVMOption(name).getValue();
        } catch (RuntimeException e) {
            return "";
        }
    }
}
