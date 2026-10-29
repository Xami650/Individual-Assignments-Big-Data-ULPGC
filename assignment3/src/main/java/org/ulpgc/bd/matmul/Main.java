package org.ulpgc.bd.matmul;

import com.sun.management.HotSpotDiagnosticMXBean;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.ulpgc.bd.matmul.ParallelMatMul.Schedule;

/**
 * Benchmark driver for Assignment 3, adapted from the Assignment 2 driver (tag
 * assignment2-v1.0): the same pre-warm, warm-up, batching and output format,
 * for the sequential {@code ikj} and the parallel {@code par_ikj}.
 *
 * <pre>
 *   java -Xmx8g -jar target/matmul.jar --method ikj|par_ikj --n N --a PATH --b PATH
 *        [--threads P] [--schedule dynamic|static] [--pre-warm-seconds PW] [--warmup W]
 *        [--reps R] [--min-sample-seconds S] [--min-warmup-seconds SW] [--out PATH]
 * </pre>
 * {@code ikj} runs on the main thread. {@code par_ikj} creates a pool of P worker
 * threads before any timing (default schedule: dynamic) and shuts it down at the end.
 *
 * <p>Before any sample, the method runs for PW seconds (default 1) on small
 * random inputs (n = min(N, 256)) with the same pool, so that the JIT compiles it
 * with a complete profile. Each sample then times {@code batch} back-to-back
 * multiplications with {@link System#nanoTime()}: the first warm-up uses
 * batch = 1 and every later sample k = max(1, ceil(S / t)), t being the time
 * per multiplication of the latest warm-up; warm-up continues beyond W samples
 * until SW seconds of warm-up work have run.
 *
 * <p>Output on stdout is CSV with the header {@link #HEADER} and one row per
 * sample. Besides the time and a checksum of C, each row holds the change of
 * the process counters over the sample ({@link ProcessCounters}): CPU time,
 * CPU time on performance cores, instructions and cycles (total and on
 * performance cores) and context switches, summed over all threads; these
 * fields are empty where the counters are unavailable. Lines starting with '#'
 * carry key=value metadata: settings, pre-warm counts, the thread counts of the
 * process, and the JVM's own activity (garbage collections, JIT compilation,
 * heap allocation) over the measured samples and since start-up. With --out, C is
 * written as raw dense float64 after the last sample. Exit status 3 means the
 * JVM ran out of heap (the memory budget, set by -Xmx).
 */
public final class Main {

    static final List<String> METHODS = List.of("ikj", "par_ikj");
    static final String HEADER =
            "kind,index,batch,elapsed_ns,checksum,cpu_ns,cpu_p_ns,instructions,cycles,instructions_p,cycles_p,"
                    + "context_switches";

    private static final int PRE_WARM_N = 256;
    private static final long PRE_WARM_SEED = 2026;

    private static volatile String stage = "start-up";

    private Main() {
    }

    /** Command-line options; schedule is null for ikj. */
    record Options(String method, int n, Path a, Path b, int threads, Schedule schedule, double preWarmSeconds,
                   int warmup, int reps, double minSampleSeconds, double minWarmupSeconds, Path out) {
    }

    /** Elapsed time of one sample and the change of the process counters (null if unavailable). */
    private record Sample(long elapsedNs, ProcessCounters.Snapshot counters) {
    }

    public static void main(String[] args) throws IOException {
        Options opt;
        try {
            opt = parse(args);
        } catch (IllegalArgumentException e) {
            System.err.println("error: " + e.getMessage());
            System.err.println("usage: --method " + String.join("|", METHODS) + " --n N --a PATH --b PATH "
                    + "[--threads P] [--schedule dynamic|static] [--pre-warm-seconds PW] [--warmup W] [--reps R] "
                    + "[--min-sample-seconds S] [--min-warmup-seconds SW] [--out PATH]");
            System.exit(2);
            return;
        }
        try {
            run(opt, System.out);
        } catch (OutOfMemoryError e) {
            System.out.flush();
            System.err.println("error: out of memory during " + stage + " (" + e.getMessage() + ")");
            System.exit(3);
        }
    }

    static Options parse(String[] args) {
        String method = null, schedule = null;
        Integer n = null;
        Path a = null, b = null, out = null;
        int threads = 1, warmup = 1, reps = 5;
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
        if (n == null || n < 1 || a == null || b == null || threads < 1 || warmup < 0 || reps < 0
                || warmup + reps < 1 || minWarmup < 0 || preWarm < 0) {
            throw new IllegalArgumentException("need --n >= 1, --a, --b, --threads >= 1 and at least one sample");
        }
        Schedule parsed = null;
        if (method.equals("par_ikj")) {
            parsed = schedule == null ? Schedule.DYNAMIC : Schedule.parse(schedule);
        } else if (threads != 1 || schedule != null) {
            throw new IllegalArgumentException("ikj is sequential: --threads and --schedule apply only to par_ikj");
        }
        return new Options(method, n, a, b, threads, parsed, preWarm, warmup, reps, minSample, minWarmup, out);
    }

    static void run(Options opt, PrintStream out) throws IOException {
        stage = "loading inputs";
        int n = opt.n();
        double[][] a = MatrixIO.loadDense(opt.a(), n);
        double[][] b = MatrixIO.loadDense(opt.b(), n);
        double[][] c = DenseMatMul.zeros(n);

        stage = "creating the thread pool";
        try (ParallelMatMul pool = opt.method().equals("par_ikj") ? new ParallelMatMul(opt.threads()) : null) {
            out.println("# threads=" + opt.threads() + ",schedule=" + (opt.schedule() == null ? "" : opt.schedule().label())
                    + ",available_processors=" + Runtime.getRuntime().availableProcessors()
                    + ",superword=" + vmOption("UseSuperWord") + ",max_vector_size=" + vmOption("MaxVectorSize")
                    + ",counters=" + (ProcessCounters.available() ? "available" : "unavailable"));
            out.println("# multiply_adds=" + (long) n * n * n);

            stage = "JIT pre-warm";
            preWarm(opt, pool, out);

            Runnable op = operation(opt, pool, a, b, c);
            out.println(HEADER);
            out.flush();
            stage = "warm-up";
            int batch = 1;
            long warmupNs = 0;
            for (int i = 0; i < opt.warmup() || (opt.warmup() > 0 && warmupNs < opt.minWarmupSeconds() * 1e9); i++) {
                Sample s = runSample(op, batch);
                out.println(row("warmup", i, batch, s, DenseMatMul.checksum(c)));
                out.flush();
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
            }
            long[] gcAfter = gcTotals();
            long allocatedAfter = allocatedBytes();
            out.println("# jvm_threads=" + jvmThreads + ",os_threads=" + osThreads
                    + ",gc_collections=" + (gcAfter[0] - gcBefore[0])
                    + ",gc_ms=" + (gcAfter[1] - gcBefore[1]) + ",jit_ms=" + (jitMillis() - jitBefore)
                    + ",allocated_bytes_measured=" + (allocatedAfter - allocatedBefore)
                    + ",allocated_bytes_total=" + allocatedAfter + ",gc_collections_total=" + gcAfter[0]);

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
        }
    }

    /** One multiplication C = A * B with the method and pool of the options. */
    private static Runnable operation(Options opt, ParallelMatMul pool, double[][] a, double[][] b, double[][] c) {
        int n = c.length;
        if (pool == null) {
            return () -> DenseMatMul.ikj(a, b, c, n);
        }
        Schedule schedule = opt.schedule();
        return () -> pool.ikj(a, b, c, n, schedule);
    }

    /** Runs the method on small random inputs for the requested time (not recorded as samples). */
    private static void preWarm(Options opt, ParallelMatMul pool, PrintStream out) {
        if (opt.preWarmSeconds() <= 0) {
            return;
        }
        int m = Math.min(opt.n(), PRE_WARM_N);
        Random rng = new Random(PRE_WARM_SEED);
        double[][] x = randomMatrix(m, rng), y = randomMatrix(m, rng), z = DenseMatMul.zeros(m);
        Runnable op = operation(opt, pool, x, y, z);
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

    private static Sample runSample(Runnable op, int batch) {
        ProcessCounters.Snapshot before = ProcessCounters.read();
        long t0 = System.nanoTime();
        for (int r = 0; r < batch; r++) {
            op.run();
        }
        long elapsed = System.nanoTime() - t0;
        ProcessCounters.Snapshot after = ProcessCounters.read();
        return new Sample(elapsed, before == null || after == null ? null : after.minus(before));
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
        return sb.toString();
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
