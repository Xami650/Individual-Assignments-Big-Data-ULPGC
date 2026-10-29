package org.ulpgc.bd.matmul;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleSupplier;
import java.util.function.Supplier;

/**
 * Benchmark driver for every method of Assignment 2.
 *
 * <pre>
 *   java -Xmx8g -jar target/matmul.jar --method M --n N --a PATH --b PATH [--block BS]
 *        [--pre-warm-seconds P] [--warmup W] [--reps R] [--min-sample-seconds S]
 *        [--min-warmup-seconds SW] [--out PATH]
 * </pre>
 * Methods: {@code basic}, {@code ikj}, {@code blocked} (dense kernels), {@code csr}
 * (sparse product), {@code to_csr} (dense to CSR conversion of A and B) and
 * {@code to_dense} (CSR to dense conversion of the product C). Files ending in
 * {@code .csr} are read as CSR, all others as raw dense (see {@link MatrixIO}).
 * Inputs are converted to the representation of the method before any timing,
 * so the timed region holds only the operation itself. Methods that allocate
 * their result drop the previous one before each call, so that at most one
 * result is reachable at a time.
 *
 * <p>Before any sample, the operation runs for P seconds (default 1) on small
 * random inputs (n = min(N, 256), densities 2% and 50%), so that the JIT
 * compiles the method with a complete profile. Without this step a single long
 * call can stay in on-stack-replacement or C1 code, or be deoptimised when its
 * loops exit for the first time.
 *
 * <p>Output on stdout follows Assignment 1: CSV with the header
 * {@code kind,index,batch,elapsed_ns,checksum} and one row per sample, where a
 * sample times {@code batch} back-to-back operations with {@link System#nanoTime()}.
 * The first warm-up uses batch = 1 and every later sample uses
 * k = max(1, ceil(S / t)), t being the per-operation time of the latest warm-up;
 * warm-up continues beyond W samples until SW seconds of warm-up work have run.
 * Lines starting with '#' carry key=value metadata (nonzero counts, multiply-adds).
 * With --out, the product C is written as raw dense float64 after the last sample.
 * Exit status 3 means the JVM ran out of heap (the memory budget, set by -Xmx).
 */
public final class Main {

    static final List<String> METHODS = List.of("basic", "ikj", "blocked", "csr", "to_csr", "to_dense");

    private static final int PRE_WARM_N = 256;
    private static final double[] PRE_WARM_DENSITIES = {0.02, 0.5};
    private static final long PRE_WARM_SEED = 2026;

    private static volatile String stage = "start-up";

    private Main() {
    }

    record Options(String method, int n, Path a, Path b, int block, double preWarmSeconds, int warmup, int reps,
                   double minSampleSeconds, double minWarmupSeconds, Path out) {
    }

    /**
     * One timed operation, a checksum of its latest result, the metadata of that
     * result (nonzero counts) and, if the method computes C, a dense view of C
     * (built only for --out, so that it never adds to the measured memory).
     */
    private record Task(Runnable op, DoubleSupplier checksum, Supplier<String> resultInfo, ProductView product) {
    }

    @FunctionalInterface
    private interface ProductView {
        double[][] dense();
    }

    public static void main(String[] args) throws IOException {
        Options opt;
        try {
            opt = parse(args);
        } catch (IllegalArgumentException e) {
            System.err.println("error: " + e.getMessage());
            System.err.println("usage: --method " + String.join("|", METHODS) + " --n N --a PATH --b PATH [--block BS] "
                    + "[--pre-warm-seconds P] [--warmup W] [--reps R] [--min-sample-seconds S] "
                    + "[--min-warmup-seconds SW] [--out PATH]");
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
        String method = null;
        Integer n = null;
        Path a = null, b = null, out = null;
        int block = DenseMatMul.DEFAULT_BLOCK, warmup = 1, reps = 5;
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
                case "--block" -> block = Integer.parseInt(v);
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
        if (n == null || n < 1 || a == null || b == null || block < 1 || warmup < 0 || reps < 0
                || warmup + reps < 1 || minWarmup < 0 || preWarm < 0) {
            throw new IllegalArgumentException("need --n >= 1, --a, --b, --block >= 1 and at least one sample");
        }
        if (out != null && method.equals("to_csr")) {
            throw new IllegalArgumentException("--out is not available for to_csr, which does not compute C");
        }
        return new Options(method, n, a, b, block, preWarm, warmup, reps, minSample, minWarmup, out);
    }

    static void run(Options opt, PrintStream out) throws IOException {
        stage = "loading inputs";
        int n = opt.n();
        Object a = load(opt.a(), n);
        Object b = load(opt.b(), n);
        out.println("# nnz_a=" + nnz(a) + ",nnz_b=" + nnz(b));

        stage = "converting inputs";
        Task task = prepare(opt, a, b, out);
        a = null;
        b = null;

        stage = "JIT pre-warm";
        preWarm(opt, out);

        out.println("kind,index,batch,elapsed_ns,checksum");
        out.flush();
        stage = "warm-up";
        int batch = 1;
        long warmupNs = 0;
        for (int i = 0; i < opt.warmup() || (opt.warmup() > 0 && warmupNs < opt.minWarmupSeconds() * 1e9); i++) {
            long elapsed = runSample(task.op(), batch);
            out.println("warmup," + i + "," + batch + "," + elapsed + "," + task.checksum().getAsDouble());
            out.flush();
            warmupNs += elapsed;
            batch = (int) Math.max(1, Math.ceil(opt.minSampleSeconds() * 1e9 * batch / Math.max(elapsed, 1)));
        }

        stage = "measured samples";
        for (int i = 0; i < opt.reps(); i++) {
            long elapsed = runSample(task.op(), batch);
            out.println("measure," + i + "," + batch + "," + elapsed + "," + task.checksum().getAsDouble());
            out.flush();
        }

        stage = "writing the result";
        if (task.resultInfo() != null) {
            out.println("# " + task.resultInfo().get());
        }
        if (opt.out() != null) {
            MatrixIO.writeDense(opt.out(), task.product().dense());
        }
        out.flush();
    }

    /** Converts the inputs for the method (outside the timed region) and builds its timed operation. */
    private static Task prepare(Options opt, Object a, Object b, PrintStream out) {
        int n = opt.n();
        switch (opt.method()) {
            case "basic", "ikj", "blocked" -> {
                double[][] da = dense(a), db = dense(b), c = DenseMatMul.zeros(n);
                out.println("# multiply_adds=" + (long) n * n * n);
                int bs = opt.block();
                Runnable op = switch (opt.method()) {
                    case "basic" -> () -> DenseMatMul.basic(da, db, c, n);
                    case "ikj" -> () -> DenseMatMul.ikj(da, db, c, n);
                    default -> () -> DenseMatMul.blocked(da, db, c, n, bs);
                };
                return new Task(op, () -> DenseMatMul.checksum(c),
                        () -> "nnz_c_nonzero=" + DenseMatMul.countNonzeros(c), () -> c);
            }
            case "csr" -> {
                CsrMatrix sa = csr(a), sb = csr(b);
                out.println("# multiply_adds=" + CsrMatrix.multiplyAdds(sa, sb));
                CsrMatrix[] c = new CsrMatrix[1];
                Runnable op = () -> {
                    c[0] = null;
                    c[0] = CsrMatrix.multiply(sa, sb);
                };
                return new Task(op, () -> c[0].checksum(),
                        () -> "nnz_c=" + c[0].nnz() + ",nnz_c_nonzero=" + nonzeroValues(c[0]), () -> c[0].toDense());
            }
            case "to_csr" -> {
                double[][] da = dense(a), db = dense(b);
                CsrMatrix[] r = new CsrMatrix[2];
                Runnable op = () -> {
                    r[0] = null;
                    r[1] = null;
                    r[0] = CsrMatrix.fromDense(da);
                    r[1] = CsrMatrix.fromDense(db);
                };
                return new Task(op, () -> r[0].checksum() + r[1].checksum(), null, null);
            }
            case "to_dense" -> {
                CsrMatrix product = CsrMatrix.multiply(csr(a), csr(b));
                out.println("# nnz_c=" + product.nnz());
                double[][][] r = new double[1][][];
                Runnable op = () -> {
                    r[0] = null;
                    r[0] = product.toDense();
                };
                return new Task(op, () -> DenseMatMul.checksum(r[0]),
                        () -> "nnz_c_nonzero=" + DenseMatMul.countNonzeros(r[0]), () -> r[0]);
            }
            default -> throw new IllegalStateException(opt.method());
        }
    }

    /** Runs the method's operation on small random inputs for the requested time (not recorded as samples). */
    private static void preWarm(Options opt, PrintStream out) {
        if (opt.preWarmSeconds() <= 0) {
            return;
        }
        int m = Math.min(opt.n(), PRE_WARM_N);
        Options small = new Options(opt.method(), m, opt.a(), opt.b(), opt.block(), 0, 0, 1, 0, 0, null);
        PrintStream sink = new PrintStream(OutputStream.nullOutputStream());
        Random rng = new Random(PRE_WARM_SEED);
        Task[] tasks = new Task[PRE_WARM_DENSITIES.length];
        for (int t = 0; t < tasks.length; t++) {
            double[][] x = randomMatrix(m, PRE_WARM_DENSITIES[t], rng);
            double[][] y = randomMatrix(m, PRE_WARM_DENSITIES[t], rng);
            boolean sparse = opt.method().equals("csr") || opt.method().equals("to_dense");
            tasks[t] = prepare(small, sparse ? CsrMatrix.fromDense(x) : x, sparse ? CsrMatrix.fromDense(y) : y, sink);
        }
        long ops = 0, t0 = System.nanoTime(), limit = (long) (opt.preWarmSeconds() * 1e9);
        double consumed = 0.0;
        while (System.nanoTime() - t0 < limit) {
            for (Task task : tasks) {
                runSample(task.op(), 2);
                consumed += task.checksum().getAsDouble();
                ops += 2;
            }
        }
        out.println("# pre_warm_n=" + m + ",pre_warm_ops=" + ops + ",pre_warm_ns=" + (System.nanoTime() - t0)
                + ",pre_warm_checksum=" + consumed);
    }

    /** Random m x m matrix with each entry nonzero (uniform on [-1, 1)) with the given probability. */
    private static double[][] randomMatrix(int m, double density, Random rng) {
        double[][] x = new double[m][m];
        for (double[] row : x) {
            for (int j = 0; j < m; j++) {
                if (rng.nextDouble() < density) {
                    row[j] = rng.nextDouble(-1.0, 1.0);
                }
            }
        }
        return x;
    }

    private static Object load(Path path, int n) throws IOException {
        return path.toString().endsWith(".csr") ? MatrixIO.loadCsr(path, n) : MatrixIO.loadDense(path, n);
    }

    private static long nnz(Object m) {
        return m instanceof CsrMatrix s ? s.nnz() : DenseMatMul.countNonzeros((double[][]) m);
    }

    private static long nonzeroValues(CsrMatrix m) {
        long count = 0;
        for (double x : m.values) {
            if (x != 0.0) {
                count++;
            }
        }
        return count;
    }

    private static double[][] dense(Object m) {
        return m instanceof CsrMatrix s ? s.toDense() : (double[][]) m;
    }

    private static CsrMatrix csr(Object m) {
        return m instanceof CsrMatrix s ? s : CsrMatrix.fromDense((double[][]) m);
    }

    private static long runSample(Runnable op, int batch) {
        long t0 = System.nanoTime();
        for (int r = 0; r < batch; r++) {
            op.run();
        }
        return System.nanoTime() - t0;
    }
}
