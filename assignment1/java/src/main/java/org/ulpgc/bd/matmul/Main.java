package org.ulpgc.bd.matmul;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;

/**
 * Benchmark driver for the Java basic matrix multiplication.
 *
 * <p>Common command-line contract shared with the Python and C drivers:
 * <pre>
 *   java -Xmx4g -jar java/target/matmul.jar --n N [--a PATH] [--b PATH] [--warmup W]
 *        [--reps R] [--min-sample-seconds S] [--out PATH]
 * </pre>
 * Inputs default to data/inputs/A_&lt;n&gt;.bin and B_&lt;n&gt;.bin relative to the working
 * directory (run from assignment1/). Output on stdout is CSV with the header
 * {@code kind,index,batch,elapsed_ns,checksum} and one row per sample. Each sample
 * times {@code batch} back-to-back multiplications with {@link System#nanoTime()}
 * (kernel only). Warm-up samples let the JIT compile the kernel; the first uses
 * batch = 1 and every later sample uses k = max(1, ceil(S / t)), with t the
 * per-multiplication time of the most recent warm-up. The checksum is the sum
 * of all entries of C, printed with {@link Double#toString(double)}, which round-trips.
 * With --out, C is written as raw little-endian float64 after the last sample.
 */
public final class Main {

    private Main() {
    }

    record Options(int n, Path a, Path b, int warmup, int reps, double minSampleSeconds, Path out) {
    }

    public static void main(String[] args) throws IOException {
        Options opt;
        try {
            opt = parse(args);
        } catch (IllegalArgumentException e) {
            System.err.println("error: " + e.getMessage());
            System.err.println("usage: --n N [--a PATH] [--b PATH] [--warmup W] [--reps R] "
                    + "[--min-sample-seconds S] [--out PATH]");
            System.exit(2);
            return;
        }
        run(opt, System.out);
    }

    static Options parse(String[] args) {
        Integer n = null;
        Path a = null, b = null, out = null;
        int warmup = 2, reps = 5;
        double minSample = 0.1;
        for (int i = 0; i < args.length; i += 2) {
            if (i + 1 >= args.length) {
                throw new IllegalArgumentException("missing value for " + args[i]);
            }
            String v = args[i + 1];
            switch (args[i]) {
                case "--n" -> n = Integer.parseInt(v);
                case "--a" -> a = Path.of(v);
                case "--b" -> b = Path.of(v);
                case "--warmup" -> warmup = Integer.parseInt(v);
                case "--reps" -> reps = Integer.parseInt(v);
                case "--min-sample-seconds" -> minSample = Double.parseDouble(v);
                case "--out" -> out = Path.of(v);
                default -> throw new IllegalArgumentException("unknown option " + args[i]);
            }
        }
        if (n == null || n < 1 || warmup < 0 || reps < 0 || warmup + reps < 1) {
            throw new IllegalArgumentException("need --n >= 1 and at least one warm-up or measured sample");
        }
        Path inputs = Path.of("data", "inputs");
        return new Options(n,
                a != null ? a : inputs.resolve("A_" + n + ".bin"),
                b != null ? b : inputs.resolve("B_" + n + ".bin"),
                warmup, reps, minSample, out);
    }

    static void run(Options opt, PrintStream out) throws IOException {
        int n = opt.n();
        double[][] a = MatMul.load(opt.a(), n);
        double[][] b = MatMul.load(opt.b(), n);
        double[][] c = MatMul.zeros(n);

        out.println("kind,index,batch,elapsed_ns,checksum");
        out.flush();
        int batch = 1;
        for (int i = 0; i < opt.warmup(); i++) {
            long elapsed = runSample(a, b, c, n, batch);
            out.println("warmup," + i + "," + batch + "," + elapsed + "," + MatMul.checksum(c));
            out.flush();
            batch = (int) Math.max(1, Math.ceil(opt.minSampleSeconds() * 1e9 * batch / Math.max(elapsed, 1)));
        }

        for (int i = 0; i < opt.reps(); i++) {
            long elapsed = runSample(a, b, c, n, batch);
            out.println("measure," + i + "," + batch + "," + elapsed + "," + MatMul.checksum(c));
            out.flush();
        }

        if (opt.out() != null) {
            MatMul.write(opt.out(), c);
        }
    }

    private static long runSample(double[][] a, double[][] b, double[][] c, int n, int batch) {
        long t0 = System.nanoTime();
        for (int r = 0; r < batch; r++) {
            MatMul.multiply(a, b, c, n);
        }
        return System.nanoTime() - t0;
    }
}
