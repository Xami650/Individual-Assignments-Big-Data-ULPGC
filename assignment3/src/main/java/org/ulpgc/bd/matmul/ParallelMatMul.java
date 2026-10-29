package org.ulpgc.bd.matmul;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Row-parallel version of {@link DenseMatMul#ikj}: the rows of C are distributed
 * among a fixed pool of p worker threads, and every row is computed exactly as
 * in the sequential method ({@link #ikjRow}).
 *
 * <p>Shared data: A and B are only read, and every row of C is written by
 * exactly one task. {@link #ikj} returns only after every task has completed,
 * and the actions of a task happen-before the return of its
 * {@code Future.get()} (memory consistency properties of
 * {@code java.util.concurrent}), so the caller sees every entry of C. Each
 * C[i][j] receives the same operations in the same order as in the sequential
 * method, so the result is bitwise identical for any p and either schedule.
 *
 * <p>The pool is created, and its p threads are started, in the constructor, so
 * that timing a call measures the hand-off of the work, the computation and
 * the wait, but not the creation of threads. The calling thread only submits
 * the tasks and waits; it never computes a row.
 */
public final class ParallelMatMul implements AutoCloseable {

    /** How the rows of C are assigned to the worker threads. */
    public enum Schedule {
        /** Each worker claims the next unclaimed row from a shared counter until none are left. */
        DYNAMIC,
        /** Worker t computes the contiguous band of rows [t n / p, (t + 1) n / p), fixed in advance. */
        STATIC;

        /** Parses "dynamic" or "static" (any case). */
        public static Schedule parse(String name) {
            for (Schedule s : values()) {
                if (s.name().equalsIgnoreCase(name)) {
                    return s;
                }
            }
            throw new IllegalArgumentException("unknown schedule " + name + ", expected dynamic or static");
        }

        /** The name used on the command line and in the results ("dynamic", "static"). */
        public String label() {
            return name().toLowerCase(Locale.ROOT);
        }
    }

    private final int threads;
    private final ThreadPoolExecutor pool;

    /** Creates a pool of exactly {@code threads} worker threads and starts them. */
    public ParallelMatMul(int threads) {
        if (threads < 1) {
            throw new IllegalArgumentException("threads must be >= 1, got " + threads);
        }
        this.threads = threads;
        AtomicInteger created = new AtomicInteger();
        ThreadFactory factory = r -> {
            Thread t = new Thread(r, "matmul-worker-" + created.getAndIncrement());
            t.setDaemon(true);
            return t;
        };
        pool = new ThreadPoolExecutor(threads, threads, 0L, TimeUnit.MILLISECONDS, new LinkedBlockingQueue<>(),
                factory);
        pool.prestartAllCoreThreads();
    }

    /** Number of worker threads. */
    public int threads() {
        return threads;
    }

    /**
     * C = A * B, with the rows of C computed by the worker threads according to
     * the schedule. Every row of c is overwritten.
     */
    public void ikj(double[][] a, double[][] b, double[][] c, int n, Schedule schedule) {
        List<Callable<Void>> tasks = new ArrayList<>(threads);
        if (schedule == Schedule.STATIC) {
            for (int t = 0; t < threads; t++) {
                int lo = bandStart(t, n, threads);
                int hi = bandStart(t + 1, n, threads);
                tasks.add(() -> {
                    for (int i = lo; i < hi; i++) {
                        ikjRow(a, b, c, n, i);
                    }
                    return null;
                });
            }
        } else {
            AtomicInteger next = new AtomicInteger();
            Callable<Void> worker = () -> {
                for (int i = next.getAndIncrement(); i < n; i = next.getAndIncrement()) {
                    ikjRow(a, b, c, n, i);
                }
                return null;
            };
            for (int t = 0; t < threads; t++) {
                tasks.add(worker);
            }
        }
        runAll(tasks);
    }

    /** First row of band t when n rows are split into p contiguous bands whose sizes differ by at most one. */
    static int bandStart(int t, int n, int p) {
        return (int) ((long) t * n / p);
    }

    /** Row i of C = row i of A times B: the loop body of {@link DenseMatMul#ikj} for one i. */
    static void ikjRow(double[][] a, double[][] b, double[][] c, int n, int i) {
        double[] ai = a[i];
        double[] ci = c[i];
        Arrays.fill(ci, 0.0);
        for (int k = 0; k < n; k++) {
            double aik = ai[k];
            double[] bk = b[k];
            for (int j = 0; j < n; j++) {
                ci[j] += aik * bk[j];
            }
        }
    }

    /** Runs the tasks on the pool, waits for all of them and rethrows the first failure. */
    private void runAll(List<Callable<Void>> tasks) {
        List<Future<Void>> futures;
        try {
            futures = pool.invokeAll(tasks);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted while waiting for the workers", e);
        }
        for (Future<Void> f : futures) {
            try {
                f.get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted while waiting for the workers", e);
            } catch (ExecutionException e) {
                Throwable cause = e.getCause();
                if (cause instanceof RuntimeException r) {
                    throw r;
                }
                if (cause instanceof Error err) {
                    throw err;
                }
                throw new IllegalStateException(cause);
            }
        }
    }

    /** Stops the worker threads; later calls are rejected. */
    @Override
    public void close() {
        pool.shutdownNow();
    }
}
