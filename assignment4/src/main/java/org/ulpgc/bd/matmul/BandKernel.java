package org.ulpgc.bd.matmul;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The computation of one cluster member: C_band = A_band * B for a band of
 * consecutive rows, with k compute threads.
 *
 * <p>Every row is computed by {@link ParallelMatMul#ikjRow}, the loop body of
 * the validated {@code ikj} for one row (Assignment 3), so each entry of C
 * receives the same operations in the same order as in the sequential method
 * and the result is bitwise identical for any number of members and threads.
 *
 * <p>With k = 1 the rows are computed on the calling thread (the member's
 * executor thread), with no hand-off. With k &gt; 1 the band is split like the
 * static schedule of {@code par_ikj} ({@link ParallelMatMul#bandStart}) among a
 * fixed pool of k threads, created and started in the constructor, so before
 * any timing; the calling thread only waits.
 */
public class BandKernel implements AutoCloseable {

    private final int threads;
    private final ThreadPoolExecutor pool;

    /** A kernel with {@code threads} compute threads (1 = the calling thread). */
    public BandKernel(int threads) {
        if (threads < 1) {
            throw new IllegalArgumentException("threads must be >= 1, got " + threads);
        }
        this.threads = threads;
        if (threads == 1) {
            pool = null;
        } else {
            AtomicInteger created = new AtomicInteger();
            pool = new ThreadPoolExecutor(threads, threads, 0L, TimeUnit.MILLISECONDS, new LinkedBlockingQueue<>(),
                    r -> {
                        Thread t = new Thread(r, "band-kernel-" + created.getAndIncrement());
                        t.setDaemon(true);
                        return t;
                    });
            pool.prestartAllCoreThreads();
        }
    }

    /** Number of compute threads. */
    public int threads() {
        return threads;
    }

    /**
     * Row r of c = row r of a times b, for every row of the band; a and c have
     * the same number of rows, each of length n, and b is n x n.
     */
    public void multiply(double[][] a, double[][] b, double[][] c, int n) {
        int rows = a.length;
        if (c.length != rows || b.length != n) {
            throw new IllegalArgumentException("band shapes do not match: A " + rows + " rows, C " + c.length
                    + " rows, B " + b.length + " rows, n " + n);
        }
        if (pool == null) {
            for (int i = 0; i < rows; i++) {
                ParallelMatMul.ikjRow(a, b, c, n, i);
            }
            return;
        }
        List<Callable<Void>> tasks = new ArrayList<>(threads);
        for (int t = 0; t < threads; t++) {
            int lo = ParallelMatMul.bandStart(t, rows, threads);
            int hi = ParallelMatMul.bandStart(t + 1, rows, threads);
            tasks.add(() -> {
                for (int i = lo; i < hi; i++) {
                    ParallelMatMul.ikjRow(a, b, c, n, i);
                }
                return null;
            });
        }
        runAll(tasks);
    }

    /** Runs the tasks on the pool, waits for all of them and rethrows the first failure. */
    private void runAll(List<Callable<Void>> tasks) {
        List<Future<Void>> futures;
        try {
            futures = pool.invokeAll(tasks);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted while waiting for the kernel threads", e);
        }
        for (Future<Void> f : futures) {
            try {
                f.get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted while waiting for the kernel threads", e);
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

    /** Stops the compute threads. */
    @Override
    public void close() {
        if (pool != null) {
            pool.shutdownNow();
        }
    }
}
