package org.ulpgc.bd.matmul;

import java.util.Arrays;

/**
 * The three dense methods compared in Part A. All of them take an n x n
 * {@code double[][]} (an array of row arrays) and write C = A * B into c.
 *
 * <p>For every C[i][j] all three methods add the products A[i][k] * B[k][j] in
 * increasing k, starting from 0.0, and Java never fuses a multiply and an add.
 * They therefore perform the same IEEE-754 operations in the same order and
 * produce bitwise-identical results; only the order in which the (i, j, k)
 * iterations are visited differs.
 */
public final class DenseMatMul {

    /** Tile size of {@link #blocked} used unless another one is given (see config/bench.json). */
    public static final int DEFAULT_BLOCK = 512;

    private DenseMatMul() {
    }

    /** Returns an n x n matrix of 0.0. */
    public static double[][] zeros(int n) {
        return new double[n][n];
    }

    /**
     * Baseline: the basic i-j-k triple loop with a scalar accumulator, copied
     * unchanged from Assignment 1 (tag assignment1-v1.0,
     * assignment1/java/src/main/java/org/ulpgc/bd/matmul/MatMul.java, method multiply).
     * The inner loop walks a column of B, with a stride of one row.
     * Every C[i][j] is assigned, so c does not need to be cleared between calls.
     */
    public static void basic(double[][] a, double[][] b, double[][] c, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++) {
                    s += a[i][k] * b[k][j];
                }
                c[i][j] = s;
            }
        }
    }

    /**
     * Loop reordering (i-k-j): the inner loop adds A[i][k] times row k of B to
     * row i of C. Both rows are read with unit stride, and the iterations of
     * the inner loop are independent of each other, so the JIT may vectorise
     * it. Clearing row i of C is part of the method.
     */
    public static void ikj(double[][] a, double[][] b, double[][] c, int n) {
        for (int i = 0; i < n; i++) {
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
    }

    /**
     * Cache blocking on top of i-k-j: the loops over i, k and j are split into
     * tiles of {@code bs} indices and the i-k-j loop runs on one tile triple at a
     * time, so a bs x bs tile of each matrix is reused while it is still in
     * cache. The tile loop over k is ascending for every (i, j), which keeps the
     * order of the additions of {@link #basic}. Clearing C is part of the method.
     */
    public static void blocked(double[][] a, double[][] b, double[][] c, int n, int bs) {
        if (bs < 1) {
            throw new IllegalArgumentException("block size must be >= 1, got " + bs);
        }
        for (double[] row : c) {
            Arrays.fill(row, 0.0);
        }
        for (int ii = 0; ii < n; ii += bs) {
            int iEnd = Math.min(ii + bs, n);
            for (int kk = 0; kk < n; kk += bs) {
                int kEnd = Math.min(kk + bs, n);
                for (int jj = 0; jj < n; jj += bs) {
                    int jEnd = Math.min(jj + bs, n);
                    for (int i = ii; i < iEnd; i++) {
                        double[] ai = a[i];
                        double[] ci = c[i];
                        for (int k = kk; k < kEnd; k++) {
                            double aik = ai[k];
                            double[] bk = b[k];
                            for (int j = jj; j < jEnd; j++) {
                                ci[j] += aik * bk[j];
                            }
                        }
                    }
                }
            }
        }
    }

    /** Sum of all entries in row-major order, used to consume the result. */
    public static double checksum(double[][] c) {
        double s = 0.0;
        for (double[] row : c) {
            for (double x : row) {
                s += x;
            }
        }
        return s;
    }

    /** Number of entries different from 0.0. */
    public static long countNonzeros(double[][] m) {
        long count = 0;
        for (double[] row : m) {
            for (double x : row) {
                if (x != 0.0) {
                    count++;
                }
            }
        }
        return count;
    }
}
