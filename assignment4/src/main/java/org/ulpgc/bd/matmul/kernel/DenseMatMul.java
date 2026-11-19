package org.ulpgc.bd.matmul.kernel;

import java.util.Arrays;

/**
 * The sequential methods of Assignment 4, both reused unchanged:
 * <ul>
 *   <li>{@link #basic}: the validated basic version of Assignment 1, as kept in
 *       Assignment 2 (tag assignment2-v1.0,
 *       assignment2/src/main/java/org/ulpgc/bd/matmul/DenseMatMul.java);</li>
 *   <li>{@link #ikj}: the sequential baseline of Assignment 3 (tag
 *       assignment3-v1.0, assignment3/src/main/java/org/ulpgc/bd/matmul/DenseMatMul.java),
 *       itself copied from Assignment 2.</li>
 * </ul>
 * For every C[i][j] both methods add the products A[i][k] * B[k][j] in
 * increasing k, starting from 0.0, and Java never fuses a multiply and an add,
 * so they produce bitwise-identical results (verified in Assignment 2).
 *
 * <p>Matrices are n x n {@code double[][]} (an array of row arrays).
 */
public final class DenseMatMul {

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
}
