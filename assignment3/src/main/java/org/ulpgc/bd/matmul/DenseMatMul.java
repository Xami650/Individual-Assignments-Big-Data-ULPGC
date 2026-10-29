package org.ulpgc.bd.matmul;

import java.util.Arrays;

/**
 * The sequential baseline of Assignment 3. The method {@link #ikj} is copied
 * unchanged from Assignment 2 (tag assignment2-v1.0,
 * assignment2/src/main/java/org/ulpgc/bd/matmul/DenseMatMul.java), where it was
 * validated against NumPy and bitwise against the Assignment 1 baseline. The
 * other dense methods of Assignment 2 (basic, blocked) are not part of this study.
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
