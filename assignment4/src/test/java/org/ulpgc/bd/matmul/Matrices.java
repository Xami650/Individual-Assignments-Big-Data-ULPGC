package org.ulpgc.bd.matmul;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;
import org.ulpgc.bd.matmul.kernel.DenseMatMul;

/** Matrices and assertions shared by the tests of every package. */
public final class Matrices {

    private Matrices() {
    }

    /** A * B with the validated sequential ikj (Assignment 3), the reference of every test. */
    public static double[][] ikj(double[][] a, double[][] b) {
        int n = a.length;
        double[][] c = DenseMatMul.zeros(n);
        DenseMatMul.ikj(a, b, c, n);
        return c;
    }

    /** n x n matrix of uniform values in [-1, 1) from a fixed seed. */
    public static double[][] random(int n, long seed) {
        Random rng = new Random(seed);
        double[][] m = new double[n][n];
        for (double[] row : m) {
            for (int j = 0; j < n; j++) {
                row[j] = rng.nextDouble(-1.0, 1.0);
            }
        }
        return m;
    }

    public static double[][] identity(int n) {
        double[][] m = DenseMatMul.zeros(n);
        for (int i = 0; i < n; i++) {
            m[i][i] = 1.0;
        }
        return m;
    }

    /** Exact equality of every entry, including the sign of zeros. */
    public static void assertBitwiseEquals(double[][] expected, double[][] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i].length, actual[i].length);
            for (int j = 0; j < expected[i].length; j++) {
                assertEquals(Double.doubleToRawLongBits(expected[i][j]), Double.doubleToRawLongBits(actual[i][j]),
                        "entry (" + i + ", " + j + ")");
            }
        }
    }
}
