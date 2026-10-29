package org.ulpgc.bd.matmul;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.math.BigDecimal;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** The sequential baseline (the ikj tests of Assignment 2) and helpers shared by the other tests. */
class DenseMatMulTest {

    private static final double U = Math.scalb(1.0, -53);

    static double[][] ikj(double[][] a, double[][] b) {
        int n = a.length;
        double[][] c = DenseMatMul.zeros(n);
        DenseMatMul.ikj(a, b, c, n);
        return c;
    }

    static double[][] random(int n, long seed) {
        Random rng = new Random(seed);
        double[][] m = new double[n][n];
        for (double[] row : m) {
            for (int j = 0; j < n; j++) {
                row[j] = rng.nextDouble(-1.0, 1.0);
            }
        }
        return m;
    }

    static double[][] identity(int n) {
        double[][] m = DenseMatMul.zeros(n);
        for (int i = 0; i < n; i++) {
            m[i][i] = 1.0;
        }
        return m;
    }

    /** Exact equality of every entry, including the sign of zeros. */
    static void assertBitwiseEquals(double[][] expected, double[][] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i].length, actual[i].length);
            for (int j = 0; j < expected[i].length; j++) {
                assertEquals(Double.doubleToRawLongBits(expected[i][j]), Double.doubleToRawLongBits(actual[i][j]),
                        "entry (" + i + ", " + j + ")");
            }
        }
    }

    @Test
    void handComputed() {
        assertBitwiseEquals(new double[][]{{19, 22}, {43, 50}},
                ikj(new double[][]{{1, 2}, {3, 4}}, new double[][]{{5, 6}, {7, 8}}));
        double[][] a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        double[][] b = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
        assertBitwiseEquals(new double[][]{{30, 24, 18}, {84, 69, 54}, {138, 114, 90}}, ikj(a, b));
        assertBitwiseEquals(new double[][]{{90, 114, 138}, {54, 69, 84}, {18, 24, 30}}, ikj(b, a));
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 7, 64})
    void identityAndZeroAreExact(int n) {
        double[][] a = random(n, n);
        assertBitwiseEquals(a, ikj(a, identity(n)));
        assertBitwiseEquals(a, ikj(identity(n), a));
        assertBitwiseEquals(DenseMatMul.zeros(n), ikj(a, DenseMatMul.zeros(n)));
    }

    @Test
    void outputBufferIsOverwritten() {
        double[][] c = {{99, -99}, {Double.NaN, 1e300}};
        DenseMatMul.ikj(new double[][]{{1, 2}, {3, 4}}, new double[][]{{5, 6}, {7, 8}}, c, 2);
        assertBitwiseEquals(new double[][]{{19, 22}, {43, 50}}, c);
    }

    /** Exact reference with BigDecimal; the double result must satisfy the 2*gamma_n bound. */
    @Test
    void matchesExactReferenceWithinTolerance() {
        int n = 33;
        double[][] a = random(n, 1);
        double[][] b = random(n, 2);
        double[][] c = ikj(a, b);
        double gamma = n * U / (1 - n * U);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                BigDecimal exact = BigDecimal.ZERO;
                double abs = 0.0;
                for (int k = 0; k < n; k++) {
                    exact = exact.add(new BigDecimal(a[i][k]).multiply(new BigDecimal(b[k][j])));
                    abs += Math.abs(a[i][k]) * Math.abs(b[k][j]);
                }
                double err = new BigDecimal(c[i][j]).subtract(exact).abs().doubleValue();
                assertTrue(err <= 2 * gamma * abs, "entry (" + i + ", " + j + ")");
            }
        }
    }

    @Test
    void checksum() {
        double[][] m = {{1.5, 0.0}, {-0.0, 2.0}};
        assertEquals(3.5, DenseMatMul.checksum(m));
        assertArrayEquals(new double[]{0, 0, 0}, DenseMatMul.zeros(3)[2]);
    }
}
