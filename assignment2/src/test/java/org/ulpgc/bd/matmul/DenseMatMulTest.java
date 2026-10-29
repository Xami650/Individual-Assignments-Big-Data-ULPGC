package org.ulpgc.bd.matmul;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.math.BigDecimal;
import java.util.Random;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.api.Test;

class DenseMatMulTest {

    private static final double U = Math.scalb(1.0, -53);

    /** Runs one dense method by name; blocked uses the given tile size. */
    static double[][] product(String method, double[][] a, double[][] b, int bs) {
        int n = a.length;
        double[][] c = DenseMatMul.zeros(n);
        switch (method) {
            case "basic" -> DenseMatMul.basic(a, b, c, n);
            case "ikj" -> DenseMatMul.ikj(a, b, c, n);
            case "blocked" -> DenseMatMul.blocked(a, b, c, n, bs);
            default -> throw new IllegalArgumentException(method);
        }
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

    @ParameterizedTest
    @CsvSource({"basic, 64", "ikj, 64", "blocked, 1", "blocked, 2", "blocked, 64"})
    void handComputed(String method, int bs) {
        assertBitwiseEquals(new double[][]{{19, 22}, {43, 50}},
                product(method, new double[][]{{1, 2}, {3, 4}}, new double[][]{{5, 6}, {7, 8}}, bs));
        double[][] a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        double[][] b = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
        assertBitwiseEquals(new double[][]{{30, 24, 18}, {84, 69, 54}, {138, 114, 90}}, product(method, a, b, bs));
        assertBitwiseEquals(new double[][]{{90, 114, 138}, {54, 69, 84}, {18, 24, 30}}, product(method, b, a, bs));
    }

    @ParameterizedTest
    @CsvSource({"basic, 1", "ikj, 1", "ikj, 7", "blocked, 3", "blocked, 7"})
    void identityAndZeroAreExact(String method, int n) {
        double[][] a = random(n, n);
        assertBitwiseEquals(a, product(method, a, identity(n), 3));
        assertBitwiseEquals(a, product(method, identity(n), a, 3));
        assertBitwiseEquals(DenseMatMul.zeros(n), product(method, a, DenseMatMul.zeros(n), 3));
    }

    /** HA3: the optimised methods perform the same additions in the same order as basic. */
    @ParameterizedTest
    @ValueSource(ints = {1, 2, 3, 7, 33, 64, 65, 100})
    void optimisedMethodsAreBitwiseEqualToBasic(int n) {
        double[][] a = random(n, 100 + n);
        double[][] b = random(n, 200 + n);
        double[][] expected = product("basic", a, b, 1);
        assertBitwiseEquals(expected, product("ikj", a, b, 1));
        for (int bs : new int[]{1, 3, 16, 64, 512}) {
            assertBitwiseEquals(expected, product("blocked", a, b, bs));
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"basic", "ikj", "blocked"})
    void outputBufferIsOverwritten(String method) {
        double[][] a = {{1, 2}, {3, 4}};
        double[][] b = {{5, 6}, {7, 8}};
        double[][] c = {{99, -99}, {Double.NaN, 1e300}};
        switch (method) {
            case "basic" -> DenseMatMul.basic(a, b, c, 2);
            case "ikj" -> DenseMatMul.ikj(a, b, c, 2);
            default -> DenseMatMul.blocked(a, b, c, 2, 1);
        }
        assertBitwiseEquals(new double[][]{{19, 22}, {43, 50}}, c);
    }

    /** Exact reference with BigDecimal; the double result must satisfy the 2*gamma_n bound. */
    @Test
    void matchesExactReferenceWithinTolerance() {
        int n = 33;
        double[][] a = random(n, 1);
        double[][] b = random(n, 2);
        double[][] c = product("blocked", a, b, 8);
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
    void rejectsTileSizeBelowOne() {
        assertThrows(IllegalArgumentException.class, () -> product("blocked", identity(2), identity(2), 0));
    }

    @Test
    void checksumAndNonzeroCount() {
        double[][] m = {{1.5, 0.0}, {-0.0, 2.0}};
        assertEquals(3.5, DenseMatMul.checksum(m));
        assertEquals(2, DenseMatMul.countNonzeros(m));
        assertArrayEquals(new double[]{0, 0, 0}, DenseMatMul.zeros(3)[2]);
    }
}
