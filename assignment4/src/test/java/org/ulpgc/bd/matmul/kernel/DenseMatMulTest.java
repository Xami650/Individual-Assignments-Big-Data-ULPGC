package org.ulpgc.bd.matmul.kernel;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.ulpgc.bd.matmul.Matrices.assertBitwiseEquals;
import static org.ulpgc.bd.matmul.Matrices.identity;
import static org.ulpgc.bd.matmul.Matrices.ikj;
import static org.ulpgc.bd.matmul.Matrices.random;

import java.math.BigDecimal;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * The sequential methods: the ikj tests copied from Assignment 3 (tag
 * assignment3-v1.0), plus basic (Assignment 2). The shared helpers are in
 * {@link org.ulpgc.bd.matmul.Matrices}.
 */
class DenseMatMulTest {

    private static final double U = Math.scalb(1.0, -53);

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

    /** basic (Assignment 1) adds the same products in the same order as ikj: bitwise-identical results. */
    @ParameterizedTest
    @ValueSource(ints = {1, 2, 3, 7, 33, 64, 100})
    void basicIsBitwiseIdenticalToIkj(int n) {
        double[][] a = random(n, 10 + n);
        double[][] b = random(n, 20 + n);
        double[][] c = new double[n][n];
        for (double[] row : c) {
            Arrays.fill(row, Double.NaN);
        }
        DenseMatMul.basic(a, b, c, n);
        assertBitwiseEquals(ikj(a, b), c);
    }

    @Test
    void basicHandComputed() {
        double[][] c = new double[2][2];
        DenseMatMul.basic(new double[][]{{1, 2}, {3, 4}}, new double[][]{{5, 6}, {7, 8}}, c, 2);
        assertBitwiseEquals(new double[][]{{19, 22}, {43, 50}}, c);
    }

    @Test
    void checksum() {
        double[][] m = {{1.5, 0.0}, {-0.0, 2.0}};
        assertEquals(3.5, DenseMatMul.checksum(m));
        assertArrayEquals(new double[]{0, 0, 0}, DenseMatMul.zeros(3)[2]);
    }
}
