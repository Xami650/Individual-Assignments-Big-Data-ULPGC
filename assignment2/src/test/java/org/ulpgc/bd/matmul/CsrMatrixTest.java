package org.ulpgc.bd.matmul;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.ulpgc.bd.matmul.DenseMatMulTest.assertBitwiseEquals;

import java.util.Random;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

class CsrMatrixTest {

    /** Random n x n matrix whose entries are nonzero with probability density. */
    static double[][] randomSparse(int n, double density, long seed) {
        Random rng = new Random(seed);
        double[][] m = new double[n][n];
        for (double[] row : m) {
            for (int j = 0; j < n; j++) {
                if (rng.nextDouble() < density) {
                    row[j] = rng.nextDouble(-1.0, 1.0);
                }
            }
        }
        return m;
    }

    private static void assertCanonical(CsrMatrix m) {
        assertEquals(0, m.rowPtr[0]);
        for (int i = 0; i < m.n; i++) {
            assertTrue(m.rowPtr[i] <= m.rowPtr[i + 1], "row pointers must not decrease");
            for (int p = m.rowPtr[i] + 1; p < m.rowPtr[i + 1]; p++) {
                assertTrue(m.colIdx[p - 1] < m.colIdx[p], "columns of row " + i + " must increase");
            }
        }
    }

    @Test
    void conversionRoundTripKeepsOnlyNonzeros() {
        double[][] dense = {{0, 1.5, 0}, {0, 0, 0}, {-2, 0, 3}};
        CsrMatrix m = CsrMatrix.fromDense(dense);
        assertArrayEquals(new int[]{0, 1, 1, 3}, m.rowPtr);
        assertArrayEquals(new int[]{1, 0, 2}, m.colIdx);
        assertArrayEquals(new double[]{1.5, -2, 3}, m.values);
        assertEquals(3, m.nnz());
        assertEquals(12L * 3 + 4L * 4, m.storageBytes());
        assertBitwiseEquals(dense, m.toDense());
    }

    @Test
    void handComputedProduct() {
        double[][] a = {{1, 0, 2}, {0, 0, 0}, {0, 3, 0}};
        double[][] b = {{0, 4, 0}, {5, 0, 0}, {0, 0, 6}};
        CsrMatrix c = CsrMatrix.multiply(CsrMatrix.fromDense(a), CsrMatrix.fromDense(b));
        assertBitwiseEquals(new double[][]{{0, 4, 12}, {0, 0, 0}, {15, 0, 0}}, c.toDense());
        assertEquals(3, c.nnz());
        assertEquals(3, CsrMatrix.multiplyAdds(CsrMatrix.fromDense(a), CsrMatrix.fromDense(b)));
        assertCanonical(c);
    }

    /** A structural nonzero whose value cancels to 0.0 stays stored. */
    @Test
    void cancellationKeepsStructuralNonzero() {
        CsrMatrix a = CsrMatrix.fromDense(new double[][]{{1, 1}, {0, 0}});
        CsrMatrix b = CsrMatrix.fromDense(new double[][]{{1, 0}, {-1, 0}});
        CsrMatrix c = CsrMatrix.multiply(a, b);
        assertEquals(1, c.nnz());
        assertEquals(0.0, c.values[0]);
        assertEquals(0L, Double.doubleToRawLongBits(c.values[0]), "+0.0, as in the dense product");
    }

    @Test
    void emptyAndIdentity() {
        int n = 6;
        double[][] a = randomSparse(n, 0.5, 1);
        CsrMatrix empty = CsrMatrix.fromDense(DenseMatMul.zeros(n));
        assertEquals(0, CsrMatrix.multiply(CsrMatrix.fromDense(a), empty).nnz());
        assertEquals(0, CsrMatrix.multiply(empty, CsrMatrix.fromDense(a)).nnz());
        CsrMatrix eye = CsrMatrix.fromDense(DenseMatMulTest.identity(n));
        assertBitwiseEquals(a, CsrMatrix.multiply(CsrMatrix.fromDense(a), eye).toDense());
        assertBitwiseEquals(a, CsrMatrix.multiply(eye, CsrMatrix.fromDense(a)).toDense());
    }

    /**
     * HA3 for the sparse product: equal bit for bit to the dense baseline, with
     * empty rows (low density) and rows long enough for the scan path (high density).
     */
    @ParameterizedTest
    @CsvSource({"1, 1.0", "5, 0.3", "40, 0.02", "40, 0.2", "100, 0.01", "100, 0.1", "100, 0.6", "100, 1.0"})
    void productIsBitwiseEqualToDenseBaseline(int n, double density) {
        double[][] a = randomSparse(n, density, 10 + n);
        double[][] b = randomSparse(n, density, 20 + n);
        CsrMatrix sa = CsrMatrix.fromDense(a);
        CsrMatrix sb = CsrMatrix.fromDense(b);
        CsrMatrix c = CsrMatrix.multiply(sa, sb);
        double[][] expected = DenseMatMulTest.product("basic", a, b, 1);
        assertBitwiseEquals(expected, c.toDense());
        assertEquals(Double.doubleToRawLongBits(DenseMatMul.checksum(expected)), Double.doubleToRawLongBits(c.checksum()));
        assertCanonical(c);

        long structural = 0;
        long adds = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                boolean reached = false;
                for (int k = 0; k < n; k++) {
                    if (a[i][k] != 0.0 && b[k][j] != 0.0) {
                        reached = true;
                        adds++;
                    }
                }
                structural += reached ? 1 : 0;
            }
        }
        assertEquals(structural, c.nnz());
        assertEquals(adds, CsrMatrix.multiplyAdds(sa, sb));
    }

    @Test
    void rejectsInconsistentArrays() {
        assertThrows(IllegalArgumentException.class, () -> new CsrMatrix(2, new int[]{0, 1}, new int[]{0}, new double[]{1}));
        assertThrows(IllegalArgumentException.class, () -> new CsrMatrix(1, new int[]{0, 2}, new int[]{0}, new double[]{1}));
        CsrMatrix one = CsrMatrix.fromDense(new double[][]{{1}});
        CsrMatrix two = CsrMatrix.fromDense(new double[][]{{1, 0}, {0, 1}});
        assertThrows(IllegalArgumentException.class, () -> CsrMatrix.multiply(one, two));
    }
}
