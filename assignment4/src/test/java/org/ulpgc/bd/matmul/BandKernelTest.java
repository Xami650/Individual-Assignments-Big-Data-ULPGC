package org.ulpgc.bd.matmul;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.ulpgc.bd.matmul.DenseMatMulTest.assertBitwiseEquals;
import static org.ulpgc.bd.matmul.DenseMatMulTest.ikj;
import static org.ulpgc.bd.matmul.DenseMatMulTest.random;

import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** The kernel of a member: a band of rows, with 1 to 4 threads, bitwise equal to the rows of ikj. */
class BandKernelTest {

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 3, 4})
    void bandsAreBitwiseEqualToIkjRows(int threads) {
        int n = 37;
        double[][] a = random(n, 1), b = random(n, 2), expected = ikj(a, b);
        try (BandKernel kernel = new BandKernel(threads)) {
            assertEquals(threads, kernel.threads());
            for (int[] band : new int[][]{{0, n}, {0, 0}, {5, 6}, {3, 5}, {10, 30}, {36, 37}}) {
                double[][] rows = Arrays.copyOfRange(a, band[0], band[1]);
                double[][] c = new double[rows.length][n];
                for (double[] row : c) {
                    Arrays.fill(row, Double.NaN);
                }
                kernel.multiply(rows, b, c, n);
                assertBitwiseEquals(Arrays.copyOfRange(expected, band[0], band[1]), c);
            }
        }
    }

    @Test
    void rejectsMismatchedShapes() {
        try (BandKernel kernel = new BandKernel(2)) {
            assertThrows(IllegalArgumentException.class,
                    () -> kernel.multiply(new double[3][4], new double[4][4], new double[2][4], 4));
            assertThrows(IllegalArgumentException.class,
                    () -> kernel.multiply(new double[3][4], new double[3][4], new double[3][4], 4));
        }
        assertThrows(IllegalArgumentException.class, () -> new BandKernel(0));
    }

    @Test
    void exceptionsOfThreadsArePropagated() {
        try (BandKernel kernel = new BandKernel(3)) {
            double[][] a = new double[6][4];
            a[4] = new double[2];  // a short row makes one thread fail
            assertThrows(ArrayIndexOutOfBoundsException.class,
                    () -> kernel.multiply(a, new double[4][4], new double[6][4], 4));
        }
    }
}
