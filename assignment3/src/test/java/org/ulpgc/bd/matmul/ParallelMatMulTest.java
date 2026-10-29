package org.ulpgc.bd.matmul;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.ulpgc.bd.matmul.DenseMatMulTest.assertBitwiseEquals;

import java.util.concurrent.RejectedExecutionException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.ulpgc.bd.matmul.ParallelMatMul.Schedule;

class ParallelMatMulTest {

    private static final int[] THREADS = {1, 2, 3, 4, 7, 10, 16, 20};

    static double[][] parallel(ParallelMatMul pm, double[][] a, double[][] b, Schedule schedule) {
        int n = a.length;
        double[][] c = DenseMatMul.zeros(n);
        pm.ikj(a, b, c, n, schedule);
        return c;
    }

    /** H1: for every p (also p > n) and both schedules, the result equals the sequential ikj bit for bit. */
    @ParameterizedTest
    @ValueSource(ints = {1, 2, 3, 7, 33, 64, 65, 100})
    void bitwiseEqualToSequential(int n) {
        double[][] a = DenseMatMulTest.random(n, 100 + n);
        double[][] b = DenseMatMulTest.random(n, 200 + n);
        double[][] expected = DenseMatMulTest.ikj(a, b);
        for (int p : THREADS) {
            try (ParallelMatMul pm = new ParallelMatMul(p)) {
                for (Schedule schedule : Schedule.values()) {
                    assertBitwiseEquals(expected, parallel(pm, a, b, schedule));
                }
            }
        }
    }

    @ParameterizedTest
    @EnumSource(Schedule.class)
    void handComputedIdentityAndZero(Schedule schedule) {
        try (ParallelMatMul pm = new ParallelMatMul(3)) {
            assertBitwiseEquals(new double[][]{{19, 22}, {43, 50}},
                    parallel(pm, new double[][]{{1, 2}, {3, 4}}, new double[][]{{5, 6}, {7, 8}}, schedule));
            double[][] a = DenseMatMulTest.random(9, 9);
            assertBitwiseEquals(a, parallel(pm, a, DenseMatMulTest.identity(9), schedule));
            assertBitwiseEquals(DenseMatMul.zeros(9), parallel(pm, a, DenseMatMul.zeros(9), schedule));
        }
    }

    /** Safe access to shared data: many calls on one pool, a stale output buffer each time, always the same bits. */
    @Test
    void repeatedCallsOnOnePoolAreIdentical() {
        int n = 65;
        double[][] a = DenseMatMulTest.random(n, 1);
        double[][] b = DenseMatMulTest.random(n, 2);
        double[][] expected = DenseMatMulTest.ikj(a, b);
        double[][] c = DenseMatMulTest.random(n, 3);
        try (ParallelMatMul pm = new ParallelMatMul(20)) {
            for (int call = 0; call < 200; call++) {
                for (double[] row : c) {
                    row[call % n] = Double.NaN;
                }
                pm.ikj(a, b, c, n, call % 2 == 0 ? Schedule.DYNAMIC : Schedule.STATIC);
                assertBitwiseEquals(expected, c);
            }
        }
    }

    /** The static bands are contiguous, cover rows 0..n-1 exactly once and differ in size by at most one. */
    @Test
    void staticBandsPartitionTheRows() {
        for (int n : new int[]{0, 1, 2, 3, 10, 65, 4096}) {
            for (int p = 1; p <= 20; p++) {
                assertEquals(0, ParallelMatMul.bandStart(0, n, p));
                assertEquals(n, ParallelMatMul.bandStart(p, n, p));
                int min = Integer.MAX_VALUE, max = 0;
                for (int t = 0; t < p; t++) {
                    int size = ParallelMatMul.bandStart(t + 1, n, p) - ParallelMatMul.bandStart(t, n, p);
                    min = Math.min(min, size);
                    max = Math.max(max, size);
                }
                assertTrue(min >= 0 && max - min <= 1, "n=" + n + ", p=" + p);
            }
        }
    }

    @ParameterizedTest
    @EnumSource(Schedule.class)
    void exceptionsPropagateAndThePoolStaysUsable(Schedule schedule) {
        double[][] a = DenseMatMulTest.random(4, 1);
        double[][] b = DenseMatMulTest.random(4, 2);
        double[][] broken = {a[0], null, a[2], a[3]};
        try (ParallelMatMul pm = new ParallelMatMul(2)) {
            assertThrows(NullPointerException.class, () -> parallel(pm, broken, b, schedule));
            assertBitwiseEquals(DenseMatMulTest.ikj(a, b), parallel(pm, a, b, schedule));
        }
    }

    @Test
    void threadCountIsCheckedAndClosedPoolsRejectWork() {
        assertThrows(IllegalArgumentException.class, () -> new ParallelMatMul(0));
        ParallelMatMul pm = new ParallelMatMul(2);
        assertEquals(2, pm.threads());
        pm.close();
        assertThrows(RejectedExecutionException.class,
                () -> parallel(pm, DenseMatMulTest.identity(2), DenseMatMulTest.identity(2), Schedule.DYNAMIC));
    }

    @Test
    void scheduleNames() {
        assertSame(Schedule.DYNAMIC, Schedule.parse("dynamic"));
        assertSame(Schedule.STATIC, Schedule.parse("STATIC"));
        assertEquals("static", Schedule.STATIC.label());
        assertThrows(IllegalArgumentException.class, () -> Schedule.parse("guided"));
    }
}
