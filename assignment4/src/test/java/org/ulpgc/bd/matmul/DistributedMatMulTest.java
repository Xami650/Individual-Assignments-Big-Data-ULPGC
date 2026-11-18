package org.ulpgc.bd.matmul;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.ulpgc.bd.matmul.DenseMatMulTest.assertBitwiseEquals;
import static org.ulpgc.bd.matmul.DenseMatMulTest.identity;
import static org.ulpgc.bd.matmul.DenseMatMulTest.ikj;
import static org.ulpgc.bd.matmul.DenseMatMulTest.random;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * The distributed multiplication on embedded clusters of 1 to 4 members (all
 * in the test JVM): bitwise agreement with ikj, explicit placement, phases and
 * bytes, repeated calls and error propagation.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class DistributedMatMulTest {

    private static final int[] SIZES = {1, 2, 3, 7, 33, 64, 65, 100};

    /** One cluster per number of members, shared by the tests; the kernels are swapped for k. */
    private final Map<Integer, TestCluster> clusters = new HashMap<>();

    private TestCluster cluster(int workers, int threads) throws Exception {
        TestCluster c = clusters.get(workers);
        if (c == null) {
            c = new TestCluster(workers, threads);
            clusters.put(workers, c);
        } else if (c.kernels.get(0).threads() != threads) {
            c.useKernels(m -> new BandKernel(threads));
        }
        return c;
    }

    @AfterAll
    void stopClusters() {
        clusters.values().forEach(TestCluster::close);
    }

    static Stream<Arguments> layouts() {
        return Stream.of(1, 2, 3, 4).flatMap(w -> Stream.of(1, 2).map(k -> Arguments.of(w, k)));
    }

    /** H1: every row is computed by the ikj row kernel, so C is bitwise identical, also with w > n (empty bands). */
    @ParameterizedTest
    @MethodSource("layouts")
    void bitwiseIdenticalToIkj(int workers, int threads) throws Exception {
        TestCluster c = cluster(workers, threads);
        for (int n : SIZES) {
            double[][] a = random(n, 100 + n);
            double[][] b = random(n, 200 + n);
            double[][] result = new double[n][n];
            DistributedMatMul.Result r = c.dist.multiply(a, b, result, n);
            assertBitwiseEquals(ikj(a, b), result);
            assertEquals(workers, r.reports().length);
            int rows = 0;
            for (TaskReport t : r.reports()) {
                rows += t.rows;
                assertEquals(threads, t.kernelThreads);
            }
            assertEquals(n, rows, "every row computed once");
        }
    }

    @Test
    void identityAndZeroAreExact() throws Exception {
        TestCluster c = cluster(3, 2);
        int n = 50;
        double[][] a = random(n, 7);
        double[][] result = new double[n][n];
        c.dist.multiply(a, identity(n), result, n);
        assertBitwiseEquals(a, result);
        c.dist.multiply(a, new double[n][n], result, n);
        assertBitwiseEquals(new double[n][n], result);
    }

    /** Band t is stored on and computed by member t: the task runs where its key's partition is owned. */
    @Test
    void placementIsExplicit() throws Exception {
        TestCluster c = cluster(4, 1);
        int n = 40;
        DistributedMatMul.Result r = c.dist.multiply(random(n, 1), random(n, 2), new double[n][n], n);
        Set<Integer> indices = new HashSet<>();
        for (int t = 0; t < 4; t++) {
            TaskReport report = r.reports()[t];
            assertEquals(t, report.band);
            assertEquals(t, report.memberIndex, "band t runs on member t");
            assertTrue(report.local, "the task's member owns the band's partition");
            assertEquals(DistributedMatMul.bandStart(t + 1, n, 4) - DistributedMatMul.bandStart(t, n, 4), report.rows);
            indices.add(report.memberIndex);
            String owner = c.client.getPartitionService().getPartition(c.dist.keys().get(t)).getOwner()
                    .getAttribute(HazelcastSetup.ATTRIBUTE_INDEX);
            assertEquals(Integer.toString(t), owner);
        }
        assertEquals(4, indices.size());
    }

    /** Payload sizes follow the band format; the socket counters see at least the payload. */
    @Test
    void phasesAndBytes() throws Exception {
        TestCluster c = cluster(2, 1);
        int n = 64;
        DistributedMatMul.Result r = c.dist.multiply(random(n, 3), random(n, 4), new double[n][n], n);
        long header = 17;  // Hazelcast's Data header (8) + identified-serializable header (9)
        long band = header + 20 + 32L * (4 + 8L * n);
        assertEquals(2 * band, r.aPayloadBytes());
        assertEquals(2 * (header + 20 + n * (4 + 8L * n)), r.bPayloadBytes(), "B is sent once per member");
        assertEquals(2 * band, r.cPayloadBytes());
        if (DistributedMatMul.socketCountersAvailable()) {
            assertTrue(r.scatterSocketOut() >= r.aPayloadBytes());
            assertTrue(r.broadcastSocketOut() >= r.bPayloadBytes());
            assertTrue(r.gatherSocketIn() >= r.cPayloadBytes());
            assertTrue(r.socketOut() >= r.aPayloadBytes() + r.bPayloadBytes());
        }
        for (long phase : new long[]{r.serializeNs(), r.scatterNs(), r.broadcastNs(), r.computeNs(), r.gatherNs()}) {
            assertTrue(phase > 0);
        }
        for (TaskReport t : r.reports()) {
            assertTrue(t.startNs >= r.computeStartNs(), "one clock for driver and members");
            assertTrue(t.kernelNs > 0 && t.taskNs >= t.kernelNs);
        }
    }

    /** A and B are taken by the tasks and C is overwritten, so repeated calls leave exactly w entries (C). */
    @Test
    void repeatedCallsLeaveNoExtraEntries() throws Exception {
        TestCluster c = cluster(3, 1);
        int n = 20;
        double[][] a = random(n, 5), b = random(n, 6), expected = ikj(a, b);
        for (int i = 0; i < 30; i++) {
            double[][] result = new double[n][n];
            c.dist.multiply(a, b, result, n);
            assertBitwiseEquals(expected, result);
        }
        assertArrayEquals(new int[]{0, 0, 3}, c.dist.mapSizes());
        c.dist.clear();
        assertArrayEquals(new int[]{0, 0, 0}, c.dist.mapSizes());
    }

    @Test
    void pingReachesEveryMember() throws Exception {
        TestCluster c = cluster(2, 1);
        long[][] times = c.dist.ping(5);
        assertEquals(2, times.length);
        for (long[] member : times) {
            assertEquals(5, member.length);
            for (long t : member) {
                assertTrue(t > 0);
            }
        }
        assertTrue(DistributedMatMul.pingSummary(times).startsWith("ping_rounds=5,ping_median_ns="));
    }

    /** A failing member makes the multiplication fail with the member's error, not hang or return garbage. */
    @Test
    void errorsPropagate() throws Exception {
        try (TestCluster c = new TestCluster(2, m -> m == 1 ? new FailingKernel() : new BandKernel(1))) {
            int n = 8;
            DistributedMatMul.ClusterFailure e = assertThrows(DistributedMatMul.ClusterFailure.class,
                    () -> c.dist.multiply(random(n, 1), random(n, 2), new double[n][n], n));
            assertTrue(e.getMessage().contains("band 1"), e.getMessage());
            assertTrue(e.getMessage().contains("injected failure"), e.getMessage());
        }
    }

    @Test
    void rejectsWrongShapes() throws Exception {
        TestCluster c = cluster(2, 1);
        assertThrows(IllegalArgumentException.class, () -> c.dist.multiply(new double[3][3], new double[2][2],
                new double[3][3], 3));
    }

    private static final class FailingKernel extends BandKernel {
        FailingKernel() {
            super(1);
        }

        @Override
        public void multiply(double[][] a, double[][] b, double[][] c, int n) {
            throw new IllegalStateException("injected failure");
        }
    }
}
