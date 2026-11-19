package org.ulpgc.bd.matmul.dist.protocol;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.ulpgc.bd.matmul.Matrices.assertBitwiseEquals;
import static org.ulpgc.bd.matmul.Matrices.random;

import com.hazelcast.internal.serialization.Data;
import com.hazelcast.internal.serialization.InternalSerializationService;
import com.hazelcast.internal.serialization.impl.DefaultSerializationServiceBuilder;
import com.hazelcast.nio.serialization.HazelcastSerializationException;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.ulpgc.bd.matmul.metrics.ProcessCounters;

/** Round trips of the classes that travel between the driver and the members, with Hazelcast's serializer. */
class SerializationTest {

    private final InternalSerializationService ss = new DefaultSerializationServiceBuilder()
            .addDataSerializableFactory(MatMulSerialization.FACTORY_ID, new MatMulSerialization()).build();

    @Test
    void rowBandRoundTripIsExact() {
        double[][] m = random(9, 3);
        m[2][4] = -0.0;
        m[5][1] = Double.MIN_VALUE;
        RowBand band = RowBand.slice(42, m, 2, 7, 9);
        Data data = ss.toData(band);
        assertEquals(17 + 20 + 5 * (4 + 8 * 9), data.totalSize(), "header + fields + rows");
        RowBand copy = ss.toObject(data);
        assertEquals(42, copy.job());
        assertEquals(2, copy.first());
        assertEquals(9, copy.columns());
        assertBitwiseEquals(Arrays.copyOfRange(m, 2, 7), copy.rows());
        RowBand empty = ss.toObject(ss.toData(new RowBand(1, 3, 9, new double[0][])));
        assertEquals(0, empty.rows().length);
        assertEquals(9, empty.columns());
    }

    @Test
    void bandWithARowOfTheWrongLengthIsRejected() {
        RowBand bad = new RowBand(1, 0, 3, new double[][]{{1, 2, 3}, {4, 5}});
        assertThrows(HazelcastSerializationException.class, () -> ss.toData(bad));
    }

    @Test
    void keysAreEqualAfterRoundTripAndCarryTheirPartitionKey() {
        BandKey key = new BandKey(3, 17);
        BandKey copy = ss.toObject(ss.toData(key));
        assertEquals(key, copy);
        assertEquals(key.hashCode(), copy.hashCode());
        assertEquals(17, copy.getPartitionKey());
        assertTrue(!key.equals(new BandKey(3, 18)) && !key.equals(new BandKey(4, 17)));
    }

    @Test
    void taskAndReportRoundTrip() {
        Object task = ss.toObject(ss.toData(new BandTask(7, new BandKey(1, 2))));
        assertInstanceOf(BandTask.class, task);
        TaskReport r = new TaskReport();
        r.band = 2;
        r.memberIndex = 2;
        r.kernelThreads = 5;
        r.rows = 100;
        r.pid = 1234;
        r.local = true;
        r.readANs = 1;
        r.readBNs = 2;
        r.allocNs = 3;
        r.kernelNs = 4;
        r.storeNs = 5;
        r.taskNs = 6;
        r.startNs = 7;
        r.endNs = 8;
        r.cBytes = 9;
        r.taskCpuNs = 10;
        r.counters(new ProcessCounters.Snapshot(11, 12, 13, 14, 15, 16, 17));
        TaskReport c = ss.toObject(ss.toData(r));
        assertEquals(List.of(2, 2, 5, 100), List.of(c.band, c.memberIndex, c.kernelThreads, c.rows));
        assertEquals(1234, c.pid);
        assertTrue(c.local);
        long[] expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
        long[] actual = {c.readANs, c.readBNs, c.allocNs, c.kernelNs, c.storeNs, c.taskNs, c.startNs, c.endNs,
                c.cBytes, c.taskCpuNs, c.cpuNs, c.cpuPNs, c.instructions, c.cycles, c.instructionsP, c.cyclesP,
                c.contextSwitches};
        assertEquals(Arrays.toString(expected), Arrays.toString(actual));
        r.counters(null);
        assertEquals(-1, r.cycles);
    }

    @Test
    void unknownClassIdIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new MatMulSerialization().create(99));
    }
}
