package org.ulpgc.bd.matmul.dist.protocol;

import com.hazelcast.nio.serialization.DataSerializableFactory;
import com.hazelcast.nio.serialization.IdentifiedDataSerializable;
import org.ulpgc.bd.matmul.dist.HazelcastSetup;

/**
 * Factory of the classes that travel between the driver and the members. They
 * implement Hazelcast's {@link IdentifiedDataSerializable}: each is written as
 * a factory id, a class id and its own fields, with no class names and no
 * reflection, and is created on the receiving side by this factory. The
 * factory is registered in the member and the client configuration
 * ({@link HazelcastSetup}).
 */
public final class MatMulSerialization implements DataSerializableFactory {

    public static final int FACTORY_ID = 4;
    public static final int ROW_BAND = 1;
    public static final int BAND_KEY = 2;
    public static final int BAND_TASK = 3;
    public static final int TASK_REPORT = 4;
    public static final int PING_TASK = 5;

    @Override
    public IdentifiedDataSerializable create(int classId) {
        return switch (classId) {
            case ROW_BAND -> new RowBand();
            case BAND_KEY -> new BandKey();
            case BAND_TASK -> new BandTask();
            case TASK_REPORT -> new TaskReport();
            case PING_TASK -> new PingTask();
            default -> throw new IllegalArgumentException("unknown class id " + classId);
        };
    }
}
