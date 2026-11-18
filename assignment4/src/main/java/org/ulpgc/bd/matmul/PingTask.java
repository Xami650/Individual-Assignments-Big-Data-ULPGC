package org.ulpgc.bd.matmul;

import com.hazelcast.nio.ObjectDataInput;
import com.hazelcast.nio.ObjectDataOutput;
import com.hazelcast.nio.serialization.IdentifiedDataSerializable;
import java.util.concurrent.Callable;

/**
 * An empty task: it returns the member's {@link System#nanoTime()} and does
 * nothing else. The driver times a series of them to measure the round trip
 * of one task through the executor (submit, transfer, queueing on the
 * member's executor thread, response) without any data.
 */
public final class PingTask implements Callable<Long>, IdentifiedDataSerializable {

    @Override
    public Long call() {
        return System.nanoTime();
    }

    @Override
    public void writeData(ObjectDataOutput out) {
    }

    @Override
    public void readData(ObjectDataInput in) {
    }

    @Override
    public int getFactoryId() {
        return MatMulSerialization.FACTORY_ID;
    }

    @Override
    public int getClassId() {
        return MatMulSerialization.PING_TASK;
    }
}
