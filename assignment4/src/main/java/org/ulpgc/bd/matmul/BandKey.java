package org.ulpgc.bd.matmul;

import com.hazelcast.nio.ObjectDataInput;
import com.hazelcast.nio.ObjectDataOutput;
import com.hazelcast.nio.serialization.IdentifiedDataSerializable;
import com.hazelcast.partition.PartitionAware;
import java.io.IOException;

/**
 * The key of band t in the maps of A, B and C. Hazelcast places an entry in
 * the partition of its key; for a {@link PartitionAware} key it uses the
 * partition of {@link #getPartitionKey()} instead. The driver chooses, for
 * every member, a partition key whose partition that member owns
 * ({@link DistributedMatMul}), so band t of A, the copy of B for band t and
 * band t of C are all stored on member t, and the task that computes band t
 * is sent to the same member ({@code submitToKeyOwner}). Placement is thus
 * explicit rather than left to the hash of the band number.
 */
public final class BandKey implements IdentifiedDataSerializable, PartitionAware<Integer> {

    private int band;
    private int partitionKey;

    /** For deserialization only. */
    public BandKey() {
    }

    public BandKey(int band, int partitionKey) {
        this.band = band;
        this.partitionKey = partitionKey;
    }

    public int band() {
        return band;
    }

    @Override
    public Integer getPartitionKey() {
        return partitionKey;
    }

    @Override
    public void writeData(ObjectDataOutput out) throws IOException {
        out.writeInt(band);
        out.writeInt(partitionKey);
    }

    @Override
    public void readData(ObjectDataInput in) throws IOException {
        band = in.readInt();
        partitionKey = in.readInt();
    }

    @Override
    public int getFactoryId() {
        return MatMulSerialization.FACTORY_ID;
    }

    @Override
    public int getClassId() {
        return MatMulSerialization.BAND_KEY;
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof BandKey k && k.band == band && k.partitionKey == partitionKey;
    }

    @Override
    public int hashCode() {
        return 31 * band + partitionKey;
    }

    @Override
    public String toString() {
        return "band " + band + " (partition key " + partitionKey + ")";
    }
}
