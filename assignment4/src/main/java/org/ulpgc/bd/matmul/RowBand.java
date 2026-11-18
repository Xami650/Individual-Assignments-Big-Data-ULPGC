package org.ulpgc.bd.matmul;

import com.hazelcast.nio.ObjectDataInput;
import com.hazelcast.nio.ObjectDataOutput;
import com.hazelcast.nio.serialization.IdentifiedDataSerializable;
import java.io.IOException;
import java.util.Arrays;

/**
 * A band of consecutive rows of an n-column matrix, the unit that is stored
 * in the cluster: a band of A, the whole of B, or a band of C. It carries the
 * number of the multiplication ({@code job}) and the index of its first row,
 * so that the receiver can check that it holds the right data.
 *
 * <p>Serialized format (Hazelcast {@link IdentifiedDataSerializable}, Hazelcast's
 * default big-endian byte order): job (8 bytes), first row (4), columns (4),
 * row count (4), then every row as Hazelcast's double array (a 4-byte length
 * and 8 bytes per value). The payload of a band of r rows is therefore
 * 20 + r (4 + 8 n) bytes, inside Hazelcast's own header.
 */
public final class RowBand implements IdentifiedDataSerializable {

    private long job;
    private int first;
    private int columns;
    private double[][] rows;

    /** For deserialization only. */
    public RowBand() {
    }

    /** A band that refers to (does not copy) the given row arrays, each of length {@code columns}. */
    public RowBand(long job, int first, int columns, double[][] rows) {
        this.job = job;
        this.first = first;
        this.columns = columns;
        this.rows = rows;
    }

    /** Rows [lo, hi) of m, sharing the row arrays (only the array of references is new). */
    public static RowBand slice(long job, double[][] m, int lo, int hi, int columns) {
        return new RowBand(job, lo, columns, Arrays.copyOfRange(m, lo, hi));
    }

    public long job() {
        return job;
    }

    public int first() {
        return first;
    }

    public int columns() {
        return columns;
    }

    public double[][] rows() {
        return rows;
    }

    @Override
    public void writeData(ObjectDataOutput out) throws IOException {
        out.writeLong(job);
        out.writeInt(first);
        out.writeInt(columns);
        out.writeInt(rows.length);
        for (double[] row : rows) {
            if (row.length != columns) {
                throw new IOException("row of length " + row.length + " in a band of " + columns + " columns");
            }
            out.writeDoubleArray(row);
        }
    }

    @Override
    public void readData(ObjectDataInput in) throws IOException {
        job = in.readLong();
        first = in.readInt();
        columns = in.readInt();
        rows = new double[in.readInt()][];
        for (int i = 0; i < rows.length; i++) {
            rows[i] = in.readDoubleArray();
            if (rows[i] == null || rows[i].length != columns) {
                throw new IOException("corrupt band: row " + i + " does not have " + columns + " columns");
            }
        }
    }

    @Override
    public int getFactoryId() {
        return MatMulSerialization.FACTORY_ID;
    }

    @Override
    public int getClassId() {
        return MatMulSerialization.ROW_BAND;
    }
}
