package org.ulpgc.bd.matmul;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Basic dense matrix multiplication in Java.
 *
 * <p>Storage: a matrix is a {@code double[][]}, i.e. an array of n references to
 * row arrays; each row is a contiguous block of n primitive doubles, but rows are
 * separate heap objects and need not be adjacent in memory.
 */
public final class MatMul {

    private MatMul() {
    }

    /** Returns an n x n matrix of 0.0. */
    public static double[][] zeros(int n) {
        return new double[n][n];
    }

    /**
     * C = A * B with the basic i-j-k triple loop and a scalar accumulator.
     * Performs n^3 multiplications and n^3 additions. Every C[i][j] is assigned,
     * so c does not need to be cleared between calls.
     */
    public static void multiply(double[][] a, double[][] b, double[][] c, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++) {
                    s += a[i][k] * b[k][j];
                }
                c[i][j] = s;
            }
        }
    }

    /** Sum of all entries in row-major order, used to consume the result. */
    public static double checksum(double[][] c) {
        double s = 0.0;
        for (double[] row : c) {
            for (double x : row) {
                s += x;
            }
        }
        return s;
    }

    /** Reads a raw little-endian float64 row-major n x n file, one row at a time. */
    public static double[][] load(Path path, int n) throws IOException {
        long expected = 8L * n * n;
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            if (ch.size() != expected) {
                throw new IOException(path + ": " + ch.size() + " bytes, expected " + expected + " for n=" + n);
            }
            double[][] m = new double[n][];
            ByteBuffer buf = ByteBuffer.allocate(8 * n).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n; i++) {
                buf.clear();
                while (buf.hasRemaining()) {
                    if (ch.read(buf) < 0) {
                        throw new IOException(path + ": unexpected end of file");
                    }
                }
                buf.flip();
                m[i] = new double[n];
                buf.asDoubleBuffer().get(m[i]);
            }
            return m;
        }
    }

    /** Writes a matrix as raw little-endian float64, row-major. */
    public static void write(Path path, double[][] c) throws IOException {
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.CREATE,
                StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            for (double[] row : c) {
                ByteBuffer buf = ByteBuffer.allocate(8 * row.length).order(ByteOrder.LITTLE_ENDIAN);
                buf.asDoubleBuffer().put(row);
                while (buf.hasRemaining()) {
                    ch.write(buf);
                }
            }
        }
    }
}
