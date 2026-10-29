package org.ulpgc.bd.matmul;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Dense binary matrix files, the Assignment 1 format: n*n little-endian
 * float64 in row-major order, no header; n is given by the caller and checked
 * against the size. The dense part of Assignment 2's MatrixIO (tag
 * assignment2-v1.0), unchanged.
 */
public final class MatrixIO {

    private MatrixIO() {
    }

    /** Reads a raw little-endian float64 row-major n x n file, one row at a time. */
    public static double[][] loadDense(Path path, int n) throws IOException {
        long expected = 8L * n * n;
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            if (ch.size() != expected) {
                throw new IOException(path + ": " + ch.size() + " bytes, expected " + expected + " for n=" + n);
            }
            double[][] m = new double[n][];
            ByteBuffer buf = ByteBuffer.allocate(8 * n).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n; i++) {
                readFully(ch, buf.clear(), path);
                m[i] = new double[n];
                buf.flip().asDoubleBuffer().get(m[i]);
            }
            return m;
        }
    }

    /** Writes a matrix as raw little-endian float64, row-major. */
    public static void writeDense(Path path, double[][] c) throws IOException {
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

    private static void readFully(FileChannel ch, ByteBuffer buf, Path path) throws IOException {
        while (buf.hasRemaining()) {
            if (ch.read(buf) < 0) {
                throw new IOException(path + ": unexpected end of file");
            }
        }
    }
}
