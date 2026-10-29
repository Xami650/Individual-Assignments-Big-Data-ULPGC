package org.ulpgc.bd.matmul;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Binary matrix files (little-endian, no compression).
 *
 * <ul>
 *   <li>Dense ({@code .bin}, the Assignment 1 format): n*n float64 in row-major
 *   order, no header; n is given by the caller and checked against the size.</li>
 *   <li>CSR ({@code .csr}): int64 n, int64 nnz, then int32 rowPtr[n + 1],
 *   int32 colIdx[nnz] and float64 values[nnz].</li>
 * </ul>
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

    /** Reads a CSR file and checks that its size matches n and its header. */
    public static CsrMatrix loadCsr(Path path, int n) throws IOException {
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer header = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
            readFully(ch, header, path);
            header.flip();
            long fileN = header.getLong();
            long nnz = header.getLong();
            if (fileN != n || nnz < 0 || nnz > Integer.MAX_VALUE - 8) {
                throw new IOException(path + ": header n=" + fileN + ", nnz=" + nnz + ", expected n=" + n);
            }
            long expected = 16 + 4L * (n + 1) + 12 * nnz;
            if (ch.size() != expected) {
                throw new IOException(path + ": " + ch.size() + " bytes, expected " + expected);
            }
            int[] rowPtr = new int[n + 1];
            int[] colIdx = new int[(int) nnz];
            double[] values = new double[(int) nnz];
            readInts(ch, rowPtr, path);
            readInts(ch, colIdx, path);
            readDoubles(ch, values, path);
            return new CsrMatrix(n, rowPtr, colIdx, values);
        }
    }

    private static final int CHUNK = 1 << 20;

    private static void readInts(FileChannel ch, int[] dst, Path path) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(4 * CHUNK).order(ByteOrder.LITTLE_ENDIAN);
        for (int off = 0; off < dst.length; off += CHUNK) {
            int len = Math.min(CHUNK, dst.length - off);
            readFully(ch, buf.clear().limit(4 * len), path);
            buf.flip().asIntBuffer().get(dst, off, len);
        }
    }

    private static void readDoubles(FileChannel ch, double[] dst, Path path) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(8 * CHUNK).order(ByteOrder.LITTLE_ENDIAN);
        for (int off = 0; off < dst.length; off += CHUNK) {
            int len = Math.min(CHUNK, dst.length - off);
            readFully(ch, buf.clear().limit(8 * len), path);
            buf.flip().asDoubleBuffer().get(dst, off, len);
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
