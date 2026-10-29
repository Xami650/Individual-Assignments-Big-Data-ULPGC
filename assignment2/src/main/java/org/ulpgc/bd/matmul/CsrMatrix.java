package org.ulpgc.bd.matmul;

import java.util.Arrays;

/**
 * Square n x n sparse matrix in compressed sparse row (CSR) format, and the
 * sparse product used in Part B. All code is our own; no library is used.
 *
 * <p>Row i holds the entries {@code values[p]} at columns {@code colIdx[p]} for
 * {@code rowPtr[i] <= p < rowPtr[i + 1]}. Within a row the column indices are
 * strictly increasing (canonical CSR). An entry that is stored is a structural
 * nonzero; its value may still be 0.0 after cancellation in a product.
 * Element storage: 8 bytes per value and 4 per column index, plus 4 (n + 1)
 * bytes of row pointers. Indices are int, so nnz is limited to 2^31 - 1.
 */
public final class CsrMatrix {

    public final int n;
    public final int[] rowPtr;
    public final int[] colIdx;
    public final double[] values;

    public CsrMatrix(int n, int[] rowPtr, int[] colIdx, double[] values) {
        if (rowPtr.length != n + 1 || colIdx.length != values.length || rowPtr[n] != values.length) {
            throw new IllegalArgumentException("inconsistent CSR arrays for n=" + n);
        }
        this.n = n;
        this.rowPtr = rowPtr;
        this.colIdx = colIdx;
        this.values = values;
    }

    /** Number of stored entries. */
    public int nnz() {
        return values.length;
    }

    /** Bytes of the three arrays' elements: 12 nnz + 4 (n + 1). */
    public long storageBytes() {
        return 12L * nnz() + 4L * (n + 1);
    }

    /** Dense to CSR: stores every entry different from 0.0, row by row. */
    public static CsrMatrix fromDense(double[][] m) {
        int n = m.length;
        int[] rowPtr = new int[n + 1];
        long nnz = 0;
        for (int i = 0; i < n; i++) {
            for (double x : m[i]) {
                if (x != 0.0) {
                    nnz++;
                }
            }
            rowPtr[i + 1] = checkedSize(nnz);
        }
        int[] colIdx = new int[(int) nnz];
        double[] values = new double[(int) nnz];
        int p = 0;
        for (double[] row : m) {
            for (int j = 0; j < n; j++) {
                if (row[j] != 0.0) {
                    colIdx[p] = j;
                    values[p++] = row[j];
                }
            }
        }
        return new CsrMatrix(n, rowPtr, colIdx, values);
    }

    /** CSR to dense: a new n x n matrix with the stored entries and 0.0 elsewhere. */
    public double[][] toDense() {
        double[][] m = new double[n][n];
        for (int i = 0; i < n; i++) {
            double[] row = m[i];
            for (int p = rowPtr[i]; p < rowPtr[i + 1]; p++) {
                row[colIdx[p]] = values[p];
            }
        }
        return m;
    }

    /** Number of scalar multiply-adds of A * B: the sum over the entries A[i][k] of nnz(row k of B). */
    public static long multiplyAdds(CsrMatrix a, CsrMatrix b) {
        long count = 0;
        for (int p = 0; p < a.nnz(); p++) {
            int k = a.colIdx[p];
            count += b.rowPtr[k + 1] - b.rowPtr[k];
        }
        return count;
    }

    /**
     * C = A * B with Gustavson's row-by-row algorithm, in two passes.
     *
     * <p>Symbolic pass: for each row i, mark the columns j reached through any
     * A[i][k] and row k of B, which gives nnz of every row of C, and allocate C
     * exactly. Numeric pass: accumulate row i of C in a dense array {@code acc}
     * of length n, again through the rows of B selected by row i of A, then
     * write the touched columns in increasing order: by sorting them, or, when
     * the row holds more than n/16 entries, by scanning the marker array.
     *
     * <p>Because the columns of each row of A are increasing, every C[i][j]
     * receives its products in increasing k, starting from 0.0, as in
     * {@link DenseMatMul#basic}; the skipped products are exact zeros, so the
     * result equals the dense one bit for bit.
     */
    public static CsrMatrix multiply(CsrMatrix a, CsrMatrix b) {
        if (a.n != b.n) {
            throw new IllegalArgumentException("size mismatch: " + a.n + " vs " + b.n);
        }
        int n = a.n;
        int[] marker = new int[n];

        Arrays.fill(marker, -1);
        int[] rowPtr = new int[n + 1];
        long total = 0;
        for (int i = 0; i < n; i++) {
            for (int p = a.rowPtr[i]; p < a.rowPtr[i + 1]; p++) {
                int k = a.colIdx[p];
                for (int q = b.rowPtr[k]; q < b.rowPtr[k + 1]; q++) {
                    int j = b.colIdx[q];
                    if (marker[j] != i) {
                        marker[j] = i;
                        total++;
                    }
                }
            }
            rowPtr[i + 1] = checkedSize(total);
        }

        int[] colIdx = new int[(int) total];
        double[] values = new double[(int) total];
        double[] acc = new double[n];
        Arrays.fill(marker, -1);
        int denseRow = n >>> 4;
        for (int i = 0; i < n; i++) {
            int start = rowPtr[i];
            int len = 0;
            for (int p = a.rowPtr[i]; p < a.rowPtr[i + 1]; p++) {
                int k = a.colIdx[p];
                double aik = a.values[p];
                for (int q = b.rowPtr[k]; q < b.rowPtr[k + 1]; q++) {
                    int j = b.colIdx[q];
                    if (marker[j] != i) {
                        marker[j] = i;
                        colIdx[start + len++] = j;
                        acc[j] = 0.0;
                    }
                    acc[j] += aik * b.values[q];
                }
            }
            if (len > denseRow) {
                int t = start;
                for (int j = 0; j < n; j++) {
                    if (marker[j] == i) {
                        colIdx[t++] = j;
                    }
                }
            } else {
                Arrays.sort(colIdx, start, start + len);
            }
            for (int t = start; t < start + len; t++) {
                values[t] = acc[colIdx[t]];
            }
        }
        return new CsrMatrix(n, rowPtr, colIdx, values);
    }

    /** Sum of the stored values in row order, used to consume the result. */
    public double checksum() {
        double s = 0.0;
        for (double x : values) {
            s += x;
        }
        return s;
    }

    private static int checkedSize(long nnz) {
        if (nnz > Integer.MAX_VALUE - 8) {
            throw new ArithmeticException("nnz " + nnz + " exceeds the int index range of this CSR implementation");
        }
        return (int) nnz;
    }
}
