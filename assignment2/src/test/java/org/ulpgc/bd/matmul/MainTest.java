package org.ulpgc.bd.matmul;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.ulpgc.bd.matmul.DenseMatMulTest.assertBitwiseEquals;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** File formats and the command-line contract of the driver. */
class MainTest {

    @TempDir
    Path dir;

    /** Writes a CSR file in the format documented in {@link MatrixIO} (the Python generator's layout). */
    static void writeCsr(Path path, CsrMatrix m) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(16 + 4 * (m.n + 1) + 12 * m.nnz()).order(ByteOrder.LITTLE_ENDIAN);
        buf.putLong(m.n).putLong(m.nnz());
        for (int x : m.rowPtr) {
            buf.putInt(x);
        }
        for (int x : m.colIdx) {
            buf.putInt(x);
        }
        for (double x : m.values) {
            buf.putDouble(x);
        }
        Files.write(path, buf.array());
    }

    @Test
    void denseAndCsrFilesRoundTrip() throws IOException {
        double[][] m = CsrMatrixTest.randomSparse(9, 0.3, 5);
        Path dense = dir.resolve("m.bin");
        MatrixIO.writeDense(dense, m);
        assertEquals(8 * 81, Files.size(dense));
        assertBitwiseEquals(m, MatrixIO.loadDense(dense, 9));

        Path csr = dir.resolve("m.csr");
        writeCsr(csr, CsrMatrix.fromDense(m));
        assertBitwiseEquals(m, MatrixIO.loadCsr(csr, 9).toDense());
        assertThrows(IOException.class, () -> MatrixIO.loadCsr(csr, 8));
        assertThrows(IOException.class, () -> MatrixIO.loadDense(dense, 8));
    }

    @Test
    void parseValidatesArguments() {
        List<String> base = List.of("--n", "4", "--a", "a.bin", "--b", "b.bin");
        assertThrows(IllegalArgumentException.class, () -> Main.parse(base.toArray(String[]::new)));
        assertThrows(IllegalArgumentException.class, () -> Main.parse(concat(base, "--method", "strassen")));
        assertThrows(IllegalArgumentException.class, () -> Main.parse(concat(base, "--method", "to_csr", "--out", "c.bin")));
        assertThrows(IllegalArgumentException.class, () -> Main.parse(concat(base, "--method", "blocked", "--block", "0")));
        Main.Options opt = Main.parse(concat(base, "--method", "blocked"));
        assertEquals(DenseMatMul.DEFAULT_BLOCK, opt.block());
        assertEquals(1, opt.warmup());
    }

    /** Every method runs end to end on CSR and dense files; the written product equals the dense baseline. */
    @ParameterizedTest
    @ValueSource(strings = {"basic", "ikj", "blocked", "csr", "to_csr", "to_dense"})
    void runWritesSamplesAndProduct(String method) throws IOException {
        int n = 12;
        double[][] a = CsrMatrixTest.randomSparse(n, 0.25, 1);
        double[][] b = CsrMatrixTest.randomSparse(n, 0.25, 2);
        Path pa = dir.resolve("a.csr");
        Path pb = dir.resolve("b.bin");
        writeCsr(pa, CsrMatrix.fromDense(a));
        MatrixIO.writeDense(pb, b);
        Path out = dir.resolve("c.bin");
        String[] args = {"--method", method, "--n", "12", "--a", pa.toString(), "--b", pb.toString(), "--block", "5",
                "--pre-warm-seconds", "0.01", "--warmup", "1", "--reps", "3", "--min-sample-seconds", "0.001",
                "--min-warmup-seconds", "0"};
        if (!method.equals("to_csr")) {
            args = concat(List.of(args), "--out", out.toString());
        }
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        Main.run(Main.parse(args), new PrintStream(bytes, true, StandardCharsets.UTF_8));
        List<String> lines = bytes.toString(StandardCharsets.UTF_8).lines().toList();

        List<String> csv = lines.stream().filter(l -> !l.startsWith("#")).toList();
        assertEquals("kind,index,batch,elapsed_ns,checksum", csv.get(0));
        assertEquals(1, csv.stream().filter(l -> l.startsWith("warmup,")).count());
        assertEquals(3, csv.stream().filter(l -> l.startsWith("measure,")).count());
        Map<String, String> meta = new HashMap<>();
        lines.stream().filter(l -> l.startsWith("# ")).flatMap(l -> Arrays.stream(l.substring(2).split(",")))
                .forEach(kv -> meta.put(kv.split("=")[0], kv.split("=")[1]));
        assertEquals(Long.toString(CsrMatrix.fromDense(a).nnz()), meta.get("nnz_a"));
        assertEquals(Long.toString(DenseMatMul.countNonzeros(b)), meta.get("nnz_b"));
        assertEquals("12", meta.get("pre_warm_n"));

        double[][] expected = DenseMatMulTest.product("basic", a, b, 1);
        if (!method.equals("to_csr")) {
            assertBitwiseEquals(expected, MatrixIO.loadDense(out, n));
            assertEquals(Long.toString(DenseMatMul.countNonzeros(expected)), meta.get("nnz_c_nonzero"));
        }
        String checksum = csv.get(csv.size() - 1).split(",")[4];
        if (!method.equals("to_csr")) {
            assertEquals(Double.toString(DenseMatMul.checksum(expected)), checksum);
        }
    }

    private static String[] concat(List<String> base, String... more) {
        return java.util.stream.Stream.concat(base.stream(), Arrays.stream(more)).toArray(String[]::new);
    }
}
