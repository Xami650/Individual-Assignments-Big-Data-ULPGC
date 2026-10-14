package org.ulpgc.bd.matmul;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.math.BigDecimal;
import java.math.MathContext;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class MatMulTest {

    private static final double U = Math.scalb(1.0, -53);

    private static double[][] product(double[][] a, double[][] b) {
        int n = a.length;
        double[][] c = MatMul.zeros(n);
        MatMul.multiply(a, b, c, n);
        return c;
    }

    private static double[][] identity(int n) {
        double[][] m = MatMul.zeros(n);
        for (int i = 0; i < n; i++) {
            m[i][i] = 1.0;
        }
        return m;
    }

    private static double[][] random(int n, long seed) {
        Random rng = new Random(seed);
        double[][] m = new double[n][n];
        for (double[] row : m) {
            for (int j = 0; j < n; j++) {
                row[j] = rng.nextDouble(-1.0, 1.0);
            }
        }
        return m;
    }

    private static void assertMatrixEquals(double[][] expected, double[][] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], actual[i], 0.0, "row " + i);
        }
    }

    @Test
    void handComputed2x2() {
        double[][] c = product(new double[][]{{1, 2}, {3, 4}}, new double[][]{{5, 6}, {7, 8}});
        assertMatrixEquals(new double[][]{{19, 22}, {43, 50}}, c);
    }

    @Test
    void handComputed3x3NotCommutative() {
        double[][] a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        double[][] b = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
        assertMatrixEquals(new double[][]{{30, 24, 18}, {84, 69, 54}, {138, 114, 90}}, product(a, b));
        assertMatrixEquals(new double[][]{{90, 114, 138}, {54, 69, 84}, {18, 24, 30}}, product(b, a));
    }

    @Test
    void oneByOne() {
        assertMatrixEquals(new double[][]{{-7.5}}, product(new double[][]{{3.0}}, new double[][]{{-2.5}}));
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 4, 7})
    void identityIsExact(int n) {
        double[][] a = random(n, n);
        assertMatrixEquals(a, product(a, identity(n)));
        assertMatrixEquals(a, product(identity(n), a));
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 5})
    void zero(int n) {
        double[][] a = random(n, 10 + n);
        assertMatrixEquals(MatMul.zeros(n), product(a, MatMul.zeros(n)));
        assertMatrixEquals(MatMul.zeros(n), product(MatMul.zeros(n), a));
    }

    @Test
    void outputBufferIsOverwritten() {
        double[][] a = {{1, 2}, {3, 4}};
        double[][] b = {{5, 6}, {7, 8}};
        double[][] c = {{99, 99}, {99, 99}};
        MatMul.multiply(a, b, c, 2);
        MatMul.multiply(a, b, c, 2);
        assertMatrixEquals(new double[][]{{19, 22}, {43, 50}}, c);
    }

    /** Exact reference with BigDecimal; the double result must satisfy the 2*gamma_n bound. */
    @ParameterizedTest
    @ValueSource(ints = {10, 33})
    void matchesExactReferenceWithinTolerance(int n) {
        double[][] a = random(n, 100 + n);
        double[][] b = random(n, 200 + n);
        double[][] c = product(a, b);
        double gamma = n * U / (1 - n * U);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                BigDecimal exact = BigDecimal.ZERO;
                double abs = 0.0;
                for (int k = 0; k < n; k++) {
                    exact = exact.add(new BigDecimal(a[i][k]).multiply(new BigDecimal(b[k][j])));
                    abs += Math.abs(a[i][k]) * Math.abs(b[k][j]);
                }
                double err = new BigDecimal(c[i][j]).subtract(exact).abs().round(MathContext.DECIMAL64).doubleValue();
                assertTrue(err <= 2 * gamma * abs, "C[" + i + "][" + j + "] error " + err);
            }
        }
    }

    @Test
    void checksumSumsAllEntries() {
        assertEquals(10.5, MatMul.checksum(new double[][]{{1, 2}, {3, 4.5}}));
    }

    @Test
    void writeLoadRoundTrip(@TempDir Path tmp) throws IOException {
        double[][] a = random(6, 7);
        Path path = tmp.resolve("m.bin");
        MatMul.write(path, a);
        assertEquals(8L * 36, Files.size(path));
        assertMatrixEquals(a, MatMul.load(path, 6));
    }

    @Test
    void loadRejectsWrongSize(@TempDir Path tmp) throws IOException {
        Path path = tmp.resolve("m.bin");
        MatMul.write(path, MatMul.zeros(3));
        assertThrows(IOException.class, () -> MatMul.load(path, 4));
    }

    @Test
    void cliOutputContract(@TempDir Path tmp) throws IOException {
        MatMul.write(tmp.resolve("a.bin"), new double[][]{{1, 2}, {3, 4}});
        MatMul.write(tmp.resolve("b.bin"), new double[][]{{5, 6}, {7, 8}});
        Path out = tmp.resolve("c.bin");
        Main.Options opt = Main.parse(new String[]{"--n", "2", "--a", tmp.resolve("a.bin").toString(),
                "--b", tmp.resolve("b.bin").toString(), "--warmup", "2", "--reps", "3",
                "--min-sample-seconds", "0.001", "--min-warmup-seconds", "0", "--out", out.toString()});
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        Main.run(opt, new PrintStream(bytes, true, StandardCharsets.UTF_8));

        List<String> lines = bytes.toString(StandardCharsets.UTF_8).lines().toList();
        assertEquals("kind,index,batch,elapsed_ns,checksum", lines.get(0));
        assertEquals(List.of("warmup", "warmup", "measure", "measure", "measure"),
                lines.stream().skip(1).map(l -> l.split(",")[0]).toList());
        for (String line : lines.subList(1, lines.size())) {
            String[] f = line.split(",");
            assertTrue(Integer.parseInt(f[2]) >= 1 && Long.parseLong(f[3]) > 0, line);
            assertEquals(134.0, Double.parseDouble(f[4]), 0.0, line);
        }
        assertEquals("1", lines.get(1).split(",")[2]);
        assertEquals(1, lines.stream().skip(3).map(l -> l.split(",")[2]).distinct().count());
        assertMatrixEquals(new double[][]{{19, 22}, {43, 50}}, MatMul.load(out, 2));
    }

    @Test
    void cliMinWarmupExtendsWarmups(@TempDir Path tmp) throws IOException {
        MatMul.write(tmp.resolve("a.bin"), new double[][]{{1, 2}, {3, 4}});
        MatMul.write(tmp.resolve("b.bin"), new double[][]{{5, 6}, {7, 8}});
        Main.Options opt = Main.parse(new String[]{"--n", "2", "--a", tmp.resolve("a.bin").toString(),
                "--b", tmp.resolve("b.bin").toString(), "--warmup", "1", "--reps", "1",
                "--min-sample-seconds", "0.01", "--min-warmup-seconds", "0.05"});
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        Main.run(opt, new PrintStream(bytes, true, StandardCharsets.UTF_8));
        List<String[]> warmups = bytes.toString(StandardCharsets.UTF_8).lines()
                .map(l -> l.split(",")).filter(f -> f[0].equals("warmup")).toList();
        assertTrue(warmups.size() >= 2, "warm-up samples: " + warmups.size());
        assertTrue(warmups.stream().mapToLong(f -> Long.parseLong(f[3])).sum() >= 50_000_000L);
    }

    @Test
    void parseRejectsMissingSize() {
        assertThrows(IllegalArgumentException.class, () -> Main.parse(new String[]{"--reps", "3"}));
    }
}
