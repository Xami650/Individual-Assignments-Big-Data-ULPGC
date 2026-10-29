package org.ulpgc.bd.matmul;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.ulpgc.bd.matmul.DenseMatMulTest.assertBitwiseEquals;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.ulpgc.bd.matmul.ParallelMatMul.Schedule;

/** The command-line contract of the driver. */
class MainTest {

    @TempDir
    Path dir;

    @Test
    void parseValidatesArguments() {
        List<String> base = List.of("--n", "4", "--a", "a.bin", "--b", "b.bin");
        assertThrows(IllegalArgumentException.class, () -> Main.parse(base.toArray(String[]::new)));
        assertThrows(IllegalArgumentException.class, () -> Main.parse(concat(base, "--method", "blocked")));
        assertThrows(IllegalArgumentException.class, () -> Main.parse(concat(base, "--method", "ikj", "--threads", "2")));
        assertThrows(IllegalArgumentException.class,
                () -> Main.parse(concat(base, "--method", "ikj", "--schedule", "static")));
        assertThrows(IllegalArgumentException.class,
                () -> Main.parse(concat(base, "--method", "par_ikj", "--threads", "0")));
        assertThrows(IllegalArgumentException.class,
                () -> Main.parse(concat(base, "--method", "par_ikj", "--schedule", "guided")));
        Main.Options seq = Main.parse(concat(base, "--method", "ikj"));
        assertEquals(1, seq.threads());
        assertNull(seq.schedule());
        Main.Options par = Main.parse(concat(base, "--method", "par_ikj", "--threads", "6"));
        assertEquals(6, par.threads());
        assertEquals(Schedule.DYNAMIC, par.schedule());
        assertEquals(5, par.reps());
    }

    /** Every configuration runs end to end; the written product and the checksum equal the sequential ikj. */
    @ParameterizedTest
    @CsvSource({"ikj, 1, ''", "par_ikj, 3, dynamic", "par_ikj, 2, static", "par_ikj, 16, dynamic"})
    void runWritesSamplesMetadataAndProduct(String method, int threads, String schedule) throws IOException {
        int n = 12;
        double[][] a = DenseMatMulTest.random(n, 1);
        double[][] b = DenseMatMulTest.random(n, 2);
        Path pa = dir.resolve("a.bin");
        Path pb = dir.resolve("b.bin");
        MatrixIO.writeDense(pa, a);
        MatrixIO.writeDense(pb, b);
        Path out = dir.resolve("c.bin");
        List<String> args = List.of("--method", method, "--n", "12", "--a", pa.toString(), "--b", pb.toString(),
                "--pre-warm-seconds", "0.01", "--warmup", "1", "--reps", "3", "--min-sample-seconds", "0.001",
                "--min-warmup-seconds", "0", "--out", out.toString());
        if (method.equals("par_ikj")) {
            args = Arrays.asList(concat(args, "--threads", Integer.toString(threads), "--schedule", schedule));
        }
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        Main.run(Main.parse(args.toArray(String[]::new)), new PrintStream(bytes, true, StandardCharsets.UTF_8));
        List<String> lines = bytes.toString(StandardCharsets.UTF_8).lines().toList();

        List<String> csv = lines.stream().filter(l -> !l.startsWith("#")).toList();
        assertEquals(Main.HEADER, csv.get(0));
        assertEquals(1, csv.stream().filter(l -> l.startsWith("warmup,")).count());
        assertEquals(3, csv.stream().filter(l -> l.startsWith("measure,")).count());
        Map<String, String> meta = new HashMap<>();
        lines.stream().filter(l -> l.startsWith("# ")).flatMap(l -> Arrays.stream(l.substring(2).split(",")))
                .forEach(kv -> meta.put(kv.split("=", 2)[0], kv.split("=", 2)[1]));
        assertEquals(Integer.toString(threads), meta.get("threads"));
        assertEquals(schedule, meta.get("schedule"));
        assertEquals(Long.toString(12L * 12 * 12), meta.get("multiply_adds"));
        assertEquals("12", meta.get("pre_warm_n"));
        for (String key : List.of("available_processors", "superword", "jvm_threads", "os_threads", "gc_collections",
                "jit_ms", "allocated_bytes_measured", "allocated_bytes_total")) {
            assertTrue(meta.containsKey(key), key);
        }

        double[][] expected = DenseMatMulTest.ikj(a, b);
        assertBitwiseEquals(expected, MatrixIO.loadDense(out, n));
        String[] last = csv.get(csv.size() - 1).split(",", -1);
        assertEquals(Main.HEADER.split(",").length, last.length);
        assertEquals(Double.toString(DenseMatMul.checksum(expected)), last[4]);
        if (ProcessCounters.available()) {
            assertEquals("available", meta.get("counters"));
            assertFalse(last[5].isEmpty());
            assertTrue(Long.parseLong(last[7]) > 0, "instructions");
        }
    }

    private static String[] concat(List<String> base, String... more) {
        return Stream.concat(base.stream(), Arrays.stream(more)).toArray(String[]::new);
    }
}
