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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
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

/** The command-line contract of the driver (adapted from Assignment 3's MainTest). */
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
                () -> Main.parse(concat(base, "--method", "basic", "--schedule", "static")));
        assertThrows(IllegalArgumentException.class,
                () -> Main.parse(concat(base, "--method", "par_ikj", "--threads", "0")));
        assertThrows(IllegalArgumentException.class,
                () -> Main.parse(concat(base, "--method", "par_ikj", "--schedule", "guided")));
        assertThrows(IllegalArgumentException.class, () -> Main.parse(concat(base, "--method", "dist")));
        assertThrows(IllegalArgumentException.class,
                () -> Main.parse(concat(base, "--method", "dist", "--workers", "0", "--cluster", "x")));
        assertThrows(IllegalArgumentException.class,
                () -> Main.parse(concat(base, "--method", "ikj", "--workers", "2")));
        assertThrows(IllegalArgumentException.class,
                () -> Main.parse(concat(base, "--method", "par_ikj", "--cluster", "x")));
        Main.Options seq = Main.parse(concat(base, "--method", "ikj"));
        assertEquals(1, seq.threads());
        assertNull(seq.schedule());
        Main.Options par = Main.parse(concat(base, "--method", "par_ikj", "--threads", "6"));
        assertEquals(6, par.threads());
        assertEquals(Schedule.DYNAMIC, par.schedule());
        assertEquals(5, par.reps());
        Main.Options dist = Main.parse(concat(base, "--method", "dist", "--workers", "3", "--cluster", "c",
                "--threads-per-worker", "2", "--base-port", "6001"));
        assertEquals(3, dist.workers());
        assertEquals(2, dist.threadsPerWorker());
        assertEquals(6001, dist.basePort());
        assertEquals(HazelcastSetup.DEFAULT_BASE_PORT, Main.parse(concat(base, "--method", "dist", "--workers", "1",
                "--cluster", "c")).basePort());
    }

    /** Every in-process method runs end to end; the written product and the checksum equal the sequential ikj. */
    @ParameterizedTest
    @CsvSource({"basic, 1, ''", "ikj, 1, ''", "par_ikj, 3, dynamic", "par_ikj, 2, static", "par_ikj, 16, dynamic"})
    void runWritesSamplesMetadataAndProduct(String method, int threads, String schedule) throws IOException {
        List<String> extra = method.equals("par_ikj")
                ? List.of("--threads", Integer.toString(threads), "--schedule", schedule) : List.of();
        Map<String, String> meta = new HashMap<>();
        List<String> csv = runDriver(method, extra, meta);
        assertEquals(Integer.toString(threads), meta.get("threads"));
        assertEquals(schedule, meta.get("schedule"));
        String[] last = csv.get(csv.size() - 1).split(",", -1);
        for (int i = 12; i < last.length; i++) {
            assertTrue(last[i].isEmpty(), "no distributed fields for " + method);
        }
    }

    /** dist through an embedded cluster: phases, bytes, member log, ping, clean-up and the exact product. */
    @Test
    void distRunsThroughTheCluster() throws Exception {
        try (TestCluster cluster = new TestCluster(2, 2)) {
            Path log = dir.resolve("members.csv");
            Map<String, String> meta = new HashMap<>();
            List<String> csv = runDriver("dist", List.of("--workers", "2", "--threads-per-worker", "2", "--cluster",
                    cluster.name, "--base-port", Integer.toString(cluster.basePort), "--member-log", log.toString(),
                    "--ping-rounds", "3"), meta);
            assertEquals("1", meta.get("connected"));
            assertEquals("2", meta.get("workers"));
            assertEquals("3", meta.get("ping_rounds"));
            assertEquals("0", meta.get("map_a_entries"));
            assertEquals("0", meta.get("map_b_entries"));
            assertEquals("2", meta.get("map_c_entries"));
            assertArrayEquals0(cluster.dist.mapSizes(), "the driver clears the maps at the end");

            List<String> header = Arrays.asList(Main.HEADER.split(","));
            String[] last = csv.get(csv.size() - 1).split(",", -1);
            long elapsed = Long.parseLong(last[3]), sum = 0;
            for (String phase : List.of("serialize_ns", "scatter_ns", "broadcast_ns", "compute_ns", "gather_ns",
                    "assemble_ns")) {
                sum += Long.parseLong(last[header.indexOf(phase)]);
            }
            assertTrue(sum <= elapsed && sum > 0.5 * elapsed, "phases add up to the end-to-end time");
            long n = 12, batch = Long.parseLong(last[2]);
            assertEquals(batch * 2 * (17 + 20 + n * (4 + 8 * n)), Long.parseLong(last[header.indexOf("b_payload_bytes")]));

            List<String> rows = Files.readAllLines(log);
            assertEquals(Main.MEMBER_HEADER, rows.get(0));
            assertEquals(2 * 4, rows.size() - 1, "one row per member and sample (1 warm-up + 3 measured)");
            for (String row : rows.subList(1, rows.size())) {
                String[] f = row.split(",", -1);
                assertEquals(f[2], f[3], "band t on member t");
                assertEquals("2", f[6], "kernel threads");
                assertEquals("true", f[8], "local");
            }
        }
    }

    private List<String> runDriver(String method, List<String> extra, Map<String, String> meta) throws IOException {
        int n = 12;
        double[][] a = DenseMatMulTest.random(n, 1);
        double[][] b = DenseMatMulTest.random(n, 2);
        Path pa = dir.resolve("a.bin");
        Path pb = dir.resolve("b.bin");
        MatrixIO.writeDense(pa, a);
        MatrixIO.writeDense(pb, b);
        Path out = dir.resolve("c.bin");
        List<String> args = new ArrayList<>(List.of("--method", method, "--n", "12", "--a", pa.toString(), "--b",
                pb.toString(), "--pre-warm-seconds", "0.01", "--warmup", "1", "--reps", "3", "--min-sample-seconds",
                "0.001", "--min-warmup-seconds", "0", "--out", out.toString()));
        args.addAll(extra);
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        Main.run(Main.parse(args.toArray(String[]::new)), new PrintStream(bytes, true, StandardCharsets.UTF_8));
        List<String> lines = bytes.toString(StandardCharsets.UTF_8).lines().toList();

        List<String> csv = lines.stream().filter(l -> !l.startsWith("#")).toList();
        assertEquals(Main.HEADER, csv.get(0));
        assertEquals(1, csv.stream().filter(l -> l.startsWith("warmup,")).count());
        assertEquals(3, csv.stream().filter(l -> l.startsWith("measure,")).count());
        lines.stream().filter(l -> l.startsWith("# ")).flatMap(l -> Arrays.stream(l.substring(2).split(",")))
                .forEach(kv -> meta.put(kv.split("=", 2)[0], kv.split("=", 2)[1]));
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
        return csv;
    }

    private static void assertArrayEquals0(int[] sizes, String message) {
        assertEquals("[0, 0, 0]", Arrays.toString(sizes), message);
    }

    private static String[] concat(List<String> base, String... more) {
        return Stream.concat(base.stream(), Arrays.stream(more)).toArray(String[]::new);
    }
}
