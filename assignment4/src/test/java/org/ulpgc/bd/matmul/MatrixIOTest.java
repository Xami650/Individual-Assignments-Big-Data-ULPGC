package org.ulpgc.bd.matmul;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.ulpgc.bd.matmul.DenseMatMulTest.assertBitwiseEquals;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class MatrixIOTest {

    @TempDir
    Path dir;

    @Test
    void denseFilesRoundTripAndSizeIsChecked() throws IOException {
        double[][] m = DenseMatMulTest.random(9, 5);
        m[3][4] = -0.0;
        Path file = dir.resolve("m.bin");
        MatrixIO.writeDense(file, m);
        assertEquals(8 * 81, Files.size(file));
        assertBitwiseEquals(m, MatrixIO.loadDense(file, 9));
        assertThrows(IOException.class, () -> MatrixIO.loadDense(file, 8));
    }
}
