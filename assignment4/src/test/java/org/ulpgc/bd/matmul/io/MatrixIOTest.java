package org.ulpgc.bd.matmul.io;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.ulpgc.bd.matmul.Matrices.assertBitwiseEquals;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.ulpgc.bd.matmul.Matrices;
import org.junit.jupiter.api.io.TempDir;

class MatrixIOTest {

    @TempDir
    Path dir;

    @Test
    void denseFilesRoundTripAndSizeIsChecked() throws IOException {
        double[][] m = Matrices.random(9, 5);
        m[3][4] = -0.0;
        Path file = dir.resolve("m.bin");
        MatrixIO.writeDense(file, m);
        assertEquals(8 * 81, Files.size(file));
        assertBitwiseEquals(m, MatrixIO.loadDense(file, 9));
        assertThrows(IOException.class, () -> MatrixIO.loadDense(file, 8));
    }
}
