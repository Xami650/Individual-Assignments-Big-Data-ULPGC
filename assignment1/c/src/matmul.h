#ifndef MATMUL_H
#define MATMUL_H

#include <stddef.h>

/*
 * Basic dense matrix multiplication in C.
 *
 * Storage: an n x n matrix is one contiguous row-major block of n*n doubles,
 * element (i, j) at m[i*n + j].
 */

/* Returns a zero-filled n x n matrix (calloc), or NULL on allocation failure. */
double *mm_zeros(size_t n);

/* C = A * B with the basic i-j-k triple loop and a scalar accumulator.
 * Performs n^3 multiplications and n^3 additions. Every C[i][j] is assigned,
 * so c does not need to be cleared between calls. */
void mm_multiply(const double *a, const double *b, double *c, size_t n);

/* Sum of all entries in row-major order, used to consume the result. */
double mm_checksum(const double *c, size_t n);

/* Reads a raw little-endian float64 row-major n x n file. Returns a malloc'd
 * matrix, or NULL (with a message on stderr) if the file is missing, has the
 * wrong size or cannot be read. */
double *mm_load(const char *path, size_t n);

/* Writes a matrix as raw little-endian float64, row-major. Returns 0 on success. */
int mm_write(const char *path, const double *c, size_t n);

#endif
