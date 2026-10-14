#include "matmul.h"

#include <stdio.h>
#include <stdlib.h>

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the raw input format is little-endian; this loader assumes a little-endian host"
#endif

double *mm_zeros(size_t n) {
    return calloc(n * n, sizeof(double));
}

void mm_multiply(const double *a, const double *b, double *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double s = 0.0;
            for (size_t k = 0; k < n; k++) {
                s += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = s;
        }
    }
}

double mm_checksum(const double *c, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n * n; i++) {
        s += c[i];
    }
    return s;
}

double *mm_load(const char *path, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return NULL;
    }
    long expected = (long)(n * n * sizeof(double));
    if (fseek(f, 0, SEEK_END) != 0 || ftell(f) != expected) {
        fprintf(stderr, "%s: size does not match %ld bytes expected for n=%zu\n", path, expected, n);
        fclose(f);
        return NULL;
    }
    rewind(f);
    double *m = malloc(n * n * sizeof(double));
    if (!m || fread(m, sizeof(double), n * n, f) != n * n) {
        fprintf(stderr, "%s: read failed\n", path);
        free(m);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return m;
}

int mm_write(const char *path, const double *c, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        perror(path);
        return -1;
    }
    size_t written = fwrite(c, sizeof(double), n * n, f);
    return (fclose(f) == 0 && written == n * n) ? 0 : -1;
}
