/*
 * Tests for the C implementation: run with `make test` from c/.
 * The CLI test runs build/matmul, so the working directory must be c/.
 */
#include "matmul.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures = 0;
static int checks = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        checks++;                                                         \
        if (!(cond)) {                                                    \
            failures++;                                                   \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                 \
    } while (0)

static double *product(const double *a, const double *b, size_t n) {
    double *c = mm_zeros(n);
    mm_multiply(a, b, c, n);
    return c;
}

static int equal(const double *x, const double *y, size_t n) {
    return memcmp(x, y, n * n * sizeof(double)) == 0;
}

static double *identity(size_t n) {
    double *m = mm_zeros(n);
    for (size_t i = 0; i < n; i++) {
        m[i * n + i] = 1.0;
    }
    return m;
}

/* Deterministic values: uniform in [-1, 1) if !integers, else integers in [-100, 100]. */
static double *random_matrix(size_t n, uint64_t seed, int integers) {
    double *m = malloc(n * n * sizeof(double));
    uint64_t x = seed * 0x9E3779B97F4A7C15ULL + 1;
    for (size_t i = 0; i < n * n; i++) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        m[i] = integers ? (double)((int64_t)(x % 201) - 100) : (double)(x >> 11) * 0x1.0p-52 - 1.0;
    }
    return m;
}

static void test_hand_computed(void) {
    double a2[] = {1, 2, 3, 4}, b2[] = {5, 6, 7, 8}, c2[] = {19, 22, 43, 50};
    double *c = product(a2, b2, 2);
    CHECK(equal(c, c2, 2));
    free(c);

    double a3[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double b3[] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    double ab[] = {30, 24, 18, 84, 69, 54, 138, 114, 90};
    double ba[] = {90, 114, 138, 54, 69, 84, 18, 24, 30};
    c = product(a3, b3, 3);
    CHECK(equal(c, ab, 3));
    free(c);
    c = product(b3, a3, 3);
    CHECK(equal(c, ba, 3));
    free(c);

    double a1[] = {3.0}, b1[] = {-2.5}, c1[] = {-7.5};
    c = product(a1, b1, 1);
    CHECK(equal(c, c1, 1));
    free(c);
}

static void test_identity_and_zero(void) {
    size_t sizes[] = {1, 4, 7};
    for (size_t t = 0; t < 3; t++) {
        size_t n = sizes[t];
        double *a = random_matrix(n, n, 0), *id = identity(n), *z = mm_zeros(n);
        double *c1 = product(a, id, n), *c2 = product(id, a, n);
        double *c3 = product(a, z, n), *c4 = product(z, a, n);
        CHECK(equal(c1, a, n));
        CHECK(equal(c2, a, n));
        CHECK(equal(c3, z, n));
        CHECK(equal(c4, z, n));
        free(a); free(id); free(z); free(c1); free(c2); free(c3); free(c4);
    }
}

static void test_output_buffer_is_overwritten(void) {
    double a[] = {1, 2, 3, 4}, b[] = {5, 6, 7, 8}, expected[] = {19, 22, 43, 50};
    double c[] = {99, 99, 99, 99};
    mm_multiply(a, b, c, 2);
    mm_multiply(a, b, c, 2);
    CHECK(equal(c, expected, 2));
}

/* Integer-valued inputs: every product and partial sum is an integer below 2^53,
 * so double arithmetic is exact and must equal an int64 reference bit for bit. */
static void test_exact_integer_reference(void) {
    size_t n = 50;
    double *a = random_matrix(n, 1, 1), *b = random_matrix(n, 2, 1);
    double *c = product(a, b, n);
    int ok = 1;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            int64_t s = 0;
            for (size_t k = 0; k < n; k++) {
                s += (int64_t)a[i * n + k] * (int64_t)b[k * n + j];
            }
            ok &= c[i * n + j] == (double)s;
        }
    }
    CHECK(ok);
    free(a); free(b); free(c);
}

static void test_checksum(void) {
    double c[] = {1, 2, 3, 4.5};
    CHECK(mm_checksum(c, 2) == 10.5);
}

static void test_io(const char *dir) {
    char path[512];
    snprintf(path, sizeof path, "%s/m.bin", dir);
    double *a = random_matrix(6, 7, 0);
    CHECK(mm_write(path, a, 6) == 0);
    double *back = mm_load(path, 6);
    CHECK(back != NULL && equal(a, back, 6));
    fprintf(stderr, "(expected size-mismatch message follows)\n");
    CHECK(mm_load(path, 5) == NULL);
    free(a); free(back);
}

static void test_cli_contract(const char *dir) {
    char a_path[512], b_path[512], out_path[512], cmd[2048];
    snprintf(a_path, sizeof a_path, "%s/a.bin", dir);
    snprintf(b_path, sizeof b_path, "%s/b.bin", dir);
    snprintf(out_path, sizeof out_path, "%s/c.bin", dir);
    double a[] = {1, 2, 3, 4}, b[] = {5, 6, 7, 8}, expected[] = {19, 22, 43, 50};
    mm_write(a_path, a, 2);
    mm_write(b_path, b, 2);
    snprintf(cmd, sizeof cmd, "./build/matmul --n 2 --a %s --b %s --warmup 2 --reps 3 "
             "--min-sample-seconds 0.001 --min-warmup-seconds 0 --out %s", a_path, b_path, out_path);

    FILE *p = popen(cmd, "r");
    CHECK(p != NULL);
    if (!p) return;
    char line[256], kinds[5][16];
    long batches[5];
    int rows = 0, header_ok = 0, sums_ok = 1, times_ok = 1;
    if (fgets(line, sizeof line, p)) {
        header_ok = strcmp(line, "kind,index,batch,elapsed_ns,checksum\n") == 0;
    }
    while (fgets(line, sizeof line, p) && rows < 5) {
        long index, batch;
        long long elapsed;
        double sum;
        if (sscanf(line, "%15[a-z],%ld,%ld,%lld,%lf", kinds[rows], &index, &batch, &elapsed, &sum) != 5) {
            break;
        }
        batches[rows] = batch;
        sums_ok &= sum == 134.0;
        times_ok &= batch >= 1 && elapsed > 0;
        rows++;
    }
    CHECK(pclose(p) == 0);
    CHECK(header_ok);
    CHECK(rows == 5);
    if (rows == 5) {
        CHECK(strcmp(kinds[0], "warmup") == 0 && strcmp(kinds[1], "warmup") == 0);
        CHECK(strcmp(kinds[2], "measure") == 0 && strcmp(kinds[4], "measure") == 0);
        CHECK(batches[0] == 1);
        CHECK(batches[2] == batches[3] && batches[3] == batches[4]);
    }
    CHECK(sums_ok && times_ok);
    double *c = mm_load(out_path, 2);
    CHECK(c != NULL && equal(c, expected, 2));
    free(c);

    /* --min-warmup-seconds keeps warming up beyond --warmup until enough work has run. */
    snprintf(cmd, sizeof cmd, "./build/matmul --n 2 --a %s --b %s --warmup 1 --reps 1 "
             "--min-sample-seconds 0.01 --min-warmup-seconds 0.05", a_path, b_path);
    p = popen(cmd, "r");
    CHECK(p != NULL);
    if (!p) return;
    int warmups = 0;
    long long warmup_ns = 0;
    while (fgets(line, sizeof line, p)) {
        long index, batch;
        long long elapsed;
        if (sscanf(line, "warmup,%ld,%ld,%lld,", &index, &batch, &elapsed) == 3) {
            warmups++;
            warmup_ns += elapsed;
        }
    }
    CHECK(pclose(p) == 0);
    CHECK(warmups >= 2);
    CHECK(warmup_ns >= 50000000LL);
}

int main(void) {
    char dir[] = "/tmp/matmul_test_XXXXXX";
    if (!mkdtemp(dir)) {
        perror("mkdtemp");
        return 1;
    }
    test_hand_computed();
    test_identity_and_zero();
    test_output_buffer_is_overwritten();
    test_exact_integer_reference();
    test_checksum();
    test_io(dir);
    test_cli_contract(dir);

    char cmd[600];
    snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
    if (system(cmd) != 0) {
        fprintf(stderr, "warning: could not remove %s\n", dir);
    }
    printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
