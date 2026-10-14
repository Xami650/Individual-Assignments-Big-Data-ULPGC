/*
 * Benchmark driver for the C basic matrix multiplication.
 *
 * Common command-line contract shared with the Python and Java drivers:
 *
 *   c/build/matmul --n N [--a PATH] [--b PATH] [--warmup W] [--reps R]
 *                  [--min-sample-seconds S] [--min-warmup-seconds SW] [--out PATH]
 *
 * Inputs default to data/inputs/A_<n>.bin and B_<n>.bin relative to the working
 * directory (run from assignment1/). Output on stdout is CSV with the header
 * kind,index,batch,elapsed_ns,checksum and one row per sample. Each sample times
 * `batch` back-to-back multiplications with clock_gettime(CLOCK_MONOTONIC_RAW)
 * (kernel only). The first warm-up uses batch = 1; every later sample uses
 * k = max(1, ceil(S / t)), with t the per-multiplication time of the most recent
 * warm-up. The checksum is the sum of all entries of C, printed with %.17g, which
 * round-trips. When W >= 1, warm-up samples continue beyond W until at least SW
 * seconds of warm-up work have run, so that the CPU has left its low-frequency
 * ramp. With --out, C is written as raw little-endian float64 at the end.
 *
 * The kernel lives in a separate translation unit and the build uses no LTO, so
 * the compiler cannot elide or merge the repeated calls of a batch.
 */
#include "matmul.h"

#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static int64_t run_sample(const double *a, const double *b, double *c, size_t n, long batch) {
    int64_t t0 = now_ns();
    for (long r = 0; r < batch; r++) {
        mm_multiply(a, b, c, n);
    }
    return now_ns() - t0;
}

static void usage(void) {
    fprintf(stderr, "usage: --n N [--a PATH] [--b PATH] [--warmup W] [--reps R] "
                    "[--min-sample-seconds S] [--min-warmup-seconds SW] [--out PATH]\n");
}

int main(int argc, char **argv) {
    long n = 0, warmup = 1, reps = 5;
    double min_sample = 0.1, min_warmup = 1.0;
    const char *a_path = NULL, *b_path = NULL, *out_path = NULL;

    static const struct option options[] = {
        {"n", required_argument, NULL, 'n'},
        {"a", required_argument, NULL, 'a'},
        {"b", required_argument, NULL, 'b'},
        {"warmup", required_argument, NULL, 'w'},
        {"reps", required_argument, NULL, 'r'},
        {"min-sample-seconds", required_argument, NULL, 's'},
        {"min-warmup-seconds", required_argument, NULL, 'W'},
        {"out", required_argument, NULL, 'o'},
        {NULL, 0, NULL, 0},
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "", options, NULL)) != -1) {
        switch (opt) {
            case 'n': n = strtol(optarg, NULL, 10); break;
            case 'a': a_path = optarg; break;
            case 'b': b_path = optarg; break;
            case 'w': warmup = strtol(optarg, NULL, 10); break;
            case 'r': reps = strtol(optarg, NULL, 10); break;
            case 's': min_sample = strtod(optarg, NULL); break;
            case 'W': min_warmup = strtod(optarg, NULL); break;
            case 'o': out_path = optarg; break;
            default: usage(); return 2;
        }
    }
    if (optind != argc || n < 1 || warmup < 0 || reps < 0 || warmup + reps < 1 || min_warmup < 0) {
        fprintf(stderr, "error: need --n >= 1 and at least one warm-up or measured sample\n");
        usage();
        return 2;
    }

    char a_default[64], b_default[64];
    snprintf(a_default, sizeof a_default, "data/inputs/A_%ld.bin", n);
    snprintf(b_default, sizeof b_default, "data/inputs/B_%ld.bin", n);
    size_t un = (size_t)n;
    double *a = mm_load(a_path ? a_path : a_default, un);
    double *b = mm_load(b_path ? b_path : b_default, un);
    double *c = mm_zeros(un);
    if (!a || !b || !c) {
        fprintf(stderr, "error: could not load inputs or allocate output\n");
        return 1;
    }

    printf("kind,index,batch,elapsed_ns,checksum\n");
    fflush(stdout);
    long batch = 1;
    int64_t warmup_ns = 0;
    for (long i = 0; i < warmup || (warmup > 0 && (double)warmup_ns < min_warmup * 1e9); i++) {
        int64_t elapsed = run_sample(a, b, c, un, batch);
        printf("warmup,%ld,%ld,%lld,%.17g\n", i, batch, (long long)elapsed, mm_checksum(c, un));
        fflush(stdout);
        warmup_ns += elapsed;
        double k = ceil(min_sample * 1e9 * (double)batch / (double)(elapsed > 0 ? elapsed : 1));
        batch = k < 1.0 ? 1 : (long)k;
    }
    for (long i = 0; i < reps; i++) {
        int64_t elapsed = run_sample(a, b, c, un, batch);
        printf("measure,%ld,%ld,%lld,%.17g\n", i, batch, (long long)elapsed, mm_checksum(c, un));
        fflush(stdout);
    }

    int status = 0;
    if (out_path && mm_write(out_path, c, un) != 0) {
        fprintf(stderr, "error: could not write %s\n", out_path);
        status = 1;
    }
    free(a);
    free(b);
    free(c);
    return status;
}
