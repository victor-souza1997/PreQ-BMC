#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

extern int qnn_input_dim(void);
extern int qnn_output_dim(void);
extern void qnn_forward_fixed(const int64_t *, int64_t *);

#ifndef PREQBMC_COMPILER_ID
#define PREQBMC_COMPILER_ID "host-unknown"
#endif
#ifndef PREQBMC_COMPILER_VERSION
#define PREQBMC_COMPILER_VERSION "host-unknown"
#endif
#ifndef PREQBMC_NDK_VERSION
#define PREQBMC_NDK_VERSION "not-an-ndk-build"
#endif
#ifndef PREQBMC_ANDROID_ABI
#define PREQBMC_ANDROID_ABI "host"
#endif
#ifndef PREQBMC_ANDROID_PLATFORM
#define PREQBMC_ANDROID_PLATFORM "host"
#endif

static const unsigned char MAGIC[8] = {'P','Q','N','N','V','0','0','1'};
static volatile uint64_t output_sink;

static int read_exact(FILE *file, void *buffer, size_t bytes) {
    return fread(buffer, 1, bytes, file) == bytes;
}

static uint32_t little_u32(const unsigned char bytes[4]) {
    return ((uint32_t)bytes[0]) | ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static int64_t little_i64(const unsigned char bytes[8]) {
    uint64_t value = 0;
    for (int i = 7; i >= 0; --i) value = (value << 8) | bytes[i];
    if (value <= INT64_MAX) return (int64_t)value;
    return -1 - (int64_t)(UINT64_MAX - value);
}

static uint64_t nanoseconds(clockid_t clock) {
    struct timespec value;
    if (clock_gettime(clock, &value) != 0) {
        perror("clock_gettime");
        exit(2);
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
}

static int compare_u64(const void *left, const void *right) {
    const uint64_t a = *(const uint64_t *)left, b = *(const uint64_t *)right;
    return (a > b) - (a < b);
}

static long positive_long(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || !end || *end || value <= 0) {
        fprintf(stderr, "%s must be a positive integer\n", name);
        exit(2);
    }
    return value;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 5) {
        fprintf(stderr, "usage: %s vectors.bin [warmup=10000] [latency_samples=10000] [throughput_iterations=100000]\n", argv[0]);
        return 2;
    }
    const long warmup = argc > 2 ? positive_long(argv[2], "warmup") : 10000;
    const long samples = argc > 3 ? positive_long(argv[3], "latency_samples") : 10000;
    const long iterations = argc > 4 ? positive_long(argv[4], "throughput_iterations") : 100000;
    FILE *file = fopen(argv[1], "rb");
    if (!file) { perror("vectors"); return 2; }
    unsigned char magic[8], header[12];
    if (!read_exact(file, magic, sizeof(magic)) || memcmp(magic, MAGIC, sizeof(magic)) ||
        !read_exact(file, header, sizeof(header))) {
        fprintf(stderr, "invalid vector corpus header\n");
        fclose(file); return 2;
    }
    const uint32_t count = little_u32(header), ni = little_u32(header + 4), no = little_u32(header + 8);
    if (!count || ni != (uint32_t)qnn_input_dim() || no != (uint32_t)qnn_output_dim()) {
        fprintf(stderr, "vector dimensions differ from compiled QNN\n");
        fclose(file); return 2;
    }
    int64_t *input = calloc(ni, sizeof(*input));
    int64_t *expected = calloc(no, sizeof(*expected));
    int64_t *output = calloc(no, sizeof(*output));
    int64_t *benchmark_inputs = calloc((count < 256 ? count : 256) * ni, sizeof(*benchmark_inputs));
    unsigned char raw8[8], raw4[4];
    if (!input || !expected || !output || !benchmark_inputs) {
        fprintf(stderr, "allocation failure\n"); fclose(file); return 2;
    }
    const uint32_t benchmark_count = count < 256 ? count : 256;
    const uint32_t stride = (count + benchmark_count - 1) / benchmark_count;
    uint32_t stored = 0, mismatches = 0, correct = 0;
    for (uint32_t row = 0; row < count; ++row) {
        if (!read_exact(file, raw4, 4)) { fprintf(stderr, "truncated label\n"); return 2; }
        const int32_t label = (int32_t)little_u32(raw4);
        for (uint32_t i = 0; i < ni; ++i) {
            if (!read_exact(file, raw8, 8)) { fprintf(stderr, "truncated input\n"); return 2; }
            input[i] = little_i64(raw8);
        }
        for (uint32_t i = 0; i < no; ++i) {
            if (!read_exact(file, raw8, 8)) { fprintf(stderr, "truncated output\n"); return 2; }
            expected[i] = little_i64(raw8);
        }
        qnn_forward_fixed(input, output);
        if (memcmp(output, expected, no * sizeof(*output))) ++mismatches;
        uint32_t selected = 0;
        for (uint32_t i = 1; i < no; ++i) if (output[i] > output[selected]) selected = i;
        correct += selected == (uint32_t)label;
        if (stored < benchmark_count && (row % stride == 0 || count - row <= benchmark_count - stored)) {
            memcpy(benchmark_inputs + stored * ni, input, ni * sizeof(*input));
            ++stored;
        }
    }
    if (fgetc(file) != EOF || stored != benchmark_count) {
        fprintf(stderr, "invalid vector corpus length\n"); return 2;
    }
    fclose(file);
    if (mismatches) {
        printf("{\"schema\":\"preqbmc_android_benchmark_v1\",\"parity_status\":\"MISMATCH\",\"vectors\":%u,\"mismatches\":%u}\n", count, mismatches);
        return 1;
    }
    for (long i = 0; i < warmup; ++i) {
        qnn_forward_fixed(benchmark_inputs + (i % benchmark_count) * ni, output);
        output_sink ^= (uint64_t)output[i % no];
    }
    uint64_t *latencies = calloc((size_t)samples, sizeof(*latencies));
    if (!latencies) { fprintf(stderr, "latency allocation failure\n"); return 2; }
    for (long i = 0; i < samples; ++i) {
        const uint64_t start = nanoseconds(CLOCK_MONOTONIC);
        qnn_forward_fixed(benchmark_inputs + (i % benchmark_count) * ni, output);
        latencies[i] = nanoseconds(CLOCK_MONOTONIC) - start;
        output_sink ^= (uint64_t)output[i % no];
    }
    qsort(latencies, (size_t)samples, sizeof(*latencies), compare_u64);
    const uint64_t wall_start = nanoseconds(CLOCK_MONOTONIC);
    const uint64_t cpu_start = nanoseconds(CLOCK_PROCESS_CPUTIME_ID);
    for (long i = 0; i < iterations; ++i) {
        qnn_forward_fixed(benchmark_inputs + (i % benchmark_count) * ni, output);
        output_sink ^= (uint64_t)output[i % no];
    }
    const uint64_t cpu_ns = nanoseconds(CLOCK_PROCESS_CPUTIME_ID) - cpu_start;
    const uint64_t wall_ns = nanoseconds(CLOCK_MONOTONIC) - wall_start;
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) memset(&usage, 0, sizeof(usage));
    const size_t p95 = (size_t)((samples - 1) * 95 / 100);
    printf("{\"schema\":\"preqbmc_android_benchmark_v1\","
           "\"parity_status\":\"EXACT_MATCH\",\"vectors\":%u,\"mismatches\":0,"
           "\"device_test_accuracy\":%.12g,\"benchmark_vectors\":%u,\"warmup\":%ld,"
           "\"latency_samples\":%ld,\"latency_median_ns\":%" PRIu64 ",\"latency_p95_ns\":%" PRIu64 ","
           "\"throughput_iterations\":%ld,\"throughput_per_second\":%.12g,"
           "\"cpu_time_over_wall_percent\":%.12g,\"peak_process_rss_kib\":%ld,"
           "\"checksum\":%" PRIu64 ",\"scope\":\"qnn_forward_fixed_only\","
           "\"compiler_id\":\"%s\",\"compiler_version\":\"%s\",\"ndk_version\":\"%s\","
           "\"compiled_abi\":\"%s\",\"android_platform\":\"%s\"}\n",
           count, (double)correct / count, benchmark_count, warmup, samples,
           latencies[samples / 2], latencies[p95], iterations,
           wall_ns ? (double)iterations * 1e9 / wall_ns : 0.0,
           wall_ns ? (double)cpu_ns * 100.0 / wall_ns : 0.0,
           usage.ru_maxrss, output_sink, PREQBMC_COMPILER_ID, PREQBMC_COMPILER_VERSION,
           PREQBMC_NDK_VERSION, PREQBMC_ANDROID_ABI, PREQBMC_ANDROID_PLATFORM);
    free(latencies); free(benchmark_inputs); free(output); free(expected); free(input);
    return 0;
}
