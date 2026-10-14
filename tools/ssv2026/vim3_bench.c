/* VIM3 benchmark for one exported QNN C file (compile with -DQNN_SOURCE='"path/qnn_compact.c"'),
   or for the float32 baseline (add -DFLOAT_MODEL).

   images.bin:   "QNNI" | uint32 count | per image: uint16 height, uint16 width, height*width*3 RGB bytes
   expected.bin: "QNNE" | uint32 count | uint32 classes | count*classes int64 logits | count int32 labels
   (float)       "QNNF" | uint32 count | count int32 host predictions | count int32 labels

   Every image runs encode + forward once per pass. Pass 1 checks the logits against the host's
   certified-semantics reference (float: the predicted class against the host float run); all passes
   are timed. Prints one JSON line. */
#define _POSIX_C_SOURCE 199309L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include QNN_SOURCE

static void *slurp(const char *path, size_t *size) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(2); }
  fseek(f, 0, SEEK_END);
  *size = (size_t)ftell(f);
  fseek(f, 0, SEEK_SET);
  void *data = malloc(*size);
  if (!data || fread(data, 1, *size, f) != *size) { fprintf(stderr, "read %s\n", path); exit(2); }
  fclose(f);
  return data;
}

static double now_us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

/* Peak RSS of this process image. getrusage's ru_maxrss survives execve on Linux, so it would
   report the launching shell's or Python's peak instead. */
static long peak_rss_kib(const char *field) {
  FILE *f = fopen("/proc/self/status", "r");
  char line[256];
  long value = -1;
  size_t length = strlen(field);
  while (f && fgets(line, sizeof line, f))
    if (!strncmp(line, field, length)) value = atol(line + length);
  if (f) fclose(f);
  return value;
}

static int by_value(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s images.bin expected.bin [passes] [label]\n", argv[0]); return 2; }
  int passes = argc > 3 ? atoi(argv[3]) : 3;
  const char *label = argc > 4 ? argv[4] : "qnn";
  size_t image_size, expected_size;
  uint8_t *images = slurp(argv[1], &image_size);
  uint8_t *expected = slurp(argv[2], &expected_size);
  uint32_t count, expected_count, classes;
  memcpy(&count, images + 4, 4);
  memcpy(&expected_count, expected + 4, 4);
#ifdef FLOAT_MODEL
  if (memcmp(images, "QNNI", 4) || memcmp(expected, "QNNF", 4)) { fprintf(stderr, "bad magic\n"); return 2; }
  classes = (uint32_t)fnn_output_dim();
  const int32_t *host_predictions = (const int32_t *)(expected + 8);
  const int32_t *labels = host_predictions + count;
  float output[64];
#else
  if (memcmp(images, "QNNI", 4) || memcmp(expected, "QNNE", 4)) { fprintf(stderr, "bad magic\n"); return 2; }
  memcpy(&classes, expected + 8, 4);
  const int64_t *logits = (const int64_t *)(expected + 12);
  const int32_t *labels = (const int32_t *)(expected + 12 + (size_t)count * classes * 8);
  int64_t *input = malloc(qnn_input_dim() * sizeof *input);
  int64_t output[64];
#endif
  if (count != expected_count || classes > 64) { fprintf(stderr, "bad header\n"); return 2; }
  uint8_t **offsets = malloc(count * sizeof *offsets);
  uint8_t *p = images + 8;
  for (uint32_t i = 0; i < count; ++i) {
    offsets[i] = p;
    uint16_t h, w;
    memcpy(&h, p, 2);
    memcpy(&w, p + 2, 2);
    p += 4 + (size_t)h * w * 3;
  }
  if ((size_t)(p - images) != image_size) { fprintf(stderr, "bad image stream\n"); return 2; }
  double *times = malloc((size_t)count * passes * sizeof *times);
  long mismatches = 0, correct = 0, encoder_rejects = 0;
  double start = now_us();
  for (int pass = 0; pass < passes; ++pass) {
    for (uint32_t i = 0; i < count; ++i) {
      uint16_t h, w;
      memcpy(&h, offsets[i], 2);
      memcpy(&w, offsets[i] + 2, 2);
      double t0 = now_us();
#ifdef FLOAT_MODEL
      int rc = fnn_forward_bytes(offsets[i] + 4, h, w, output);
#else
      int rc = qnn_encode_bytes(offsets[i] + 4, h, w, input);
      qnn_forward_fixed(input, output);
#endif
      times[(size_t)pass * count + i] = now_us() - t0;
      if (pass) continue;
      encoder_rejects += rc != 0;
      int best = 0;
      for (uint32_t c = 1; c < classes; ++c) if (output[c] > output[best]) best = (int)c;
#ifdef FLOAT_MODEL
      mismatches += best != host_predictions[i];
#else
      mismatches += memcmp(output, logits + (size_t)i * classes, classes * 8) != 0;
#endif
      correct += best == labels[i];
    }
  }
  double total = now_us() - start;
  size_t n = (size_t)count * passes;
  double sum = 0;
  for (size_t i = 0; i < n; ++i) sum += times[i];
  qsort(times, n, sizeof *times, by_value);
#ifdef FLOAT_MODEL
  const char *check = "predicted_class_vs_host_float";
#else
  const char *check = "int64_logits_vs_host_certified";
#endif
  printf("{\"label\":\"%s\",\"check\":\"%s\",\"images\":%u,\"passes\":%d,\"mismatches\":%ld,\"encoder_rejects\":%ld,"
         "\"correct\":%ld,\"accuracy\":%.6f,\"latency_us_mean\":%.2f,\"latency_us_median\":%.2f,"
         "\"latency_us_p95\":%.2f,\"latency_us_min\":%.2f,\"latency_us_max\":%.2f,\"wall_seconds\":%.3f,"
         "\"peak_rss_kib\":%ld,\"image_file_bytes\":%zu}\n",
         label, check, count, passes, mismatches, encoder_rejects, correct, (double)correct / count, sum / n,
         times[n / 2], times[(size_t)(n * 0.95)], times[0], times[n - 1], total / 1e6, peak_rss_kib("VmHWM:"),
         image_size);
  return mismatches || encoder_rejects ? 1 : 0;
}
