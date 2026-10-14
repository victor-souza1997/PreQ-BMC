#!/bin/sh
# Run on the VIM3 from the bundle directory (Linux shell, or AAOS via: adb shell; cd /data/local/tmp/vim3).
# Appends one JSON line per run to results_vim3.jsonl; board facts go to board_vim3.txt.
set -u
out=results_vim3.jsonl
{ uname -a; cat /proc/cpuinfo | grep -m4 -iE "model name|Hardware|CPU part"; getprop ro.build.fingerprint 2>/dev/null
  date; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c; } > board_vim3.txt 2>&1
sha256sum bench_*_aarch64 images_*.bin expected_*.bin >> board_vim3.txt 2>/dev/null
for variant in int16_compact int64_certified float32_baseline; do
  expected=expected; [ "$variant" = float32_baseline ] && expected=expected_float
  chmod +x "bench_${variant}_aarch64"
  # Consistency + accuracy over all 7,830 validation images, then latency over 3 more passes.
  "./bench_${variant}_aarch64" images_all.bin "${expected}_all.bin" 1 "${variant}_consistency" >> "$out"
  "./bench_${variant}_aarch64" images_all.bin "${expected}_all.bin" 3 "${variant}_latency" >> "$out"
  # Resident memory with one image loaded, so the RSS is the network, not the image file.
  "./bench_${variant}_aarch64" images_one.bin "${expected}_one.bin" 1000 "${variant}_rss_one_image" >> "$out"
done
cat "$out"
