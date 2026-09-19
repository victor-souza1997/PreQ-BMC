"""Named, uncertified TFLite float32 and post-training int8 baselines.

Uses the frozen shared convolution parameters, not the dense integer lowering.
Calibration is train-only. No retraining or Android measurements are performed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from backends.image_encoder import ByteImageEncoder
from datasets.gtsrb_study import load_crop, sha256
from models.restricted_conv import RestrictedCNN
from scripts.prepare_ssv_gtsrb import validate_config
from scripts.run_ssv_cnn_gate import write_new_json


def calibration_rows(records, count, seed=2026):
    if type(count) is not int or count <= 0:
        raise ValueError("Calibration count must be positive")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate image identity across dataset splits")
    train = sorted((r for r in records if r["split"] == "train"), key=lambda r: r["id"])
    if len(train) < count:
        raise ValueError("Not enough training images for the requested calibration set")
    indices = np.random.default_rng(seed).choice(len(train), size=count, replace=False)
    return [train[int(i)] for i in sorted(indices)]


def quantize_input(x, detail):
    scale, zero = detail["quantization"]
    if detail["dtype"] != np.int8 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Expected a positive per-tensor int8 input scale")
    # This is a declared baseline adapter, not the certified QIF encoder.
    return np.clip(np.rint(np.asarray(x, dtype=np.float64) / scale) + zero, -128, 127).astype(np.int8)


def export(study, output, *, calibration_count=500, evaluate_test=False):
    import tensorflow as tf
    geom = validate_config(study["config"])
    if sha256(study["model_path"]) != study["model_sha256"]:
        raise ValueError("Source model identity mismatch")
    chosen = calibration_rows(study["records"], calibration_count)
    with np.load(study["model_path"], allow_pickle=False) as weights:
        cnn = RestrictedCNN(geom, **dict(weights),
                            max_affine_entries=study["config"]["model"]["max_affine_entries"])
    enc = ByteImageEncoder(geom.input_shape)

    def normalized(row):
        return enc.resize(load_crop(study["dataset_root"], row)).astype(np.float32)[None, ...] / 256

    # Constants avoid retraining or a second mutable Keras weight identity.
    class Source(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([1, *geom.input_shape], tf.float32)])
        def __call__(self, x):
            h = tf.nn.conv2d(x, tf.constant(cnn.conv_kernel), strides=[1, *geom.strides, 1], padding=geom.padding)
            h = tf.nn.relu(tf.nn.bias_add(h, tf.constant(cnn.conv_bias)))
            return tf.matmul(tf.reshape(h, [1, -1]), tf.constant(cnn.dense_kernel)) + tf.constant(cnn.dense_bias)

    source = Source()
    concrete = source.__call__.get_concrete_function()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    variants = []
    for name in ("tflite_float32", "tflite_ptq_int8"):
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], source)
        if name == "tflite_ptq_int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = lambda: ([normalized(r)] for r in chosen)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        path = output / f"{name}.tflite"
        path.write_bytes(converter.convert())
        interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1,
                                         experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
        interpreter.allocate_tensors()
        inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        if name == "tflite_ptq_int8" and (inp["dtype"] != np.int8 or out["dtype"] != np.int8
                or any(np.issubdtype(t["dtype"], np.floating) for t in interpreter.get_tensor_details())):
            raise ValueError("PTQ baseline contains floating-point tensors")
        n_test = correct = 0
        if evaluate_test:
            with (output / f"{name}_device_vectors.jsonl").open("x") as vectors:
                for row in study["records"]:
                    if row["split"] != "test":
                        continue
                    x = normalized(row)
                    value = quantize_input(x, inp) if name == "tflite_ptq_int8" else x
                    interpreter.set_tensor(inp["index"], value)
                    interpreter.invoke()
                    logits = interpreter.get_tensor(out["index"])[0]
                    correct += int(np.argmax(logits) == row["class_id"])
                    n_test += 1
                    vectors.write(json.dumps({"image_id": row["id"], "image_sha256": row["sha256"],
                                              "input": value.tolist(), "output": logits.tolist(),
                                              "prediction": int(np.argmax(logits))}) + "\n")
        variants.append({"method": name, "artifact": path.name, "artifact_sha256": sha256(path),
                         "artifact_bytes": path.stat().st_size, "certification_status": "UNCERTIFIED_BASELINE",
                         "input_dtype": np.dtype(inp["dtype"]).name, "input_quantization": list(inp["quantization"]),
                         "output_dtype": np.dtype(out["dtype"]).name, "output_quantization": list(out["quantization"]),
                         "n_test_images": n_test, "host_test_accuracy": correct / n_test if n_test else None})
    report = {"schema": "ssv_tflite_baselines_v1", "source_model_sha256": study["model_sha256"],
              "tensorflow_version": tf.__version__, "calibration_seed": 2026,
              "calibration_images": [{"id": r["id"], "sha256": r["sha256"]} for r in chosen],
              "calibration_split": "train", "methods": variants,
              "preprocessing": "nearest_floor_top_left RGB /256; int8 adapter uses nearest-even and clipping",
              "host_runtime": "TFLite reference kernels, one thread", "android_status": "NOT_MEASURED",
              "latency": "NOT_MEASURED", "energy": "NOT_MEASURED",
              "scope": "Same shared source coefficients; different uncertified arithmetic and native convolution"}
    write_new_json(output / "baseline_summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-count", type=int, default=500)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    export(json.loads(args.study.read_text()), args.output,
           calibration_count=args.calibration_count, evaluate_test=args.evaluate_test)


if __name__ == "__main__":
    main()
