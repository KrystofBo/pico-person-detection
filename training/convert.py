"""Convert a trained model to int8 TFLite and evaluate it against the baseline.

    python training/convert.py --run training/runs/run0-noaug-20260917-171354

Steps, each checked:
  1. copy weights into a batch_size=1 copy   (a dynamic batch emits SHAPE/STRIDED_SLICE/PACK)
  2. convert with post-training int8 quantisation using real training images
  3. assert the op set is within the five the firmware registers
  4. assert the input quantisation is scale 1/128, zero point 0, so the bytes the
     host already sends (pixel - 128) are exactly what this model expects
  5. evaluate float and int8 on the test split, and report against the baseline

LiteRT is pinned to BUILTIN_REF: its default XNNPACK delegate disagrees with the
device by up to 3 int8 units (see docs/journal/03-host-reference-streaming.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from tensorflow import keras

import data as D
import model as M

import sys
sys.path.insert(0, str(D.REPO / "tools"))
from check_ops import PICO_OPS, model_ops  # noqa: E402

BASELINE = {"accuracy": 0.760, "precision": 0.797, "recall": 0.698, "f1": 0.745}


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": (tp + tn) / len(y), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def evaluate_tflite(blob: bytes, images: np.ndarray, labels: np.ndarray) -> dict:
    interp = Interpreter(model_content=blob, experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    pred = np.empty(len(labels), dtype=np.int8)
    for i in range(len(images)):
        interp.set_tensor(inp["index"],
                          (images[i].astype(np.int16) - 128).astype(np.int8).reshape(inp["shape"]))
        interp.invoke()
        scores = interp.get_tensor(out["index"])[0]
        pred[i] = int(scores[1] > scores[0])
        if (i + 1) % 3000 == 0:
            print(f"  int8 eval {i + 1}/{len(images)}", flush=True)
    return metrics(labels, pred)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="training/runs/<name> directory")
    ap.add_argument("--alpha", type=float, default=None, help="default: read from config.json")
    args = ap.parse_args()

    config = json.loads((args.run / "config.json").read_text())
    alpha = args.alpha if args.alpha is not None else config["alpha"]

    # compile=False: the checkpoint references SparsePrecision/SparseRecall, which are
    # not registered for serialisation, and loading them would fail. Conversion and
    # evaluation need only the architecture and weights.
    trained = keras.models.load_model(args.run / "best.keras", compile=False)
    export = M.build_for_export(alpha=alpha, weights_from=trained)

    converter = tf.lite.TFLiteConverter.from_keras_model(export)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: D.representative_dataset(500)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    blob = converter.convert()

    tflite_path = args.run / "model_int8.tflite"
    tflite_path.write_bytes(blob)

    unsupported = sorted(set(model_ops(blob)) - PICO_OPS)
    if unsupported:
        raise SystemExit(f"model uses ops the firmware does not register: {unsupported}")
    print(f"ops: {', '.join(model_ops(blob))}  (all registered)")

    interp = Interpreter(model_content=blob, experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    interp.allocate_tensors()
    scale, zero = interp.get_input_details()[0]["quantization"]
    if not (abs(scale - 1 / 128) < 1e-6 and zero == 0):
        raise SystemExit(
            f"input quantisation is scale={scale}, zero_point={zero}, expected 1/128 and 0.\n"
            "The host sends pixel-128 bytes; with different quantisation the device would be "
            "fed wrong values. Fix the training input scaling in training/data.py.")
    print(f"input quantisation scale={scale:.8f} zero_point={zero}  (int8 = pixel - 128)")

    images, labels = D.load_split("test")
    float_pred = np.argmax(trained.predict(D.scale(images)[..., None], batch_size=256, verbose=0), axis=1)
    float_metrics = metrics(labels, float_pred)
    int8_metrics = evaluate_tflite(blob, images, labels)

    print(f"\n{'':<12}{'accuracy':>10}{'precision':>11}{'recall':>9}{'F1':>8}")
    for name, m in (("baseline", BASELINE), ("ours float", float_metrics), ("ours int8", int8_metrics)):
        print(f"{name:<12}{m['accuracy']:>9.1%}{m['precision']:>11.1%}{m['recall']:>9.1%}{m['f1']:>8.1%}")
    print(f"\nquantisation cost: {float_metrics['accuracy'] - int8_metrics['accuracy']:+.1%} accuracy")
    print(f"vs baseline:       {int8_metrics['accuracy'] - BASELINE['accuracy']:+.1%} accuracy, "
          f"{int8_metrics['recall'] - BASELINE['recall']:+.1%} recall")
    print(f"model: {len(blob):,} bytes -> {tflite_path}")

    (args.run / "eval.json").write_text(json.dumps(
        {"float": float_metrics, "int8": int8_metrics, "baseline": BASELINE,
         "model_bytes": len(blob)}, indent=2))


if __name__ == "__main__":
    main()
