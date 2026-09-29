"""Run the person detection model on the host, as a reference for the Pico.

    python tools/host_reference.py --samples            # the two embedded arrays
    python tools/host_reference.py data/samples/*.jpg   # image files

Same .tflite the Pico runs, so the int8 scores should match it exactly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

import model_io

# Output indices, matching model_settings.h in pico-tflmicro.
NOT_A_PERSON, PERSON = 0, 1


def make_interpreter(model_path: Path | None = None) -> Interpreter:
    """Reference interpreter. Defaults to the pretrained baseline; pass a path
    to check one of our own trained .tflite models instead.

    BUILTIN_REF, not the default: LiteRT otherwise applies the XNNPACK
    delegate, whose int8 path disagrees with the Pico. On the no_person
    sample XNNPACK scores -60/60 where the device (and the reference kernels)
    score -57/57. A reference that silently differs from the device is worse
    than no reference, so this pins the canonical integer arithmetic.
    """
    content = (Path(model_path).read_bytes() if model_path
               else bytes(model_io.model_bytes_for_litert()))
    interp = Interpreter(model_content=content,
                         experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    interp.allocate_tensors()
    scale, zero = interp.get_input_details()[0]["quantization"]
    # Two mappings are valid, and both mean the host should send `pixel - 128`:
    #   baseline      scale 1/127.5, zero point -1   (TOCO-converted)
    #   ours          scale 1/128,   zero point  0   (see training/data.py)
    # They differ by half a quantisation step. Anything else means the bytes
    # model_io.preprocess produces would be wrong for this model.
    known = (np.isclose(scale, model_io.INPUT_SCALE) and zero == model_io.INPUT_ZERO_POINT) \
        or (np.isclose(scale, 1 / 128) and zero == 0)
    if not known:
        raise SystemExit(
            f"input quantisation is scale={scale}, zero_point={zero}, which is neither "
            f"the baseline's ({model_io.INPUT_SCALE}, {model_io.INPUT_ZERO_POINT}) nor "
            f"(1/128, 0). tools/model_io.py sends pixel-128 bytes, which this model "
            "would interpret differently.")
    return interp


def infer(interp: Interpreter, tensor: np.ndarray) -> tuple[int, int, float]:
    """-> (person int8, no_person int8, P(person))"""
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    interp.set_tensor(inp["index"], tensor.reshape(inp["shape"]))
    interp.invoke()
    scores = interp.get_tensor(out["index"])[0]
    scale, zero = out["quantization"]
    return int(scores[PERSON]), int(scores[NOT_A_PERSON]), (int(scores[PERSON]) - zero) * scale


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", nargs="*", type=Path)
    ap.add_argument("--samples", action="store_true",
                    help="run the two int8 sample arrays embedded in pico-tflmicro")
    ap.add_argument("--model", type=Path, help="a .tflite to use instead of the baseline")
    args = ap.parse_args()
    if not args.images and not args.samples:
        ap.error("give image paths, or --samples")

    interp = make_interpreter(args.model)
    print(f"{'input':<22}{'person':>8}{'no_person':>11}{'P(person)':>11}")
    if args.samples:
        for name in ("person", "no_person"):
            p, n, prob = infer(interp, model_io.embedded_sample(name))
            print(f"{'<embedded ' + name + '>':<22}{p:>8}{n:>11}{prob:>11.3f}")
    for path in args.images:
        p, n, prob = infer(interp, model_io.preprocess(path))
        print(f"{path.name:<22}{p:>8}{n:>11}{prob:>11.3f}")


if __name__ == "__main__":
    main()
