"""Shared plumbing for the host-side tools: the model, and turning images into
the int8 tensor the Pico expects.

The model and the two sample images only exist in this repo as C arrays inside
pico-tflmicro, so they are parsed out of those rather than duplicated as binary
files. Parsing is cheap (~300 KB) and keeps a single source of truth.
"""
from __future__ import annotations

import functools
import re
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
MODEL_DIR = (REPO / "third_party/pico-tflmicro/examples/person_detection"
             / "tensorflow/lite/micro/tools/make/downloads/person_model")

INPUT_H = INPUT_W = 96

# From the model's own input tensor: scale 1/127.5, zero point -1. Asserted in
# host_reference so a different model fails loudly instead of being fed garbage.
INPUT_SCALE = 0.007843137718737125
INPUT_ZERO_POINT = -1


def _c_array_bytes(path: Path, symbol: str) -> bytes:
    """Pull the byte list out of `const unsigned char <symbol>[] = { 0x.., ... };`"""
    src = path.read_text()
    start = src.index("{", src.index(symbol))
    end = src.index("};", start)
    # Literals are not zero-padded: the image arrays contain both 0x9 and 0x1f.
    return bytes(int(b, 16) for b in re.findall(r"0x([0-9a-fA-F]{1,2})\b", src[start:end]))


@functools.lru_cache(maxsize=None)
def model_bytes() -> bytes:
    """The person detection .tflite, exactly as it is flashed to the Pico."""
    return _c_array_bytes(MODEL_DIR / "person_detect_model_data.cpp",
                          "g_person_detect_model_data")


@functools.lru_cache(maxsize=None)
def embedded_sample(name: str) -> np.ndarray:
    """One of the two int8 sample images that ship with pico-tflmicro.

    These are known-good ground truth: the Pico scores them 113/-113 and -57/57.
    """
    symbol = {"person": "g_person_image_data",
              "no_person": "g_no_person_image_data"}[name]
    raw = _c_array_bytes(MODEL_DIR / f"{name}_image_data.cpp", symbol)
    if len(raw) != INPUT_H * INPUT_W:
        raise ValueError(f"{name}: parsed {len(raw)} bytes, expected {INPUT_H * INPUT_W}")
    return np.frombuffer(raw, dtype=np.int8).reshape(INPUT_H, INPUT_W, 1)


def preprocess(path: Path) -> np.ndarray:
    """Image file -> (96, 96, 1) int8, the exact bytes the Pico's input tensor wants.

    Centre-crops to a square before resizing, so the aspect ratio is preserved
    rather than squashed.

    The int8 mapping is `pixel - 128`, which is what pico-tflmicro's own
    image_provider does. The model's quantisation (real = (q+1)/127.5) implies
    `pixel - 128.5` instead, but that lands exactly on a rounding tie for every
    integer pixel; the two differ by at most half a quantisation step, and
    matching upstream keeps our bytes identical to the embedded samples'.
    """
    im = Image.open(path).convert("L")
    side = min(im.size)
    left, top = (im.width - side) // 2, (im.height - side) // 2
    im = im.crop((left, top, left + side, top + side))
    im = im.resize((INPUT_W, INPUT_H), Image.BILINEAR)
    pixels = np.asarray(im, dtype=np.int16)
    return (pixels - 128).astype(np.int8).reshape(INPUT_H, INPUT_W, 1)


def model_bytes_for_litert() -> bytearray:
    """The same model, with one piece of metadata corrected so LiteRT will load it.

    The model is TOCO-converted and declares `quantized_dimension = 3` on its 14
    per-channel bias tensors, which are rank 1. TFLite Micro ignores the field
    for biases, so the Pico runs the model as-is; modern LiteRT validates it and
    refuses to load ("quantized_dimension must be in range [0, 1). Was 3").

    Zero is the only in-range value, and it is what the scales already mean: one
    scale per element of the single dimension. So this rewrites the field to 0
    in place. It is an int32 stored inline in the flatbuffer, so the edit is
    size-preserving and touches no weights, scales or zero points - the
    arithmetic is identical, which the sample images confirm by still scoring
    113/-113 and -57/57.
    """
    import struct

    import tflite

    buf = bytearray(model_bytes())
    model = tflite.Model.GetRootAsModel(bytes(buf), 0)
    patched = 0
    for s in range(model.SubgraphsLength()):
        graph = model.Subgraphs(s)
        for i in range(graph.TensorsLength()):
            tensor = graph.Tensors(i)
            quant = tensor.Quantization()
            if quant is None:
                continue
            rank = tensor.ShapeLength()
            if quant.QuantizedDimension() < rank:
                continue
            offset = quant._tab.Offset(16)  # QuantizationParameters.quantized_dimension
            if offset == 0:
                continue  # absent means it already defaults to 0
            struct.pack_into("<i", buf, quant._tab.Pos + offset, 0)
            patched += 1
    if patched != 14:
        raise ValueError(f"expected to patch 14 bias tensors, patched {patched}")
    return buf
