"""MCUNet's searched architecture, at 96x96x1.

The block table below is the real configuration of `mcunet-5fps_vww`, fetched from
https://hanlab18.mit.edu/projects/tinyml/mcunet/release/mcunet-5fps_vww.json
(MIT HAN Lab, MCUNet, NeurIPS 2020). Kernel sizes and expansion ratios are what
TinyNAS searched; they are not a guess.

**This is MCUNet's architecture, not MCUNet.** MCUNet is a system: TinyNAS, which
searches architecture and input resolution jointly against a memory budget, plus
TinyEngine, a runtime that a large part of the published speedup comes from. We
run neither - TinyNAS needs ImageNet-scale supernet training, and we are on
TFLite Micro with CMSIS-NN. Their published numbers are not comparable to ours.

One deliberate deviation: the searched resolution is 80, and this builds at 96 so
the comparison against MobileNet v1 and v3 holds input constant and varies only
architecture. Resolution is part of what TinyNAS co-designed, so this is not the
design point they found; it costs (96/80)^2 = 1.44x the intended MACs.

What makes it promising here, unlike v3: relu6 throughout and no squeeze-excite,
so the only operator beyond the base five is ADD for the residuals - measured at
2% of inference time, against h-swish's 36%.

alpha defaults to 0.6, the widest that fits. Not capacity parity - 0.7 matched v1
to 0.5% - because alpha cannot buy what actually constrains this model.

Peak activation memory is 221,184 bytes at every alpha from 0.4 to 0.7, flat,
because the largest tensor is block 1's expansion: the stem width floors at 8
channels and block 1 expands it 6x at 48x48, and neither term scales with alpha.
Measured arena on device is ~180 KB against MobileNet v1's 82,308. So alpha only
shrinks the weights, and they have to shrink enough that weights plus a fixed
~180 KB arena fit in 520 KB of SRAM.

At 0.6 the trained model is 301,960 bytes and runs in SRAM with 32 KB to spare;
0.7 is 356,288 and does not fit. The cost is capacity: 167,098 weights against
v1's 207,968, so this is not a like-for-like comparison.

An untrained 0.6 model hard-faulted the board twice. That was not memory - more
margin changed nothing - but quantisation scales near 1e-12 from random weights;
tools/check_scales.py now refuses such a model before it is flashed.
"""
from __future__ import annotations

from tensorflow import keras

INPUT_SIZE = 96
STEM_FILTERS = 16

# (kernel, expand_ratio, out_channels, stride), from mcunet-5fps_vww.
BLOCKS = [
    (3, 1, 8, 1), (3, 6, 16, 2), (3, 3, 16, 1), (3, 3, 16, 1),
    (7, 3, 24, 2), (3, 6, 24, 1), (5, 5, 24, 1),
    (7, 6, 40, 2), (7, 6, 40, 1), (3, 6, 48, 1), (3, 4, 48, 1),
    (5, 5, 96, 2), (3, 5, 96, 1), (3, 4, 96, 1), (7, 3, 160, 1),
]


def _width(channels: int, alpha: float) -> int:
    return max(8, int(round(channels * alpha / 8)) * 8)


def _inverted_residual(x, kernel: int, expand: int, filters: int, stride: int,
                       alpha: float, index: int):
    in_ch = x.shape[-1]
    out_ch = _width(filters, alpha)
    shortcut = x

    if expand != 1:
        x = keras.layers.Conv2D(in_ch * expand, 1, padding="same", use_bias=False,
                                name=f"b{index}_expand")(x)
        x = keras.layers.BatchNormalization(name=f"b{index}_expand_bn")(x)
        x = keras.layers.ReLU(6.0, name=f"b{index}_expand_relu")(x)

    x = keras.layers.DepthwiseConv2D(kernel, strides=stride, padding="same", use_bias=False,
                                     name=f"b{index}_dw")(x)
    x = keras.layers.BatchNormalization(name=f"b{index}_dw_bn")(x)
    x = keras.layers.ReLU(6.0, name=f"b{index}_dw_relu")(x)

    # Linear bottleneck: no activation after the projection, per MobileNetV2.
    x = keras.layers.Conv2D(out_ch, 1, padding="same", use_bias=False, name=f"b{index}_project")(x)
    x = keras.layers.BatchNormalization(name=f"b{index}_project_bn")(x)

    if stride == 1 and in_ch == out_ch:
        x = keras.layers.Add(name=f"b{index}_residual")([shortcut, x])
    return x


def _build(alpha: float, batch_size: int | None) -> keras.Model:
    inp = keras.Input((INPUT_SIZE, INPUT_SIZE, 1), batch_size=batch_size, name="image")
    # Depthwise with depth_multiplier, not Conv2D: identical with one input
    # channel, and 46.3 ns/MAC against 104.6 on device.
    x = keras.layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False,
                                     depth_multiplier=_width(STEM_FILTERS, alpha), name="stem")(inp)
    x = keras.layers.BatchNormalization(name="stem_bn")(x)
    x = keras.layers.ReLU(6.0, name="stem_relu")(x)

    for i, (kernel, expand, filters, stride) in enumerate(BLOCKS):
        x = _inverted_residual(x, kernel, expand, filters, stride, alpha, i)

    x = keras.layers.AveragePooling2D(pool_size=x.shape[1], name="pool")(x)
    x = keras.layers.Conv2D(2, 1, padding="same", name="logits")(x)
    x = keras.layers.Reshape((2,), name="squeeze")(x)
    out = keras.layers.Softmax(name="probs")(x)
    return keras.Model(inp, out, name=f"mcunet_vww_a{alpha:g}")


def build(alpha: float = 0.6) -> keras.Model:
    return _build(alpha, batch_size=None)


def build_for_export(alpha: float = 0.6, weights_from: keras.Model | None = None):
    model = _build(alpha, batch_size=1)
    if weights_from is not None:
        model.set_weights(weights_from.get_weights())
    return model
