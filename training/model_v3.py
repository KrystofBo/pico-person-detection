"""MobileNetV3-Small for the Pico, at 96x96x1.

Follows the MobileNetV3-Small table from the paper, with the spatial progression
retargeted from 224 to 96 (96 -> 48 -> 24 -> 12 -> 6 -> 3) and a classifier head
built from ops the firmware registers.

Two departures from a textbook implementation, both forced by the device:

  * Squeeze-excite uses AveragePooling2D + two 1x1 convolutions rather than
    GlobalAveragePooling2D + Dense. Arithmetically the same, but the Dense form
    emits MEAN and FULLY_CONNECTED, which the firmware does not register.
  * The stem is a depthwise convolution with depth_multiplier rather than a
    3x3 Conv2D. With one input channel these are identical, but CONV_2D 3x3
    measured 104.6 ns/MAC on device against DEPTHWISE_CONV_2D's 46.3.

Needs ADD, MUL and HARD_SWISH registered in the firmware (they are).

alpha defaults to 0.35, not 0.5: with the paper's 1024-wide classifier bottleneck
included, 0.5 comes to 411k parameters and 546 KB of int8, which exceeds the
~400 KB the device can hold in SRAM. At 0.35 the model is 205,730 parameters,
within 6% of MobileNet v1 alpha=0.25's 218,914, which makes the comparison a
fair one on capacity.
"""
from __future__ import annotations

from tensorflow import keras

INPUT_SIZE = 96


def _divisible(v: float, divisor: int = 8) -> int:
    """Channel counts rounded to a multiple of 8, as MobileNet does."""
    new = max(divisor, int(v + divisor / 2) // divisor * divisor)
    return int(new + divisor) if new < 0.9 * v else int(new)


def _act(x, kind: str, name: str):
    return keras.layers.Activation("hard_silu" if kind == "HS" else "relu6", name=name)(x)


def _squeeze_excite(x, channels: int, name: str):
    s = keras.layers.AveragePooling2D(pool_size=x.shape[1], name=f"{name}_pool")(x)
    s = keras.layers.Conv2D(_divisible(channels / 4), 1, activation="relu",
                            name=f"{name}_reduce")(s)
    s = keras.layers.Conv2D(channels, 1, activation="hard_sigmoid", name=f"{name}_expand")(s)
    return keras.layers.Multiply(name=f"{name}_scale")([x, s])


def _bneck(x, kernel: int, exp: int, out: int, se: bool, act: str, stride: int,
           alpha: float, name: str):
    exp_ch, out_ch = _divisible(exp * alpha), _divisible(out * alpha)
    shortcut, in_ch = x, x.shape[-1]

    if exp_ch != in_ch:                                   # expansion, skipped when 1x
        x = keras.layers.Conv2D(exp_ch, 1, padding="same", use_bias=False, name=f"{name}_exp")(x)
        x = keras.layers.BatchNormalization(name=f"{name}_exp_bn")(x)
        x = _act(x, act, f"{name}_exp_act")

    x = keras.layers.DepthwiseConv2D(kernel, strides=stride, padding="same", use_bias=False,
                                     name=f"{name}_dw")(x)
    x = keras.layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = _act(x, act, f"{name}_dw_act")

    if se:
        x = _squeeze_excite(x, x.shape[-1], f"{name}_se")

    x = keras.layers.Conv2D(out_ch, 1, padding="same", use_bias=False, name=f"{name}_proj")(x)
    x = keras.layers.BatchNormalization(name=f"{name}_proj_bn")(x)

    if stride == 1 and in_ch == out_ch:
        x = keras.layers.Add(name=f"{name}_residual")([shortcut, x])
    return x


# (kernel, expansion, out channels, squeeze-excite, nonlinearity, stride),
# MobileNetV3-Small with strides retargeted for a 96x96 input.
BNECKS = [
    (3, 16, 16, True, "RE", 2),
    (3, 72, 24, False, "RE", 2),
    (3, 88, 24, False, "RE", 1),
    (5, 96, 40, True, "HS", 2),
    (5, 240, 40, True, "HS", 1),
    (5, 240, 40, True, "HS", 1),
    (5, 120, 48, True, "HS", 1),
    (5, 144, 48, True, "HS", 1),
    (5, 288, 96, True, "HS", 2),
    (5, 576, 96, True, "HS", 1),
    (5, 576, 96, True, "HS", 1),
]
STEM_FILTERS, LAST_CONV, LAST_POINT = 16, 576, 1024


def _build(alpha: float, batch_size: int | None) -> keras.Model:
    inp = keras.Input((INPUT_SIZE, INPUT_SIZE, 1), batch_size=batch_size, name="image")
    stem = _divisible(STEM_FILTERS * alpha)
    x = keras.layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False,
                                     depth_multiplier=stem, name="stem")(inp)
    x = keras.layers.BatchNormalization(name="stem_bn")(x)
    x = _act(x, "HS", "stem_act")

    for i, (kernel, exp, out, se, act, stride) in enumerate(BNECKS):
        x = _bneck(x, kernel, exp, out, se, act, stride, alpha, f"bneck{i}")

    x = keras.layers.Conv2D(_divisible(LAST_CONV * alpha), 1, padding="same", use_bias=False,
                            name="head_conv")(x)
    x = keras.layers.BatchNormalization(name="head_bn")(x)
    x = _act(x, "HS", "head_act")

    x = keras.layers.AveragePooling2D(pool_size=x.shape[1], name="pool")(x)
    x = keras.layers.Conv2D(_divisible(LAST_POINT * alpha), 1, padding="same", name="head_point")(x)
    x = _act(x, "HS", "head_point_act")
    x = keras.layers.Conv2D(2, 1, padding="same", name="logits")(x)
    x = keras.layers.Reshape((2,), name="squeeze")(x)
    out = keras.layers.Softmax(name="probs")(x)
    return keras.Model(inp, out, name=f"mobilenetv3small_a{alpha:g}")


def build(alpha: float = 0.35) -> keras.Model:
    return _build(alpha, batch_size=None)


def build_for_export(alpha: float = 0.35, weights_from: keras.Model | None = None):
    model = _build(alpha, batch_size=1)
    if weights_from is not None:
        model.set_weights(weights_from.get_weights())
    return model
