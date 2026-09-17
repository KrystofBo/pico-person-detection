"""MobileNet v1 for the Pico, at 96x96x1.

Reproduces the topology of the pretrained baseline documented in
docs/journal/04-train-our-model.md: a 3x3 stem then 13 depthwise-separable
blocks, with a classifier head built from ops the firmware registers.

Two builders, deliberately:

  build()            flexible batch, for training
  build_for_export() batch_size=1, for int8 conversion

They must stay identical in everything but batch size. A fixed batch of 1 cannot
be trained on, and a dynamic batch makes the converter emit SHAPE, STRIDED_SLICE
and PACK, none of which the firmware registers. Train with the first, copy the
weights into the second, convert that. See the `tflite-for-pico` skill.
"""
from __future__ import annotations

from tensorflow import keras

INPUT_SIZE = 96

# (output channels at alpha=1.0, stride) for each depthwise-separable block,
# i.e. standard MobileNet v1. At alpha=0.25 this gives the baseline's
# 8 -> 16 -> 32 -> 64 -> 128 -> 256 progression.
BLOCKS = [
    (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2),
    (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),
    (1024, 2), (1024, 1),
]
STEM_FILTERS = 32


def _width(channels: int, alpha: float) -> int:
    return max(8, int(channels * alpha))


def _separable(x, filters: int, stride: int, alpha: float, index: int):
    """Depthwise 3x3 then pointwise 1x1, each with BatchNorm folded in and ReLU6 fused."""
    x = keras.layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False,
                                     name=f"block{index}_dw")(x)
    x = keras.layers.BatchNormalization(name=f"block{index}_dw_bn")(x)
    x = keras.layers.ReLU(6.0, name=f"block{index}_dw_relu")(x)
    x = keras.layers.Conv2D(_width(filters, alpha), 1, padding="same", use_bias=False,
                            name=f"block{index}_pw")(x)
    x = keras.layers.BatchNormalization(name=f"block{index}_pw_bn")(x)
    return keras.layers.ReLU(6.0, name=f"block{index}_pw_relu")(x)


def _build(alpha: float, blocks, batch_size: int | None) -> keras.Model:
    inp = keras.Input((INPUT_SIZE, INPUT_SIZE, 1), batch_size=batch_size, name="image")
    x = keras.layers.Conv2D(_width(STEM_FILTERS, alpha), 3, strides=2, padding="same",
                            use_bias=False, name="stem")(inp)
    x = keras.layers.BatchNormalization(name="stem_bn")(x)
    x = keras.layers.ReLU(6.0, name="stem_relu")(x)

    for i, (filters, stride) in enumerate(blocks):
        x = _separable(x, filters, stride, alpha, i)

    # Classifier built from registered ops only: GlobalAveragePooling2D would
    # emit MEAN and Dense would emit FULLY_CONNECTED.
    x = keras.layers.AveragePooling2D(pool_size=x.shape[1], name="pool")(x)
    x = keras.layers.Conv2D(2, 1, padding="same", name="logits")(x)
    x = keras.layers.Reshape((2,), name="squeeze")(x)
    out = keras.layers.Softmax(name="probs")(x)
    return keras.Model(inp, out, name=f"mobilenetv1_a{alpha:g}")


def build(alpha: float = 0.25, blocks=None) -> keras.Model:
    """Trainable model, batch size left free."""
    return _build(alpha, blocks or BLOCKS, batch_size=None)


def build_for_export(alpha: float = 0.25, blocks=None, weights_from: keras.Model | None = None):
    """Conversion-ready copy with batch_size=1, optionally taking trained weights."""
    model = _build(alpha, blocks or BLOCKS, batch_size=1)
    if weights_from is not None:
        model.set_weights(weights_from.get_weights())
    return model
