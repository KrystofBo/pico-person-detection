---
name: tflite-for-pico
description: Convert a Keras model to an int8 .tflite that actually runs on this project's Pico firmware. Use when training, converting, or changing a model's architecture, before spending time on a training run.
---

# Getting a trained model onto the Pico

The firmware registers **exactly five operators** (`firmware/*/main.cpp`):

```
AVERAGE_POOL_2D   CONV_2D   DEPTHWISE_CONV_2D   RESHAPE   SOFTMAX
```

Anything else fails at `AllocateTensors()` on the device, *after* training. Check op coverage
**before** a training run, not after.

## The recipe that works

Verified by conversion, 2026-09-16:

```python
inp = keras.Input((96, 96, 1), batch_size=1)      # fixed batch - see trap 2
...
x   = keras.layers.AveragePooling2D(pool_size=S)(x)   # S = final spatial size
x   = keras.layers.Conv2D(2, 1, padding="same")(x)    # 1x1 conv as the classifier
x   = keras.layers.Reshape((2,))(x)
out = keras.layers.Softmax()(x)

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_ds              # real images, not random noise
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type  = tf.int8
conv.inference_output_type = tf.int8
```

This mirrors the pretrained baseline's own head, which is why the baseline needs only five ops.

## Two traps, both real

1. **`GlobalAveragePooling2D` + `Dense` emits `MEAN` and `FULLY_CONNECTED`.** Neither is registered.
   Use `AveragePooling2D` over the full spatial extent, and a 1x1 `Conv2D` as the classifier instead.

2. **A dynamic batch dimension emits `SHAPE`, `STRIDED_SLICE` and `PACK`.** With `keras.Input((96,96,1))`
   the converter builds a runtime shape computation to preserve the unknown batch size, and `Reshape`
   drags in three unsupported ops. Passing `batch_size=1` makes the shape static and they disappear.

## Checking coverage

```python
import tflite
mod = tflite.Model.GetRootAsModel(blob, 0); g = mod.Subgraphs(0)
names = {v: k for k, v in vars(tflite.BuiltinOperator).items() if isinstance(v, int)}
codes = [mod.OperatorCodes(i).BuiltinCode() for i in range(mod.OperatorCodesLength())]
ops = {names[codes[g.Operators(i).OpcodeIndex()]] for i in range(g.OperatorsLength())}
print(sorted(ops - {"AVERAGE_POOL_2D","CONV_2D","DEPTHWISE_CONV_2D","RESHAPE","SOFTMAX"}))
```

If that prints anything, either change the architecture or add the kernel to the firmware's
`MicroMutableOpResolver<N>` (and bump `N`).

## Designing for this chip

Measured on the Pico 2 after the step 02 dual-core fix:

| | ns per MAC | relative |
|---|---|---|
| Pointwise 1x1 conv | 10.4 | 1.0x |
| Depthwise conv | 35.3 | **3.4x** |

So **minimising MACs is the wrong objective here**. Depthwise-separable blocks trade pointwise MACs for
depthwise ones, and a depthwise MAC costs 3.4x more on this hardware. Roughly: 2.2 M MACs ~ 30 ms,
0.7 M MACs ~ 10 ms, against the baseline's 7.16 M MACs ~ 99 ms.

Also check `arena_used_bytes()` on device: the firmware allocates 88 KB and the baseline uses 82,308 B.
