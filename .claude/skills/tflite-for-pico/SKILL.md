---
name: tflite-for-pico
description: Converts a Keras model to an int8 .tflite that runs on this project's Pico firmware, which registers only five operators. Use before any training run, and when changing a model's architecture, head, or conversion settings.
---

# Converting a model for the Pico

The firmware registers exactly five operators:

```
AVERAGE_POOL_2D  CONV_2D  DEPTHWISE_CONV_2D  RESHAPE  SOFTMAX
```

Anything else fails at `AllocateTensors()` on device, after training. Check coverage before training.

## Recipe

```python
inp = keras.Input((96, 96, 1), batch_size=1)          # fixed batch: see trap 2
...
x   = keras.layers.AveragePooling2D(pool_size=S)(x)   # S = final spatial size
x   = keras.layers.Conv2D(2, 1, padding="same")(x)    # 1x1 conv, not Dense
x   = keras.layers.Reshape((2,))(x)
out = keras.layers.Softmax()(x)

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_ds                  # real images, not noise
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type  = tf.int8
conv.inference_output_type = tf.int8
```

## Traps

1. `GlobalAveragePooling2D` + `Dense` emit `MEAN` and `FULLY_CONNECTED`. Use the head above instead.
2. An unspecified batch size emits `SHAPE`, `STRIDED_SLICE`, `PACK`: the converter builds a runtime
   shape computation and `Reshape` pulls them in. `batch_size=1` makes it static.

## Verify

```bash
python tools/check_ops.py model.tflite     # prints unsupported ops, exit 1 if any
```

Non-empty output means either change the architecture, or register the kernel in the firmware's
`MicroMutableOpResolver<N>` and bump `N`.

## Cost model

Measured on device after the step 02 dual-core fix:

| | ns/MAC |
|---|---|
| Pointwise 1x1 conv | 10.4 |
| Depthwise conv | 35.3 |

Depthwise MACs cost 3.4x pointwise ones, so minimising total MACs is the wrong objective here.
Baseline for comparison: 7.16 M MACs, 99 ms, 300,568 B, 82,308 B arena (firmware allocates 88 KB).
