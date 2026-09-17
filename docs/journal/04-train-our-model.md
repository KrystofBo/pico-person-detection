# Step 04 — Train our own model

Experiment report. Written before training; results sections are filled as the step runs.

## Objective

Beat the pretrained baseline's **accuracy** on our own data. Size and latency are budgets to stay
within, not quantities to minimise. Success = accuracy above baseline on a held-out test set, with the
model still running on the Pico 2.

Baseline accuracy is currently **unmeasured**, so establishing it is task 1.

## Baseline architecture

MobileNet v1, width multiplier α=0.25, input 96×96×1 (greyscale), int8, 2 classes.
Source: `third_party/pico-tflmicro/.../person_model/person_detect_model_data.cpp`.

Design structure — a 3×3 stem, 13 depthwise-separable blocks, classifier:

| stage | block | output |
|---|---|---|
| stem | conv 3×3 s2 | 48×48×8 |
| 1 | dw 3×3 s1 + pw 1×1 | 48×48×16 |
| 2 | dw 3×3 s2 + pw 1×1 | 24×24×32 |
| 3 | dw 3×3 s1 + pw 1×1 | 24×24×32 |
| 4 | dw 3×3 s2 + pw 1×1 | 12×12×64 |
| 5 | dw 3×3 s1 + pw 1×1 | 12×12×64 |
| 6 | dw 3×3 s2 + pw 1×1 | 6×6×128 |
| 7-11 | dw 3×3 s1 + pw 1×1, ×5 identical | 6×6×128 |
| 12 | dw 3×3 s2 + pw 1×1 | 3×3×256 |
| 13 | dw 3×3 s1 + pw 1×1 | 3×3×256 |
| head | avgpool 3×3 → conv 1×1 → reshape → softmax | 2 |

Every conv has BatchNorm folded in and ReLU6 fused; there are no separate activation ops.
The stem is stored as a **depthwise** conv with `depth_multiplier=8` — with a 1-channel input that is
arithmetically identical to an 8-filter standard conv, and is how TOCO emitted it.

### As executed, with measured time

31 operators. Times are per-op on the Pico 2 at 150 MHz, dual-core CMSIS-NN, model in SRAM.

| # | op | k | stride | output | MACs | weight B | ms |
|---|---|---|---|---|---|---|---|
| 0 | dwconv | 3x3 | s2 m8 | 48x48x8 | 165,888 | 104 | 7.7 |
| 1 | dwconv | 3x3 | s1 | 48x48x8 | 165,888 | 104 | 6.0 |
| 2 | conv | 1x1 | s1 | 48x48x16 | 294,912 | 192 | 8.1 |
| 3 | dwconv | 3x3 | s2 | 24x24x16 | 82,944 | 208 | 2.8 |
| 4 | conv | 1x1 | s1 | 24x24x32 | 294,912 | 640 | 4.8 |
| 5 | dwconv | 3x3 | s1 | 24x24x32 | 165,888 | 416 | 5.3 |
| 6 | conv | 1x1 | s1 | 24x24x32 | 589,824 | 1,152 | 6.8 |
| 7 | dwconv | 3x3 | s2 | 12x12x32 | 41,472 | 416 | 1.4 |
| 8 | conv | 1x1 | s1 | 12x12x64 | 294,912 | 2,304 | 3.4 |
| 9 | dwconv | 3x3 | s1 | 12x12x64 | 82,944 | 832 | 2.6 |
| 10 | conv | 1x1 | s1 | 12x12x64 | 589,824 | 4,352 | 5.5 |
| 11 | dwconv | 3x3 | s2 | 6x6x64 | 20,736 | 832 | 0.7 |
| 12 | conv | 1x1 | s1 | 6x6x128 | 294,912 | 8,704 | 2.8 |
| 13 | dwconv | 3x3 | s1 | 6x6x128 | 41,472 | 1,664 | 1.2 |
| 14 | conv | 1x1 | s1 | 6x6x128 | 589,824 | 16,896 | 5.0 |
| 15 | dwconv | 3x3 | s1 | 6x6x128 | 41,472 | 1,664 | 1.2 |
| 16 | conv | 1x1 | s1 | 6x6x128 | 589,824 | 16,896 | 5.0 |
| 17 | dwconv | 3x3 | s1 | 6x6x128 | 41,472 | 1,664 | 1.2 |
| 18 | conv | 1x1 | s1 | 6x6x128 | 589,824 | 16,896 | 5.0 |
| 19 | dwconv | 3x3 | s1 | 6x6x128 | 41,472 | 1,664 | 1.2 |
| 20 | conv | 1x1 | s1 | 6x6x128 | 589,824 | 16,896 | 5.0 |
| 21 | dwconv | 3x3 | s1 | 6x6x128 | 41,472 | 1,664 | 1.2 |
| 22 | conv | 1x1 | s1 | 6x6x128 | 589,824 | 16,896 | 5.0 |
| 23 | dwconv | 3x3 | s2 | 3x3x128 | 10,368 | 1,664 | 0.6 |
| 24 | conv | 1x1 | s1 | 3x3x256 | 294,912 | 33,792 | 2.9 |
| 25 | dwconv | 3x3 | s1 | 3x3x256 | 20,736 | 3,328 | 1.0 |
| 26 | conv | 1x1 | s1 | 3x3x256 | 589,824 | 66,560 | 5.4 |
| 27 | avgpool | - | - | 1x1x256 | 0 | 0 | 0.2 |
| 28 | conv | 1x1 | s1 | 1x1x2 | 512 | 520 | 0.0 |
| 29 | reshape | - | - | 2 | 0 | 0 | 0.0 |
| 30 | softmax | - | - | 2 | 0 | 0 | 0.1 |

**Totals:** 7,157,888 MACs · 218,920 B weights (210,708 params) · 300,568 B model file
(81,648 B of that is flatbuffer metadata) · 82,308 B arena · **99.1 ms**.

Quantisation: input scale 1/127.5, zero point −1 (byte = pixel − 128); all hidden activations share
scale 6/255, zero point −128 (ReLU6 range); weights per-channel int8, biases int32; output scale 1/256.

### Cost distribution

| | MACs | share | ms | share | ns/MAC |
|---|---|---|---|---|---|
| conv 1×1 (pointwise) | 6,193,664 | 87% | 64.7 | 65% | 10.4 |
| depthwise 3×3 | 964,224 | 13% | 34.0 | 34% | **35.3** |
| avgpool + reshape + softmax | — | — | 0.3 | <1% | — |

Depthwise MACs cost 3.4× pointwise ones, so total MACs is a poor proxy for latency on this chip.
Blocks 7-11 (five identical 6×6×128) are 44% of all MACs and 31 ms.

**Gap:** no measurement exists for a 3×3 *standard* conv, because this model contains none — the stem
is a depthwise. The 10.4 ns/MAC figure covers 1×1 convs only. Any design that substitutes plain convs
for depthwise-separable blocks needs that measured first.

## Hardware budget

Pico 2: 520 KB SRAM, 4 MB flash, 150 MHz. Current use: 294 KB model copy + 88 KB arena + ~23 KB
SDK/stack = 405 KB SRAM.

The model is copied to SRAM because both cores share one 16 KB XIP cache; reading weights from flash
makes the dual-core split *slower* than single-core on layers whose weights exceed the cache
(see the step 02 fix). So model size is bounded by SRAM, not flash.

Extrapolated cost of scaling width (α² for weights and MACs, ~α for activations). **Estimates, not
measurements:**

| α | weights | model file | MACs | est. ms | arena | fits SRAM? |
|---|---|---|---|---|---|---|
| 0.25 (baseline) | 219 KB | 294 KB | 7.2 M | 99 (measured) | 82 KB | yes, 405 KB used |
| 0.35 | ~429 KB | ~510 KB | ~14 M | ~194 | ~115 KB | **no** (~650 KB) |
| 0.50 | ~876 KB | ~960 KB | ~28.6 M | ~396 | ~165 KB | no |

**Consequence: α=0.25 is already at the practical ceiling for the model-in-SRAM strategy.** Widening
the network is not an available route to higher accuracy without either giving up the SRAM copy (which
costs roughly 2× latency on top of the MAC increase) or raising the latency budget several-fold.

Expected source of accuracy gains is therefore **training on our own data**, not scaling: the baseline
was trained on Visual Wake Words (COCO) and has never seen our distribution.

## Layer constraints

The firmware registers five operators. Verified by converting each pattern to int8 and inspecting the
resulting op set (`tools/check_ops.py`):

| supported, no firmware change | needs a registered op |
|---|---|
| `Conv2D` any kernel/stride/padding | `Add` → `ADD` (**blocks residual connections**) |
| `DepthwiseConv2D`, `SeparableConv2D` | `MaxPooling2D` → `MAX_POOL_2D` |
| `BatchNormalization` (folded into conv) | `Concatenate` → `CONCATENATION` |
| `ReLU`, `ReLU6` (fused into conv) | |
| `ZeroPadding2D` (folded into conv) | |
| `Dropout` (inference no-op) | |
| `AveragePooling2D`, `Reshape`, `Softmax` | |

Avoid `swish` (→ `LOGISTIC` + `MUL`) and `hard_sigmoid` (→ `MUL`).

Two conversion requirements, both verified: use `batch_size=1` on the Input (a dynamic batch emits
`SHAPE`/`STRIDED_SLICE`/`PACK`), and use `AveragePooling2D` + 1×1 `Conv2D` as the classifier head
(`GlobalAveragePooling2D` + `Dense` emit `MEAN` + `FULLY_CONNECTED`). See the `tflite-for-pico` skill.

## Method

```
1. Split the dataset, hold out a test set     -> verify: disjoint, class balance recorded
2. Score the pretrained baseline on it        -> verify: gives the accuracy bar to beat
3. Reproduce MobileNet v1 α=0.25 in Keras,
   train on our data, convert to int8         -> verify: check_ops.py clean; int8 vs float
                                                        agreement within a stated margin
4. Compare against the baseline, same test set-> verify: accuracy above baseline
5. Run on the Pico                            -> verify: device scores match host bit-exactly;
                                                        latency and arena recorded
6. Ablate from there                          -> verify: each variant measured, not estimated
```

Step 5 needs a `.tflite` → C array converter and a profiling firmware target, so any candidate can be
timed per-op on device. Neither exists yet.

## Open

- Dataset not yet supplied: path, layout, label form (presence flags or boxes) unknown.
- Latency budget undefined. 99 ms is the current cost; no target frame rate has been set.
- Preprocessing: centre-crop beat squash and letterbox on the 10-image sample set (10/10 vs 9/10 vs
  7/10), but n=10 is not a result. Train and deploy with whichever is chosen, so the two match.

## Results

Not yet run.
