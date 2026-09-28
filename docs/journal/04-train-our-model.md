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

## Dataset

Wake Vision (`Harvard-Edge/Wake-Vision`), streamed and preprocessed by `tools/build_wake_vision.py`.
The full dataset is 365 GB against ~39 GB free, so nothing is stored but the 96x96 mono result.

| split | source | images | balance |
|---|---|---|---|
| train | `train_quality` | 40,000 | 20,000 / 20,000 |
| val | official `validation` | 3,000 | 1,500 / 1,500 |
| test | official `test` | 9,000 | 4,500 / 4,500 |

Official validation and test splits are used as-is rather than carved from train, so results stay
comparable to the published benchmark. 387 MB on disk, gitignored.

Kept only images with aspect ratio within 25% of 1:1, then centre-cropped to square and resized to
96x96 greyscale - the same preprocessing as inference, so train and deploy match.

**Filtering to exactly 1:1 was not viable.** Measured yield on `train_quality` is 4.6%, which would
need 2.17 M rows from a 1.20 M-row split. Separately, all 104 square images sampled across two splits
were the same 447x447 size, so square images here are a systematically processed subset, not a
representative slice. The 0.25 tolerance caps crop loss at 20% and admits the 4:3 and 3:2 aspects a
camera actually produces.

Build cost: 268,552 rows streamed for the 40,000 training images (14.9% yield) in 147 min. Yield falls
towards the end of a run because the commoner class fills first and its rows are then discarded.

### Integrity

| check | result |
|---|---|
| counts and balance | exact, all three splits |
| train vs val, train vs test | **0 shared images** |
| val vs test | 1 shared image (0.01%) |
| internal duplicates | 0 train, 0 val, 1 test |

Verified by sha1 over raw pixels. Train is clean against both evaluation sets, so evaluation is
unbiased.

## Baseline measurement

The bar to beat, measured with `tools/host_reference.py` (bit-exact against the device).

| | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| pretrained baseline, test (n=9,000) | **76.0%** | 79.7% | 69.8% | 74.5% |
| pretrained baseline, validation (n=3,000) | 75.5% | 78.6% | 70.1% | 74.1% |
| trivial mean-brightness threshold | 55.8% | - | - | - |

Confusion on test: tp 3,143 · tn 3,700 · fp 800 · **fn 1,357**.

Two observations:

- **Recall is 10 points below precision.** The model misses 30% of people. Whatever we train should
  close that gap; it is the more useful direction for a wake-word-style trigger.
- **76% is a soft bar.** The pretrained model was trained on Visual Wake Words (COCO) and is being
  evaluated on Wake Vision (Open Images), which uses a broader person definition - body parts count.
  Beating it by training on the target distribution is expected rather than impressive. The
  informative comparison will be against the published Wake Vision results, not against this number.

The trivial floor of 55.8% is tuned on the training data itself, so it is optimistic; the ~11-level
difference in mean brightness between classes is not an exploitable shortcut.

## Training pipeline

```
training/model.py    MobileNet v1 builder, alpha and block list as parameters
training/data.py     npz -> tf.data, scaling, augmentation, calibration samples
training/train.py    training loop, metrics, TensorBoard
training/convert.py  trained Keras -> int8 .tflite -> evaluation vs baseline
training/logs/       TensorBoard runs (gitignored)
training/runs/       checkpoints, exported .tflite, eval.json (gitignored)
```

```bash
python training/train.py --name run0-noaug
tensorboard --logdir training/logs
python training/convert.py --run training/runs/run0-noaug-<stamp>
```

Architecture reproduces the baseline exactly: **conv weights 207,968 against the baseline's 207,968**,
and 2,738 bias slots against 2,738. The Keras parameter count is higher (218,914) only because
BatchNorm parameters exist before conversion folds them into the convs.

### Monitoring

TensorBoard per run under `training/logs/<name>-<timestamp>`, so runs overlay for comparison.
Scalars: loss, accuracy, precision, recall, F1, learning rate. Histograms: weight distributions.
Images: a grid of validation predictions each epoch, wrong ones outlined in red - the place where
label noise from crop damage will show itself. Plus the model graph.

### Two things that are checked rather than assumed

**Input quantisation.** Training scales inputs to `(pixel - 128) / 128`, which makes post-training
quantisation choose scale 1/128 and zero point 0 - so the int8 the device receives is exactly
`pixel - 128`, the bytes `tools/model_io.py` already produces. `convert.py` asserts this and refuses
to continue otherwise, because a model with different input quantisation would be silently fed wrong
values by the existing firmware. Measured on the first conversion: scale 0.0078125, zero point 0.

**Operator coverage.** `convert.py` runs the model through `tools/check_ops.py` and exits non-zero if
anything falls outside the five the firmware registers. First conversion emitted exactly
AVERAGE_POOL_2D, CONV_2D, DEPTHWISE_CONV_2D, RESHAPE, SOFTMAX.

### Conversion to int8

Run 2 converted successfully, which validates the deployment path end to end on a real model. Both
guards passed: the op set came out as exactly the five registered ops, and input quantisation as scale
1/128 zero point 0, so the `pixel - 128` bytes the host already sends are what this model expects.

| | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| baseline | **76.0%** | 79.7% | 69.8% | 74.5% |
| run2 float | 72.2% | 74.3% | 68.0% | 71.0% |
| run2 int8 | 72.0% | 74.1% | 67.6% | 70.7% |

**Quantisation costs 0.2 points of accuracy.** Post-training quantisation is therefore good enough;
quantisation-aware training is not worth pursuing for this model.

Model size is 303,496 B against the baseline's 300,568 B - the same architecture, so no size win
either. Nothing has been flashed: at 4.0 points below the baseline this model would make the device
worse, and the deployment tooling (a .tflite to C array converter and a firmware target that embeds
it) does not exist yet.

### Flashed to the device

Run 2's int8 model was flashed purely to validate the path, not because it is worth deploying.
`tools/tflite_to_c.py` emits the same symbols pico-tflmicro's model file declares, so the
`person_detect_serial_custom` target links it with `main.cpp` untouched.

**Device and host agree bit-exactly on all 10 sample images**, and arena use is identical to the
baseline at 82,308 B. The whole chain - Keras, int8 conversion, C array, flash, USB - is sound.

### The 3x3 convolution cost, finally measured

Our model ran at **108.2 ms against the baseline's 98.6 ms** despite identical topology. Per-op
profiling found the entire 9.6 ms in one operator:

| stem | op | time | MACs | ns/MAC |
|---|---|---|---|---|
| baseline | `DEPTHWISE_CONV_2D`, depth_multiplier 8 | 7,686 us | 165,888 | 46.3 |
| ours (Keras `Conv2D`) | `CONV_2D` 3x3 | **17,351 us** | 165,888 | **104.6** |

Every other operator matched within noise. This fills the gap flagged earlier: there was no
measurement for a 3x3 standard convolution because the baseline contains none.

| | ns/MAC |
|---|---|
| conv 1x1 (pointwise) | 10.4 |
| depthwise 3x3 | 35.3 |
| depthwise 3x3, depth_multiplier 8, 1 input channel | 46.3 |
| **conv 3x3, 1 input channel** | **104.6** |

With one input channel the two stem formulations are arithmetically identical - each output channel is
the input convolved with its own 3x3 kernel - so `model.py` now builds the stem as
`DepthwiseConv2D(depth_multiplier=8)`. Verified to convert to `DEPTHWISE_CONV_2D [1,3,3,8]` with the
parameter count unchanged. **9% of inference time for no change in arithmetic.**

Caveat on generalising: this measures a 3x3 convolution over a *single* input channel, a degenerate
case for im2col. It does not establish the cost of a 3x3 convolution over many channels.

### Experiment: MobileNetV3-Small

Branch `experiment/mobilenetv3`, not merged. Recorded here because the result is evidence either way.

MobileNetV3-Small, faithful - squeeze-excite, h-swish, inverted residuals - retargeted from 224 to 96
and trained with **the same recipe as run 2**, so architecture is the only variable. alpha=0.35 gives
205,730 parameters against v1's 218,914, so capacity matches within 6%.

Two adaptations were forced by the device, both arithmetically neutral: squeeze-excite built from
`AveragePooling2D` + two 1x1 convolutions rather than `GlobalAveragePooling2D` + `Dense` (which emits
`MEAN` and `FULLY_CONNECTED`), and a depthwise stem. The firmware gained `ADD`, `MUL` and `HARD_SWISH`,
+30 KB of flash.

| | params | MACs | int8 bytes | device ms | test acc | recall |
|---|---|---|---|---|---|---|
| v1 pretrained baseline | 210,708 | 7.16 M | 300,568 | **99.1** | **76.0%** | 69.8% |
| v1 ours, run 2 | 218,914 | 7.16 M | 303,496 | 108.7 | 72.0% | 67.6% |
| **v3-Small ours, run 3** | 205,730 | ~1.2 M | 341,312 | **112.3** | **70.2%** | 63.2% |

**MobileNetV3 lost on every axis.** Less accurate, slower, and larger on disk despite six times fewer
MACs and fewer parameters.

Per-operator profile on device, one inference:

| op | count | ms | share | MACs |
|---|---|---|---|---|
| `HARD_SWISH` | 19 | **40.6** | **36%** | none |
| `DEPTHWISE_CONV_2D` | 12 | 37.7 | 34% | |
| `CONV_2D` | 42 | 20.0 | 18% | |
| `MUL` (squeeze-excite rescale) | 18 | 10.2 | 9% | none |
| `ADD` (residuals) | 7 | 2.3 | 2% | none |
| `AVERAGE_POOL_2D` | 10 | 1.3 | 1% | |

**h-swish alone is 36% of inference for zero arithmetic**, and with squeeze-excite's `MUL` it is 45%
of the time in zero-MAC elementwise operations. Only 18% goes to `CONV_2D`. MobileNetV3's innovations
buy accuracy per MAC, which is the right currency on a mobile CPU or NPU; on a Cortex-M33 with
CMSIS-NN it is the wrong one, because convolution has good kernels and elementwise work over whole
feature maps does not. ReLU6, by contrast, is free - it folds into the convolution.

Model size is also worse for a structural reason: 110 operators carry much more flatbuffer metadata
than v1's 31, which outweighs having fewer weights.

**How far the conclusion goes.** The latency result is architectural and solid: h-swish and the
squeeze-excite rescale cost what they cost regardless of how the network was trained. The accuracy
result is narrower - v3 was trained with v1's recipe (Adam, cosine decay, the same augmentation) on
40,000 images, where MobileNetV3 was originally trained with RMSProp, dropout and label smoothing on
far more data. A recipe tuned for v3 might well close the 1.8 point gap. What can be said is that
v3-Small at matched capacity, given v1's recipe and this much data, is worse on both axes.

### Experiment: MCUNet

Branch `experiment/mcunet`. The block table is the real `mcunet-5fps_vww` configuration from MIT HAN
Lab, so the kernel sizes (3/5/7) and expansion ratios (1,3,4,5,6) are what TinyNAS searched rather
than a guess. alpha=0.7 gives 208,922 weights against v1's 207,968, matching capacity to 0.5%.

**This is MCUNet's architecture, not MCUNet.** MCUNet is TinyNAS plus TinyEngine; we run neither, so
their published numbers are not comparable to these. One deliberate deviation: the searched resolution
is 80 and this was built at 96, to hold input constant against v1 and v3.

| | test acc | precision | recall | F1 | model | arena | device ms |
|---|---|---|---|---|---|---|---|
| baseline (pretrained) | **76.0%** | 79.7% | 69.8% | 74.5% | 300,568 | 82,308 | **99.1** |
| v1 ours, run 2 | 72.0% | 74.1% | 67.6% | 70.7% | 303,496 | 82,308 | 108.7 |
| v3-Small, run 3 | 70.2% | 73.6% | 63.2% | 68.0% | 341,312 | 79,428 | 112.3 |
| **MCUNet, run 4** | **74.9%** | 77.9% | 69.5% | 73.5% | 356,288 | **180,916** | **251.0** |

**Best accuracy of anything we have trained**: +2.9 points over v1, 1.1 below the baseline, with
recall essentially matched (69.5% against 69.8%). Quantisation cost 0.0%. Only 59 operators, and the
sole addition beyond the base five is `ADD` - the architecture suits this chip, unlike v3.

**But it does not fit.** Peak activation memory is 180,916 bytes against v1's 82,308, so
356,288 + 180,916 + ~23,000 = 560,204 bytes against 520 KB of SRAM. The model has to stay in flash,
where both cores contend for the one 16 KB XIP cache (the step 02 fix), and inference takes 251 ms
rather than 99.

**The deviation caused this, and that is the interesting part.** Peak activation memory is exactly
what TinyNAS optimises, and resolution is one of the variables it searches to control it. At the
searched resolution of 80, activations scale by (80/96)^2 and the arena would be roughly 125 KB,
which together with the model would fit in SRAM. Building at 96 for comparability broke the very
budget the architecture was designed around. The lesson is not that MCUNet is too big; it is that
MCUNet's resolution is not a free parameter.

Untried follow-up: a narrower MCUNet, or one built at resolution 80. Both would change the input
pipeline or the capacity match, so neither is a like-for-like comparison with the runs above.

### Problems found while building it

**Keras precision and recall were silently wrong.** `keras.metrics.Precision(class_id=1)` slices
`y_true[..., 1]` as well as `y_pred`, but our labels are sparse with shape `(batch,)`, so it indexes
the batch axis. On a hand-computed case whose true values are 0.667 / 0.667 it reported 1.000 / 0.500;
without `class_id` it raises a shape error instead. `train.py` now defines `SparsePrecision` and
`SparseRecall`, which take the class-1 probability and compare it against the sparse label. Verified
against the hand-computed case. Recall is the metric we most want to improve, so a wrong one would
have misdirected the whole step.

**Training and export need different batch sizes.** Conversion requires `batch_size=1` or the
converter emits SHAPE, STRIDED_SLICE and PACK; training with a fixed batch of 1 is useless. `model.py`
therefore builds the architecture twice and copies weights across.

**Paths were relative to the working directory**, so the scripts only ran from the repo root.
`data.py` now anchors to `__file__`, matching `tools/model_io.py`.

### Plan for the first runs

Run 0 without augmentation, as the control: with 40,000 images and 218,914 parameters it should
overfit, and the size of that gap is what says how much augmentation is worth. Augmentation
(horizontal flip, brightness, contrast, +-8 px translation) is ablation 1. Defaults: batch 128, Adam
with cosine decay from 1e-3, 60 epochs, early stopping on validation accuracy with patience 12.
About 15 s per epoch on the GTX 1650.

## Results

All figures are validation (n=3,000), so they compare like-for-like with the baseline measured on the
same split (75.5%). Test numbers come from `convert.py` and are recorded when a model is worth
deploying.

| run | augment | best val acc | val precision | val recall | train acc at end | gap | stopped |
|---|---|---|---|---|---|---|---|
| baseline (pretrained) | - | 75.5% | 78.6% | 70.1% | - | - | - |
| run0-noaug | no | 63.8% | - | - | 84.7% | **23 pts** | epoch 26 (early) |
| run1-aug | yes | **72.4%** | 77.2% | 63.6% | 78.6% | **6 pts** | epoch 60 (ran out) |

**Augmentation is worth +8.6 points** and closes the train/val gap from 23 to 6. Run 0 confirmed the
failure mode was memorisation of 40,000 images, not a broken pipeline, which is what the control was
for.

Two things the numbers say about where to go next:

- **Run 1 never converged.** Its best epoch was 56 of 60 and validation accuracy was still climbing
  when the schedule ran out. Some of that late gain is the cosine decay annealing to zero, but the
  trajectory (0.668 at 21, 0.708 at 31, 0.717 at 51, 0.724 at 56) does not look finished. A longer
  schedule is the cheapest untried lever.
- **Recall is the weak axis, and worse than the baseline's.** 63.6% against 70.1%, while precision is
  close (77.2% vs 78.6%). The model is biased towards "no person" - the same failure the baseline has,
  slightly worse. Accuracy alone would hide this.

| run2-aug-long | yes | **74.6%** | 77.0% | 70.0% | 80.4% | 5.8 pts | epoch 83 (early, best 58) |

Longer training is worth a further **+2.2 points**, and recall recovers from 63.6% to 70.0%, matching
the baseline. But the lever is spent: validation peaked at epoch 58 and oscillated flat
(0.733, 0.741, 0.737, 0.733, 0.736) until early stopping fired at 83. 150 epochs was not the binding
constraint.

### On test, which is the fair comparison

Validation accuracy is optimistically biased for our runs because the checkpoint is *selected* on
validation; the baseline had no such selection. Test is the honest number.

| | accuracy | precision | recall | F1 | false negatives |
|---|---|---|---|---|---|
| baseline (pretrained) | **76.0%** | 79.7% | 69.8% | 74.5% | 1,357 |
| run2-aug-long (float) | 72.2% | 74.3% | 68.0% | 71.0% | 1,442 |

**We have not beaten the baseline: 3.8 points short on test.** Note our model drops 2.4 points from
validation to test while the baseline gains 0.5, which is the checkpoint-selection bias showing.

Levers tried and their value: augmentation +8.6, longer schedule +2.2, both on validation. Untried:
initialising from the pretrained weights instead of from scratch, and more data. The plateau at
epoch 58 says the recipe is no longer the constraint.

### Problems found

**Checkpoints could not be reloaded.** `SparsePrecision` and `SparseRecall` are not registered for
Keras serialisation, so `load_model` raised `TypeError: Could not locate class 'SparsePrecision'` -
which meant every run was unconvertible and the whole deployment path was blocked. `convert.py` now
loads with `compile=False`, since conversion and evaluation need only architecture and weights. It
surfaced from evaluating a checkpoint by hand; going straight to conversion would have hit the same
wall.
