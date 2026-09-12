# Step 02 — External model: TFLM person detection with embedded images

## Goal
Prove that a real, known-good person detection model runs on the Pico 2, and measure its latency and memory use. This becomes the baseline our own model has to beat.

## What we did
- Added **pico-tflmicro** (Raspberry Pi's port of TensorFlow Lite Micro) as a submodule in `third_party/pico-tflmicro`. It has no release tags, so it's pinned to commit `dfea53b` (2024-12-26, "Updated to latest upstream changes").
- It ships the pretrained **person detection model** (Visual Wake Words, 96×96 grayscale int8 input, 2 outputs: `notperson`, `person`) and two sample images already converted to int8 C arrays.
- **Sanity check with unmodified upstream code**: built pico-tflmicro's own `person_detection_test` against our pinned SDK and flashed it. It passed.
- **Our own firmware**, `firmware/person_detect_embedded/`: reuses the model, the sample images and `model_settings` from the example; only `main.cpp` is ours. It loops forever over both images and prints the raw int8 scores, the dequantized probabilities, the time per `Invoke()` and `arena_used_bytes()`.
- Our `firmware/CMakeLists.txt` pulls in pico-tflmicro's CMakeLists with `add_subdirectory(... EXCLUDE_FROM_ALL)`. This gives us the `pico-tflmicro` library target without building its ~100 tests. Its nested `pico_sdk_import` / `pico_sdk_init()` calls are harmless because the SDK guards against double initialization (`_pico_sdk_pre_init_marker`, `_pico_sdk_inclusion_marker`).
- Op resolver: only the 5 ops the model uses (AveragePool2D, Conv2D, DepthwiseConv2D, Reshape, Softmax), with the **CMSIS-NN int8 kernels**.

## Commands
```bash
# Upstream test, unmodified source (in a scratch build dir). The define makes stdio_init_all()
# wait for a terminal, because the test prints once at boot. It's passed via CFLAGS/CXXFLAGS
# env vars so it is *added* to the SDK's CPU flags, see "Problems".
W=-DPICO_STDIO_USB_CONNECT_WAIT_TIMEOUT_MS=30000
CFLAGS=$W CXXFLAGS=$W cmake -S third_party/pico-tflmicro -B <scratch>/tflm-build \
    -DPICO_SDK_PATH=$PWD/third_party/pico-sdk -Dpicotool_DIR=$HOME/opt/picotool/lib/cmake/picotool
cmake --build <scratch>/tflm-build -j8 --target person_detection_test

# Our firmware: see the pico-build-flash skill, target person_detect_embedded
cmake --build firmware/build -j8
~/opt/picotool/bin/picotool load -f -x firmware/build/person_detect_embedded/person_detect_embedded.uf2
```

## Results

Upstream `person_detection_test`:
```
person data.     person score:  113, no person score: -113
no person data.  person score:  -57, no person score:   57
~~~ALL TESTS PASSED~~~
```

Our `person_detect_embedded`:
```
--- model=300568 bytes, arena used=82308 of 139264 bytes, binary=435996 bytes
person    person= 113 (0.941) no_person=-113 (0.059) time=190433 us
no_person person= -57 (0.277) no_person=  57 (0.723) time=190511 us
```

| Metric | Value |
|---|---|
| Classification of the 2 samples | 2/2 correct: person image P(person) = 0.941; no-person image P(person) = 0.277 |
| Latency per inference | **~190.4 ms** (stable across runs, 150 MHz, CMSIS-NN, dual-core conv) |
| Model size (flash) | **300,568 bytes** (293.5 KB) |
| Tensor arena actually used | **82,308 bytes** (80.4 KB); 136 KB allocated |
| Whole firmware in flash | 435,996 bytes (of 4 MB) |
| Static RAM (`bss`) | 145,156 bytes (arena + SDK/TFLM state), of 520 KB |

The Pico's scores match upstream's exactly (same int8 model, same kernels). The output is quantized with scale 1/256 and zero point -128, so int8 113 → (113+128)/256 = 0.941.

## Observations
- The arena could shrink from 136 KB to about 81 KB for this model. It's left as is for now.
- pico-tflmicro's CMakeLists sets `COMPILE_FLAGS` five times in one `set_target_properties`; only the last one (`-nostdlib`) survives. So the intended `-Os` is **not** applied, and the library builds with the Release default **`-O3`** (checked in `flags.make`). That makes it faster, but the code is larger than intended. We could tune this in an optimization step.
- `arm_nn_mat_mult_nt_t_s8.c` has `#define TF_LITE_PICO_MULTICORE` enabled ("experimental dual core support"). The int8 matrix multiply inside Conv2D launches **core 1** as a worker on every call, so core 1 isn't free for other work while inference is running.

## Problems & fixes
- **Upstream test printed nothing**: it runs once right at boot, before a terminal is attached, and USB stdio drops that output. Fix without touching upstream code: build with `PICO_STDIO_USB_CONNECT_WAIT_TIMEOUT_MS`.
- **`-DCMAKE_C_FLAGS=...` broke the build** (`#error no SW_SPIN_LOCK_LOCK available`): on an already-configured build dir it *replaces* the SDK's CPU flags (`-mcpu=cortex-m33 ...`), so the code was compiled for the wrong architecture. Fix: a fresh build dir with `CFLAGS` / `CXXFLAGS` environment variables, which CMake *prepends* to the toolchain's flags.
- **Our firmware seemed to hang**: nothing showed up with `timeout 8 cat /dev/ttyACM0 | tr -d '\r' | grep ... | head`. It took a debug build with stage markers to prove the firmware was fine. The culprit was the reading pipeline: when `timeout` fired, the whole pipeline was killed (exit 143) while `tr`/`grep` still held their buffered output. Capturing to a file (`timeout 8 cat /dev/ttyACM0 > file`) works reliably. The pitfall is now written into the `pico-build-flash` skill.

## Next
Step 03: a Python host tool that preprocesses arbitrary images to 96×96 int8, runs the same `.tflite` on the PC as a reference, and streams images to a new `person_detect_serial` firmware over USB so any folder of images can be tested without reflashing.

---

# Fix: inference latency — 190 ms → 99 ms

Branch `fix/02-inference-latency`. Revises the latency and memory numbers measured above.

## Goal
Step 02 recorded ~190 ms per inference without checking whether that was near the hardware's limit. Find out where the time goes, and fix what is actually wrong.

## What we found
Built `firmware/person_detect_profile/`, which hooks a 64-slot `MicroProfilerInterface` into the interpreter and prints per-op timings plus `clk_sys`. Per-op times sum to within 140 µs of `Invoke()`, and `clk_sys` is confirmed at 150 MHz.

| | time | share | MACs | share | cyc/MAC |
|---|---|---|---|---|---|
| CONV_2D (all 1×1 pointwise) | 126.0 ms | 66% | 6.19 M | 87% | 2.40 |
| DEPTHWISE_CONV_2D | 63.9 ms | 34% | 0.96 M | 13% | 9.94 |
| AvgPool + Reshape + Softmax | 0.36 ms | 0.2% | — | — | — |

The model is 7,157,888 MACs (MobileNet v1, α=0.25, 96×96×1) — cross-checked against upstream's own per-op RP2040 timings in `third_party/pico-tflmicro/benchmark_results.txt`, which are linear in our per-layer MAC counts. Fitted cost model for pointwise: **2.40 cycles/MAC + 30 cycles per output element**.

Three things were wrong in the step 02 write-up and in our reasoning:

1. **Core 1 was idle for the whole inference**, not busy as the step 02 observation claimed. Upstream's `TF_LITE_PICO_MULTICORE` block sits in the plain-C `#else` branch of `arm_nn_mat_mult_nt_t_s8()`, unreachable whenever `ARM_MATH_DSP` is defined — so it only ever ran on the RP2040. Commenting the define out changed our runtime by 0.07%, which is how we found out.
2. **The kernels are not far off their own ceiling single-threaded.** Counting the `ARM_MATH_DSP` inner loop gives ~1.3-1.5 cycles/MAC (16 MACs per ~20 instructions), not the 1-2 MAC/cycle we had assumed. The gap to 2.40 is requantisation and loop prologue.
3. **Flash (XIP) looked irrelevant and was not.** Reading 219 KB of weights once per inference costs ~6 ms of 190 ms, so single-core it does not matter. But both cores share one 16 KB XIP cache and one QSPI port, so it dominates as soon as a layer is split — see below.

So the headroom was never in the arithmetic: it was that half the chip was switched off.

## What we changed
Three patches to the pinned `pico-tflmicro`, in `third_party/patches/`, applied by `firmware/CMakeLists.txt` at configure time (idempotent: a patch that reverse-applies cleanly is skipped). The submodule stays pinned to upstream's commit, so `git status` reports its working tree as modified — that is expected.

- `0001-dual-core-dsp-matmul.patch` — adds `pico_nn_multicore.h` (a `pico_nn_run_pair()` dispatcher that launches core 1 once and parks it on the inter-core FIFO) and splits the `ARM_MATH_DSP` row loop of `arm_nn_mat_mult_nt_t_s8()` by rhs row. Guarded by its own `PICO_NN_DUAL_CORE`, not upstream's `TF_LITE_PICO_MULTICORE`, which means something else.
- `0002-dual-core-depthwise-3x3.patch` — splits `arm_depthwise_conv_3x3_s8()` by output row (13 of our 14 depthwise ops).
- `0003-dual-core-depthwise-generic.patch` — adds an output-row range to `depthwise_conv_s8_mult_4()`, for op 0 (`ch_mult=8`, the one depthwise op that does not take the 3×3 path).

Each split writes disjoint output bytes, so no locking is needed: RP2350 SRAM has byte write enables, so two cores storing adjacent bytes cannot lose an update. Layers with fewer than 4 rows/pairs stay single-core; the hand-off costs more than they save.

Firmware side: `person_detect_embedded` and the profiler now **copy the model into SRAM** and the arena is cut from 136 KB to 88 KB to make room.

## Results
Measured on the person image, all four configurations on the same board at 150 MHz.

| Configuration | Latency | vs baseline |
|---|---|---|
| Step 02 baseline (single core, model in flash) | 190.4 ms | — |
| Dual-core matmul, model in **flash** | 185.1 ms | 1.03× |
| Dual-core matmul, model in **SRAM** | 125.3 ms | 1.52× |
| + dual-core depthwise 3×3 | 104.6 ms | 1.82× |
| + dual-core generic depthwise | **99.0 ms** | **1.92×** |

| | step 02 | now | |
|---|---|---|---|
| CONV_2D total | 126.0 ms | 64.7 ms | 1.95× |
| DEPTHWISE_CONV_2D total | 63.9 ms | 34.0 ms | 1.88× |
| Whole `Invoke()` | 190.4 ms | 99.0 ms | 1.92× |

Both sample images score **bit-identically to step 02** — person `113 / -113`, no-person `-57 / 57`. That is the correctness gate for every split.

## Problems & fixes
- **Dual-core made the big layers slower** (op 26: 16.0 → 19.8 ms) while small ones improved. The split is exactly at 16 KB of weights — the RP2350's shared XIP cache size. Two cores streaming different halves of a >16 KB weight tensor thrash it and serialise on QSPI. Fixed by copying the model into SRAM, after which pointwise scales at 1.96× (essentially perfect).
- **Depthwise patch had no effect at first.** `TF_LITE_PICO_MULTICORE` is `#define`d inside `arm_nn_mat_mult_nt_t_s8.c`, so it is per-translation-unit; the depthwise file silently compiled its single-core path. Fixed by introducing `PICO_NN_DUAL_CORE` in the shared header.
- **Then it would not link**: the matmul file included that header *inside* the `#if PICO_NN_DUAL_CORE` guard, so the switch was undefined when tested and `pico_nn_run_pair()` was never defined. The include has to precede the guard.

## Constraints this introduces
- **Core 1 now belongs to CMSIS-NN** for the lifetime of the program. Anything else wanting core 1 (a camera driver, say) has to be reconciled with `pico_nn_run_pair()`.
- SRAM use is up to ~405 KB of 520 KB (294 KB model copy + 88 KB arena). A larger model will not fit this way; our own smaller model should make the copy cheaper.
- The patches are pinned to `pico-tflmicro` @ `dfea53b`. Bumping the submodule means rebasing them; the CMake step fails loudly if they no longer apply.

## Remaining headroom (not pursued)
- Ops 23 and 25 (output_y=3) and op 28 (2 rhs rows) stay single-core under the 4-row threshold: ~1.6 ms.
- Ops 24 and 26 have an odd pixel count (3×3=9), so the kernel's 2-row tiling drops one row onto a slow path: ~9 ms at baseline, less now.
- Pointwise is still ~2.4 cyc/MAC against a ~1.3-1.5 inner-loop ceiling; the rest is requantisation per output element.
- The real lever is fewer MACs, which belongs to the "train our own model" step, not here.
