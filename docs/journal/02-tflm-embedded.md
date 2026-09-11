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
