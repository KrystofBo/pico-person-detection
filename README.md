# pico-person-detection

Person / no-person image classification running on a **Raspberry Pi Pico 2** (RP2350, 520 KB SRAM, 4 MB flash).

There's no camera yet — images are currently embedded in the firmware, with host streaming over USB serial planned.
The plan is to first run an existing model (TensorFlow Lite Micro person detection) to prove the pipeline end to end,
then train our own model that fits the Pico's memory.

The project is built step by step. Each step has its own branch, tag and journal entry. A fix to an earlier
step does not take a new step number: it goes on `fix/NN-short-name` and is written up inside that step's
existing journal entry, so the corrected numbers sit next to the baseline they supersede.

## Working in this repo

- **Firmware**: one CMake project in `firmware/`, one executable per experiment. Build, flash and serial-read
  commands are in the `pico-build-flash` skill; one-time toolchain setup is in
  [01-toolchain-hello](docs/journal/01-toolchain-hello.md).
- **Host Python**: the conda env `pico-person-detection` (Python 3.12), specified in `environment.yml`.
  See the `python-env` skill.
- **`third_party/pico-tflmicro` is patched at build time** from `third_party/patches/`, so its working tree
  always shows as modified while the submodule itself stays pinned to an upstream commit. That is expected —
  don't commit the pointer change, and don't `git submodule update --force`.

## Progress

| Step | Description | Status | Journal |
|------|-------------|--------|---------|
| 00 | Repo foundation: conventions, docs, .gitignore | done | [00-project-setup](docs/journal/00-project-setup.md) |
| 01 | Toolchain + "hello" firmware on the Pico 2 | done | [01-toolchain-hello](docs/journal/01-toolchain-hello.md) |
| 02 | External model: TFLM person detection, embedded images | done | [02-tflm-embedded](docs/journal/02-tflm-embedded.md) |
| 03 | Host reference + USB image streaming | planned | |
| 04 | Train our own model (host) | planned | |
| 05 | Deploy our model on the Pico | planned | |

## Results

Measured on the Pico 2 unless stated otherwise.

| Model | Input | Accuracy | Latency | Flash (model) | Tensor arena |
|-------|-------|----------|---------|---------------|--------------|
| TFLM person detection (pretrained, pico-tflmicro) | 96×96 gray int8 | not measured yet (2/2 samples correct) | **98.9 ms** | 300,568 B | 82,308 B |

The model is 7.16 M MACs (MobileNet v1, α=0.25). Latency was 190.4 ms as first measured in step 02;
[fix/02-inference-latency](docs/journal/02-tflm-embedded.md#fix-inference-latency--190-ms--99-ms) brought it to 98.9 ms
(1.92×) by splitting the CMSIS-NN kernels across both cores and copying the model into SRAM, with bit-identical outputs.
Total SRAM use is ~405 KB of 520 KB.
