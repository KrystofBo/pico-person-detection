# pico-person-detection

Person / no-person image classification running on a **Raspberry Pi Pico 2** (RP2350, 520 KB SRAM, 4 MB flash).

> **The task is classification, not object detection.** The model answers one question per image — "is a
> person present?" — as a 2-class softmax. It has no bounding boxes: a global average pool before the
> classifier discards all spatial information by construction, so the network cannot represent *where*
> anyone is. This is the [Visual Wake Words](https://arxiv.org/abs/1906.05721) task, which its own authors
> frame as binary classification. The project name follows TensorFlow Lite Micro's `person_detection`
> example, which we vendored; "detection" there means detecting *presence*.
> Accordingly the metrics are accuracy / precision / recall, never mAP or IoU.

There's no camera on the Pico yet — images are embedded in the firmware, streamed from the host over USB serial (step 03),
or streamed live from the laptop's webcam (step 05).
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
| 03 | Host reference + USB image streaming | done | [03-host-reference-streaming](docs/journal/03-host-reference-streaming.md) |
| 04 | Train our own model (host) | done | [04-train-our-model](docs/journal/04-train-our-model.md) |
| 05 | Live laptop webcam: frames preprocessed to 96×96 greyscale, streamed to the Pico, which answers person / no person | in progress | [05-live-webcam](docs/journal/05-live-webcam.md) |

## Results

Measured on the Pico 2 unless stated otherwise.

| Model | Input | Accuracy | Latency | Flash (model) | Tensor arena |
|-------|-------|----------|---------|---------------|--------------|
| TFLM person detection (pretrained, pico-tflmicro) | 96×96 gray int8 | 76.1% | 98.6 ms | 300,568 B | 82,308 B |
| Ours, main: MobileNet v1 α=0.25, trained on 80k Wake Vision images | 96×96 gray int8 | 76.1% | 98.5 ms | 303,552 B | 82,308 B |
| Ours, optional: MCUNet α=0.6, same data (`--arch mcunet`) | 96×96 gray int8 | **77.9%** | 198.8 ms | 301,960 B | 177,284 B |

Accuracy is int8 on a balanced 9,000-image Wake Vision test split, one prediction per image by argmax
(a tie counts as no person). Latency is the device's own timing of `Invoke()` in `person_detect_serial`,
mean over the same 100 test images, all three measured in one session.

The model is 7.16 M MACs (MobileNet v1, α=0.25). Latency was 190.4 ms as first measured in step 02;
[fix/02-inference-latency](docs/journal/02-tflm-embedded.md#fix-inference-latency--190-ms--99-ms) brought it to 98.9 ms
(1.92×) by splitting the CMSIS-NN kernels across both cores and copying the model into SRAM, with bit-identical outputs.
SRAM use is 411,456 of the 524,288 bytes that hold data (the chip's other 8 KB are the two cores' stacks). That 98.9 ms came from the profile firmware, whose per-operator timing reads
0.3-0.4 ms higher than the serial firmware in the table.

Our models are trained from scratch on 80,000 Wake Vision images (step 04). v1 is the baseline's architecture and ties
it. MCUNet beats it by 1.9 points (paired test, p = 1.2e-4) at twice the latency; it needs the firmware built with
`EXTENDED_OPS` and leaves 19 KB of SRAM free, against 110 KB for v1.

Live (step 05), the laptop's webcam streams to the Pico, which answers person / no person over USB and on its LED.
With v1: **7.8 fps, 152 ms from capture to verdict**, and the Pico's scores match the host reference bit-exactly
on live frames.

Since step 03 the Pico's scores can be checked against a host reference running the same `.tflite`:
they agree **exactly** on all 12 inputs tested (`results/step03-samples.csv`). Accuracy is measured on a
balanced 9,000-image Wake Vision test split built in step 04; note the pretrained model was trained on
Visual Wake Words, not Wake Vision, so 76.1% reflects that distribution mismatch.
