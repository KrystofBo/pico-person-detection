# pico-person-detection

Person / no-person image classification running on a **Raspberry Pi Pico 2** (RP2350, 520 KB SRAM, 4 MB flash).

There's no camera yet, so images are either embedded in the firmware or streamed from a host PC over USB serial.
The plan is to first run an existing model (TensorFlow Lite Micro person detection) to prove the pipeline end to end,
then train our own model that fits the Pico's memory.

The project is built step by step. Each step has its own branch, tag and journal entry.

## Progress

| Step | Description | Status | Journal |
|------|-------------|--------|---------|
| 00 | Repo foundation: conventions, docs, .gitignore | done | [00-project-setup](docs/journal/00-project-setup.md) |
| 01 | Toolchain + "hello" firmware on the Pico 2 | done | [01-toolchain-hello](docs/journal/01-toolchain-hello.md) |
| 02 | External model: TFLM person detection, embedded images | planned | |
| 03 | Host reference + USB image streaming | planned | |
| 04 | Train our own model (host) | planned | |
| 05 | Deploy our model on the Pico | planned | |

## Results

Measured on the Pico 2 unless stated otherwise.

| Model | Input | Accuracy | Latency | Flash (model) | Tensor arena |
|-------|-------|----------|---------|---------------|--------------|
| *(filled in from step 02)* | | | | | |
