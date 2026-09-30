---
name: live-webcam
description: Runs the step 05 live demo - the laptop webcam streamed to the Pico 2, which answers person / no person over USB and on its LED - including attaching the webcam and the Pico to WSL and flashing the firmware it needs. Use when asked to run, demo, test or troubleshoot the live webcam.
---

# Live webcam demo

The laptop streams its webcam to the Pico, which runs the main model and answers person / no person over
USB and on its LED (`docs/journal/05-live-webcam.md`). Commands run from the repo root, on a branch that
contains step 05.

## 1. Attach both devices to WSL

Windows, admin PowerShell. Bus ids come from `usbipd list`; each device needs `usbipd bind --busid <id>`
once before its first attach.

```
usbipd attach --wsl --busid 1-1 --auto-attach    # Pico; leave this window open
usbipd attach --wsl --busid 1-12                 # webcam; Windows cannot use it while attached
```

In WSL, `/dev/ttyACM0` is the Pico and `/dev/video0` the webcam. `/dev/video1` is the webcam's metadata
node and delivers no frames.

## 2. Flash the firmware - only if the Pico runs something else

Its flash keeps the last program, so this is needed only after flashing a different one. The main model on
the base five operators, named explicitly because both settings stay cached in the build directory; the
`pico-build-flash` skill explains the `*_custom` targets.

```bash
export PICO_TOOLCHAIN_PATH=$HOME/opt/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi
cmake -S firmware -B firmware/build -Dpicotool_DIR=$HOME/opt/picotool/lib/cmake/picotool \
      -DCUSTOM_MODEL=models/v1-80k.tflite -DCUSTOM_EXTENDED_OPS=OFF
cmake --build firmware/build -j8 --target person_detect_serial_custom
~/opt/picotool/bin/picotool load -f -x firmware/build/person_detect_serial/person_detect_serial_custom.uf2
```

MCUNet instead, at roughly half the frame rate:
`-DCUSTOM_MODEL=training/runs/run8-mcunet-80k-20260928-202517/model_int8.tflite -DCUSTOM_EXTENDED_OPS=ON`.

## 3. Run

```bash
/home/krystof/miniconda3/envs/pico-person-detection/bin/python tools/webcam_pico.py
```

The preview opens through WSLg. q, Esc, closing the window or Ctrl+C stops it and prints frames per second,
Pico inference, round trip and capture-to-verdict latency. `--camera N` and `--port` if the devices came up
under other numbers.

For a timed run from a script, stop it with SIGINT so the summary still prints, and run it in the
foreground: `timeout -s INT 30 <python> tools/webcam_pico.py`. Under a background task wrapper the SIGINT
killed it without a summary.

## 4. Give the webcam back

`usbipd detach --busid 1-12` on Windows.

## Troubleshooting

| symptom | cause |
|---|---|
| `no camera at /dev/video0` | the webcam is not attached (step 1) |
| the window freezes; q and the close button stop responding | the camera stopped delivering frames (detached, laptop slept); Ctrl+C still stops it |
| `device error: ERR model_too_large` or `ERR allocate_tensors` | firmware built with the wrong `CUSTOM_EXTENDED_OPS` for its model |
| no frames in another pixel format | over usbipd only MJPEG arrives, which the tool requests |
