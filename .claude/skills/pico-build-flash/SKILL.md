---
name: pico-build-flash
description: Build the firmware in firmware/, flash a target to the Pico 2 with picotool, and read its USB serial output. Use whenever firmware needs to be compiled, flashed, or run on the board.
---

# Build, flash and read the Pico 2

The toolchain and picotool live outside the repo in `~/opt`. One-time setup is in `docs/journal/01-toolchain-hello.md`.

## 1. Build
```bash
git submodule update --init --recursive   # only needed after a fresh clone or a submodule bump
export PICO_TOOLCHAIN_PATH=$HOME/opt/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi
cmake -S firmware -B firmware/build -Dpicotool_DIR=$HOME/opt/picotool/lib/cmake/picotool
cmake --build firmware/build -j8
```
Each executable ends up at `firmware/build/<target>/<target>.uf2`. `PICO_BOARD` defaults to `pico2` in `firmware/CMakeLists.txt`.

The configure step applies `third_party/patches/*.patch` to the pico-tflmicro submodule (dual-core CMSIS-NN kernels; see the step 02 journal). Consequences:
- `git status` permanently shows `m third_party/pico-tflmicro`. That is expected - the submodule stays pinned to an upstream commit and must **not** be committed as a pointer change.
- Re-configuring is safe; a patch that reverse-applies cleanly is skipped.
- `git submodule update --force` or a submodule bump reverts/breaks the patches. Re-run the configure step to reapply; if it fails with "the patches need rebasing", regenerate them against the new commit.

## 2. Flash
```bash
~/opt/picotool/bin/picotool load -f -x firmware/build/<target>/<target>.uf2
```
- `-f` makes the running program reboot into BOOTSEL by itself, so there's no button to press. `-x` starts the program after flashing.
- If nothing is running yet (empty flash, or a crashed program), hold BOOTSEL while plugging in and drop `-f`.

## 3. Read serial output
```bash
timeout 20 bash -c 'until [ -e /dev/ttyACM0 ]; do sleep 0.5; done'   # wait for re-enumeration
timeout 8 cat /dev/ttyACM0 > /tmp/serial.txt; tr -d '\r' < /tmp/serial.txt
```
- **Don't pipe `timeout ... cat` into filters** (`| tr | grep | head`). When the timeout fires, the whole pipeline is killed (exit 143) while `tr`/`grep` still hold buffered output, so you see nothing even though the Pico printed fine. Capture to a file first, then filter it.
- Output printed before the port is opened is lost; USB stdio doesn't buffer it. Firmware should print in a loop, or wait for a host with `PICO_STDIO_USB_CONNECT_WAIT_TIMEOUT_MS`.
- For the user's interactive terminal: `stty -F /dev/ttyACM0 raw -echo && cat /dev/ttyACM0`. Without the `stty`, you get blank lines between messages because the SDK sends `\r\n`.

## Troubleshooting
The Pico reaches WSL through usbipd-win. On Windows, `usbipd attach --wsl --busid 1-1 --auto-attach` must be running.
- `picotool` finds no device, or `/dev/ttyACM0` never appears: the USB device isn't attached to WSL. Check `lsusb | grep 2e8a`. BOOTSEL mode is `2e8a:000f`, the running program is `2e8a:0009`, and each must have been `usbipd bind`-ed once on Windows (admin). Ask the user to run `usbipd list` / `usbipd bind --busid 1-1`, since this can't be done from WSL.
- A firmware with USB stdio disabled can't be rebooted with `-f`; it needs the BOOTSEL button.
