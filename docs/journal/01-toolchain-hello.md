# Step 01 — Toolchain and "hello" on the Pico 2

## Goal
Build a firmware for the Pico 2, flash it from WSL, and read its output over USB serial, before any ML code is involved.

## What we did
- **Compiler**: ARM GNU Toolchain **14.2.rel1**, from the official tarball (checksum verified), in `~/opt/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi`.
  Ubuntu 22.04's `gcc-arm-none-eabi` package is 10.3, older than the SDK recommends. 15.2.rel1 also exists, but we picked 14.2 because it's what Raspberry Pi's VS Code extension installs, so it's the most tested combination with pico-sdk.
- **SDK**: `pico-sdk` **2.3.1** (latest release) as a submodule in `third_party/pico-sdk`. Only its `tinyusb` sub-submodule is actually needed (for USB serial); the wireless ones (btstack, cyw43, lwip, mbedtls) come along with `--recursive` but aren't used on a plain Pico 2.
- **picotool** **2.3.1** (matches the SDK), built from source in `~/opt/src/picotool` and installed to `~/opt/picotool`. The SDK build uses it to generate the `.uf2` files, and we use it to flash.
- **USB into WSL**: usbipd-win on Windows + picotool's udev rules in WSL (WSL here runs systemd, so udev rules apply; the user is already in `plugdev` and `dialout`).
- **Firmware**: `firmware/` is one CMake project. `firmware/hello/` blinks the LED and prints once per second over USB CDC.
- **Skill**: the build → flash → read-serial loop is written up as the project skill `.claude/skills/pico-build-flash/`. `CLAUDE.md` only points to the skills folder.

## Commands

### One-time host setup (WSL)
```bash
sudo apt install -y libusb-1.0-0-dev pkg-config

# Toolchain
mkdir -p ~/opt && cd ~/opt
V=14.2.rel1; F=arm-gnu-toolchain-$V-x86_64-arm-none-eabi.tar.xz
B=https://developer.arm.com/-/media/Files/downloads/gnu/$V/binrel
curl -LO $B/$F && curl -LO $B/$F.sha256asc && sha256sum -c $F.sha256asc && tar xf $F

# picotool
mkdir -p ~/opt/src && cd ~/opt/src
git clone --depth 1 --branch 2.3.1 https://github.com/raspberrypi/picotool.git
cmake -S picotool -B picotool/build -DPICO_SDK_PATH=<repo>/third_party/pico-sdk \
      -DCMAKE_INSTALL_PREFIX=$HOME/opt/picotool -DCMAKE_BUILD_TYPE=Release
cmake --build picotool/build -j8 && cmake --install picotool/build

# udev rules so picotool works without sudo
sudo cp ~/opt/src/picotool/udev/60-picotool.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### One-time host setup (Windows, admin PowerShell)
```powershell
winget install --exact dorssel.usbipd-win
# open a NEW PowerShell window afterwards, otherwise `usbipd` is not on PATH

# Pico plugged in while holding BOOTSEL:
usbipd list                                   # "RP2350 Boot" (2e8a:000f) was on BUSID 1-1
usbipd bind --busid 1-1
usbipd attach --wsl --busid 1-1 --auto-attach # leave running

# after the first flash the running app shows up as a different device (2e8a:0009),
# which must be bound once as well:
usbipd bind --busid 1-1
```

### Build and flash
```bash
git submodule update --init --recursive
export PICO_TOOLCHAIN_PATH=$HOME/opt/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi
cmake -S firmware -B firmware/build -Dpicotool_DIR=$HOME/opt/picotool/lib/cmake/picotool
cmake --build firmware/build -j8
~/opt/picotool/bin/picotool load -f -x firmware/build/hello/hello.uf2
cat /dev/ttyACM0
```
`-f` makes a running program reboot into BOOTSEL by itself (through the SDK's USB reset interface), and `-x` starts it after flashing. So after the first flash, no BOOTSEL button press is needed.

## Results
Serial output:
```
hello from RP2350 #154: clk_sys=150000000 Hz, binary=28180 bytes
```
- Default system clock: **150 MHz**.
- Flash image (`__flash_binary_end - XIP_BASE`): **28,180 bytes**.
- `arm-none-eabi-size hello.elf`: text 32,272 / data 0 / bss 2,640 bytes. `.uf2` file: 57,344 bytes (UF2 has 512-byte blocks carrying 256 bytes of payload each).
- Reflash without touching the board (`picotool load -f -x`): works.

## Problems & fixes
- `usbipd` "not recognized" right after `winget install`: the PowerShell session still had the old PATH. Fixed by opening a new window.
- After flashing, `/dev/ttyACM0` didn't appear. The running app enumerates as `2e8a:0009` (the BOOTSEL device is `2e8a:000f`), and usbipd treats that as a separate device that needs its own `usbipd bind`. After binding it once, `--auto-attach` handles both modes.
- `cat /dev/ttyACM0` shows an empty line between messages. The SDK sends `\r\n` and the tty's default settings turn `\r` into another newline. Only cosmetic; the Python tools in step 03 will read the port raw.

## Next
Step 02: add `pico-tflmicro` and run the pretrained TFLM person detection model on the two sample images bundled with it; measure latency, arena usage and flash size.
