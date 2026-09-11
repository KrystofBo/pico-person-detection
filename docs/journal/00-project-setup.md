# Step 00 — Project setup

## Goal
Set up the repository and the working conventions before writing any code: how we branch, commit and document each step.

## What we did
- Wrote down the overall plan: toolchain → external model with embedded images → USB image streaming → own model → deploy → optimize.
- Made these decisions:
  - **Board**: Raspberry Pi Pico 2 (`PICO_BOARD=pico2`).
  - **Toolchain**: set up by hand and pinned. `pico-sdk` and `pico-tflmicro` become git submodules under `third_party/`; the compiler is the official ARM GNU toolchain (Ubuntu 22.04's apt package is gcc 10.3, which is older than the SDK recommends).
  - **Image input**: embedded images first (the TFLM example ships two), then USB serial streaming as a separate step.
  - **USB on WSL2**: usbipd-win attaches the Pico to WSL.
  - **Python**: a Python 3.12 venv, because the system Python 3.14 probably has no TensorFlow / LiteRT wheels yet.
- Extended `.gitignore` with firmware artifacts (`*.uf2`, `*.elf`, `*.bin`, ...) and `datasets/`.
- Added a project section to `CLAUDE.md` (goal, hardware, layout, conventions).
- Added `README.md` with a progress table and a results table.

## Git conventions
- One branch per step: `step/NN-short-name`.
- Small commits with an imperative subject; the body explains *why*.
- End of step: review, merge to `main` with `git merge --no-ff`, tag `step-NN`.

## Commands
```bash
git switch -c step/00-project-setup
# ... commits ...
git switch main
git merge --no-ff step/00-project-setup
git tag step-00
```

## Results
Repository skeleton only, nothing to measure yet.

## Problems & notes
- The GitHub Python `.gitignore` template ignores `lib/` at any depth. If we ever need a directory called `lib/`, we'll have to un-ignore it. Submodule contents aren't affected.

## Next
Step 01: install the ARM toolchain, add `pico-sdk` as a submodule, build `picotool`, attach the Pico to WSL with usbipd-win, then build and flash a "hello" firmware.
