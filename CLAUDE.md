# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

# Project: person detection on Raspberry Pi Pico 2

## Goal
A person / no-person image classifier that runs inference on a Raspberry Pi Pico 2.
No camera yet: images are embedded in firmware or streamed from the host over USB serial.
First prove the pipeline with an existing model (TFLite Micro person detection), then train our own smaller one.

## Hardware & environment
- Board: Raspberry Pi Pico 2 (RP2350, Cortex-M33 @ 150 MHz, 520 KB SRAM, 4 MB flash), `PICO_BOARD=pico2`.
- Host: WSL2 (Ubuntu). The Pico is attached to WSL with usbipd-win, so `picotool` and `/dev/ttyACM0` work from Linux.
- Python tooling uses a Python 3.12 venv (system Python 3.14 lacks TF/LiteRT wheels).

## Layout (directories are created only when a step needs them)
- `docs/journal/NN-title.md` - one write-up per step
- `third_party/` - pico-sdk, pico-tflmicro as pinned git submodules
- `firmware/` - one CMake project, one executable per experiment
- `tools/` - host Python scripts
- `training/` - model training (later)
- `data/samples/` - small, licensed test images; `results/` - CSVs and summaries

## Skills
Repeatable procedures (building, flashing, reading serial, ...) are project skills in `.claude/skills/`. Use them instead of reconstructing commands.

## Workflow conventions
- Work in numbered steps. One branch per step: `step/NN-short-name`.
- Fixes never consume a step number. Branch `fix/NN-short-name`, where `NN` is the step whose
  results the fix revises. The write-up is a new section appended to that step's existing
  `docs/journal/NN-*.md`, not a new journal file, so each step's journal stays the single
  narrative for its own numbers. Merge with `--no-ff`; no tag.
- Small, focused commits: imperative subject, body explains *why*.
- Each step adds `docs/journal/NN-title.md` (Goal, What we did, Commands, Results, Problems & fixes, Next) and updates the progress table in `README.md`.
- End of step: summarize the diff for review, merge to `main` with `--no-ff`, tag `step-NN`. Never push unless asked.
- Record real measured numbers (latency, flash, arena, accuracy) in the journal and the README results table - no estimates presented as measurements.