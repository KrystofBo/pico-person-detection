---
name: python-env
description: Set up and use the project's conda environment, and install Python packages. Use BEFORE running any pip/conda install or any host Python script (model inspection, training, image preprocessing, serial tooling).
---

# Python environment

This project uses **conda**, not venv.

## The rule

**Never install into a base, system, or global interpreter.** Not conda `base`, not `/usr/bin/python3`, not `pip install --user`. No exceptions, and never "just for a quick check" — a throwaway analysis is exactly how packages end up somewhere they were not wanted.

Before any `pip install` or `conda install`, pick one of these, in order:

1. **Use this project's environment** if it exists: the conda env `pico-person-detection`.
2. **Ask** if there is any ambiguity. `conda env list` here shows ~18 environments; do not guess which one a package belongs in.
3. **Create a new environment, named after the project**, only when it is unambiguous that none exists.

## This project

Conda env **`pico-person-detection`** (Python 3.12). The spec is tracked in `environment.yml`.

```bash
# Create (only if `conda env list` does not show it)
conda env create -f environment.yml

# Use - call the interpreter by full path. Each tool call is a fresh shell, so
# `conda activate` does not persist between commands.
/home/krystof/miniconda3/envs/pico-person-detection/bin/python tools/<script>.py

# Or activate within a single command
source /home/krystof/miniconda3/etc/profile.d/conda.sh && conda activate pico-person-detection && python tools/<script>.py
```

After adding a dependency, add it to `environment.yml` (under `dependencies:` for conda packages, under the nested `pip:` list for pip-only ones, pinned to an exact version) so the environment stays reproducible. Then:
```bash
conda env update -f environment.yml --prune
```

## Checking before you remove anything

If packages did land in the wrong environment, confirm nothing else needs them before uninstalling — the point is to restore the previous state, not to trade one mess for another:

```bash
<interp> -m pip show <pkg> | grep Required-by
```
Uninstall only when `Required-by:` is empty (or lists only other packages you are also removing).
