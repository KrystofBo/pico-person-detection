"""The architectures the training pipeline can build.

`v1` is the project's architecture: MobileNet v1, what the baseline uses and
what everything is measured against.

`v3` is MobileNetV3-Small, an experiment that lost on every axis - less
accurate, slower and larger (docs/journal/04-train-our-model.md). It is kept
because it stays pluggable at no cost to v1, and because being able to re-run
the comparison is worth more than the file it occupies. Nothing depends on it.

`mcunet` is MCUNet's searched architecture for Visual Wake Words. Unlike v3 it
uses relu6 and no squeeze-excite, so the only extra operator is ADD.

Adding an architecture means writing a module with `build(alpha)` and
`build_for_export(alpha, weights_from)` and adding one line here.
"""
from __future__ import annotations

import model
import model_mcunet
import model_v3

# name -> (module, default alpha)
ARCHITECTURES = {
    "v1": (model, 0.25),
    "v3": (model_v3, 0.35),
    "mcunet": (model_mcunet, 0.5),
}
NAMES = tuple(ARCHITECTURES)

# v3 needs ADD, MUL and HARD_SWISH, which the firmware only registers when built
# with EXTENDED_OPS=1. v1 needs none of them. tools/check_ops.py reports which
# of the two firmware configurations a converted model requires.
NEEDS_EXTENDED_OPS = {"v3", "mcunet"}   # v3 needs all three, mcunet only ADD


def _entry(arch: str):
    if arch not in ARCHITECTURES:
        raise SystemExit(f"unknown architecture {arch!r}; choose from {', '.join(NAMES)}")
    return ARCHITECTURES[arch]


def default_alpha(arch: str) -> float:
    return _entry(arch)[1]


def build(arch: str, alpha: float | None = None):
    module, default = _entry(arch)
    return module.build(alpha if alpha is not None else default)


def build_for_export(arch: str, alpha: float | None = None, weights_from=None):
    module, default = _entry(arch)
    return module.build_for_export(alpha if alpha is not None else default,
                                   weights_from=weights_from)
