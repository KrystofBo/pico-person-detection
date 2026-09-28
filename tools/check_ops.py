"""Report any operators a .tflite uses that the Pico firmware does not register.

    python tools/check_ops.py model.tflite

Exits 1 if the model would fail at AllocateTensors() on the device.
"""
import sys
from pathlib import Path

import tflite

# What firmware/*/main.cpp registers. Keep in step with those files: this is what
# decides whether a model can run on the device.
BASE_OPS = {"AVERAGE_POOL_2D", "CONV_2D", "DEPTHWISE_CONV_2D", "RESHAPE", "SOFTMAX"}

# Only registered when the firmware is built with EXTENDED_OPS=1, which the
# person_detect_*_custom targets do. MobileNetV3 needs all three; MobileNet v1
# needs none. Enabling them costs ~30 KB of flash and ~50 KB of SRAM.
EXTENDED_OPS = {"ADD", "MUL", "HARD_SWISH"}

PICO_OPS = BASE_OPS | EXTENDED_OPS


def model_ops(blob: bytes) -> list[str]:
    model = tflite.Model.GetRootAsModel(blob, 0)
    names = {v: k for k, v in vars(tflite.BuiltinOperator).items() if isinstance(v, int)}
    codes = [model.OperatorCodes(i).BuiltinCode() for i in range(model.OperatorCodesLength())]
    ops = set()
    for s in range(model.SubgraphsLength()):
        g = model.Subgraphs(s)
        ops |= {names[codes[g.Operators(i).OpcodeIndex()]] for i in range(g.OperatorsLength())}
    return sorted(ops)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    ops = set(model_ops(Path(sys.argv[1]).read_bytes()))
    unsupported = sorted(ops - PICO_OPS)
    needs_extended = sorted(ops & EXTENDED_OPS)
    print("ops        :", ", ".join(sorted(ops)))
    if unsupported:
        print("UNSUPPORTED:", ", ".join(unsupported))
        print("             register these in firmware/*/main.cpp and raise the resolver size")
        sys.exit(1)
    if needs_extended:
        print("UNSUPPORTED: none, but needs EXTENDED_OPS=1 for:", ", ".join(needs_extended))
        print("             the person_detect_*_custom targets build with it")
    else:
        print("UNSUPPORTED: none - runs on the default firmware")


if __name__ == "__main__":
    main()
