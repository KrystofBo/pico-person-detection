"""Report any operators a .tflite uses that the Pico firmware does not register.

    python tools/check_ops.py model.tflite

Exits 1 if the model would fail at AllocateTensors() on the device.
"""
import sys
from pathlib import Path

import tflite

# Registered in firmware/*/main.cpp via MicroMutableOpResolver<5>.
PICO_OPS = {"AVERAGE_POOL_2D", "CONV_2D", "DEPTHWISE_CONV_2D", "RESHAPE", "SOFTMAX"}


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
    ops = model_ops(Path(sys.argv[1]).read_bytes())
    unsupported = sorted(set(ops) - PICO_OPS)
    print("ops        :", ", ".join(ops))
    if unsupported:
        print("UNSUPPORTED:", ", ".join(unsupported))
        sys.exit(1)
    print("UNSUPPORTED: none")


if __name__ == "__main__":
    main()
