"""Reject a .tflite whose quantisation scales would fault the device.

    python tools/check_scales.py model.tflite

Random or barely-trained weights produce channels with a near-zero range, and
scales around 1e-12 make the requantisation shift go out of range. Shifting by
32 or more bits is undefined behaviour, so the firmware hard-faults before it
can print anything and only the BOOTSEL button recovers it.

Exits 1 if the model should not be flashed.
"""
import sys
from pathlib import Path

import tflite

# Trained models here sit around 1e-3 to 1e-6; 1e-9 is far below anything real.
MIN_SCALE = 1e-9


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    model = tflite.Model.GetRootAsModel(Path(sys.argv[1]).read_bytes(), 0)
    worst, bad, total = float("inf"), 0, 0
    for s in range(model.SubgraphsLength()):
        g = model.Subgraphs(s)
        for i in range(g.TensorsLength()):
            q = g.Tensors(i).Quantization()
            if q is None:
                continue
            for j in range(q.ScaleLength()):
                scale = q.Scale(j)
                total += 1
                worst = min(worst, scale)
                if scale < MIN_SCALE:
                    bad += 1
    print(f"{total:,} scales, smallest {worst:.3e}")
    if bad:
        print(f"REJECT: {bad:,} scales below {MIN_SCALE:g}. Flashing this will hard-fault "
              f"the board and need BOOTSEL to recover. Is the model trained?")
        sys.exit(1)
    print("ok to flash")


if __name__ == "__main__":
    main()
