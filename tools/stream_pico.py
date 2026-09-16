"""Stream images to person_detect_serial on the Pico and compare with the host.

    python tools/stream_pico.py data/samples/*.jpg
    python tools/stream_pico.py --samples --csv ../results/step03.csv

Both sides run the same int8 model, so every score should match exactly. Any
mismatch is a real finding, not rounding, and is reported as such.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import serial

import host_reference
import model_io

MAGIC = b"PIMG"
REPLY_TIMEOUT_S = 10.0


def send_frame(port: serial.Serial, tensor: np.ndarray) -> tuple[int, int, int]:
    """-> (person, no_person, device microseconds). Raises on a bad reply."""
    payload = tensor.astype(np.int8).tobytes()
    if len(payload) != model_io.INPUT_H * model_io.INPUT_W:
        raise ValueError(f"expected {model_io.INPUT_H * model_io.INPUT_W} bytes, got {len(payload)}")
    port.write(MAGIC + payload)
    port.flush()

    deadline = time.monotonic() + REPLY_TIMEOUT_S
    while time.monotonic() < deadline:
        line = port.readline().decode("ascii", "replace").strip()
        if not line:
            continue
        if line.startswith("OK "):
            fields = dict(kv.split("=", 1) for kv in line[3:].split())
            return int(fields["person"]), int(fields["no_person"]), int(fields["time"])
        if line.startswith("ERR"):
            raise RuntimeError(f"device error: {line}")
        # READY banner or leftover noise: ignore and keep waiting.
    raise TimeoutError("no reply from device")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", nargs="*", type=Path)
    ap.add_argument("--samples", action="store_true",
                    help="also send the two embedded int8 arrays (known ground truth)")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--csv", type=Path, help="write per-image results here")
    args = ap.parse_args()
    if not args.images and not args.samples:
        ap.error("give image paths, or --samples")

    interp = host_reference.make_interpreter()

    jobs: list[tuple[str, str, np.ndarray]] = []
    if args.samples:
        for name in ("person", "no_person"):
            jobs.append((f"<embedded {name}>", name.replace("no_person", "nonperson"),
                         model_io.embedded_sample(name)))
    for path in args.images:
        label = "person" if path.name.startswith("person") else "nonperson"
        jobs.append((path.name, label, model_io.preprocess(path)))

    with serial.Serial(args.port, 115200, timeout=1.0) as port:
        time.sleep(0.3)
        port.reset_input_buffer()  # drop the READY banner if it is still queued

        rows, mismatches = [], 0
        print(f"{'input':<22}{'label':>10}{'device':>9}{'host':>7}{'P(person)':>11}{'ms':>8}  ")
        for name, label, tensor in jobs:
            dev_p, dev_n, dev_us = send_frame(port, tensor)
            host_p, host_n, prob = host_reference.infer(interp, tensor)
            agree = (dev_p, dev_n) == (host_p, host_n)
            mismatches += not agree
            flag = "" if agree else "  <-- MISMATCH"
            print(f"{name:<22}{label:>10}{dev_p:>9}{host_p:>7}{prob:>11.3f}{dev_us/1000:>8.1f}{flag}")
            rows.append({"input": name, "label": label, "device_person": dev_p,
                         "device_no_person": dev_n, "host_person": host_p,
                         "host_no_person": host_n, "p_person": round(prob, 4),
                         "device_us": dev_us, "agree": agree})

    correct = sum((r["p_person"] >= 0.5) == (r["label"] == "person") for r in rows)
    print(f"\n{correct}/{len(rows)} classified correctly at threshold 0.5")
    print(f"device/host agreement: {len(rows) - mismatches}/{len(rows)}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")

    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
