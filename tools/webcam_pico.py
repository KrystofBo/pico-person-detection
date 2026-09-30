"""Stream a live webcam to the Pico and show its verdict.

    python tools/webcam_pico.py                   # /dev/video0 -> /dev/ttyACM0
    python tools/webcam_pico.py --camera 1 --port /dev/ttyACM1

The webcam has to be attached to WSL first, like the Pico (usbipd; see
docs/journal/05-live-webcam.md). Each frame gets the same preprocessing as the
training data - centre square, greyscale, 96x96 bilinear (tools/model_io.py) -
and goes to person_detect_serial, which replies with its verdict and shows it on
the Pico's LED. The preview shows the camera with the crop outlined, the 96x96
frame the Pico actually saw, the verdict, P(person), the Pico's inference time
and the frame rate. q, Esc or closing the window stops it and prints a summary.
"""
from __future__ import annotations

import argparse
import os
import statistics
import threading
import time

import cv2
import numpy as np
import serial
from PIL import Image

import model_io
from stream_pico import exchange

WINDOW = "Pico person detection"
GREEN, GREY = (0, 200, 0), (200, 200, 200)

# Importing cv2 points Qt at a font directory the wheel does not ship
# (cv2/config-3.py), and Qt warns about it on every start. Qt reads the
# variable when the first window opens, so overriding it here still works.
SYSTEM_FONTS = "/usr/share/fonts/truetype/dejavu"
if os.path.isdir(SYSTEM_FONTS):
    os.environ["QT_QPA_FONTDIR"] = SYSTEM_FONTS


class LatestFrame:
    """Reads the camera on its own thread and keeps only the newest frame.

    Reading on demand would take frames from the driver's queue, already a few
    frames old by the time the Pico is free again. The verdict should be about
    what is in front of the camera now.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._lock = threading.Lock()
        self._latest: tuple[int, float, np.ndarray] | None = None
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        seq = 0
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            seq += 1
            with self._lock:
                self._latest = (seq, time.monotonic(), frame)

    def newer_than(self, seq: int) -> tuple[int, float, np.ndarray] | None:
        """-> (sequence number, capture time, BGR frame), or None if nothing new."""
        with self._lock:
            return self._latest if self._latest and self._latest[0] > seq else None

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2)


def label(frame: np.ndarray, text: str, y: int, scale: float, colour: tuple) -> None:
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)


def draw(frame: np.ndarray, tensor: np.ndarray, reply: dict[str, str], fps: float) -> None:
    person = reply["verdict"] == "person"
    colour = GREEN if person else GREY
    h, w = frame.shape[:2]
    left, top, side = model_io.centre_square(w, h)
    cv2.rectangle(frame, (left, top), (left + side - 1, top + side - 1), colour, 2)

    # What the Pico saw: its 96x96 input, doubled, in the top-right corner.
    seen = (tensor[..., 0].astype(np.int16) + 128).astype(np.uint8)
    seen = cv2.resize(seen, (192, 192), interpolation=cv2.INTER_NEAREST)
    frame[8:200, w - 200:w - 8] = cv2.cvtColor(seen, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(frame, (w - 201, 7), (w - 8, 200), GREY, 1)

    # Darken the corner behind the text so it reads on any background.
    frame[:122, :330] //= 3
    p_person = (int(reply["person"]) + 128) / 256  # int8 softmax: scale 1/256, zero point -128
    label(frame, "PERSON" if person else "no person", 42, 1.2, colour)
    label(frame, f"P(person) {p_person:.2f}", 78, 0.7, colour)
    label(frame, f"Pico {int(reply['time']) / 1000:.1f} ms   {fps:.1f} fps", 108, 0.7, GREY)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="N in /dev/videoN")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"no camera at /dev/video{args.camera}. Attach the webcam to WSL first: "
                         "usbipd attach --wsl --busid <id> (docs/journal/05-live-webcam.md).")
    # MJPEG keeps the USB traffic small, which matters over usbipd.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
    print(f"camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
          f"{fourcc} at {cap.get(cv2.CAP_PROP_FPS):.0f} fps")
    frames = LatestFrame(cap)

    pico_ms, round_ms, age_ms, persons = [], [], [], 0
    seq, fps, last = 0, 0.0, None
    start = time.monotonic()
    try:
        with serial.Serial(args.port, 115200, timeout=1.0) as port:
            time.sleep(0.3)
            port.reset_input_buffer()  # drop the READY banner if it is still queued
            while True:
                latest = frames.newer_than(seq)
                if latest is None:
                    time.sleep(0.002)
                    continue
                seq, captured, frame = latest
                rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tensor = model_io.preprocess_image(rgb)
                sent = time.monotonic()
                reply = exchange(port, tensor)
                now = time.monotonic()

                pico_ms.append(int(reply["time"]) / 1000)
                round_ms.append((now - sent) * 1000)
                age_ms.append((now - captured) * 1000)
                persons += reply["verdict"] == "person"
                if last is not None:
                    rate = 1 / (now - last)
                    fps = rate if fps == 0 else 0.9 * fps + 0.1 * rate
                last = now

                draw(frame, tensor, reply, fps)
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        pass  # Ctrl+C stops it like q does, summary included
    finally:
        frames.stop()
        cap.release()
        cv2.destroyAllWindows()

    if pico_ms:
        n, elapsed = len(pico_ms), time.monotonic() - start
        print(f"{n} frames in {elapsed:.1f} s = {n / elapsed:.1f} fps; person in {persons / n:.0%} of them")
        print(f"  Pico inference      {statistics.mean(pico_ms):6.1f} ms")
        print(f"  round trip          {statistics.mean(round_ms):6.1f} ms  (send, infer, reply)")
        print(f"  capture to verdict  {statistics.mean(age_ms):6.1f} ms")


if __name__ == "__main__":
    main()
