# Step 05 — Live laptop webcam

## Goal

Stream the laptop's webcam to the Pico 2 in real time, each frame preprocessed on the laptop to 96x96
greyscale, with the Pico deciding person / no person and signalling it back. The model is step 04's
main one: v1 trained on 80,000 images, 76.1% on the test split, 98.5 ms per inference.

## What we did

**The webcam in WSL.** The camera belongs to Windows and the Pico is attached to WSL through usbipd.
WSL's kernel (6.18) ships the UVC webcam driver as a module, so the camera attaches the same way, and
capture, preprocessing and the serial link stay in one process and one conda environment. It appears as
`/dev/video0` (`/dev/video1` is its metadata node), and the preview window opens through WSLg.

What the camera delivers over usbipd, measured over 3 s each:

| format | resolution | driver claims | delivered |
|---|---|---|---|
| MJPEG | 640x480 | 30 fps | 16.6 fps |
| MJPEG | 1280x720 | 30 fps | 16.6 fps |
| YUYV, uncompressed | 640x480 | 30 fps | **none** |

Uncompressed video needs ~18 MB/s of isochronous transfer and nothing arrives, so the tool asks for
MJPEG. 16.6 fps is well above what the Pico consumes.

**The Pico decides.** `person_detect_serial` now replies with `verdict=<person|no_person>` - the argmax
of its two scores, so a tie is not a person - and drives the onboard LED with it. The LED goes dark
after a second without frames, so a stale verdict is not left lit. The reply only gained a field, so
`stream_pico.py` works as before.

**The laptop side**, `tools/webcam_pico.py`:

- A capture thread keeps only the newest frame, so each verdict is about what is in front of the camera
  now rather than a frame queued in the driver.
- Preprocessing is the training pipeline itself - centre square, greyscale, 96x96 bilinear - through
  `model_io.preprocess_image()`, split out of `preprocess()` and verified identical on 110 images.
- The preview shows the camera with the crop outlined, the 96x96 frame the Pico saw, the verdict,
  P(person), the Pico's inference time and the frame rate. q, Esc, Ctrl+C or closing the window stops it
  and prints a timing summary.

**The main model is in the repo**: `models/v1-80k.tflite`, 303,552 B, byte-identical to step 04's run 6,
so the firmware builds from any checkout. The C array is still generated from it.

## Commands

The `live-webcam` skill holds the full procedure, with troubleshooting.

```bash
# Windows, admin PowerShell. Bus id from `usbipd list`; bind once, attach per session.
usbipd bind --busid 1-12
usbipd attach --wsl --busid 1-12       # Windows loses the camera until: usbipd detach --busid 1-12

# Firmware with the main model: the base five operators (pico-build-flash skill)
export PICO_TOOLCHAIN_PATH=$HOME/opt/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi
conda activate pico-person-detection
python tools/tflite_to_c.py models/v1-80k.tflite -o firmware/generated/model_data.cpp
cmake -S firmware -B firmware/build -Dpicotool_DIR=$HOME/opt/picotool/lib/cmake/picotool -DCUSTOM_EXTENDED_OPS=OFF
cmake --build firmware/build -j8 --target person_detect_serial_custom
~/opt/picotool/bin/picotool load -f -x firmware/build/person_detect_serial/person_detect_serial_custom.uf2

python tools/webcam_pico.py
```

## Results

| check | result |
|---|---|
| 100 test images: Pico scores against the host reference | 100/100 bit-exact |
| 100 test images: Pico verdict against the argmax of its scores | 100/100 |
| 50 live webcam frames: Pico scores against the host reference | 50/50 bit-exact |
| empty room, 107 live frames over two runs | "person" on 1 |
| in view and stepping out, 360 live frames | "person" on 87%; the verdict and the LED followed |
| in view, 235 live frames | "person" on 100% |

A third, 30-frame run in the empty room was not counted, but its P(person) peaked at 0.54, so it held at
least one more false "person". That the verdict and the LED followed was observed by the user in front of
the camera; the rest is measured.

Latency on the live webcam, v1:

| | byte-at-a-time read | block read | 512-byte USB transfers |
|---|---|---|---|
| frames per second | 6.0 | 7.8 | **8.4** |
| capture to verdict | 186 ms | 152 ms | **129 ms** |
| round trip: send, infer, reply | 155.5 ms | 118.4 ms | 109.7 ms |
| of which Pico inference | 98.5 ms | 98.6 ms | 98.5 ms |
| of which moving the frame and reply | 56.3 ms | 20.1 ms | 11.2 ms |

The first three rows are the tool's own summaries over 60, 30 and 30 s; the last two send one fixed frame
40 times, the same with the camera closed or streaming. Of the final 129 ms, inference is 98.5, the link
11, and the remaining ~19 is the frame's age when it is sent - the camera delivers one every 60 ms - plus
preprocessing.

## Problems & fixes

**Moving a frame cost 56 ms, a third of the round trip.** Not the webcam: the same with the camera
closed. The Pico's USB FIFO holds 64 bytes at full speed, and the host can send the next packet only
once it is drained; the firmware drained it with one `getchar_timeout_us()` per byte, each a full pass
through the stdio drivers. `stdio_get_until()` returns whatever is buffered, and the cost fell to
20.1 ms. A frame that stalls mid-send still fails after 2 s with `ERR short_frame`, and the next frame is
read normally - both tested.

**That left 20 ms, and the obvious next fix did nothing.** Enlarging the receive FIFO from 64 bytes to
4 KB: 19.1 ms to send, as before. Logging the receive loop on the Pico showed why: the frame arrived as
145 single 64-byte packets, one about every 120 us, never more than one waiting. The limit was the
endpoint buffer, one packet per USB transfer. At 512 bytes a transfer carries eight: the frame arrives in
19 transfers, 8.8 ms on the Pico and 10.1 ms to send, against 7.6 ms for full-speed USB at its maximum
of 19 packets per millisecond. TinyUSB re-arms the endpoint only while the FIFO has a whole transfer free
(`cdc_device.c`), so the FIFO holds two - 1 KB, which measured as fast as 4 KB. Cost: 1,856 B of SRAM.
Rechecked afterwards: bit-exact on the 100 test images, and a cut-off frame still recovers.

**Uncompressed video does not survive usbipd** (table above); MJPEG does.

**"No person" on every frame looked like a failure, but the room was empty.** The same check found the
empty room's image far darker and flatter than the training data: mean brightness 43 against the test
images' 118, darker than 95% of them, with a third of their contrast. With someone in view the model
finds them, but that gap is the first suspect if it misses a person in poor light.

**Qt warned `QFontDatabase: Cannot find font directory` on every start.** Importing cv2 points Qt at a
font directory the wheel does not ship (`cv2/config-3.py`), overwriting any value set beforehand. Qt only
reads it when the first window opens, so the tool now sets it to the system's DejaVu fonts after
importing cv2, and the warning is gone.

## Next

- The link is down to 11 ms, near full-speed USB's floor. Sending the next frame while the Pico infers
  would hide it entirely - about 10 fps - but each verdict would then be about a frame ~100 ms older.
- MCUNet is one rebuild away (`-DCUSTOM_EXTENDED_OPS=ON`, step 04): +1.9 points on test, at an
  estimated 4.6 fps: its 198.8 ms of inference plus the same 20 ms of link and laptop work per frame.
