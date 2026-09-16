# Step 03 — Host reference + USB image streaming

## Goal
Be able to test arbitrary images without reflashing, and have a host-side reference that says what the Pico *should* answer, so device results can be trusted rather than assumed.

## What we did
- **Sample corpus.** Ten CC0 images from Wikimedia Commons in `data/samples/`, five with people and five without, downscaled to 480 px. Candidates were filtered programmatically on the API's licence metadata (only `CC0` / `Public domain` accepted), and every label was checked by eye rather than trusted from the file title — two plausible-looking hits were sculptures and a Degas painting, which are poor tests for a model trained on photographs. `data/samples/README.md` records source page, licence and author per image.
- **`tools/model_io.py`** — parses the model and the two embedded sample images out of pico-tflmicro's C arrays, so there is one source of truth and no duplicated 300 KB binary. Also does preprocessing: centre-crop to square, resize to 96×96, greyscale, `pixel - 128`.
- **`tools/host_reference.py`** — runs that model under LiteRT and prints int8 scores.
- **`firmware/person_detect_serial/`** — reads a 96×96 int8 frame over USB CDC, runs `Invoke()`, replies with one text line.
- **`tools/stream_pico.py`** — preprocesses a folder, streams it to the board, and compares each device score against the host reference. Exits non-zero on any mismatch.

## Commands
```bash
# Host reference only
python tools/host_reference.py --samples data/samples/*.jpg

# Flash (see the pico-build-flash skill), then stream
cmake --build firmware/build -j8 --target person_detect_serial
~/opt/picotool/bin/picotool load -f -x firmware/build/person_detect_serial/person_detect_serial.uf2
python tools/stream_pico.py --samples data/samples/*.jpg --csv results/step03-samples.csv
```

## Protocol
```
host -> pico : 'P','I','M','G' + 9216 raw int8 pixels
pico -> host : OK person=<int8> no_person=<int8> time=<us>
               ERR <reason>
```
Binary one way, text the other, deliberately: the SDK's USB stdio rewrites `\n` to `\r\n` on output but leaves input untouched, so a binary *reply* would be silently corrupted while a binary *request* is safe. The magic prefix means a desynchronised host resynchronises by just sending another frame.

No software flow control proved necessary for the 9216-byte frames. USB-level NAK throttles the host while the Pico is busy for ~99 ms inside `Invoke()`; frames arrive intact. This was measured, not assumed — the fallback would have been chunked acks.

## Results
Full run in `results/step03-samples.csv`.

| Input | Label | Device | Host | P(person) | ms |
|---|---|---|---|---|---|
| `<embedded person>` | person | 113 | 113 | 0.941 | 98.7 |
| `<embedded no_person>` | nonperson | -57 | -57 | 0.277 | 98.6 |
| `nonperson_01` empty lecture room | nonperson | -84 | -84 | 0.172 | 98.7 |
| `nonperson_02` autumn forest | nonperson | -93 | -93 | 0.137 | 98.6 |
| `nonperson_03` empty apartment | nonperson | -6 | -6 | 0.477 | 98.6 |
| `nonperson_04` forest path | nonperson | -45 | -45 | 0.324 | 98.7 |
| `nonperson_05` empty train interior | nonperson | -57 | -57 | 0.277 | 98.4 |
| `person_01` portrait | person | 114 | 114 | 0.945 | 98.4 |
| `person_02` man at cafe table | person | 95 | 95 | 0.871 | 98.4 |
| `person_03` street market | person | 114 | 114 | 0.945 | 98.7 |
| `person_04` cafe, people small in frame | person | 39 | 39 | 0.652 | 98.6 |
| `person_05` group portrait | person | 59 | 59 | 0.730 | 98.6 |

- **Device/host agreement: 12/12, exact.** Not close — identical int8 scores.
- **12/12 correct at threshold 0.5.** This is a 12-image smoke set, *not* a dataset-level accuracy measurement; the README results table still says accuracy is unmeasured.
- Latency ~98.6 ms, consistent with the 98.9 ms from the step 02 latency fix.
- Closest call is `nonperson_03` (empty apartment) at 0.477 — correct, but barely. `person_04`, where the people are small in frame, is the weakest positive at 0.652.

## Problems & fixes
- **The model does not load in modern LiteRT at all.** It is TOCO-converted and declares `quantized_dimension = 3` on its 14 rank-1 per-channel bias tensors. TFLite Micro ignores the field for biases, so the Pico runs it happily; LiteRT validates it and refuses (`quantized_dimension must be in range [0, 1). Was 3`). `model_io.model_bytes_for_litert()` rewrites the field to 0 — the only in-range value, and exactly what the per-channel scales already mean. It is an int32 stored inline, so the edit is size-preserving and touches no weights, scales or zero points.
- **LiteRT's default XNNPACK delegate disagrees with the device.** On the `no_person` sample it scores -60/60 where the Pico scores -57/57. `host_reference.py` therefore pins `OpResolverType.BUILTIN_REF`, which matches bit-exactly. Only having known-good ground truth (113/-113, -57/57) exposed this; a "reference" that silently differs from the device is worse than no reference.
- **Parsing the C arrays lost 250 of 9216 bytes at first.** The byte literals are not zero-padded — the image arrays contain both `0x9` and `0x1f` — and a `0x[0-9a-f]{2}` regex silently skipped the short ones. The model array happens to be uniformly two-digit, so it parsed fine and hid the bug. There is now an explicit length check.

## Next
Step 04: train our own, smaller model on the host. The reference harness built here is what will compare it against this baseline — same images, same comparison, so the trade between size/latency and accuracy is measured rather than estimated. A real accuracy number needs a proper held-out dataset, which that step should bring.
