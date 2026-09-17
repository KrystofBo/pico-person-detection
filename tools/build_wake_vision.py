"""Build a balanced 96x96 greyscale subset of Wake Vision by streaming it.

    python tools/build_wake_vision.py --split train_quality --target 40000
    python tools/build_wake_vision.py --split validation    --target 3000
    python tools/build_wake_vision.py --split test          --target 9000

The full dataset is 365 GB, so nothing is downloaded to disk: rows are streamed,
filtered, preprocessed to 96x96 mono and only those kept. Output is ~9 KB per
image.

Images are kept only if their aspect ratio is within --tolerance of 1:1, then
centre-cropped to square. At the default 0.25 that caps crop loss at 20% while
still admitting 4:3 and 3:2. Filtering to exactly 1:1 is not viable: the yield
is 4.6% and every square image sampled was the same processed 447x447 size.

Each split is written with exactly half person and half non-person images.
Shards are flushed as they fill, so an interrupted run keeps its work; resume
with --skip-rows using the row count printed on exit.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image

SIZE = 96
SHARD = 2000
PERSON, NON_PERSON = 1, 0


def preprocess(im: Image.Image) -> np.ndarray:
    """Centre-crop to square, resize to 96x96, greyscale. Same as tools/model_io.py."""
    im = im.convert("L")
    side = min(im.size)
    left, top = (im.width - side) // 2, (im.height - side) // 2
    im = im.crop((left, top, left + side, top + side))
    return np.asarray(im.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8)


def flush(out_dir: Path, split: str, index: int, images: list, labels: list) -> None:
    if not images:
        return
    np.savez_compressed(out_dir / f"{split}_{index:04d}.npz",
                        images=np.stack(images), labels=np.array(labels, dtype=np.uint8))
    images.clear()
    labels.clear()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True,
                    choices=["train_quality", "validation", "test"])
    ap.add_argument("--target", type=int, required=True, help="total images, split 50/50")
    ap.add_argument("--out", type=Path, default=Path("data/wake-vision-96"))
    ap.add_argument("--tolerance", type=float, default=0.25,
                    help="max |aspect - 1| to accept (default 0.25)")
    ap.add_argument("--skip-rows", type=int, default=0,
                    help="resume: skip this many source rows first")
    args = ap.parse_args()

    if args.target % 2:
        ap.error("--target must be even so the split can be exactly balanced")
    per_class = args.target // 2
    args.out.mkdir(parents=True, exist_ok=True)

    shard = len(list(args.out.glob(f"{args.split}_*.npz")))
    images: list[np.ndarray] = []
    labels: list[int] = []
    kept = {PERSON: 0, NON_PERSON: 0}
    rows = args.skip_rows
    rejected_aspect = 0
    start = time.time()

    ds = load_dataset("Harvard-Edge/Wake-Vision", split=args.split, streaming=True)
    if args.skip_rows:
        ds = ds.skip(args.skip_rows)

    try:
        for row in ds:
            rows += 1
            # Must come before the filters below: gating it on accepted rows
            # only ticks when one lands on the interval, about 18% of the time.
            if rows % 5000 == 0:
                done = kept[PERSON] + kept[NON_PERSON]
                streamed = max(rows - args.skip_rows, 1)
                elapsed = time.time() - start
                eta = f"{(args.target - done) / (done / elapsed) / 60:.0f} min" if done else "?"
                print(f"  rows {rows:,}  kept {done:,}/{args.target:,} "
                      f"(person {kept[PERSON]:,} / non {kept[NON_PERSON]:,})  "
                      f"yield {done / streamed:.1%}  eta {eta}", flush=True)

            label = row["person"]
            if label not in kept or kept[label] >= per_class:
                continue                       # class already full, or unlabelled
            image = row["image"]
            width, height = image.size
            if abs(width / height - 1.0) > args.tolerance:
                rejected_aspect += 1
                continue

            images.append(preprocess(image))
            labels.append(label)
            kept[label] += 1

            if len(images) >= SHARD:
                flush(args.out, args.split, shard, images, labels)
                shard += 1
            if kept[PERSON] >= per_class and kept[NON_PERSON] >= per_class:
                break
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        flush(args.out, args.split, shard, images, labels)

    total = kept[PERSON] + kept[NON_PERSON]
    print(f"\n{args.split}: kept {total:,} of {rows - args.skip_rows:,} rows streamed "
          f"({total / max(rows - args.skip_rows, 1):.1%} yield) in "
          f"{(time.time() - start) / 60:.0f} min")
    print(f"  person {kept[PERSON]:,}  non-person {kept[NON_PERSON]:,}  "
          f"rejected on aspect {rejected_aspect:,}")
    if total < args.target:
        print(f"  SHORT of target by {args.target - total:,} - the split ran out. "
              f"Resume is not possible past the end of a split.")
    print(f"  resume with --skip-rows {rows}")

    # datasets/pyarrow background threads abort during interpreter finalisation
    # ("PyGILState_Release: thread state must be current"), which dumps core and
    # buries this summary. The data is already on disk, so leave without
    # finalising. Flush first, since os._exit does not.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
