"""Export the built Wake Vision subset to individual PNGs for manual inspection.

    python tools/export_images.py                      # everything
    python tools/export_images.py --split test --limit 200

Writes data/wake-vision-96-png/<split>/<person|nonperson>/<index>.png.

The .npz shards are a training cache: they hold exactly these pixels and load in
about two seconds instead of forty. Nothing is lost by regenerating either from
the other; the PNGs are the copy you can actually look at.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = {0: "nonperson", 1: "person"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("data/wake-vision-96"))
    ap.add_argument("--out", type=Path, default=Path("data/wake-vision-96-png"))
    ap.add_argument("--split", choices=["train_quality", "validation", "test"],
                    help="default: all splits")
    ap.add_argument("--limit", type=int, help="export at most this many images per split")
    args = ap.parse_args()

    splits = [args.split] if args.split else ["train_quality", "validation", "test"]
    for split in splits:
        shards = sorted(args.src.glob(f"{split}_*.npz"))
        if not shards:
            print(f"{split}: no shards in {args.src}, skipping")
            continue
        for name in CLASSES.values():
            (args.out / split / name).mkdir(parents=True, exist_ok=True)

        written = 0
        for shard in shards:
            data = np.load(shard)
            for image, label in zip(data["images"], data["labels"]):
                if args.limit and written >= args.limit:
                    break
                path = args.out / split / CLASSES[int(label)] / f"{written:06d}.png"
                Image.fromarray(image, mode="L").save(path, optimize=True)
                written += 1
            if args.limit and written >= args.limit:
                break
        print(f"{split}: wrote {written:,} PNGs to {args.out / split}")


if __name__ == "__main__":
    main()
