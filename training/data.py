"""Loading the Wake Vision subset for training.

Reads the .npz shards written by tools/build_wake_vision.py. The matching PNGs
under data/wake-vision-96-png/ hold the same pixels and are there to be looked
at; the shards are loaded here because decoding 52,000 PNGs costs ~40 s per run
against ~2 s for the shards.

Scaling is `(pixel - 128) / 128`, chosen so post-training quantisation lands on
scale 1/128 and zero point 0 - i.e. the int8 the Pico receives is exactly
`pixel - 128`, the same bytes tools/model_io.py already produces. convert.py
asserts that the converted model really did get that quantisation rather than
trusting it.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import tensorflow as tf

SPLITS = {"train": "train_quality", "val": "validation", "test": "test"}
AUTOTUNE = tf.data.AUTOTUNE

# Anchored to the file, not the working directory, so these work from anywhere.
REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO / "data/wake-vision-96"


def load_split(split: str, root: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """-> (uint8 images (N,96,96), uint8 labels (N,)). 1 = person."""
    root = root or DATA_ROOT
    name = SPLITS.get(split, split)
    shards = sorted(glob.glob(str(root / f"{name}_*.npz")))
    if not shards:
        raise FileNotFoundError(
            f"no {name} shards in {root}. Build them first:\n"
            f"  python tools/build_wake_vision.py --split {name} --target <n>")
    images = np.concatenate([np.load(s)["images"] for s in shards])
    labels = np.concatenate([np.load(s)["labels"] for s in shards])
    return images, labels


def scale(images: np.ndarray) -> np.ndarray:
    """uint8 pixels -> float32 in [-1, 1), matching the device's int8 mapping."""
    return (images.astype(np.float32) - 128.0) / 128.0


def _augment(x, y):
    """Light augmentation. Deliberately no vertical flip: people are rarely upside down."""
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_brightness(x, 0.2)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    # Small translation, up to 8 px each way, via pad then random crop.
    x = tf.image.resize_with_crop_or_pad(x, 96 + 16, 96 + 16)
    x = tf.image.random_crop(x, (tf.shape(x)[0], 96, 96, 1))
    return tf.clip_by_value(x, -1.0, 1.0), y


def dataset(split: str, batch_size: int, *, augment: bool = False,
            shuffle: bool | None = None, root: Path | None = None) -> tf.data.Dataset:
    root = root or DATA_ROOT
    images, labels = load_split(split, root)
    if shuffle is None:
        shuffle = split == "train"

    ds = tf.data.Dataset.from_tensor_slices((scale(images)[..., None], labels.astype(np.int32)))
    if shuffle:
        ds = ds.shuffle(len(labels), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def representative_dataset(n: int = 500, root: Path | None = None):
    """Calibration samples for post-training quantisation: real training images."""
    images, _ = load_split("train", root or DATA_ROOT)
    idx = np.random.default_rng(0).choice(len(images), size=min(n, len(images)), replace=False)
    for i in idx:
        yield [scale(images[i])[None, ..., None]]
