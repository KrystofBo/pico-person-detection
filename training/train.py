"""Train MobileNet v1 on the Wake Vision subset, with TensorBoard monitoring.

    python training/train.py --name run0-noaug
    python training/train.py --name run1-aug --augment
    tensorboard --logdir training/logs

Logged per epoch: loss, accuracy, precision, recall, F1 and learning rate as
scalars; weight histograms; the model graph; and a grid of validation
predictions with errors outlined in red, which is where label noise from crop
damage shows up. Hyperparameters go to the HParams tab so ablation runs compare
in one table.

The bar to beat is the pretrained baseline's 76.0% accuracy on the same 9,000
test images (docs/journal/04-train-our-model.md).
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

import data as D
import model as M

BASELINE_TEST_ACCURACY = 0.760


# keras.metrics.Precision(class_id=1) slices y_true[..., 1] as well as y_pred.
# Our labels are sparse, shape (batch,), so that indexes the batch axis and
# silently returns a wrong number - 1.000/0.500 on a case whose true values are
# 0.667/0.667. These take the class-1 probability and compare it against the
# sparse label, which is already the binary indicator for "person".
class SparsePrecision(keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, y_pred[:, 1], sample_weight)


class SparseRecall(keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, y_pred[:, 1], sample_weight)


class ValidationImages(keras.callbacks.Callback):
    """Logs a grid of validation predictions; wrong ones get a red border."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, writer, count: int = 32):
        super().__init__()
        self.x = D.scale(images[:count])[..., None]
        self.y = labels[:count]
        self.writer = writer

    def on_epoch_end(self, epoch, logs=None):
        pred = np.argmax(self.model.predict(self.x, verbose=0), axis=1)
        # grey -> RGB so correctness can be colour-coded
        tiles = np.repeat(((self.x[..., 0] + 1.0) / 2.0)[..., None], 3, axis=-1)
        for i, (p, t) in enumerate(zip(pred, self.y)):
            colour = (0.0, 1.0, 0.0) if p == t else (1.0, 0.0, 0.0)
            tiles[i, :3, :, :] = colour
            tiles[i, -3:, :, :] = colour
            tiles[i, :, :3, :] = colour
            tiles[i, :, -3:, :] = colour
        rows = [np.concatenate(tiles[r * 8:(r + 1) * 8], axis=1) for r in range(len(tiles) // 8)]
        grid = np.concatenate(rows, axis=0)[None]
        with self.writer.as_default():
            tf.summary.image("val/predictions", grid, step=epoch)
            tf.summary.scalar("val/f1", _f1(logs), step=epoch)
            tf.summary.scalar("train/lr",
                              float(keras.backend.get_value(self.model.optimizer.learning_rate)),
                              step=epoch)


def _f1(logs) -> float:
    p = (logs or {}).get("val_precision", 0.0)
    r = (logs or {}).get("val_recall", 0.0)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run name; used for the log and run directory")
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--patience", type=int, default=12, help="early stopping on val accuracy")
    args = ap.parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = D.REPO / "training/logs" / f"{args.name}-{stamp}"
    run_dir = D.REPO / "training/runs" / f"{args.name}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = D.dataset("train", args.batch_size, augment=args.augment)
    val_ds = D.dataset("val", args.batch_size)
    val_images, val_labels = D.load_split("val")

    model = M.build(alpha=args.alpha)
    steps = args.epochs * (len(D.load_split("train")[1]) // args.batch_size)
    schedule = keras.optimizers.schedules.CosineDecay(args.lr, decay_steps=steps)
    model.compile(
        optimizer=keras.optimizers.Adam(schedule),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            SparsePrecision(name="precision", thresholds=0.5),
            SparseRecall(name="recall", thresholds=0.5),
        ],
    )
    print(f"{model.name}: {model.count_params():,} params, "
          f"{len(D.load_split('train')[1]):,} training images, augment={args.augment}")

    writer = tf.summary.create_file_writer(str(log_dir / "custom"))
    callbacks = [
        keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1, write_graph=True),
        ValidationImages(val_images, val_labels, writer),
        keras.callbacks.ModelCheckpoint(str(run_dir / "best.keras"), monitor="val_accuracy",
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=args.patience,
                                      restore_best_weights=True, verbose=1),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        callbacks=callbacks, verbose=2)

    best = max(history.history["val_accuracy"])
    model.save(run_dir / "final.keras")
    (run_dir / "config.json").write_text(json.dumps({
        "name": args.name, "alpha": args.alpha, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "augment": args.augment,
        "best_val_accuracy": best, "log_dir": str(log_dir),
    }, indent=2))

    print(f"\nbest val accuracy {best:.1%}  (baseline test accuracy {BASELINE_TEST_ACCURACY:.1%})")
    print(f"  logs {log_dir}\n  run  {run_dir}")
    print(f"  next: python training/convert.py --run {run_dir}")


if __name__ == "__main__":
    main()
