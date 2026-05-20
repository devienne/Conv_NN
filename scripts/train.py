"""Train the ConvNetQuake model.

Usage
-----
python scripts/train.py
python scripts/train.py --config configs/experiment_convnetquake.yaml
"""

import argparse
import os

from seismic_cnn.config import load_config
from seismic_cnn.datasets import build_dataset, check_dataset
from seismic_cnn.models.convnetquake import build_model
from seismic_cnn.training.train import run_training
from seismic_cnn.evaluation.metrics import evaluate_model, confusion_matrix, plot_training_history
from seismic_cnn.db.database import init_db, log_model_run


def main(cfg):
    windows = cfg.windows_dir
    train_eq    = os.path.join(windows, "train", "earthquake")
    train_noise = os.path.join(windows, "train", "noise")
    val_eq      = os.path.join(windows, "test",  "earthquake")
    val_noise   = os.path.join(windows, "test",  "noise")

    print("Building datasets ...")
    train_ds, n_train_eq, n_train_noise = build_dataset(
        train_eq, train_noise, batch_size=cfg.batch_size, shuffle=True)
    val_ds, n_val_eq, n_val_noise = build_dataset(
        val_eq, val_noise, batch_size=cfg.batch_size, shuffle=False)

    print(f"  Train — earthquake: {n_train_eq:,}  |  noise: {n_train_noise:,}")
    print(f"  Test  — earthquake: {n_val_eq:,}  |  noise: {n_val_noise:,}\n")

    check_dataset(train_ds, cfg.figures_dir, n_show=4)

    model = build_model(n_samples=cfg.n_samples, n_channels=cfg.n_channels)
    model.summary()

    checkpoint_path = os.path.join(cfg.models_dir, "best_model.keras")
    history = run_training(
        model, train_ds, val_ds,
        n_train_eq=n_train_eq, n_train_noise=n_train_noise,
        epochs=cfg.epochs,
        checkpoint_path=checkpoint_path,
    )

    metrics = evaluate_model(model, val_ds)
    confusion_matrix(model, val_ds)
    plot_training_history(history, cfg.figures_dir)

    # Persist experiment result to SQLite
    init_db()
    log_model_run(
        model_name="ConvNetQuake",
        hyperparams={"n_samples": cfg.n_samples, "n_channels": cfg.n_channels,
                     "batch_size": cfg.batch_size, "epochs": cfg.epochs},
        metrics=metrics,
        epochs_run=len(history.history["loss"]),
    )

    print(f"\nBest model saved -> {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to experiment YAML")
    parser.add_argument("--local",  default=None, help="Path to local YAML")
    args = parser.parse_args()
    main(load_config(args.config, args.local))
