"""Training loop for seismic CNN models."""

import os
import tensorflow as tf
from tensorflow.keras import callbacks


def run_training(model, train_ds, val_ds, n_train_eq: int, n_train_noise: int,
                 epochs: int, checkpoint_path: str):
    """Compile, fit, and return the training history.

    Parameters
    ----------
    model : keras.Model
        Uncompiled model (build_model output).
    train_ds, val_ds : tf.data.Dataset
    n_train_eq, n_train_noise : int
        Class counts used to compute inverse-frequency class weights.
    epochs : int
    checkpoint_path : str
        Where to save the best model (.keras file).

    Returns
    -------
    history : keras.callbacks.History
    """
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    total = n_train_eq + n_train_noise
    class_weight = {
        1: total / (2 * n_train_eq)    if n_train_eq    else 1.0,
        0: total / (2 * n_train_noise) if n_train_noise else 1.0,
    }
    print(f"Class weights: earthquake={class_weight[1]:.3f}  "
          f"noise={class_weight[0]:.3f}\n")

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    cb = [
        callbacks.ModelCheckpoint(checkpoint_path, monitor="val_auc",
                                  mode="max", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                patience=5, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=3, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=cb,
    )
    return history
