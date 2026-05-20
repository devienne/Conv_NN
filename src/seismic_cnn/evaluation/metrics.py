"""Model evaluation: metrics, confusion matrix, and training history plots."""

import os
import numpy as np
import matplotlib.pyplot as plt


def evaluate_model(model, val_ds):
    """Print validation metrics and return them as a dict."""
    val_loss, val_acc, val_auc, val_prec, val_rec = model.evaluate(val_ds, verbose=0)
    f1 = (2 * val_prec * val_rec / (val_prec + val_rec)
          if (val_prec + val_rec) > 0 else 0.0)

    metrics = dict(loss=val_loss, accuracy=val_acc, auc=val_auc,
                   precision=val_prec, recall=val_rec, f1=f1)

    print("\n--- Validation evaluation ---")
    for name, val in metrics.items():
        print(f"  {name.capitalize():12s}: {val:.4f}")
    return metrics


def confusion_matrix(model, val_ds, threshold: float = 0.5) -> dict:
    """Collect all predictions and print + return the confusion matrix."""
    y_true, y_pred = [], []
    for x_batch, y_batch in val_ds:
        probs = model.predict(x_batch, verbose=0).flatten()
        y_pred.extend((probs >= threshold).astype(int).tolist())
        y_true.extend(y_batch.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    print(f"\nConfusion matrix (threshold = {threshold}):")
    print(f"              Predicted EQ   Predicted Noise")
    print(f"  Actual EQ   {tp:>6d}         {fn:>6d}   (TP / FN)")
    print(f"  Actual Noise{fp:>6d}         {tn:>6d}   (FP / TN)")

    return dict(tp=tp, tn=tn, fp=fp, fn=fn)


def plot_training_history(history, figures_dir: str):
    """Save a 2×2 grid of training/validation curves to *figures_dir*."""
    metrics_to_plot = [
        ("loss",      "Loss"),
        ("accuracy",  "Accuracy"),
        ("auc",       "AUC"),
        ("precision", "Precision"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (key, title) in zip(axes.flat, metrics_to_plot):
        ax.plot(history.history[key],          label="Train")
        ax.plot(history.history[f"val_{key}"], label="Validation")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("ConvNetQuake (Perol et al. 2018) — training history",
                 fontweight="bold")
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out = os.path.join(figures_dir, "training_history.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"\nTraining history saved -> {out}")
