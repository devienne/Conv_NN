"""tf.data pipeline for loading windowed seismic CSV files.

Each CSV file contains a (N_CHANNELS × N_SAMPLES) matrix — channels as rows,
time samples as columns. Conv1D expects (N_SAMPLES, N_CHANNELS), so we
transpose after loading.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

N_CHANNELS    = 3
N_SAMPLES     = 2000    # 20 s × 100 Hz
SAMPLING_RATE = 100.0   # Hz


def _load_csv(path_bytes, label):
    """numpy CSV loader called inside tf.py_function — runs in eager mode."""
    path = path_bytes.numpy().decode("utf-8")
    mat  = np.loadtxt(path, delimiter=",", dtype=np.float32)   # (C, T)
    mat  = mat.T                                                # (T, C)
    # pad/clip to exact shape
    t, c = mat.shape
    if t < N_SAMPLES:
        mat = np.pad(mat, ((0, N_SAMPLES - t), (0, 0)))
    if c < N_CHANNELS:
        mat = np.pad(mat, ((0, 0), (0, N_CHANNELS - c)))
    return mat[:N_SAMPLES, :N_CHANNELS], np.int32(label.numpy())


def parse_csv(path: tf.Tensor, label: tf.Tensor):
    """Read a (N_CHANNELS × N_SAMPLES) CSV → float32 tensor (N_SAMPLES, N_CHANNELS)."""
    mat, lbl = tf.py_function(_load_csv, [path, label], [tf.float32, tf.int32])
    mat.set_shape([N_SAMPLES, N_CHANNELS])
    lbl.set_shape([])
    return mat, lbl


def build_dataset(eq_dir: str, noise_dir: str, batch_size: int = 32,
                  shuffle: bool = True, seed: int = 42):
    """Build a batched, prefetched tf.data.Dataset from earthquake and noise dirs.

    Returns
    -------
    ds : tf.data.Dataset
    n_eq : int
    n_noise : int
    """
    eq_files    = sorted(f for f in os.listdir(eq_dir)    if f.endswith(".csv"))
    noise_files = sorted(f for f in os.listdir(noise_dir) if f.endswith(".csv"))
    n_eq, n_noise = len(eq_files), len(noise_files)

    paths  = ([os.path.join(eq_dir,    f) for f in eq_files] +
              [os.path.join(noise_dir, f) for f in noise_files])
    labels = [1] * n_eq + [0] * n_noise

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 20_000),
                        seed=seed, reshuffle_each_iteration=True)
    ds = (ds
          .map(parse_csv, num_parallel_calls=tf.data.AUTOTUNE)
          .cache()
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE))
    return ds, n_eq, n_noise


def check_dataset(ds: tf.data.Dataset, figures_dir: str, n_show: int = 4):
    """Inspect one batch and save a sample plot to *figures_dir*."""
    batch_x, batch_y = next(iter(ds))

    print("--- Dataset check ---")
    print(f"  Batch shape : {batch_x.shape}  (batch, time, channels)")
    print(f"  dtype       : {batch_x.dtype}")
    print(f"  Value range : [{float(batch_x.numpy().min()):.4f}, "
          f"{float(batch_x.numpy().max()):.4f}]")
    print(f"  Labels (first {n_show}): {batch_y[:n_show].numpy().tolist()}\n")

    n_show  = min(n_show, batch_x.shape[0])
    t       = np.arange(N_SAMPLES) / SAMPLING_RATE
    ch_lbl  = ["HHE", "HHN", "HHZ"][:N_CHANNELS]
    colors  = {1: "crimson", 0: "steelblue"}
    names   = {1: "Earthquake", 0: "Noise"}

    fig, axes = plt.subplots(n_show, N_CHANNELS,
                             figsize=(5 * N_CHANNELS, 2.5 * n_show),
                             sharex=True)
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for r in range(n_show):
        lbl = int(batch_y[r].numpy())
        for c in range(N_CHANNELS):
            axes[r, c].plot(t, batch_x[r, :, c].numpy(), lw=0.8, color=colors[lbl])
            axes[r, c].grid(alpha=0.3)
            if r == 0:
                axes[r, c].set_title(ch_lbl[c], fontweight="bold")
            if c == 0:
                axes[r, c].set_ylabel(names[lbl], color=colors[lbl],
                                      fontweight="bold", fontsize=9)
        if r == n_show - 1:
            for c in range(N_CHANNELS):
                axes[r, c].set_xlabel("Time (s)")

    fig.suptitle("Dataset check — sample windows  (red = earthquake, blue = noise)",
                 fontweight="bold")
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out = os.path.join(figures_dir, "dataset_check.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"  Saved -> {out}\n")
