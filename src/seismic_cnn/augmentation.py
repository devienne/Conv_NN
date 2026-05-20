"""Data augmentation for earthquake windows.

Only Gaussian noise augmentation is used (Perol et al. 2018).
"""

import os
import math
import numpy as np
from multiprocessing import Pool


# ---------------------------------------------------------------------------
# Augmentation primitives
# ---------------------------------------------------------------------------

def add_gaussian_noise(data: np.ndarray, sigma: float = None) -> np.ndarray:
    """Add Gaussian noise; sigma drawn uniformly from [1e-4, 5e-4] if not given."""
    if sigma is None:
        sigma = np.random.uniform(1e-5, 20e-4)
    return np.clip(data + np.random.normal(0, sigma, data.shape), -1.0, 1.0)


COMBOS = [
    ([add_gaussian_noise], "gauss"),
]


def apply_combo(data: np.ndarray, fns: list) -> np.ndarray:
    result = data.copy()
    for fn in fns:
        result = fn(result)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_csv(directory: str, originals_only: bool = False) -> int:
    files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    if originals_only:
        files = [f for f in files if "_aug" not in f]
    return len(files)


def _augment_one(args: tuple):
    """Worker: augment one earthquake file. Compatible with Pool.imap_unordered."""
    eq_dir, fname, aug_per_file = args
    try:
        data = np.loadtxt(os.path.join(eq_dir, fname), delimiter=",")
        if data.ndim == 1:
            data = data[np.newaxis, :]
        stem  = os.path.splitext(fname)[0]
        count = 0
        for j in range(aug_per_file):
            fns, tag = COMBOS[j % len(COMBOS)]
            aug_data = apply_combo(data, fns)
            out_name = f"{stem}_{tag}_{j}_aug.csv"
            np.savetxt(os.path.join(eq_dir, out_name),
                       aug_data, delimiter=",", fmt="%.6f")
            count += 1
        return count, None
    except Exception as e:
        return 0, str(e)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def augment_split(eq_dir: str, noise_dir: str, split_name: str,
                  n_workers: int) -> int:
    """Augment earthquake windows in *eq_dir* until they match *noise_dir* count.

    Returns the total number of augmented files saved.
    """
    eq_files  = [f for f in os.listdir(eq_dir)
                 if f.endswith(".csv") and "_aug" not in f]
    n_eq_orig = len(eq_files)
    n_noise   = _count_csv(noise_dir)

    print(f"[{split_name}]  earthquake (original): {n_eq_orig}  |  noise: {n_noise}")

    if n_eq_orig == 0:
        print(f"  No earthquake files found — skipping {split_name}.")
        return 0

    if n_eq_orig >= n_noise:
        print(f"  Earthquakes ({n_eq_orig}) already >= noise ({n_noise}) — no augmentation needed.")
        return 0

    aug_per_file = max(1, math.ceil((n_noise - n_eq_orig) / n_eq_orig))
    print(f"  Augmentations per file: {aug_per_file}  "
          f"(cycling through {len(COMBOS)} combination(s))\n")

    worker_args = [(eq_dir, fname, aug_per_file) for fname in eq_files]

    total_saved = 0
    with Pool(n_workers) as pool:
        for i, (count, err) in enumerate(
            pool.imap_unordered(_augment_one, worker_args), 1
        ):
            if err:
                print(f"  [{i}/{n_eq_orig}] ERROR: {err}")
            else:
                total_saved += count
            if i % 50 == 0 or i == n_eq_orig:
                print(f"  [{i}/{n_eq_orig}] files augmented ...", flush=True)

    n_eq_final = _count_csv(eq_dir)
    ratio = n_noise / n_eq_final if n_eq_final else float("inf")
    print(f"  Earthquake files total : {n_eq_final}")
    print(f"  Imbalance ratio        : {ratio:.2f}x\n")
    return total_saved
