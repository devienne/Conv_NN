"""Windowing, labeling, and per-file processing (Pass 2 of the pipeline).

Reads MiniSEED files, applies monthly normalisation, slices into fixed-length
windows, labels each window as earthquake or noise using the merged catalog,
and saves windows as CSV files into the dataset directory tree.
"""

import os
import numpy as np
import pandas as pd
from obspy import read as obspy_read


# ---------------------------------------------------------------------------
# Catalog loaders
# ---------------------------------------------------------------------------

def load_earthquake_times(csv_path: str) -> pd.Series:
    """Load OGS corrected P-wave arrival times as a sorted, tz-naive Series."""
    df = pd.read_csv(csv_path, parse_dates=["corrected_time"])
    return df["corrected_time"].dropna().sort_values().reset_index(drop=True)


def load_benz_times(csv_path: str) -> pd.Series:
    """Load Benz catalog origin times as a timezone-naive sorted Series."""
    df = pd.read_csv(csv_path)
    times = pd.to_datetime(df["origintime"], utc=True).dt.tz_convert(None)
    return times.dropna().sort_values().reset_index(drop=True)


def merge_catalogs(*series: pd.Series) -> pd.Series:
    """Merge and sort multiple time Series into one deduplicated catalog."""
    return pd.concat(list(series)).sort_values().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

def label_windows(t_start_pd: pd.Timestamp, n_windows: int,
                  eq_times: pd.Series, window_s: int) -> np.ndarray:
    """Return a bool array of length *n_windows*; True = earthquake window."""
    is_eq   = np.zeros(n_windows, dtype=bool)
    day_end = t_start_pd + pd.Timedelta(seconds=window_s * n_windows)

    lo = eq_times.searchsorted(t_start_pd)
    hi = eq_times.searchsorted(day_end)

    for eq_t in eq_times.iloc[lo:hi]:
        j = int((eq_t - t_start_pd).total_seconds() // window_s)
        if 0 <= j < n_windows:
            is_eq[j] = True

    return is_eq


# ---------------------------------------------------------------------------
# Filename helper
# ---------------------------------------------------------------------------

def fmt(utc_time) -> str:
    """Format an obspy UTCDateTime as a Windows-safe filename string."""
    return str(utc_time)[:19].replace(":", "-").replace("T", "_")


# ---------------------------------------------------------------------------
# Per-file worker (Pass 2) — picklable for multiprocessing.Pool
# ---------------------------------------------------------------------------

def process_file(args: tuple):
    """Apply monthly normalisation, window, label, and save one daily file.

    Parameters (packed as a single tuple for Pool.imap_unordered)
    -------------------------------------------------------------
    filepath          : str   — path to the MiniSEED file
    eq_times          : pd.Series — merged sorted earthquake times
    split_date        : pd.Timestamp — last date of the training split
    monthly_stats     : dict  — {channel: {mean, peak}} from Pass 1
    dirs              : dict  — {(split, label): output_dir}
    window_s          : int   — window length in seconds
    max_noise_per_file: int or None — noise window cap per daily file

    Returns
    -------
    (filepath, n_earthquake_windows, n_noise_windows, error_or_None)
    """
    filepath, eq_times, split_date, monthly_stats, dirs, window_s, max_noise_per_file = args
    station = os.path.basename(filepath).split("_")[0]

    try:
        st = obspy_read(filepath)
    except Exception as e:
        return filepath, 0, 0, str(e), []

    for tr in st:
        s = monthly_stats.get(tr.stats.channel)
        if s:
            tr.data = (tr.data.astype(np.float64) - s["mean"]) / s["peak"]

    date_str  = os.path.basename(filepath).split("_", 1)[1].replace(".mseed", "")
    file_date = pd.Timestamp(date_str)
    split_key = "train" if file_date <= split_date else "test"

    sampling_rate      = st[0].stats.sampling_rate
    samples_per_window = int(sampling_rate * window_s)

    arrays  = [tr.data for tr in st]
    min_len = min(len(a) for a in arrays)
    n_windows = min_len // samples_per_window

    if n_windows == 0:
        return filepath, 0, 0, None, []

    t_start_utc = st[0].stats.starttime
    t_start_pd  = pd.Timestamp(t_start_utc.datetime)

    is_eq = label_windows(t_start_pd, n_windows, eq_times, window_s)

    noise_idx = np.where(~is_eq)[0]
    if max_noise_per_file is not None and len(noise_idx) > max_noise_per_file:
        keep_noise = set(
            np.random.choice(noise_idx, max_noise_per_file, replace=False).tolist()
        )
    else:
        keep_noise = set(noise_idx.tolist())

    n_eq = n_noise = 0
    window_records = []
    for j in range(n_windows):
        i0 = j * samples_per_window
        i1 = i0 + samples_per_window

        label = "earthquake" if is_eq[j] else "noise"
        if label == "noise" and j not in keep_noise:
            continue

        out_dir = dirs[(split_key, label)]
        data    = np.array([a[i0:i1] for a in arrays])

        win_start_utc = t_start_utc + window_s * j
        win_end_utc   = win_start_utc + window_s
        out_file = os.path.join(out_dir, f"{fmt(win_start_utc)}_{fmt(win_end_utc)}.csv")

        np.savetxt(out_file, data, delimiter=",", fmt="%.6f")

        window_records.append({
            "station":      station,
            "start_time":   str(win_start_utc)[:19],
            "end_time":     str(win_end_utc)[:19],
            "label":        label,
            "split":        split_key,
            "is_augmented": 0,
            "filepath":     out_file,
        })

        if is_eq[j]:
            n_eq += 1
        else:
            n_noise += 1

    return filepath, n_eq, n_noise, None, window_records
