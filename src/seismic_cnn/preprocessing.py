"""Per-channel monthly normalization statistics (Pass 1 of the processing pipeline).

Implements the normalization described in Perol et al. (2018):
  x_norm = (x - monthly_mean_channel) / monthly_peak_abs_channel

Statistics are computed incrementally (sum + count + running max) so the
full month of raw data never needs to be held in memory simultaneously.
"""

import numpy as np
from collections import defaultdict
from obspy import read


def compute_monthly_stats(filepaths: list) -> dict:
    """Compute per-channel mean and absolute peak over all files in a month.

    Parameters
    ----------
    filepaths : list of str
        Paths to the MiniSEED files belonging to one calendar month.

    Returns
    -------
    dict
        ``{channel: {'mean': float, 'peak': float}}``
    """
    ch_sum   = defaultdict(float)
    ch_count = defaultdict(int)
    ch_peak  = defaultdict(float)

    for fp in filepaths:
        try:
            st = read(fp)
        except Exception:
            continue
        for tr in st:
            ch   = tr.stats.channel
            data = tr.data.astype(np.float64)
            ch_sum[ch]   += float(data.sum())
            ch_count[ch] += len(data)
            peak = float(np.abs(data).max())
            if peak > ch_peak[ch]:
                ch_peak[ch] = peak

    stats = {}
    for ch in ch_sum:
        mean = ch_sum[ch] / ch_count[ch] if ch_count[ch] > 0 else 0.0
        peak = ch_peak[ch] if ch_peak[ch] > 0 else 1.0
        stats[ch] = {"mean": mean, "peak": peak}
    return stats
