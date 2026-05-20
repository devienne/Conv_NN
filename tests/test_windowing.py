"""Unit tests for seismic_cnn.windowing."""

import numpy as np
import pandas as pd
import pytest

from seismic_cnn.windowing import label_windows, fmt, merge_catalogs


class TestLabelWindows:
    def test_no_events(self):
        t0 = pd.Timestamp("2014-03-01 00:00:00")
        eq_times = pd.Series([], dtype="datetime64[ns]")
        result = label_windows(t0, n_windows=10, eq_times=eq_times, window_s=20)
        assert not result.any()

    def test_event_falls_in_first_window(self):
        t0 = pd.Timestamp("2014-03-01 00:00:00")
        eq_times = pd.Series([pd.Timestamp("2014-03-01 00:00:05")])
        result = label_windows(t0, n_windows=5, eq_times=eq_times, window_s=20)
        assert result[0] is np.bool_(True)
        assert not result[1:].any()

    def test_event_outside_range_ignored(self):
        t0 = pd.Timestamp("2014-03-01 00:00:00")
        eq_times = pd.Series([pd.Timestamp("2014-03-02 00:00:00")])
        result = label_windows(t0, n_windows=5, eq_times=eq_times, window_s=20)
        assert not result.any()

    def test_length(self):
        t0 = pd.Timestamp("2014-03-01 00:00:00")
        eq_times = pd.Series([], dtype="datetime64[ns]")
        result = label_windows(t0, n_windows=7, eq_times=eq_times, window_s=20)
        assert len(result) == 7


class TestMergeCatalogs:
    def test_sorted_output(self):
        s1 = pd.Series([pd.Timestamp("2014-03-02"), pd.Timestamp("2014-03-01")])
        s2 = pd.Series([pd.Timestamp("2014-03-01 12:00:00")])
        merged = merge_catalogs(s1, s2)
        assert merged.is_monotonic_increasing
