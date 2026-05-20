"""Unit tests for seismic_cnn.ingest modules."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from seismic_cnn.ingest.travel_times import haversine_km, p_wave_travel_time_s, compute_travel_times


class TestHaversine:
    def test_same_point_is_zero(self):
        assert haversine_km(35.721, -97.28, 35.721, -97.28) == pytest.approx(0.0)

    def test_known_distance(self):
        # ~111 km per degree of latitude
        d = haversine_km(0.0, 0.0, 1.0, 0.0)
        assert 110 < d < 112

    def test_symmetry(self):
        d1 = haversine_km(35.7, -97.6, 36.0, -97.2)
        d2 = haversine_km(36.0, -97.2, 35.7, -97.6)
        assert d1 == pytest.approx(d2)


class TestTravelTime:
    def test_vertical_ray(self):
        # purely vertical ray: slant = depth
        t = p_wave_travel_time_s(0.0, 6.0, velocity_kms=6.0)
        assert t == pytest.approx(1.0)

    def test_horizontal_ray(self):
        t = p_wave_travel_time_s(6.0, 0.0, velocity_kms=6.0)
        assert t == pytest.approx(1.0)


class TestComputeTravelTimes:
    def test_returns_series(self):
        df = pd.DataFrame({
            "latitude":  [35.8],
            "longitude": [-97.4],
            "depth_km":  [5.0],
        })
        result = compute_travel_times(df, station_lat=35.721, station_lon=-97.28)
        assert isinstance(result, pd.Series)
        assert len(result) == 1
        assert result.iloc[0] >= 0

    def test_nan_on_bad_row(self):
        df = pd.DataFrame({
            "latitude":  [None],
            "longitude": [None],
            "depth_km":  [5.0],
        })
        result = compute_travel_times(df, station_lat=35.721, station_lon=-97.28)
        assert result.isna().all()
