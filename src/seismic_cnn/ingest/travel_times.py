"""P-wave travel-time correction for the historical earthquake catalog.

Uses a straight-ray, homogeneous half-space assumption with a crustal
P-wave velocity of 6.0 km/s (Perol et al. 2018 convention).
"""

import math
import numpy as np
import pandas as pd


P_WAVE_VELOCITY_KMS = 6.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle surface distance in km (Haversine formula)."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def p_wave_travel_time_s(surface_km: float, depth_km: float,
                          velocity_kms: float = P_WAVE_VELOCITY_KMS) -> float:
    """P-wave travel time in seconds (straight-ray, homogeneous half-space)."""
    slant_km = math.sqrt(surface_km ** 2 + depth_km ** 2)
    return slant_km / velocity_kms


def compute_travel_times(df: pd.DataFrame, station_lat: float,
                          station_lon: float) -> pd.Series:
    """Return a Series of integer travel times (s) for each event in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Catalog with columns ``latitude``, ``longitude``, ``depth_km``.
    station_lat, station_lon : float
        Receiver coordinates.
    """
    times = []
    for _, row in df.iterrows():
        try:
            dist = haversine_km(row["latitude"], row["longitude"],
                                station_lat, station_lon)
            t = p_wave_travel_time_s(dist, row["depth_km"])
            times.append(round(t))
        except Exception:
            times.append(np.nan)
    return pd.Series(times, index=df.index)


def correct_catalog(input_csv: str, output_csv: str,
                    station_lat: float, station_lon: float) -> pd.DataFrame:
    """Load *input_csv*, add ``corrected_time`` column, save to *output_csv*.

    Returns the corrected DataFrame.
    """
    df = pd.read_csv(input_csv)
    df["origintime"] = pd.to_datetime(df["origintime"])

    travel_times = compute_travel_times(df, station_lat, station_lon)
    df["corrected_time"] = df["origintime"] + pd.to_timedelta(travel_times, unit="s")

    df.to_csv(output_csv, index=False)
    return df
