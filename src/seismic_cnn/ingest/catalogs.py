"""Download and filter the OGS earthquake catalog."""

import requests
import pandas as pd
from io import StringIO

OGS_URL_TMPL = (
    "https://ogsweb.ou.edu/eq_catalog/earthquake"
    "?start={start}&end={end}&mag=0&format=csv"
)


def download_catalog(start: str, end: str) -> pd.DataFrame:
    """Download the OGS earthquake catalog and return it as a DataFrame.

    Parameters
    ----------
    start, end : str
        Time range in OGS format ``YYYYMMDDHHMI`` (e.g. ``'201402150000'``).
    """
    url = OGS_URL_TMPL.format(start=start, end=end)
    try:
        response = requests.get(url, timeout=60)
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"Catalog download failed: {e}")

    response.raise_for_status()

    if not response.text.strip() or response.text.strip().lower().startswith("error"):
        raise SystemExit(f"API returned unexpected response: {response.text[:200]}")

    return pd.read_csv(StringIO(response.text))


def filter_by_area(df: pd.DataFrame, lat_min, lat_max, lon_min, lon_max) -> pd.DataFrame:
    """Keep only earthquakes within the study-area bounding box."""
    mask = (
        (df["latitude"]  >= lat_min) & (df["latitude"]  <= lat_max) &
        (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max)
    )
    return df[mask].reset_index(drop=True)
