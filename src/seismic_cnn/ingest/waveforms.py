"""Download daily MiniSEED files from the IRIS FDSNWS dataselect service."""

import os
import time
import datetime
import requests
import pandas as pd

STATION_URL_TMPL = (
    "http://service.iris.edu/fdsnws/station/1/query"
    "?minlatitude={lat_min}&maxlatitude={lat_max}"
    "&minlongitude={lon_min}&maxlongitude={lon_max}"
    "&format=text"
)
DATASELECT_URL = "http://service.iris.edu/fdsnws/dataselect/1/query"


def list_stations(lat_min, lat_max, lon_min, lon_max):
    """Return a list of station dicts inside the bounding box from IRIS FDSNWS."""
    url = STATION_URL_TMPL.format(
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max,
    )
    response = requests.get(url, timeout=100)
    response.raise_for_status()

    stations = []
    for line in response.text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("|")
        stations.append({
            "network":   parts[0],
            "station":   parts[1],
            "latitude":  float(parts[2]),
            "longitude": float(parts[3]),
            "elevation": float(parts[4]),
            "site_name": parts[5],
        })
    return stations


def download(station, channel, start: datetime.datetime, end: datetime.datetime,
             output_dir: str, max_retries: int = 5, retry_delay: float = 10.0):
    """Download daily MiniSEED files for *station* over [start, end].

    Skips dates that already exist in *output_dir*.
    Retries up to *max_retries* times on network errors, waiting *retry_delay*
    seconds between attempts (covers transient DNS failures and timeouts).
    """
    os.makedirs(output_dir, exist_ok=True)
    daterange = pd.date_range(start, end)
    total = len(daterange)
    n_failed = 0

    for idx, date in enumerate(daterange, 1):
        date_str = date.strftime("%Y-%m-%d")
        filename = f"{station}_{date_str}.mseed"
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            print(f"[{idx}/{total}] {filename} already exists, skipping.")
            continue

        print(f"[{idx}/{total}] {filename} ...", end=" ", flush=True)

        url = (
            f"{DATASELECT_URL}"
            f"?sta={station}&cha={channel}"
            f"&starttime={date_str}T00:00:00"
            f"&endtime={date_str}T23:59:59"
            f"&format=miniseed&nodata=404"
        )

        response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url, allow_redirects=True, timeout=60)
                break
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    print(f"\n    attempt {attempt} failed ({type(e).__name__}), "
                          f"retrying in {retry_delay}s ...", end=" ", flush=True)
                    time.sleep(retry_delay)
                else:
                    print(f"FAILED after {max_retries} attempts ({type(e).__name__})")
                    n_failed += 1
                    response = None

        if response is None:
            continue

        if not response.ok or response.text[:5] == "Error":
            print("skipped (no data)")
            continue

        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"saved ({len(response.content) / 1024:.1f} KB)")

    if n_failed:
        print(f"\nWarning: {n_failed}/{total} file(s) failed after {max_retries} retries. "
              f"Re-run the pipeline to retry — existing files will be skipped.")
