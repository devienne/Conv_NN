"""End-to-end data pipeline: ingest -> travel-time correction -> windowing -> augmentation.

Usage
-----
python scripts/run_pipeline.py
python scripts/run_pipeline.py --config configs/experiment_convnetquake.yaml
"""

import argparse
import datetime
import os
from collections import defaultdict
from multiprocessing import cpu_count, Pool

from seismic_cnn.config import load_config
from seismic_cnn.ingest import waveforms as wf
from seismic_cnn.ingest import catalogs as cat
from seismic_cnn.ingest import travel_times as tt
from seismic_cnn.preprocessing import compute_monthly_stats
from seismic_cnn.windowing import (
    load_earthquake_times, load_benz_times, merge_catalogs, process_file
)
from seismic_cnn.augmentation import augment_split
from seismic_cnn.db.database import (
    init_db, insert_events, get_monthly_stats, upsert_monthly_stats, insert_windows
)

import pandas as pd


def main(cfg):
    init_db()

    # Ensure all data directories exist before any step tries to write into them
    for d in [
        cfg.raw_mseed_dir,
        cfg.raw_catalogs_dir,
        cfg.interim_dir,
        cfg.windows_dir,
        os.path.join(cfg.windows_dir, "train", "earthquake"),
        os.path.join(cfg.windows_dir, "train", "noise"),
        os.path.join(cfg.windows_dir, "test",  "earthquake"),
        os.path.join(cfg.windows_dir, "test",  "noise"),
        cfg.models_dir,
        cfg.figures_dir,
        cfg.comparisons_dir,
    ]:
        os.makedirs(d, exist_ok=True)

    # # ------------------------------------------------------------------
    # # Step 1 — download MiniSEED waveforms
    # # ------------------------------------------------------------------
    # print("=== Step 1: Download waveforms ===")
    # stations = wf.list_stations(cfg.lat_min, cfg.lat_max, cfg.lon_min, cfg.lon_max)
    # print(f"Found {len(stations)} station(s) in study area.")
    # names = [s["station"] for s in stations]
    # station = cfg.station if cfg.station in names else (names[0] if names else None)
    # if not station:
    #     raise SystemExit("No stations found in study area.")
    # wf.download(
    #     station=station,
    #     channel=cfg.channel,
    #     start=datetime.datetime.fromisoformat(cfg.start_date),
    #     end=datetime.datetime.fromisoformat(cfg.end_date),
    #     output_dir=cfg.raw_mseed_dir,
    # )

    # # ------------------------------------------------------------------
    # # Step 2 — download OGS earthquake catalog
    # # ------------------------------------------------------------------
    # historical_csv = os.path.join(cfg.raw_catalogs_dir, "historical.csv")
    # if os.path.exists(historical_csv):
    #     print(f"\n=== Step 2: Skipped (catalog already exists: {historical_csv}) ===")
    # else:
    #     print("\n=== Step 2: Download OGS catalog ===")
    #     start_ogs = cfg.start_date.replace("-", "")[:8] + "0000"
    #     end_ogs   = cfg.end_date.replace("-", "")[:8] + "2359"
    #     df_cat = cat.download_catalog(start_ogs, end_ogs)
    #     df_cat = cat.filter_by_area(df_cat, cfg.lat_min, cfg.lat_max, cfg.lon_min, cfg.lon_max)
    #     df_cat.to_csv(historical_csv, index=False)
    #     print(f"Saved {len(df_cat)} events -> {historical_csv}")
    #     insert_events([
    #         {"origin_time": str(r.get("origintime", ""))[:19],
    #          "magnitude":   r.get("magnitude"),
    #          "source":      "historical",
    #          "latitude":    r.get("latitude"),
    #          "longitude":   r.get("longitude")}
    #         for r in df_cat.to_dict("records")
    #     ])

    # ------------------------------------------------------------------
    # Step 3 — correct P-wave travel times
    # ------------------------------------------------------------------
    corrected_csv = os.path.join(cfg.interim_dir, "historical_corrected.csv")
    if os.path.exists(corrected_csv):
        print(f"\n=== Step 3: Skipped (corrected catalog already exists: {corrected_csv}) ===")
    else:
        print("\n=== Step 3: Correct travel times ===")
        tt.correct_catalog(historical_csv, corrected_csv, cfg.station_lat, cfg.station_lon)
        print(f"Saved corrected catalog -> {corrected_csv}")

    # ------------------------------------------------------------------
    # Step 4 — window, label, normalise
    # ------------------------------------------------------------------
    print("\n=== Step 4: Windowing and labeling ===")
    benz_csv = os.path.join(cfg.raw_catalogs_dir, "Benz_catalog.csv")
    eq_times_hist = load_earthquake_times(corrected_csv)
    eq_times_benz = load_benz_times(benz_csv) if os.path.exists(benz_csv) else pd.Series([], dtype="datetime64[ns]")
    eq_times = merge_catalogs(eq_times_hist, eq_times_benz)
    print(f"Loaded {len(eq_times)} earthquake times "
          f"({len(eq_times_hist)} historical + {len(eq_times_benz)} Benz)")

    # Persist Benz events to SQLite (historical events inserted in step 2)
    if os.path.exists(benz_csv):
        df_benz = pd.read_csv(benz_csv)
        insert_events([
            {"origin_time": str(r.get("origintime", ""))[:19],
             "magnitude":   r.get("Magnitude"),
             "source":      "benz",
             "latitude":    None,
             "longitude":   None}
            for r in df_benz.to_dict("records")
        ])

    mseed_files = sorted(
        os.path.join(cfg.raw_mseed_dir, f)
        for f in os.listdir(cfg.raw_mseed_dir) if f.endswith(".mseed")
    )
    if not mseed_files:
        raise SystemExit(f"No .mseed files found in {cfg.raw_mseed_dir}")

    split_idx  = int(len(mseed_files) * cfg.train_ratio)
    split_date = pd.Timestamp(
        os.path.basename(mseed_files[split_idx]).split("_", 1)[1].replace(".mseed", ""))
    print(f"Train/test split: files up to {split_date.date()} -> train ({cfg.train_ratio:.0%})")

    dirs = {
        ("train", "earthquake"): os.path.join(cfg.windows_dir, "train", "earthquake"),
        ("train", "noise"):      os.path.join(cfg.windows_dir, "train", "noise"),
        ("test",  "earthquake"): os.path.join(cfg.windows_dir, "test",  "earthquake"),
        ("test",  "noise"):      os.path.join(cfg.windows_dir, "test",  "noise"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    month_groups = defaultdict(list)
    for fp in mseed_files:
        date_str = os.path.basename(fp).split("_", 1)[1].replace(".mseed", "")
        d = pd.Timestamp(date_str)
        month_groups[(d.year, d.month)].append(fp)

    n_workers = max(1, cpu_count() - 3)
    print(f"Using {n_workers} worker process(es)\n")

    total_eq = total_noise = 0
    for (year, month), month_files in sorted(month_groups.items()):
        ym = f"{year}-{month:02d}"
        station_name = os.path.basename(month_files[0]).split("_")[0]

        # Pass 1 — monthly stats: use DB cache if available, else compute and store
        cached = get_monthly_stats(station_name, year, month)
        if cached:
            monthly_stats = cached
            print(f"=== {ym} ({len(month_files)} files) — using cached monthly stats ===")
            for ch, s in sorted(monthly_stats.items()):
                print(f"  {ch}: mean={s['mean']:+.6f}  peak={s['peak']:.6f}")
        else:
            print(f"=== {ym} ({len(month_files)} files) — computing monthly stats ===")
            monthly_stats = compute_monthly_stats(month_files)
            for ch, s in sorted(monthly_stats.items()):
                print(f"  {ch}: mean={s['mean']:+.6f}  peak={s['peak']:.6f}")
                upsert_monthly_stats(station_name, year, month, ch, s["mean"], s["peak"])

        # Pass 2 — windowing: parallel, collect window records for DB
        args = [(f, eq_times, split_date, monthly_stats, dirs,
                 cfg.window_s, cfg.max_noise_per_file) for f in month_files]
        month_eq = month_noise = 0
        with Pool(n_workers) as pool:
            for i, (filepath, n_eq, n_noise, err, window_records) in enumerate(
                pool.imap_unordered(process_file, args), 1
            ):
                fname = os.path.basename(filepath)
                if err:
                    print(f"  [{i}/{len(month_files)}] {fname}  ERROR: {err}")
                else:
                    print(f"  [{i}/{len(month_files)}] {fname}  eq={n_eq}  noise={n_noise}",
                          flush=True)
                    insert_windows(window_records)
                month_eq    += n_eq
                month_noise += n_noise
        print(f"  {ym} subtotal — eq={month_eq}  noise={month_noise}\n")
        total_eq    += month_eq
        total_noise += month_noise

    print(f"Windowing done. Earthquake windows: {total_eq}  Noise windows: {total_noise}")

    # ------------------------------------------------------------------
    # Step 5 — augmentation
    # ------------------------------------------------------------------
    print("\n=== Step 5: Augmentation (train split only) ===")
    augment_split(
        eq_dir=dirs[("train", "earthquake")],
        noise_dir=dirs[("train", "noise")],
        split_name="train",
        n_workers=n_workers,
    )

    print("\nPipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to experiment YAML")
    parser.add_argument("--local",  default=None, help="Path to local YAML")
    args = parser.parse_args()
    main(load_config(args.config, args.local))
