"""Streamlit dashboard for the SeismicCNN project.

Run with:
    streamlit run src/seismic_cnn/dashboard/app.py
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from seismic_cnn.config import load_config

cfg = load_config()

SAMPLING_RATE = 100.0
CH_LABELS     = ["HHE", "HHN", "HHZ"]

st.set_page_config(page_title="SeismicCNN Dashboard", layout="wide")
st.title("SeismicCNN — ConvNetQuake Reproduction")
st.caption("Perol, Gharbi & Denolle (2018)")

page = st.sidebar.selectbox("Page", [
    "Earthquake Catalog",
    "Raw Waveform Explorer",
    "Window Viewer",
    "Dataset Overview",
    "Model Performance",
    "Live Prediction",
])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def ensure_2d(arr: np.ndarray) -> np.ndarray:
    return arr[np.newaxis, :] if arr.ndim == 1 else arr


def gauss(data: np.ndarray, sigma: float) -> np.ndarray:
    return np.clip(data + np.random.normal(0, sigma, data.shape), -1.0, 1.0)


def plot_channels(data: np.ndarray, title: str, color: str = "steelblue",
                  time_unit: str = "s") -> plt.Figure:
    """Plot (n_channels, n_samples) array, one subplot per channel."""
    n_ch = data.shape[0]
    n_t  = data.shape[1]
    t    = np.arange(n_t) / SAMPLING_RATE
    if time_unit == "min":
        t /= 60

    fig, axes = plt.subplots(n_ch, 1, figsize=(14, 2.5 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]
    for ax, ch_idx in zip(axes, range(n_ch)):
        ax.plot(t, data[ch_idx], lw=0.4, color=color)
        ax.set_ylabel(CH_LABELS[ch_idx] if ch_idx < len(CH_LABELS) else f"Ch{ch_idx}")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (min)" if time_unit == "min" else "Time (s)")
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_eq_vs_noise(eq_data: np.ndarray, noise_data: np.ndarray) -> plt.Figure:
    """Side-by-side channel plot: earthquake (left) vs noise (right)."""
    n_ch = min(eq_data.shape[0], noise_data.shape[0])
    t    = np.arange(eq_data.shape[1]) / SAMPLING_RATE

    fig, axes = plt.subplots(n_ch, 2, figsize=(14, 2.8 * n_ch), sharex=True)
    if n_ch == 1:
        axes = axes[np.newaxis, :]

    for ch in range(n_ch):
        axes[ch, 0].plot(t, eq_data[ch],    lw=0.8, color="crimson")
        axes[ch, 0].set_ylabel(CH_LABELS[ch] if ch < len(CH_LABELS) else f"Ch{ch}")
        axes[ch, 0].grid(alpha=0.3)
        axes[ch, 1].plot(t, noise_data[ch], lw=0.8, color="steelblue")
        axes[ch, 1].grid(alpha=0.3)

    axes[0, 0].set_title("Earthquake window", fontweight="bold", color="crimson")
    axes[0, 1].set_title("Noise window",      fontweight="bold", color="steelblue")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("20-second windows: Earthquake vs Noise", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_progression(rows_data, row_labels, title, colors=None) -> plt.Figure:
    """Grid: rows = variants, columns = channels. Mirrors notebook helper."""
    n_rows = len(rows_data)
    n_ch   = rows_data[0].shape[0]
    t      = np.arange(rows_data[0].shape[1]) / SAMPLING_RATE
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 0.9, n_rows))

    fig, axes = plt.subplots(n_rows, n_ch,
                             figsize=(5 * n_ch, 2.2 * n_rows), sharex=True)
    if n_rows == 1: axes = axes[np.newaxis, :]
    if n_ch   == 1: axes = axes[:, np.newaxis]

    for r, (data, label, color) in enumerate(zip(rows_data, row_labels, colors)):
        for c in range(n_ch):
            axes[r, c].plot(t, data[c], lw=0.8, color=color)
            axes[r, c].grid(alpha=0.3)
            if r == 0:
                axes[r, c].set_title(CH_LABELS[c] if c < len(CH_LABELS) else f"Ch{c}",
                                     fontweight="bold")
            if c == n_ch - 1:
                axes[r, c].set_ylabel(label, fontsize=8, fontweight="bold",
                                      color=color, rotation=0,
                                      labelpad=5, ha="left", va="center")
                axes[r, c].yaxis.set_label_position("right")
            if r == n_rows - 1:
                axes[r, c].set_xlabel("Time (s)")
    fig.suptitle(title, fontweight="bold", fontsize=12)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_catalog(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["corrected_time"])


@st.cache_data
def compute_kmeans(coords_array: np.ndarray, k_max: int = 10):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    k_range    = range(2, k_max)
    inertias, sil_scores = [], []
    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=420, n_init=10)
        lbl = km.fit_predict(coords_array)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(coords_array, lbl))

    best_k  = list(k_range)[int(np.argmax(sil_scores))]
    km_best = KMeans(n_clusters=best_k, random_state=420, n_init=10)
    labels  = km_best.fit_predict(coords_array)
    centers = km_best.cluster_centers_
    return list(k_range), inertias, sil_scores, best_k, labels, centers, km_best


# ---------------------------------------------------------------------------
# Page: Earthquake Catalog
# ---------------------------------------------------------------------------

if page == "Earthquake Catalog":
    st.header("Earthquake Catalog")

    corrected_path = os.path.join(cfg.interim_dir, "historical_corrected.csv")
    if not os.path.exists(corrected_path):
        st.warning("Corrected catalog not found. Run the pipeline first.")
        st.stop()

    catalog     = load_catalog(corrected_path)
    STATION_LAT = cfg.station_lat
    STATION_LON = cfg.station_lon

    st.write(f"**{len(catalog)}** events  ·  study area 35.7–36.0°N, 97.2–97.6°W  ·  Station OK027")

    # --- Study area map ---
    st.subheader("Study area")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(catalog["longitude"], catalog["latitude"],
               c="crimson", s=45, alpha=0.75, edgecolors="darkred", linewidths=0.3,
               zorder=3, label=f"Earthquakes  (n = {len(catalog)})")
    ax.scatter(STATION_LON, STATION_LAT,
               marker="*", s=380, c="gold", edgecolors="black", linewidths=0.6,
               zorder=4, label="OK027 station")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Study area — earthquake catalog and seismic station\nFeb 2014 – Feb 2015",
                 fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- K-means cluster selection ---
    st.subheader("K-means cluster selection")
    coords = catalog[["longitude", "latitude"]].values
    k_range, inertias, sil_scores, best_k, labels, centers, km_best = compute_kmeans(coords)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(k_range, inertias,  "o-", color="steelblue", lw=1.5)
    axes[0].set_xlabel("Number of clusters  k")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow method", fontweight="bold")
    axes[0].grid(alpha=0.3)
    axes[1].plot(k_range, sil_scores, "o-", color="crimson", lw=1.5)
    axes[1].set_xlabel("Number of clusters  k")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette score  (higher = better)", fontweight="bold")
    axes[1].axvline(best_k, linestyle="--", color="gray", label=f"Best  k = {best_k}")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.suptitle("K-means cluster selection", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(f"Best k by silhouette score: **{best_k}**")

    # --- Clustered scatter ---
    st.subheader(f"Earthquake clusters  (k = {best_k})")
    pad   = 0.05
    lon_g = np.linspace(coords[:, 0].min() - pad, coords[:, 0].max() + pad, 300)
    lat_g = np.linspace(coords[:, 1].min() - pad, coords[:, 1].max() + pad, 300)
    xx, yy = np.meshgrid(lon_g, lat_g)
    zz     = km_best.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, zz, levels=np.arange(-0.5, best_k + 0.5),
                cmap="Pastel1", alpha=0.45)
    ax.contour(xx, yy, zz,  levels=np.arange(-0.5, best_k + 0.5),
               colors="dimgray", linewidths=1.0, linestyles="--")
    tab_colors = plt.cm.tab10.colors
    for k in range(best_k):
        mask = labels == k
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[tab_colors[k]], s=50, edgecolors="k", linewidths=0.3,
                   label=f"Cluster {k + 1}  (n = {int(mask.sum())})")
    ax.scatter(centers[:, 0], centers[:, 1],
               marker="s", s=60, c="white", edgecolors="black", linewidths=1.2,
               label="Centroids")
    ax.scatter(STATION_LON, STATION_LAT,
               marker="*", s=320, c="gold", edgecolors="black", linewidths=0.6,
               label="OK027 station")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(f"K-means clustering  (k = {best_k})\nFeb 2014 – Feb 2015",
                 fontweight="bold")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page: Raw Waveform Explorer
# ---------------------------------------------------------------------------

elif page == "Raw Waveform Explorer":
    from obspy import read as obspy_read

    st.header("Raw Waveform Explorer")

    mseed_dir   = cfg.raw_mseed_dir
    mseed_files = sorted(f for f in os.listdir(mseed_dir) if f.endswith(".mseed"))

    if not mseed_files:
        st.warning("No .mseed files found. Run the pipeline first.")
        st.stop()

    col1, col2 = st.columns([5, 1])
    with col1:
        chosen_file = st.selectbox("Select a day", mseed_files)
    with col2:
        st.write("")
        st.write("")
        if st.button("Random"):
            chosen_file = random.choice(mseed_files)

    stream = obspy_read(os.path.join(mseed_dir, chosen_file))
    st.caption(f"File: `{chosen_file}`  ·  {len(stream)} trace(s)  ·  "
               f"{stream[0].stats.sampling_rate} Hz")

    # --- Section 1: raw waveforms ---
    st.subheader("Raw waveforms — full day")
    fig, axes = plt.subplots(len(stream), 1,
                             figsize=(14, 3 * len(stream)), sharex=True)
    if len(stream) == 1:
        axes = [axes]
    for ax, tr in zip(axes, stream):
        t = np.arange(len(tr.data)) / tr.stats.sampling_rate / 60
        ax.plot(t, tr.data, lw=0.3, color="steelblue")
        ax.set_ylabel(tr.stats.channel, fontsize=9)
        ax.set_title(
            f"{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}  |  "
            f"{str(tr.stats.starttime)[:19]}",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (minutes)")
    fig.suptitle(f"Raw seismic data — {chosen_file}", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Section 2: preprocessing steps ---
    st.subheader("Effect of preprocessing  (last channel, first 5 min)")
    tr_raw     = stream[-1].copy()
    tr_norm    = tr_raw.copy();    tr_norm.normalize()
    tr_detrend = tr_norm.copy();   tr_detrend.detrend("linear")
    tr_demean  = tr_detrend.copy(); tr_demean.detrend("demean")

    seg = int(5 * 60 * SAMPLING_RATE)
    steps = [
        (tr_raw,     "Raw",                "steelblue"),
        (tr_norm,    "Normalised",         "darkorange"),
        (tr_detrend, "Detrend linear",     "seagreen"),
        (tr_demean,  "Detrend + Demean",   "crimson"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    for ax, (tr, label, color) in zip(axes, steps):
        data  = tr.data[:seg]
        t_seg = np.arange(len(data)) / SAMPLING_RATE / 60
        ax.plot(t_seg, data, lw=0.5, color=color)
        ax.set_ylabel("Amplitude")
        ax.set_title(label, fontweight="bold", color=color)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (minutes)")
    fig.suptitle(
        f"Preprocessing steps — {tr_raw.stats.channel}  (first 5 min of {chosen_file})",
        fontweight="bold",
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page: Window Viewer
# ---------------------------------------------------------------------------

elif page == "Window Viewer":
    st.header("Window Viewer")

    eq_dir    = os.path.join(cfg.windows_dir, "train", "earthquake")
    noise_dir = os.path.join(cfg.windows_dir, "train", "noise")

    eq_files    = [f for f in os.listdir(eq_dir)    if f.endswith(".csv") and "_aug" not in f]
    noise_files = [f for f in os.listdir(noise_dir) if f.endswith(".csv")]

    if not eq_files or not noise_files:
        st.warning("No window CSV files found. Run the pipeline first.")
        st.stop()

    st.write(f"Earthquake windows (original): **{len(eq_files):,}**  ·  "
             f"Noise windows: **{len(noise_files):,}**")

    if st.button("Refresh — pick new random windows"):
        st.rerun()

    # --- Section 3: EQ vs Noise ---
    st.subheader("Earthquake vs Noise")
    np.random.seed(None)
    eq_data    = ensure_2d(np.loadtxt(os.path.join(eq_dir,    random.choice(eq_files)),
                                      delimiter=","))
    noise_data = ensure_2d(np.loadtxt(os.path.join(noise_dir, random.choice(noise_files)),
                                      delimiter=","))

    fig = plot_eq_vs_noise(eq_data, noise_data)
    st.pyplot(fig)
    plt.close(fig)

    # --- Section 4a: Augmentation overview ---
    st.subheader("Data augmentation — overview")
    orig = ensure_2d(np.loadtxt(os.path.join(eq_dir, eq_files[0]), delimiter=","))

    np.random.seed(0)
    rows   = [orig, gauss(orig, sigma=50e-4)]
    labels = ["Original", "Gaussian noise  (σ = 5×10⁻³)"]
    colors = ["black", "crimson"]
    fig = plot_progression(rows, labels,
                           "Original and Gaussian noise applied once", colors)
    st.pyplot(fig)
    plt.close(fig)

    # --- Section 4b: Gaussian sigma sweep ---
    st.subheader("Gaussian noise — effect of increasing sigma")
    np.random.seed(1)
    sigmas = [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 20e-4]
    rows   = [orig] + [gauss(orig, s) for s in sigmas]
    labels = ["Original"] + [f"σ = {s:.0e}" for s in sigmas]
    colors = ["black"] + list(plt.cm.Reds(np.linspace(0.3, 0.95, len(sigmas))))
    fig = plot_progression(rows, labels,
                           "Gaussian noise — effect of increasing sigma", colors)
    st.pyplot(fig)
    plt.close(fig)

    # --- Model prediction on the displayed EQ window ---
    st.subheader("Model prediction on the displayed earthquake window")
    model_path = os.path.join(cfg.models_dir, "best_model.keras")
    if os.path.exists(model_path):
        from seismic_cnn.inference.predict import load_model, classify_window
        model  = load_model(model_path)
        result = classify_window(model, eq_data)
        label_str = result["label"].upper()
        color_str = "green" if result["label"] == "earthquake" else "red"
        st.markdown(
            f"**Predicted:** :{color_str}[{label_str}]  ·  "
            f"probability of {result['label']}: `{result['probability']:.4f}`  ·  "
            f"earthquake probability: `{result['earthquake_probability']:.4f}`"
        )
    else:
        st.info("No trained model found. Run training first.")


# ---------------------------------------------------------------------------
# Page: Dataset Overview
# ---------------------------------------------------------------------------

elif page == "Dataset Overview":
    from seismic_cnn.db.database import get_windows_summary, get_windows_monthly

    st.header("Dataset Overview")

    summary = get_windows_summary()
    monthly = get_windows_monthly()

    if not summary:
        st.warning("No windows found in the database. Run the pipeline first.")
    else:
        st.subheader("Class balance")
        df_sum = pd.DataFrame(summary)
        df_sum["category"] = (
            df_sum["split"] + " / " + df_sum["label"] + " / "
            + df_sum["is_augmented"].map({0: "original", 1: "augmented"})
        )
        df_sum = df_sum.set_index("category")[["count"]]
        st.bar_chart(df_sum)

        col1, col2, col3, col4 = st.columns(4)
        totals = df_sum["count"]
        for col, key in zip(
            [col1, col2, col3, col4],
            ["train / earthquake / original", "train / noise / original",
             "test / earthquake / original",  "test / noise / original"],
        ):
            col.metric(key, int(totals.get(key, 0)))

        if monthly:
            st.subheader("Windows per month")
            df_mon   = pd.DataFrame(monthly)
            df_pivot = (
                df_mon.pivot(index="month", columns="label", values="count")
                .fillna(0).astype(int)
            )
            st.line_chart(df_pivot)


# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------

elif page == "Model Performance":
    from seismic_cnn.db.database import list_model_runs

    st.header("Model Performance")

    runs = list_model_runs()
    if runs:
        best = runs[0]   # ordered by val_auc desc

        st.subheader("Best run")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC",       f"{best.get('val_auc', 0):.4f}")
        c2.metric("F1",        f"{best.get('val_f1',  0):.4f}")
        c3.metric("Precision", f"{best.get('val_precision', 0):.4f}")
        c4.metric("Recall",    f"{best.get('val_recall',    0):.4f}")

        st.subheader("All training runs")
        df_runs = pd.DataFrame(runs)
        cols_order = ["model_name", "val_auc", "val_f1", "val_accuracy",
                      "val_precision", "val_recall", "val_loss",
                      "epochs_run", "trained_at"]
        df_runs = df_runs[[c for c in cols_order if c in df_runs.columns]]
        st.dataframe(df_runs, use_container_width=True)
    else:
        st.info("No model runs recorded yet. Run training first.")

    fig_path = os.path.join(cfg.figures_dir, "training_history.png")
    if os.path.exists(fig_path):
        st.subheader("Training history")
        st.image(fig_path)


# ---------------------------------------------------------------------------
# Page: Live Prediction
# ---------------------------------------------------------------------------

elif page == "Live Prediction":
    st.header("Live Prediction")
    st.info("Upload a window CSV file (3 rows × 2000 columns — channels as rows).")

    uploaded = st.file_uploader("Upload window CSV", type="csv")
    if uploaded:
        data = np.loadtxt(uploaded, delimiter=",")
        st.write(f"Loaded shape: `{data.shape}`")

        fig = plot_channels(ensure_2d(data), title="Uploaded window")
        st.pyplot(fig)
        plt.close(fig)

        model_path = os.path.join(cfg.models_dir, "best_model.keras")
        if os.path.exists(model_path):
            from seismic_cnn.inference.predict import load_model, classify_window
            model  = load_model(model_path)
            result = classify_window(model, data)
            color  = "green" if result["label"] == "earthquake" else "red"
            st.markdown(
                f"**Predicted:** :{color}[{result['label'].upper()}]  ·  "
                f"probability of {result['label']}: `{result['probability']:.4f}`  ·  "
                f"earthquake probability: `{result['earthquake_probability']:.4f}`"
            )
        else:
            st.error("No trained model found. Run training first.")
