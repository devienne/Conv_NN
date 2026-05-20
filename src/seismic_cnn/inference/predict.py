"""Load a trained model and classify new seismic windows."""

import numpy as np
import tensorflow as tf

from seismic_cnn.datasets import N_SAMPLES, N_CHANNELS


def load_model(model_path: str):
    """Load a saved Keras model from *model_path*."""
    return tf.keras.models.load_model(model_path)


def classify_window(model, waveform: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Classify a single seismic window.

    Parameters
    ----------
    model : keras.Model
    waveform : np.ndarray, shape (N_SAMPLES, N_CHANNELS) or (N_CHANNELS, N_SAMPLES)
        Raw or normalised window. Transposed automatically if channels-first.
    threshold : float
        Decision boundary (default 0.5).

    Returns
    -------
    dict with keys ``label`` (str), ``probability`` (float).
    """
    arr = np.array(waveform, dtype=np.float32)

    # Accept channels-first input and transpose
    if arr.shape == (N_CHANNELS, N_SAMPLES):
        arr = arr.T

    if arr.shape != (N_SAMPLES, N_CHANNELS):
        raise ValueError(
            f"Expected shape ({N_SAMPLES}, {N_CHANNELS}), got {arr.shape}"
        )

    earthquake_prob = float(model.predict(arr[np.newaxis], verbose=0)[0, 0])
    label = "earthquake" if earthquake_prob >= threshold else "noise"
    return {
        "label":                label,
        "probability":          earthquake_prob if label == "earthquake" else 1.0 - earthquake_prob,
        "earthquake_probability": earthquake_prob,
    }
