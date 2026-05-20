"""Model routes: prediction and experiment results."""

import os
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from seismic_cnn.config import load_config
from seismic_cnn.inference.predict import load_model, classify_window
from seismic_cnn.db.database import list_model_runs

router = APIRouter()

_model = None


def _get_model():
    global _model
    if _model is None:
        cfg = load_config()
        path = os.path.join(cfg.models_dir, "best_model.keras")
        if not os.path.exists(path):
            raise HTTPException(status_code=503, detail="No trained model found.")
        _model = load_model(path)
    return _model


def _load_example_waveform() -> list[list[float]]:
    """Read a random earthquake window CSV from the test set for Swagger example data."""
    import random
    try:
        cfg = load_config()
        eq_dir = os.path.join(cfg.windows_dir, "test", "earthquake")
        files  = [f for f in os.listdir(eq_dir)
                  if f.endswith(".csv") and "_aug" not in f]
        if files:
            chosen = random.choice(files)
            print(f"[predict] Swagger example loaded from: {chosen}")
            data = np.loadtxt(os.path.join(eq_dir, chosen),
                              delimiter=",", dtype=np.float64)
            return data.tolist()          # shape (N_CHANNELS, N_SAMPLES)
    except Exception:
        pass
    return [[0.0] * 2000 for _ in range(3)]   # fallback: silence


_EXAMPLE_WAVEFORM = _load_example_waveform()


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "waveform":  _EXAMPLE_WAVEFORM,
            "threshold": 0.5,
        }]
    })

    waveform:  list[list[float]]
    threshold: float = 0.5


@router.post("/predict")
def predict(req: PredictRequest):
    """Classify a single 20-second seismic window."""
    model = _get_model()
    arr = np.array(req.waveform, dtype=np.float32)
    result = classify_window(model, arr, threshold=req.threshold)
    return result


@router.get("/runs")
def get_runs():
    """List all training runs from the SQLite model_runs table."""
    runs = list_model_runs()
    return {"count": len(runs), "runs": runs}


