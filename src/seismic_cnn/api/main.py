"""FastAPI application entry point.

Run with:
    uvicorn seismic_cnn.api.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs available at http://localhost:8000/docs
"""

from fastapi import FastAPI
from seismic_cnn.api.routes import data, model

app = FastAPI(
    title="SeismicCNN API",
    description="Earthquake detection API based on ConvNetQuake (Perol et al. 2018)",
    version="0.1.0",
)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


app.include_router(data.router,  prefix="/data",  tags=["data"])
app.include_router(model.router, prefix="/model", tags=["model"])
