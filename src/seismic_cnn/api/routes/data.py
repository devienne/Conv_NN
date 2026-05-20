"""Data routes: earthquake events and window registry."""

from fastapi import APIRouter, Query
from seismic_cnn.db.database import query_events, get_windows_summary, get_windows_monthly

router = APIRouter()


@router.get("/events")
def get_events(
    start: str = Query(..., description="ISO date, e.g. 2014-03-01"),
    end:   str = Query(..., description="ISO date, e.g. 2014-04-01"),
):
    """Return earthquake events from the SQLite catalog in [start, end]."""
    events = query_events(start, end)
    return {"count": len(events), "events": events}

@router.get("/windows/summary")
def windows_summary():
    """Window counts grouped by split, label, and augmented flag."""
    return {"data": get_windows_summary()}


@router.get("/windows/monthly")
def windows_monthly():
    """Window counts per calendar month per label (train + test combined)."""
    return {"data": get_windows_monthly()}
