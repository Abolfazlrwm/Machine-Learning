"""/v1/stock — next-day movement signal from the persisted walk-forward models."""
from __future__ import annotations

import time

from fastapi import APIRouter, Query, Request

from ..schemas import StockSignalResponse

router = APIRouter(prefix="/stock", tags=["stock"])


@router.get("/signal", response_model=StockSignalResponse)
def signal(
    request: Request,
    ticker: str = Query(min_length=1, max_length=10, examples=["AAPL"]),
):
    registry = request.app.state.registry
    started = time.perf_counter()
    result = registry.stock.signal(ticker)
    return StockSignalResponse(
        ticker=result["ticker"],
        signal=result["signal"],
        probability_up=result["probability_up"],
        model=result["model"],
        as_of=result["as_of"],
        took_ms=int((time.perf_counter() - started) * 1000),
    )
