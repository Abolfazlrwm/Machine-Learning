"""/v1/sms — spam vs ham classification."""
from __future__ import annotations

import time

from fastapi import APIRouter, Request

from ..schemas import SmsRequest, SmsResponse

router = APIRouter(prefix="/sms", tags=["sms"])


@router.post("/classify", response_model=SmsResponse)
def classify(payload: SmsRequest, request: Request):
    registry = request.app.state.registry
    started = time.perf_counter()
    result = registry.sms.classify(payload.text)
    return SmsResponse(
        label=result["label"],
        probability_spam=result["probability_spam"],
        model_version=result["model_version"],
        took_ms=int((time.perf_counter() - started) * 1000),
    )
