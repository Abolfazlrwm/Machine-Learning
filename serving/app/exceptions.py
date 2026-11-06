"""Typed API errors with a stable machine-readable shape.

Every error response looks like::

    {"error": {"code": "SERVICE_UNAVAILABLE", "message": "...", "hint": "..."}}

so clients can branch on ``code`` instead of parsing free-form text.
"""
from __future__ import annotations


class APIError(Exception):
    """Base class: HTTP status + machine readable code + optional hint."""

    status_code = 500
    code = "INTERNAL_ERROR"

    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint


class ServiceUnavailable(APIError):
    """A backing service has no local artifacts (model / index / data)."""

    status_code = 503
    code = "SERVICE_UNAVAILABLE"


class UpstreamError(APIError):
    """A backing dependency (e.g. the LLM provider) failed at request time."""

    status_code = 502
    code = "UPSTREAM_ERROR"


class NotFound(APIError):
    """Requested entity (e.g. ticker) has no artifacts."""

    status_code = 404
    code = "NOT_FOUND"
