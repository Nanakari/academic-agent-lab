"""Shared HTTP helpers for external evidence providers."""

from __future__ import annotations

import urllib.request
from typing import Protocol


class HttpResponse(Protocol):
    def read(self) -> bytes: ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...


def open_url(request: urllib.request.Request, timeout: float) -> HttpResponse:
    """Small injectable seam around urllib for deterministic tests."""
    return urllib.request.urlopen(request, timeout=timeout)
