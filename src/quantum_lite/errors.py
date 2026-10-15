"""Exception types. AppError carries the HTTP status and the `{error, detail}` body."""
from __future__ import annotations

import asyncio


class AppError(Exception):
    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


class ConfigError(Exception):
    """Invalid environment configuration."""


class ProviderError(Exception):
    """A provider failed to produce a measurement (job error, cancelled, bad result)."""


def provider_failure(exc: Exception) -> AppError:
    """A provider failure as the 502 the API returns, with the real text."""
    return AppError(502, "provider_error", f"{type(exc).__name__}: {exc}")


async def off_loop(fn, *args):
    """Run blocking provider code off the loop; failures surface as 502 with the real text."""
    try:
        return await asyncio.to_thread(fn, *args)
    except AppError:
        raise
    except Exception as exc:
        raise provider_failure(exc) from exc
