"""Shared exponential-backoff retry helper for live ShipBob mock calls.

Added after repeated, directly-observed `httpx.ReadTimeout`s against the
real (Postman-hosted) mock server during manual testing -- the mock is a
free-tier public endpoint and is genuinely, if infrequently, flaky. This
module is the one place that retry policy lives, used by both
`clients/http.py` (case/shipment/order/invoice/email/reimbursement calls)
and `clients/attachment_guard.py` (evidence photo downloads from Azure blob
storage) so the policy can't drift between the two call sites.

Design decisions:

1. **Only transient failures are retried.** Network-level exceptions
   (`httpx.TimeoutException` -- covers connect/read/write/pool timeouts --
   plus `httpx.ConnectError`/`httpx.NetworkError`) and 5xx responses are
   retried. A 4xx response (404 "not found", 422 "invoice unavailable",
   etc.) is never retried -- those are legitimate, deterministic answers
   from the server, not transient failures, and retrying one would just
   waste time re-asking a question that already has a real answer.
2. **Exponential backoff with jitter**, capped at a max delay, so repeated
   retries don't hammer an already-struggling server and concurrent
   requests don't all retry in lockstep. Delay grows as
   `base_delay * 2**attempt`, capped at `max_delay`, plus up to 25% random
   jitter on top of that capped value.
3. **Result-based retry is opt-in via `should_retry_result`.** Callers that
   only care about exceptions (e.g. `attachment_guard.fetch_attachment`,
   which returns plain bytes on success, not an `httpx.Response`) simply
   omit it. Callers that need to inspect a returned `httpx.Response`'s
   status code (e.g. `clients/http.py`, which still needs to hand a 4xx
   response through to its own `raise_for_status()`/404-to-`NotFoundError`
   logic untouched) pass a predicate.
4. **Settings-driven, not hardcoded**, matching this codebase's standing
   convention (`config.py`) that a deployment-tunable value is always an
   env-overridable `Settings` field, read at call time -- never a bare
   module-level constant, and never snapshotted into a default parameter
   value (which would freeze it at import time and ignore a later
   `monkeypatch.setattr`/real env var change).
5. **On final exhaustion**, the last exception is re-raised, or the last
   (still-bad) result is returned as-is -- callers' existing error handling
   (`raise_for_status()`, 404-to-`NotFoundError` mapping, etc.) still runs
   exactly as it did before retries existed, just after some extra attempts
   first.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Network-level exceptions worth retrying -- all subclasses of
# `httpx.TransportError`, covering connect/read/write/pool timeouts and
# lower-level connection failures. Never includes `httpx.HTTPStatusError`
# (that's a 4xx/5xx *response*, handled via `should_retry_result` instead,
# since plain `httpx.Client.get()`/`.post()` calls don't raise on those
# unless `.raise_for_status()` is called first).
RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.NetworkError,
)


async def retry_async(
    func: Callable[[], Awaitable[T]],
    *,
    max_retries: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    should_retry_result: Callable[[T], bool] | None = None,
    description: str = "request",
) -> T:
    """Call `func()`, retrying with exponential backoff + jitter on transient
    failure.

    Args:
        func: zero-argument async callable to invoke (and re-invoke on
            retry) -- e.g. `lambda: client.get(path)`.
        max_retries: max retry attempts after the first (so `max_retries=3`
            means up to 4 total attempts). Defaults to
            `settings.http_max_retries`, read at call time.
        base_delay: seconds to wait before the first retry; doubles each
            subsequent attempt. Defaults to `settings.http_retry_base_delay_seconds`.
        max_delay: hard ceiling on the backoff delay (before jitter).
            Defaults to `settings.http_retry_max_delay_seconds`.
        should_retry_result: optional predicate on a *successful* (no
            exception raised) result -- return `True` to retry anyway (e.g.
            an `httpx.Response` with a 5xx status). Omit for callables whose
            success is unambiguous (e.g. a fully-read attachment download).
        description: short label included in the retry warning log, so a
            real retry happening in production is diagnosable from logs
            alone (e.g. "GET /cases/CASE-1001").

    Returns:
        The first result that isn't flagged for retry, or the final result
        after exhausting all retries.

    Raises:
        Whatever `func()` raised on the final attempt, if it was still
        raising a retryable exception when retries ran out. A
        non-retryable exception propagates immediately, on the first
        attempt, with no retry at all.
    """
    from claimpilot.config import settings

    retries = settings.http_max_retries if max_retries is None else max_retries
    delay = settings.http_retry_base_delay_seconds if base_delay is None else base_delay
    ceiling = settings.http_retry_max_delay_seconds if max_delay is None else max_delay

    attempt = 0
    while True:
        try:
            result = await func()
        except RETRYABLE_EXCEPTIONS as exc:
            if attempt >= retries:
                raise
            wait = _backoff_delay(delay, ceiling, attempt)
            logger.warning(
                "%s failed (%s: %s), retrying in %.2fs (attempt %d/%d)",
                description,
                type(exc).__name__,
                exc,
                wait,
                attempt + 1,
                retries,
            )
            await asyncio.sleep(wait)
            attempt += 1
            continue

        if should_retry_result is not None and should_retry_result(result) and attempt < retries:
            wait = _backoff_delay(delay, ceiling, attempt)
            logger.warning(
                "%s returned a retryable result, retrying in %.2fs (attempt %d/%d)",
                description,
                wait,
                attempt + 1,
                retries,
            )
            await asyncio.sleep(wait)
            attempt += 1
            continue

        return result


def _backoff_delay(base_delay: float, max_delay: float, attempt: int) -> float:
    """Exponential backoff (`base_delay * 2**attempt`), capped at
    `max_delay`, plus up to 25% random jitter on top of the capped value --
    module docstring points 2/4.
    """
    capped = min(base_delay * (2**attempt), max_delay)
    return capped + random.uniform(0, capped * 0.25)
