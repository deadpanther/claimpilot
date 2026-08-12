"""Tests for `clients/retry.py`'s exponential-backoff retry helper.

`asyncio.sleep` is monkeypatched to a no-op recorder throughout -- these
tests assert on retry *behavior* (attempt counts, which exceptions/results
trigger a retry, backoff growth) in milliseconds, not by actually waiting
out real delays.
"""

from __future__ import annotations

import httpx
import pytest

from claimpilot.clients.retry import retry_async
from claimpilot.config import settings


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Record every `asyncio.sleep` call (delay only) instead of actually
    sleeping, so this whole test file runs near-instantly regardless of how
    many retries/backoff attempts a test drives.
    """
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("claimpilot.clients.retry.asyncio.sleep", fake_sleep)
    return sleeps


# --- success paths -----------------------------------------------------------


async def test_succeeds_first_try_with_no_retries_needed(no_real_sleep):
    calls = 0

    async def func():
        nonlocal calls
        calls += 1
        return "ok"

    result = await retry_async(func)

    assert result == "ok"
    assert calls == 1
    assert no_real_sleep == []  # never slept -- no retry needed at all


async def test_retries_on_timeout_then_succeeds(no_real_sleep):
    calls = 0

    async def func():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ReadTimeout("simulated timeout")
        return "ok"

    result = await retry_async(func, max_retries=5, base_delay=0.01, max_delay=1.0)

    assert result == "ok"
    assert calls == 3  # failed twice, succeeded on the 3rd attempt
    assert len(no_real_sleep) == 2  # slept before each of the 2 retries


async def test_exhausts_retries_and_reraises_final_exception(no_real_sleep):
    calls = 0

    async def func():
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("simulated connection failure")

    with pytest.raises(httpx.ConnectError):
        await retry_async(func, max_retries=2, base_delay=0.01, max_delay=1.0)

    assert calls == 3  # initial attempt + 2 retries, then re-raises


async def test_max_retries_zero_means_no_retry_at_all(no_real_sleep):
    calls = 0

    async def func():
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("simulated timeout")

    with pytest.raises(httpx.ReadTimeout):
        await retry_async(func, max_retries=0, base_delay=0.01, max_delay=1.0)

    assert calls == 1
    assert no_real_sleep == []


# --- non-retryable exceptions propagate immediately --------------------------


async def test_non_retryable_exception_propagates_without_retrying(no_real_sleep):
    """A `ValueError` (e.g. attachment_guard's bad-content-type/oversize
    checks) is not a network failure -- must never be retried.
    """
    calls = 0

    async def func():
        nonlocal calls
        calls += 1
        raise ValueError("not a network problem")

    with pytest.raises(ValueError):
        await retry_async(func, max_retries=3, base_delay=0.01, max_delay=1.0)

    assert calls == 1
    assert no_real_sleep == []


async def test_http_status_error_is_not_retried_by_exception_type_alone(no_real_sleep):
    """`httpx.HTTPStatusError` (raised by `response.raise_for_status()`) is
    deliberately NOT in the retryable-exception set -- retrying a definitive
    4xx/5xx status by exception type alone would retry a 404 too. Result-
    based retry (`should_retry_result`) is the intended path for a 5xx
    *response* that hasn't been raised yet -- see the `should_retry_result`
    tests below.
    """
    calls = 0

    async def func():
        nonlocal calls
        calls += 1
        request = httpx.Request("GET", "https://example.test/x")
        response = httpx.Response(500, request=request)
        raise httpx.HTTPStatusError("boom", request=request, response=response)

    with pytest.raises(httpx.HTTPStatusError):
        await retry_async(func, max_retries=3, base_delay=0.01, max_delay=1.0)

    assert calls == 1


# --- should_retry_result (5xx-response retry, clients/http.py's use case) ---


async def test_should_retry_result_retries_5xx_response_then_succeeds(no_real_sleep):
    responses = [500, 503, 200]
    calls = 0

    async def func():
        nonlocal calls
        status = responses[calls]
        calls += 1
        return httpx.Response(status, request=httpx.Request("GET", "https://example.test/x"))

    result = await retry_async(
        func,
        max_retries=5,
        base_delay=0.01,
        max_delay=1.0,
        should_retry_result=lambda r: r.is_server_error,
    )

    assert result.status_code == 200
    assert calls == 3
    assert len(no_real_sleep) == 2


async def test_should_retry_result_never_retries_4xx_response(no_real_sleep):
    """A 404 must be returned immediately, not retried -- callers (e.g.
    `HttpShipBobClient._get`) depend on getting the real 404 back to map it
    to `NotFoundError`.
    """
    calls = 0

    async def func():
        nonlocal calls
        calls += 1
        return httpx.Response(404, request=httpx.Request("GET", "https://example.test/x"))

    result = await retry_async(
        func,
        max_retries=5,
        base_delay=0.01,
        max_delay=1.0,
        should_retry_result=lambda r: r.is_server_error,
    )

    assert result.status_code == 404
    assert calls == 1
    assert no_real_sleep == []


async def test_should_retry_result_returns_final_bad_result_after_exhausting_retries(no_real_sleep):
    """Exhausting retries on a persistently-5xx result returns that result
    as-is (not an exception) -- the caller's own `raise_for_status()` then
    raises normally, same as if retries didn't exist at all.
    """
    calls = 0

    async def func():
        nonlocal calls
        calls += 1
        return httpx.Response(500, request=httpx.Request("GET", "https://example.test/x"))

    result = await retry_async(
        func,
        max_retries=2,
        base_delay=0.01,
        max_delay=1.0,
        should_retry_result=lambda r: r.is_server_error,
    )

    assert result.status_code == 500
    assert calls == 3  # initial + 2 retries, then gives up and returns it


# --- defaults come from settings, read at call time ---------------------------


async def test_defaults_read_from_settings_at_call_time(monkeypatch, no_real_sleep):
    monkeypatch.setattr(settings, "http_max_retries", 1)
    monkeypatch.setattr(settings, "http_retry_base_delay_seconds", 0.02)
    monkeypatch.setattr(settings, "http_retry_max_delay_seconds", 0.5)

    calls = 0

    async def func():
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("simulated")

    with pytest.raises(httpx.ReadTimeout):
        await retry_async(func)  # no explicit max_retries/base_delay/max_delay

    assert calls == 2  # settings.http_max_retries=1 -> initial + 1 retry


# --- backoff growth ------------------------------------------------------------


async def test_backoff_delay_grows_exponentially_and_respects_cap(no_real_sleep):
    calls = 0

    async def func():
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("simulated")

    with pytest.raises(httpx.ReadTimeout):
        await retry_async(func, max_retries=4, base_delay=1.0, max_delay=3.0)

    # base_delay=1.0 -> raw delays would be 1, 2, 4, 8 for attempts 0-3;
    # capped at max_delay=3.0 -> capped values are 1, 2, 3, 3, each with up
    # to 25% jitter added on top (never negative, never more than 1.25x cap).
    assert len(no_real_sleep) == 4
    caps = [1.0, 2.0, 3.0, 3.0]
    for delay, cap in zip(no_real_sleep, caps):
        assert cap <= delay <= cap * 1.25
