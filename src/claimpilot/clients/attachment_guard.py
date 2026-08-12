"""Shared attachment-download guardrails.

Both the fixture client and the httpx-based real client (`clients/http.py`)
fetch attachment bytes from the same Azure blob storage account, so the
SSRF/content-type/size guardrails live here as a neutral module that both
implementations depend on -- neither depends on the other.

The whole download (open the stream, read headers, stream all bytes) is
wrapped in `clients.retry.retry_async` -- a transient network failure
(timeout/connection error) at any point restarts the download from scratch
on the next attempt (these images are small and capped at
`settings.max_attachment_bytes`, so a full re-fetch is cheap; there's no
partial-resume complexity worth adding). Unlike `clients/http.py`, this does
NOT also retry on a 5xx status -- `response.raise_for_status()` raises
`httpx.HTTPStatusError` for any 4xx/5xx alike, and `retry_async`'s
exception-based retry set deliberately never includes that (it would retry
a definitive 404 too, which must never happen). The network-level failures
this module was actually written in response to (`httpx.ReadTimeout`,
directly observed against the live case/order API -- Azure blob storage
has not itself been observed flaky) are still fully covered.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from claimpilot.clients.retry import retry_async
from claimpilot.config import settings

_REPO_ROOT = Path(__file__).resolve().parents[3]
_IMAGE_CACHE_DIR = _REPO_ROOT / "fixtures" / "images"


def validate_attachment_url(url: str) -> None:
    """Guard against SSRF / unexpected hosts. Raises ValueError, never fetches.

    Reads `settings.allowed_attachment_host` at call time (not a
    module-level snapshot) so a `monkeypatch.setattr(settings, ...)`
    override in tests, or a real env var change, is honored immediately.
    """
    host = httpx.URL(url).host
    if host != settings.allowed_attachment_host:
        raise ValueError(f"attachment host not allowlisted: {host!r}")


def _cache_path_for(attachment_id: str) -> Path:
    """Resolve the on-disk cache path for an attachment_id, rejecting any
    value that could escape fixtures/images/ (path traversal / absolute
    path injection). attachment_id may originate from a live API response,
    not just the fixture client, so it must be treated as untrusted input.
    """
    if not attachment_id or attachment_id in (".", "..") or "/" in attachment_id or "\\" in attachment_id:
        raise ValueError(f"unsafe attachment_id for cache path: {attachment_id!r}")
    return _IMAGE_CACHE_DIR / attachment_id


async def fetch_attachment(url: str, attachment_id: str | None = None) -> bytes:
    """Download an attachment, enforcing host/content-type/size guardrails.

    Caches downloaded bytes on disk under fixtures/images/<attachment_id> so
    repeated test/demo runs don't re-download.
    """
    validate_attachment_url(url)

    cache_path = _cache_path_for(attachment_id) if attachment_id else None
    if cache_path is not None and cache_path.exists():
        return cache_path.read_bytes()

    async def _download() -> bytes:
        async with httpx.AsyncClient() as client, client.stream("GET", url) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                raise ValueError(f"unexpected content-type for attachment: {content_type!r}")

            max_bytes = settings.max_attachment_bytes
            chunks = bytearray()
            async for chunk in response.aiter_bytes():
                chunks.extend(chunk)
                if len(chunks) > max_bytes:
                    raise ValueError(f"attachment exceeds max size of {max_bytes} bytes")
        return bytes(chunks)

    data = await retry_async(_download, description=f"GET attachment {attachment_id or url}")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
    return data
