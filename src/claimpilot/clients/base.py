"""Shared client contract for talking to the ShipBob claims API.

Both the fixture implementation (`fixtures.FixtureClient`) and the httpx-based
implementation (`http.HttpShipBobClient`) implement this Protocol, so the
rest of the pipeline never needs to know which one it's talking to. Use
`get_client()` below to pick the right one based on `settings.use_fixtures`.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from typing import Protocol

from claimpilot.models import Attachment, Case, Invoice, Order, Shipment


class NotFoundError(Exception):
    """Raised when a requested entity can't be produced by the client.

    For case/shipment/order lookups, this mirrors the real API's
    `404 {error: "<entity>_not_found"}` shape closely enough for callers to
    build an eligibility outcome from it, without needing to parse HTTP
    responses in the pipeline layer.

    For `generate_invoice`, there is no `GET /invoices/:id` 404 in the real
    API -- the only invoice error is `422 {error: "invoice_unavailable"}`
    from `POST /invoices/generate`. By convention (see `generate_invoice`
    below), both client implementations raise `NotFoundError("invoice", ...)`
    for that case too, so pipeline code can pattern-match on one exception
    type regardless of which client is wired in.
    """

    def __init__(self, entity: str, entity_id: str) -> None:
        self.entity = entity
        self.entity_id = entity_id
        super().__init__(f"{entity} not found: {entity_id}")


class ShipBobClient(Protocol):
    """Async adapter over the ShipBob claims API (real or fixture-backed)."""

    async def list_cases(self) -> list[Case]: ...

    async def get_case(self, case_id: str) -> Case: ...

    async def list_attachments(self, case_id: str) -> list[Attachment]: ...

    async def get_attachment_bytes(self, attachment: Attachment) -> bytes: ...

    async def get_shipment(self, shipment_id: str) -> Shipment: ...

    async def get_order(self, order_id: str) -> Order: ...

    async def generate_invoice(self, shipment_id: str, user_id: str) -> Invoice:
        """Raise `NotFoundError("invoice", shipment_id)` when unavailable.

        The real API has no 404 for invoices, only `422
        {error: "invoice_unavailable"}` from `POST /invoices/generate`.
        Both the fixture and real implementations treat that as
        the not-found case and raise `NotFoundError` for it, so downstream
        eligibility/calc code has one exception contract to pattern-match
        on regardless of which client is active.
        """
        ...

    async def send_email(self, case_id: str, to: str, subject: str, body: str) -> dict:
        """Write path -- no typed error contract, unlike the GET/invoice methods above.

        `FixtureClient` can't fail here (it unconditionally writes to the
        outbox and returns success). `HttpShipBobClient` calls
        `raise_for_status()` with no normalization, so a real API error (e.g.
        `400 {error: "invalid_request"}`) propagates as a raw
        `httpx.HTTPStatusError`, not `NotFoundError`. This is a deliberate,
        documented asymmetry -- callers should not assume `NotFoundError` is
        the only failure mode on this method.
        """
        ...

    async def submit_reimbursement(
        self,
        case_id: str,
        order_id: str,
        user_id: str,
        shipment_id: str,
        product_name: str,
        amount: Decimal,
    ) -> dict:
        """Write path -- same untyped-error caveat as `send_email` above.

        `HttpShipBobClient` surfaces a real `400 {error: "invalid_request"}`
        (or any other non-2xx) as a raw `httpx.HTTPStatusError` via
        `raise_for_status()`; `FixtureClient` can never fail this way. Not
        normalized into `NotFoundError` or any other typed exception.
        """
        ...


@lru_cache(maxsize=1)
def get_client() -> ShipBobClient:
    """Pick the fixture or real HTTP implementation based on `USE_FIXTURES`.

    Cached as a process-lifetime singleton: `HttpShipBobClient` owns an
    `httpx.AsyncClient` (a real connection pool), so constructing a fresh one
    on every call -- the natural way to write per-request pipeline code --
    would leak a pool per call. `lru_cache(maxsize=1)` (this function takes no
    args) makes repeated calls return the same instance instead. There's no
    `aclose()`/app-lifecycle plumbing yet; that's future work once there's a
    real process lifecycle to hang it on.

    Note for tests: since the result is cached, flipping
    `settings.use_fixtures` between two `get_client()` calls in the same
    process won't produce a second instance unless `get_client.cache_clear()`
    is called first.

    Imports are local to avoid import cycles: `fixtures.py` and `http.py`
    both import from this module (`NotFoundError`), and `config.py` is kept
    independent of both.
    """
    from claimpilot.config import settings

    if settings.use_fixtures:
        from claimpilot.clients.fixtures import FixtureClient

        base: ShipBobClient = FixtureClient()
    else:
        from claimpilot.clients.http import HttpShipBobClient

        base = HttpShipBobClient()

    if settings.demo_controls_enabled:
        # Layer the optional demo scenarios over whichever real client is in
        # use. This does NOT add them to `list_cases()` -- the overlay
        # delegates that untouched, so a fetch still returns exactly what the
        # ShipBob API returned. It only makes the synthetic IDs *resolvable*,
        # so the review UI's "Add demo scenarios" button produces cases that
        # can actually be processed, including against the live mock (which
        # 404s those IDs).
        from claimpilot.clients.synthetic import SyntheticOverlayClient

        return SyntheticOverlayClient(base)

    return base
