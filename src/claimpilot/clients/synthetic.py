"""Optional synthetic demo scenarios, layered over whatever the real client is.

ShipBob's sample data cannot exercise three rules the brief explicitly
requires, because of what the five real cases happen to contain:

- **Insured routing** ("insured shipments are a different process entirely")
  -- all five real shipments are `is_insured: false`.
- **Corrections carrying forward** ("not just on this case, on the next one
  too") -- the five real cases belong to five *different* merchants, so no
  repeat customer exists to carry anything forward to.
- **The $100 cap** -- the largest single real line item is $59.99.

`fixtures/synthetic.json` holds three hand-written cases covering exactly
those gaps, and this module makes them available *on demand* without
polluting the real case list.

**Why an overlay rather than baking them into `FixtureClient`.** The demo
should open on what the ShipBob API actually returns -- five cases, nothing
invented -- and only add scenarios when there's a reason to. Wrapping instead
of merging keeps that split honest in three ways:

1. `list_cases()` delegates straight through, so the synthetic cases never
   appear in a normal fetch. They only enter the queue when a human clicks
   "Add demo scenarios", which creates their rows explicitly.
2. It works against the **live** HTTP client too, not just fixtures. The
   scenarios are local test data layered on top of the real API, so a demo
   can run against the live mock and still show the insured/cap/memory paths
   -- which would otherwise be impossible, since the live mock 404s these IDs.
3. Removing the scenarios later is deleting one file and one wrapper, not
   unpicking merged dictionaries from the real fixture loader.

Lookups check the synthetic data first and fall through to the wrapped client
on a miss. The synthetic IDs (`CASE-9001-*`/`9002`/`9003`) don't collide with
ShipBob's real `CASE-1xxx` numbering, so the precedence order never shadows a
real case -- but it's checked first rather than last so a synthetic case
resolves in one step instead of after a failed network round-trip.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from claimpilot.clients.base import NotFoundError, ShipBobClient
from claimpilot.models import Attachment, Case, Invoice, Order, Shipment

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SYNTHETIC_PATH = _REPO_ROOT / "fixtures" / "synthetic.json"


@lru_cache(maxsize=1)
def _raw() -> dict:
    return json.loads(_SYNTHETIC_PATH.read_text())


def synthetic_case_ids() -> list[str]:
    """The demo-scenario case IDs, in file order.

    Read from the file rather than hardcoded so adding a fourth scenario is a
    JSON edit -- the review UI's \"Add demo scenarios\" button, its button
    label's count, and the fetch endpoint's exclusion list all derive from
    this one source.
    """
    return [c["case_id"] for c in _raw()["cases"]]


class SyntheticOverlayClient:
    """Wraps a `ShipBobClient`, resolving synthetic demo cases locally and
    delegating everything else.

    Deliberately does *not* implement `send_email`/`submit_reimbursement`
    itself -- those are real side effects and belong to the wrapped client
    whatever it is, so approving a synthetic case still writes to the same
    outbox as a real one. Only read paths are overlaid.
    """

    def __init__(self, inner: ShipBobClient) -> None:
        self._inner = inner
        raw = _raw()
        self._cases = {c["case_id"]: Case(**c) for c in raw["cases"]}
        self._shipments = {s["shipment_id"]: Shipment(**s) for s in raw["shipments"]}
        self._orders = {o["order_id"]: Order(**o) for o in raw["orders"]}
        self._invoices = {i["shipment_id"]: Invoice(**i) for i in raw["invoices"]}
        self._attachments = {
            case_id: [Attachment(**a) for a in items]
            for case_id, items in raw.get("attachments", {}).items()
        }

    # --- reads: synthetic first, then delegate ------------------------------

    async def list_cases(self) -> list[Case]:
        """Straight delegation -- synthetic cases are deliberately absent from
        the case list so a normal fetch only ever returns what the real API
        returned. They're added to the queue by explicit action instead.
        """
        return await self._inner.list_cases()

    async def get_case(self, case_id: str) -> Case:
        if case_id in self._cases:
            return self._cases[case_id]
        return await self._inner.get_case(case_id)

    async def list_attachments(self, case_id: str) -> list[Attachment]:
        if case_id in self._cases:
            # Copied for the same reason `FixtureClient` copies: callers
            # downstream iterate and occasionally mutate, and this list is
            # shared across every case processed in the process's lifetime.
            return list(self._attachments.get(case_id, []))
        return await self._inner.list_attachments(case_id)

    async def get_attachment_bytes(self, attachment: Attachment) -> bytes:
        # Synthetic cases reuse real attachment URLs, so byte fetching always
        # goes through the wrapped client -- same SSRF allowlist, size cap and
        # caching as any other attachment.
        return await self._inner.get_attachment_bytes(attachment)

    async def get_shipment(self, shipment_id: str) -> Shipment:
        if shipment_id in self._shipments:
            return self._shipments[shipment_id]
        return await self._inner.get_shipment(shipment_id)

    async def get_order(self, order_id: str) -> Order:
        if order_id in self._orders:
            return self._orders[order_id]
        return await self._inner.get_order(order_id)

    async def generate_invoice(self, *, shipment_id: str, user_id: str) -> Invoice:
        if shipment_id in self._invoices:
            return self._invoices[shipment_id]
        return await self._inner.generate_invoice(shipment_id=shipment_id, user_id=user_id)

    # --- writes: always the wrapped client ----------------------------------

    async def send_email(self, case_id: str, *, to: str, subject: str, body: str) -> dict:
        return await self._inner.send_email(case_id, to=to, subject=subject, body=body)

    async def submit_reimbursement(
        self,
        case_id: str,
        *,
        order_id: str,
        user_id: str,
        shipment_id: str,
        product_name: str,
        amount,
    ) -> dict:
        return await self._inner.submit_reimbursement(
            case_id,
            order_id=order_id,
            user_id=user_id,
            shipment_id=shipment_id,
            product_name=product_name,
            amount=amount,
        )


__all__ = ["NotFoundError", "SyntheticOverlayClient", "synthetic_case_ids"]
