"""Fixture-backed ShipBobClient.

Real cases (CASE-1001..1005) are parsed directly from the saved example
responses in `docs/api/postman_collection.json` at load time -- that file
stays the single source of truth, nothing here is a regenerated snapshot.
Three synthetic cases (insured routing, guaranteed reimbursement-cap
trigger, and a same-merchant repeat case for the memory
carry-forward demo -- reuses CASE-1001's real `user_id`, not an independent
fourth merchant) come from the small hand-written `fixtures/synthetic.json`.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from claimpilot.clients.attachment_guard import fetch_attachment
from claimpilot.clients.base import NotFoundError
from claimpilot.models import Attachment, Case, Invoice, Order, Shipment

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COLLECTION_PATH = _REPO_ROOT / "docs" / "api" / "postman_collection.json"
_SYNTHETIC_PATH = _REPO_ROOT / "fixtures" / "synthetic.json"
_OUTBOX_DIR = _REPO_ROOT / "outbox"


@dataclass(frozen=True)
class _FixtureData:
    cases: dict[str, Case]
    shipments: dict[str, Shipment]
    orders: dict[str, Order]
    invoices: dict[str, Invoice]
    attachments: dict[str, list[Attachment]]


# --- Postman collection parser (~40 lines) ----------------------------------


def _folder(data: dict, name: str) -> dict:
    return next(item for item in data["item"] if item["name"] == name)


def _request(folder: dict, name: str) -> dict:
    return next(item for item in folder["item"] if item["name"] == name)


def _examples(data: dict, folder_name: str, request_name: str):
    """Yield (example_name, parsed_body) for each non-error saved response."""
    request = _request(_folder(data, folder_name), request_name)
    for example in request["response"]:
        body = json.loads(example["body"])
        if isinstance(body, dict) and "error" in body:
            continue
        yield example["name"], body


def _assert_attachments_match_cases(cases: dict[str, Case], attachments: dict[str, list]) -> None:
    """Guard against a silent wrong answer if a collection example is renamed.

    Attachments are keyed on the saved example's *name* (the response body
    itself carries no case_id) -- an unvalidated string match against case
    IDs. If that name ever drifted from the case ID, list_attachments would
    silently fall through to "no attachments", indistinguishable from a real
    missing-evidence case (e.g. CASE-1005). Fail loudly at load time instead.
    """
    if set(cases) != set(attachments):
        raise AssertionError(
            f"attachment/case name mismatch between collection examples: {set(cases) ^ set(attachments)}"
        )


@lru_cache(maxsize=1)
def _load_real_data() -> _FixtureData:
    data = json.loads(_COLLECTION_PATH.read_text())

    cases = {body["case_id"]: Case(**body) for _, body in _examples(data, "Cases", "GET /cases/:case_id")}
    shipments = {
        body["shipment_id"]: Shipment(**body)
        for _, body in _examples(data, "Shipments", "GET /shipments/:shipment_id")
    }

    order_to_shipment = {c.order_id: c.shipment_id for c in cases.values() if c.order_id}
    orders: dict[str, Order] = {}
    for _, body in _examples(data, "Orders", "GET /orders/:order_id"):
        order_id = body["order_id"]
        orders[order_id] = Order(
            order_id=order_id,
            shipment_id=order_to_shipment.get(order_id),
            line_items=body["line_items"],
        )

    invoices = {
        body["shipment_id"]: Invoice(**body)
        for _, body in _examples(data, "Invoices", "POST /invoices/generate")
    }
    attachments = {
        name: [Attachment(**a) for a in body["attachments"]]
        for name, body in _examples(data, "Attachments", "GET /cases/:case_id/attachments")
    }

    _assert_attachments_match_cases(cases, attachments)

    return _FixtureData(cases, shipments, orders, invoices, attachments)


def _load_synthetic_data() -> _FixtureData:
    raw = json.loads(_SYNTHETIC_PATH.read_text())
    return _FixtureData(
        cases={c["case_id"]: Case(**c) for c in raw["cases"]},
        shipments={s["shipment_id"]: Shipment(**s) for s in raw["shipments"]},
        orders={o["order_id"]: Order(**o) for o in raw["orders"]},
        invoices={i["shipment_id"]: Invoice(**i) for i in raw["invoices"]},
        attachments={
            case_id: [Attachment(**a) for a in items]
            for case_id, items in raw.get("attachments", {}).items()
        },
    )


# --- Outbox (send_email / submit_reimbursement side effects) ---------------


def _append_outbox(record: dict) -> None:
    _OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    with (_OUTBOX_DIR / "outbox.jsonl").open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


class FixtureClient:
    """In-memory ShipBobClient backed by the real Postman collection.

    Serves exactly the five cases the ShipBob API returns, and nothing else --
    `list_cases()` here is the same set a live `GET /cases` gives back, so a
    fixture-mode demo opens on real data rather than a padded queue.

    The three synthetic demo scenarios used to be merged in here. They now
    live behind `clients/synthetic.SyntheticOverlayClient`, added on demand,
    so they can't quietly inflate the real case list and so they work against
    the live HTTP client too -- see that module's docstring for the full
    reasoning.

    `include_synthetic=True` restores the old merged behaviour. It exists for
    tests that exercise the synthetic scenarios directly (insured routing,
    the cap, the repeat merchant) without needing to construct an overlay --
    the pipeline-level tests for those paths predate the overlay and are
    about the pipeline, not about client composition.
    """

    def __init__(self, *, include_synthetic: bool = False) -> None:
        real = _load_real_data()
        if not include_synthetic:
            self._cases = dict(real.cases)
            self._shipments = dict(real.shipments)
            self._orders = dict(real.orders)
            self._invoices = dict(real.invoices)
            self._attachments = dict(real.attachments)
            return
        synthetic = _load_synthetic_data()
        self._cases = {**real.cases, **synthetic.cases}
        self._shipments = {**real.shipments, **synthetic.shipments}
        self._orders = {**real.orders, **synthetic.orders}
        self._invoices = {**real.invoices, **synthetic.invoices}
        self._attachments = {**real.attachments, **synthetic.attachments}

    async def list_cases(self) -> list[Case]:
        return list(self._cases.values())

    async def get_case(self, case_id: str) -> Case:
        try:
            return self._cases[case_id]
        except KeyError:
            raise NotFoundError("case", case_id) from None

    async def list_attachments(self, case_id: str) -> list[Attachment]:
        if case_id not in self._cases:
            raise NotFoundError("case", case_id)
        # Copy so callers (e.g. the evidence classifier) can't mutate the
        # list held inside the lru_cache'd fixture data and corrupt it for
        # every subsequent case processed in the same run.
        return list(self._attachments.get(case_id, []))

    async def get_attachment_bytes(self, attachment: Attachment) -> bytes:
        if not attachment.url:
            raise ValueError(f"attachment {attachment.attachment_id} has no url")
        return await fetch_attachment(attachment.url, attachment.attachment_id)

    async def get_shipment(self, shipment_id: str) -> Shipment:
        try:
            return self._shipments[shipment_id]
        except KeyError:
            raise NotFoundError("shipment", shipment_id) from None

    async def get_order(self, order_id: str) -> Order:
        try:
            return self._orders[order_id]
        except KeyError:
            raise NotFoundError("order", order_id) from None

    async def generate_invoice(self, shipment_id: str, user_id: str) -> Invoice:
        try:
            return self._invoices[shipment_id]
        except KeyError:
            raise NotFoundError("invoice", shipment_id) from None

    async def send_email(self, case_id: str, to: str, subject: str, body: str) -> dict:
        response = {"success": True, "message": "Email queued", "case_id": case_id}
        _append_outbox(
            {
                "type": "email",
                "case_id": case_id,
                "to": to,
                "subject": subject,
                "body": body,
                "response": response,
            }
        )
        return response

    async def submit_reimbursement(
        self,
        case_id: str,
        order_id: str,
        user_id: str,
        shipment_id: str,
        product_name: str,
        amount: Decimal,
    ) -> dict:
        response = {
            "reimbursement_id": f"RMB-{uuid.uuid4().hex[:8].upper()}",
            "status": "submitted",
            "created_at": datetime.now(UTC).isoformat(),
        }
        _append_outbox(
            {
                "type": "reimbursement",
                "case_id": case_id,
                "order_id": order_id,
                "user_id": user_id,
                "shipment_id": shipment_id,
                "product_name": product_name,
                "amount": amount,
                "response": response,
            }
        )
        return response
