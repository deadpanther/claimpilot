"""httpx-backed ShipBobClient talking to the real Postman mock.

Endpoint shapes mirror `docs/api/postman_collection.json` (the same
collection the fixture client parses at load time -- see
`claimpilot.clients.fixtures`). GET 404s (`{error: "<entity>_not_found"}`)
and the invoice 422 (`{error: "invoice_unavailable"}`) are normalized into
`NotFoundError` per the contract documented on `ShipBobClient` in `base.py`.

Every request goes through `clients.retry.retry_async` (module docstring
there has the full policy) -- transient network failures and 5xx responses
are retried with exponential backoff; 404/422 are real answers and are
never retried, so `_get`'s/`generate_invoice`'s existing 404-to-
`NotFoundError` mapping runs exactly as before, just possibly after a few
retries first.
"""

from __future__ import annotations

from decimal import Decimal

import httpx

from claimpilot.clients.attachment_guard import fetch_attachment
from claimpilot.clients.base import NotFoundError
from claimpilot.clients.retry import retry_async
from claimpilot.models import Attachment, Case, Invoice, Order, Shipment

# 5xx is worth retrying (transient server-side failure); 4xx is never
# retried here -- it's a deterministic answer (404/422/400/etc.), not a
# transient one. `httpx.Response.is_server_error` covers exactly 500-599.
def _is_retryable_response(response: httpx.Response) -> bool:
    return response.is_server_error


class HttpShipBobClient:
    """Async ShipBobClient implementation over the real (mocked) HTTP API."""

    def __init__(self) -> None:
        from claimpilot.config import settings

        self._client = httpx.AsyncClient(
            base_url=settings.shipbob_api_base,
            headers={"x-api-key": settings.shipbob_api_key},
            timeout=settings.shipbob_http_timeout_seconds,
        )

    async def _get(self, path: str, *, entity: str, entity_id: str) -> dict:
        response = await retry_async(
            lambda: self._client.get(path),
            should_retry_result=_is_retryable_response,
            description=f"GET {path}",
        )
        if response.status_code == 404:
            raise NotFoundError(entity, entity_id)
        response.raise_for_status()
        return response.json()

    async def list_cases(self) -> list[Case]:
        response = await retry_async(
            lambda: self._client.get("/cases"),
            should_retry_result=_is_retryable_response,
            description="GET /cases",
        )
        response.raise_for_status()
        return [Case(**c) for c in response.json()["cases"]]

    async def get_case(self, case_id: str) -> Case:
        body = await self._get(f"/cases/{case_id}", entity="case", entity_id=case_id)
        return Case(**body)

    async def list_attachments(self, case_id: str) -> list[Attachment]:
        body = await self._get(f"/cases/{case_id}/attachments", entity="case", entity_id=case_id)
        return [Attachment(**a) for a in body["attachments"]]

    async def get_attachment_bytes(self, attachment: Attachment) -> bytes:
        if not attachment.url:
            raise ValueError(f"attachment {attachment.attachment_id} has no url")
        return await fetch_attachment(attachment.url, attachment.attachment_id)

    async def get_shipment(self, shipment_id: str) -> Shipment:
        body = await self._get(f"/shipments/{shipment_id}", entity="shipment", entity_id=shipment_id)
        return Shipment(**body)

    async def get_order(self, order_id: str) -> Order:
        body = await self._get(f"/orders/{order_id}", entity="order", entity_id=order_id)
        return Order(**body)

    async def generate_invoice(self, shipment_id: str, user_id: str) -> Invoice:
        response = await retry_async(
            lambda: self._client.post(
                "/invoices/generate", json={"shipment_id": shipment_id, "user_id": user_id}
            ),
            should_retry_result=_is_retryable_response,
            description="POST /invoices/generate",
        )
        if response.status_code == 422:
            raise NotFoundError("invoice", shipment_id)
        response.raise_for_status()
        return Invoice(**response.json())

    async def send_email(self, case_id: str, to: str, subject: str, body: str) -> dict:
        response = await retry_async(
            lambda: self._client.post(
                f"/cases/{case_id}/email", json={"to": to, "subject": subject, "body": body}
            ),
            should_retry_result=_is_retryable_response,
            description=f"POST /cases/{case_id}/email",
        )
        response.raise_for_status()
        return response.json()

    async def submit_reimbursement(
        self,
        case_id: str,
        order_id: str,
        user_id: str,
        shipment_id: str,
        product_name: str,
        amount: Decimal,
    ) -> dict:
        response = await retry_async(
            lambda: self._client.post(
                "/reimbursements",
                json={
                    "case_id": case_id,
                    "order_id": order_id,
                    "user_id": user_id,
                    "shipment_id": shipment_id,
                    "product_name": product_name,
                    # httpx's JSON encoder can't serialize Decimal directly, and
                    # the API expects a plain number here -- don't "fix" this
                    # back to Decimal, it'll break request serialization.
                    "amount": float(amount),
                },
            ),
            should_retry_result=_is_retryable_response,
            description="POST /reimbursements",
        )
        response.raise_for_status()
        return response.json()
