import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
import respx
from httpx import Response

from claimpilot.clients.base import NotFoundError, get_client
from claimpilot.clients.fixtures import FixtureClient
from claimpilot.clients.http import HttpShipBobClient
from claimpilot.config import Settings, settings
from claimpilot.models import Attachment

BASE = settings.shipbob_api_base


TEST_API_KEY = "test-key-123"


@pytest.fixture
def client(monkeypatch) -> HttpShipBobClient:
    # Set before construction: __init__ snapshots headers into the
    # httpx.AsyncClient at build time, so patching afterward has no effect.
    monkeypatch.setattr(settings, "shipbob_api_key", TEST_API_KEY)
    return HttpShipBobClient()


def test_client_applies_configured_timeout(monkeypatch):
    monkeypatch.setattr(settings, "shipbob_http_timeout_seconds", 42.0)
    fresh_client = HttpShipBobClient()
    assert fresh_client._client.timeout.connect == 42.0
    assert fresh_client._client.timeout.read == 42.0


def _assert_api_key_header(request) -> None:
    # Assert the literal value (not settings.shipbob_api_key, which defaults
    # to "" and would make this comparison pass trivially even if the client
    # sent the wrong field, or nothing at all).
    assert request.headers["x-api-key"] == TEST_API_KEY


# --- list_cases / get_case ---------------------------------------------------


@respx.mock
async def test_list_cases_parses_summary_shape(client: HttpShipBobClient):
    route = respx.get(f"{BASE}/cases").mock(
        return_value=Response(
            200,
            json={
                "cases": [
                    {
                        "case_id": "CASE-1001",
                        "case_number": "01838218",
                        "status": "New",
                        "subject": "ShipBob Claim",
                        "created_date": "2026-02-19T14:20:16.000+0000",
                    }
                ]
            },
        )
    )

    cases = await client.list_cases()

    assert route.called
    _assert_api_key_header(route.calls[0].request)
    assert len(cases) == 1
    assert cases[0].case_id == "CASE-1001"
    assert cases[0].status == "New"
    assert cases[0].created_date == "2026-02-19T14:20:16.000+0000"


@respx.mock
async def test_get_case_parses_detail_shape(client: HttpShipBobClient):
    route = respx.get(f"{BASE}/cases/CASE-1001").mock(
        return_value=Response(
            200,
            json={
                "case_id": "CASE-1001",
                "case_number": "01838218",
                "status": "New",
                "sub_category": "Claim | Damaged in Transit",
                "description": "Shipment ID: 342578703.",
                "order_id": "334291211",
                "user_id": "334430",
                "shipment_id": "342578703",
                "delivered_date": "2026-02-11T11:36:14.000+0000",
                "contact_email": "someone@example.com",
                "account_name": "Best Paw Nutrition",
                "origin": "Case Portal - Claim",
                "created_date": "2026-02-19T14:20:16.000+0000",
            },
        )
    )

    case = await client.get_case("CASE-1001")

    request = route.calls[0].request
    assert route.called
    assert (request.method, str(request.url)) == ("GET", f"{BASE}/cases/CASE-1001")
    _assert_api_key_header(request)
    assert case.case_id == "CASE-1001"
    assert case.order_id == "334291211"
    assert case.shipment_id == "342578703"
    assert case.account_name == "Best Paw Nutrition"


@respx.mock
async def test_get_case_404_raises_not_found(client: HttpShipBobClient):
    respx.get(f"{BASE}/cases/CASE-9999").mock(
        return_value=Response(404, json={"error": "case_not_found", "message": "No case found."})
    )

    with pytest.raises(NotFoundError) as exc_info:
        await client.get_case("CASE-9999")

    assert exc_info.value.entity == "case"
    assert exc_info.value.entity_id == "CASE-9999"


# --- list_attachments ----------------------------------------------------------


@respx.mock
async def test_list_attachments_parses_shape(client: HttpShipBobClient):
    route = respx.get(f"{BASE}/cases/CASE-1001/attachments").mock(
        return_value=Response(
            200,
            json={
                "attachments": [
                    {
                        "attachment_id": "ATT-CASE-1001-01",
                        "file_name": "kgray2.png",
                        "content_type": "image/png",
                        "url": "https://sa032101pubdevuc.blob.core.windows.net/shipbob-fde-mock/case-1001/01_kgray2.png",
                    }
                ]
            },
        )
    )

    attachments = await client.list_attachments("CASE-1001")

    assert route.called
    _assert_api_key_header(route.calls[0].request)
    assert len(attachments) == 1
    assert attachments[0].attachment_id == "ATT-CASE-1001-01"
    assert attachments[0].content_type == "image/png"


@respx.mock
async def test_list_attachments_404_raises_not_found(client: HttpShipBobClient):
    respx.get(f"{BASE}/cases/CASE-9999/attachments").mock(
        return_value=Response(404, json={"error": "case_not_found", "message": "No case found."})
    )

    with pytest.raises(NotFoundError) as exc_info:
        await client.list_attachments("CASE-9999")

    assert exc_info.value.entity == "case"
    assert exc_info.value.entity_id == "CASE-9999"


# --- get_attachment_bytes -------------------------------------------------------
# fetch_attachment itself (SSRF/content-type/size guardrails) is exercised in
# test_attachment_guard.py. Here we only cover HttpShipBobClient's own wiring:
# that it delegates with the right args, and the ValueError branch for a
# missing url.


async def test_get_attachment_bytes_delegates_to_fetch_attachment(client, monkeypatch):
    attachment = Attachment(
        attachment_id="ATT-CASE-1001-01",
        file_name="kgray2.png",
        content_type="image/png",
        url="https://sa032101pubdevuc.blob.core.windows.net/shipbob-fde-mock/case-1001/01_kgray2.png",
    )

    seen_calls = []

    async def fake_fetch_attachment(url: str, attachment_id: str | None = None) -> bytes:
        seen_calls.append((url, attachment_id))
        return b"fake-image-bytes"

    # http.py imports fetch_attachment by name, so patch the name in http's
    # own namespace, not on attachment_guard where it's merely defined.
    monkeypatch.setattr(
        "claimpilot.clients.http.fetch_attachment", fake_fetch_attachment
    )

    result = await client.get_attachment_bytes(attachment)

    assert result == b"fake-image-bytes"
    assert seen_calls == [(attachment.url, attachment.attachment_id)]


async def test_get_attachment_bytes_without_url_raises_value_error(client):
    attachment = Attachment(
        attachment_id="ATT-CASE-1001-01", file_name="kgray2.png", content_type="image/png", url=None
    )

    with pytest.raises(ValueError, match="ATT-CASE-1001-01"):
        await client.get_attachment_bytes(attachment)


# --- get_shipment ----------------------------------------------------------------


@respx.mock
async def test_get_shipment_parses_shape(client: HttpShipBobClient):
    route = respx.get(f"{BASE}/shipments/342578703").mock(
        return_value=Response(
            200,
            json={
                "shipment_id": "342578703",
                "order_id": "334291211",
                "carrier": "Royal Mail Tracked 48",
                "tracking_number": "XQ607930599GB",
                "status": "Delivered",
                "delivered_date": "2026-02-11T11:36:14.000+0000",
                "is_insured": False,
            },
        )
    )

    shipment = await client.get_shipment("342578703")

    assert route.called
    _assert_api_key_header(route.calls[0].request)
    assert shipment.shipment_id == "342578703"
    assert shipment.is_insured is False
    assert shipment.carrier == "Royal Mail Tracked 48"


@respx.mock
async def test_get_shipment_404_raises_not_found(client: HttpShipBobClient):
    respx.get(f"{BASE}/shipments/999999999").mock(
        return_value=Response(404, json={"error": "shipment_not_found", "message": "No shipment found."})
    )

    with pytest.raises(NotFoundError) as exc_info:
        await client.get_shipment("999999999")

    assert exc_info.value.entity == "shipment"
    assert exc_info.value.entity_id == "999999999"


@respx.mock
async def test_get_shipment_500_raises_http_status_error_not_not_found(client: HttpShipBobClient):
    # Locks in the 404-vs-5xx distinction: a missing entity is an eligibility
    # outcome (NotFoundError), but an upstream failure is a crash and must
    # propagate as-is, not get miscategorized as "not found". A persistent
    # 5xx still exhausts its retries first (see settings below), then raises
    # -- this test uses the real default `settings.http_max_retries` (3), so
    # asserting on `route.call_count` here doubles as a real-default check.
    route = respx.get(f"{BASE}/shipments/342578703").mock(
        return_value=Response(500, json={"error": "internal_server_error"})
    )

    with pytest.raises(httpx.HTTPStatusError):
        await client.get_shipment("342578703")

    assert route.call_count == 1 + settings.http_max_retries


# --- retry-on-transient-failure (clients/retry.py integration) -------------


@respx.mock
async def test_get_case_retries_on_timeout_then_succeeds(client: HttpShipBobClient, monkeypatch):
    """Directly reproduces the real failure mode observed against the live
    mock (`httpx.ReadTimeout`) -- the first attempt times out, the second
    succeeds, and `get_case` returns the real case with no visible error.
    """
    monkeypatch.setattr(settings, "http_retry_base_delay_seconds", 0.01)
    monkeypatch.setattr(settings, "http_retry_max_delay_seconds", 0.05)

    route = respx.get(f"{BASE}/cases/CASE-1001").mock(
        side_effect=[
            httpx.ReadTimeout("simulated read timeout"),
            Response(
                200,
                json={
                    "case_id": "CASE-1001",
                    "case_number": "01838218",
                    "status": "New",
                    "order_id": "334291211",
                    "user_id": "334430",
                    "shipment_id": "342578703",
                    "account_name": "Best Paw Nutrition",
                },
            ),
        ]
    )

    case = await client.get_case("CASE-1001")

    assert route.call_count == 2
    assert case.case_id == "CASE-1001"
    assert case.account_name == "Best Paw Nutrition"


@respx.mock
async def test_get_case_404_is_never_retried(client: HttpShipBobClient, monkeypatch):
    """A real 404 must come back immediately as `NotFoundError` -- exactly
    one request, never retried, regardless of `settings.http_max_retries`.
    """
    monkeypatch.setattr(settings, "http_retry_base_delay_seconds", 0.01)
    route = respx.get(f"{BASE}/cases/CASE-9999").mock(
        return_value=Response(404, json={"error": "case_not_found"})
    )

    with pytest.raises(NotFoundError):
        await client.get_case("CASE-9999")

    assert route.call_count == 1


@respx.mock
async def test_generate_invoice_retries_on_connect_error_then_succeeds(
    client: HttpShipBobClient, monkeypatch
):
    monkeypatch.setattr(settings, "http_retry_base_delay_seconds", 0.01)
    monkeypatch.setattr(settings, "http_retry_max_delay_seconds", 0.05)

    route = respx.post(f"{BASE}/invoices/generate").mock(
        side_effect=[
            httpx.ConnectError("simulated connection failure"),
            Response(
                200,
                json={
                    "invoice_id": "INV-342578703",
                    "shipment_id": "342578703",
                    "line_items": [],
                },
            ),
        ]
    )

    invoice = await client.generate_invoice(shipment_id="342578703", user_id="334430")

    assert route.call_count == 2
    assert invoice.invoice_id == "INV-342578703"


@respx.mock
async def test_generate_invoice_422_is_never_retried(client: HttpShipBobClient, monkeypatch):
    monkeypatch.setattr(settings, "http_retry_base_delay_seconds", 0.01)
    route = respx.post(f"{BASE}/invoices/generate").mock(
        return_value=Response(422, json={"error": "invoice_unavailable"})
    )

    with pytest.raises(NotFoundError):
        await client.generate_invoice(shipment_id="999999999", user_id="000000")

    assert route.call_count == 1


# --- get_order ---------------------------------------------------------------------


@respx.mock
async def test_get_order_parses_line_items(client: HttpShipBobClient):
    route = respx.get(f"{BASE}/orders/334291211").mock(
        return_value=Response(
            200,
            json={
                "order_id": "334291211",
                "user_id": "334430",
                "line_items": [
                    {
                        "product_id": "1374243085",
                        "name": "Additional Collagen Ampoule Duo",
                        "sku": "AMP1",
                        "quantity": 1,
                        "unit_price": 38.00,
                    }
                ],
            },
        )
    )

    order = await client.get_order("334291211")

    assert route.called
    _assert_api_key_header(route.calls[0].request)
    assert order.order_id == "334291211"
    assert len(order.line_items) == 1
    assert order.line_items[0].unit_price == Decimal("38.00")
    assert order.line_items[0].sku == "AMP1"


@respx.mock
async def test_get_order_404_raises_not_found(client: HttpShipBobClient):
    respx.get(f"{BASE}/orders/999999999").mock(
        return_value=Response(404, json={"error": "order_not_found", "message": "No order found."})
    )

    with pytest.raises(NotFoundError) as exc_info:
        await client.get_order("999999999")

    assert exc_info.value.entity == "order"
    assert exc_info.value.entity_id == "999999999"


# --- generate_invoice --------------------------------------------------------------


@respx.mock
async def test_generate_invoice_posts_body_and_parses_response(client: HttpShipBobClient):
    route = respx.post(f"{BASE}/invoices/generate").mock(
        return_value=Response(
            200,
            json={
                "invoice_id": "INV-342578703",
                "shipment_id": "342578703",
                "line_items": [
                    {
                        "product_id": "1374243085",
                        "name": "Additional Collagen Ampoule Duo",
                        "sku": "AMP1",
                        "quantity": 1,
                        "unit_price": 38.00,
                    }
                ],
                "generated_at": "2026-03-21T10:00:00.000+0000",
            },
        )
    )

    invoice = await client.generate_invoice(shipment_id="342578703", user_id="334430")

    assert route.called
    request = route.calls[0].request
    _assert_api_key_header(request)
    assert json.loads(request.content) == {"shipment_id": "342578703", "user_id": "334430"}
    assert invoice.invoice_id == "INV-342578703"
    assert invoice.line_items[0].unit_price == Decimal("38.00")


@respx.mock
async def test_generate_invoice_422_raises_not_found(client: HttpShipBobClient):
    respx.post(f"{BASE}/invoices/generate").mock(
        return_value=Response(
            422, json={"error": "invoice_unavailable", "message": "No invoice could be generated."}
        )
    )

    with pytest.raises(NotFoundError) as exc_info:
        await client.generate_invoice(shipment_id="349164073", user_id="398045")

    assert exc_info.value.entity == "invoice"
    assert exc_info.value.entity_id == "349164073"


# --- send_email ----------------------------------------------------------------------


@respx.mock
async def test_send_email_posts_body_and_parses_response(client: HttpShipBobClient):
    route = respx.post(f"{BASE}/cases/CASE-1001/email").mock(
        return_value=Response(200, json={"success": True, "message": "Email queued", "case_id": "CASE-1001"})
    )

    response = await client.send_email(
        case_id="CASE-1001", to="sakukreja@shipbob.com", subject="Hello", body="Hello Case1001"
    )

    assert route.called
    request = route.calls[0].request
    _assert_api_key_header(request)
    assert json.loads(request.content) == {
        "to": "sakukreja@shipbob.com",
        "subject": "Hello",
        "body": "Hello Case1001",
    }
    assert response == {"success": True, "message": "Email queued", "case_id": "CASE-1001"}


# --- submit_reimbursement -------------------------------------------------------------


@respx.mock
async def test_submit_reimbursement_posts_body_and_parses_response(client: HttpShipBobClient):
    route = respx.post(f"{BASE}/reimbursements").mock(
        return_value=Response(
            201, json={"reimbursement_id": "RMB-00101", "status": "approved", "created_at": "2026-03-21T10:00:00.000+0000"}
        )
    )

    response = await client.submit_reimbursement(
        case_id="CASE-1001",
        order_id="334291211",
        user_id="334430",
        shipment_id="342578703",
        product_name="Additional Collagen Ampoule Duo",
        amount=Decimal("38.00"),
    )

    assert route.called
    request = route.calls[0].request
    _assert_api_key_header(request)
    sent = json.loads(request.content)
    assert sent == {
        "case_id": "CASE-1001",
        "order_id": "334291211",
        "user_id": "334430",
        "shipment_id": "342578703",
        "product_name": "Additional Collagen Ampoule Duo",
        "amount": 38.00,
    }
    assert response["reimbursement_id"] == "RMB-00101"
    assert response["status"] == "approved"


# --- get_client() factory -------------------------------------------------------------


def test_get_client_returns_fixture_client_by_default(monkeypatch):
    # get_client() is @lru_cache(maxsize=1)'d (see base.py) so it can be
    # reused as a process-lifetime singleton in real code. That means calls
    # across tests share one cached result unless explicitly cleared -- clear
    # before and after so this test neither reads a previous test's cached
    # instance nor leaves one behind for the next.
    get_client.cache_clear()
    monkeypatch.setattr(settings, "use_fixtures", True)
    # Demo controls off so the factory returns the bare client -- with them on
    # it wraps in a SyntheticOverlayClient (covered separately below).
    monkeypatch.setattr(settings, "demo_controls_enabled", False)
    assert isinstance(get_client(), FixtureClient)
    get_client.cache_clear()


def test_get_client_returns_http_client_when_fixtures_disabled(monkeypatch):
    get_client.cache_clear()
    monkeypatch.setattr(settings, "use_fixtures", False)
    monkeypatch.setattr(settings, "demo_controls_enabled", False)
    assert isinstance(get_client(), HttpShipBobClient)
    get_client.cache_clear()


def test_get_client_caches_singleton_across_calls(monkeypatch):
    get_client.cache_clear()
    monkeypatch.setattr(settings, "use_fixtures", True)
    first = get_client()
    second = get_client()
    assert first is second
    get_client.cache_clear()


def test_settings_can_be_constructed_directly_with_overrides():
    custom = Settings(use_fixtures=False, shipbob_api_key="test-key")
    assert custom.use_fixtures is False
    assert custom.shipbob_api_key == "test-key"


# --- env-var parsing for the business-policy/infra Settings fields ----------
#
# These construct a *fresh* `Settings()` after setting an env var (mirroring
# `tests/test_llm.py`'s `test_settings_llm_provider_reads_from_env_var`
# pattern) rather than asserting against the already-constructed module-level
# `settings` singleton -- proving pydantic-settings actually parses each
# field's env var into the right Python type (Decimal/int/Path), not just
# that the Python-level default value looks right.


def test_settings_cap_parses_decimal_from_env_var(monkeypatch):
    monkeypatch.setenv("CAP", "250.00")
    assert Settings().cap == Decimal("250.00")


def test_settings_claim_window_days_parses_int_from_env_var(monkeypatch):
    monkeypatch.setenv("CLAIM_WINDOW_DAYS", "45")
    fresh = Settings()
    assert fresh.claim_window_days == 45
    assert isinstance(fresh.claim_window_days, int)


def test_settings_high_value_threshold_parses_decimal_from_env_var(monkeypatch):
    monkeypatch.setenv("HIGH_VALUE_THRESHOLD", "750.00")
    assert Settings().high_value_threshold == Decimal("750.00")


def test_settings_db_path_parses_path_from_env_var(monkeypatch, tmp_path):
    custom_path = tmp_path / "custom.db"
    monkeypatch.setenv("DB_PATH", str(custom_path))
    fresh = Settings()
    assert fresh.db_path == custom_path
    assert isinstance(fresh.db_path, Path)


def test_settings_max_attachment_bytes_parses_int_from_env_var(monkeypatch):
    monkeypatch.setenv("MAX_ATTACHMENT_BYTES", "1048576")
    assert Settings().max_attachment_bytes == 1_048_576


def test_settings_claim_frequency_window_days_parses_int_from_env_var(monkeypatch):
    monkeypatch.setenv("CLAIM_FREQUENCY_WINDOW_DAYS", "60")
    fresh = Settings()
    assert fresh.claim_frequency_window_days == 60
    assert isinstance(fresh.claim_frequency_window_days, int)


def test_settings_max_recent_notes_parses_int_from_env_var(monkeypatch):
    monkeypatch.setenv("MAX_RECENT_NOTES", "8")
    assert Settings().max_recent_notes == 8
