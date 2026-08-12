import json
from decimal import Decimal

import httpx
import pytest

from claimpilot.config import settings
from claimpilot.clients.base import NotFoundError
from claimpilot.clients.fixtures import FixtureClient, _assert_attachments_match_cases

REAL_CASE_IDS = {"CASE-1001", "CASE-1002", "CASE-1003", "CASE-1004", "CASE-1005"}
# CASE-9003-REPEAT deliberately reuses CASE-1001's real user_id
# ("334430") so it's a same-merchant repeat case for the memory carry-forward
# demo, not an independent fourth synthetic merchant.
SYNTHETIC_CASE_IDS = {"CASE-9001-INSURED", "CASE-9002-CAP", "CASE-9003-REPEAT"}


@pytest.fixture
def client() -> FixtureClient:
    """Default (real-data-only) client -- what a normal fetch sees."""
    return FixtureClient()


@pytest.fixture
def synthetic_client() -> FixtureClient:
    """Opt-in client including the three demo scenarios, for the tests below
    that assert on them specifically.
    """
    return FixtureClient(include_synthetic=True)


# --- list_cases / get_case ---------------------------------------------------


async def test_list_cases_returns_only_the_five_real_api_cases(client: FixtureClient):
    """The default client mirrors exactly what ShipBob's `GET /cases`
    returns. The synthetic demo scenarios are deliberately NOT in here --
    they're opt-in via the overlay so a fetch never shows invented cases
    alongside real ones (see `clients/synthetic.py`).
    """
    cases = await client.list_cases()
    ids = {c.case_id for c in cases}
    assert ids == REAL_CASE_IDS
    assert not (ids & SYNTHETIC_CASE_IDS)


async def test_synthetic_scenarios_are_available_when_explicitly_requested(
    synthetic_client: FixtureClient,
):
    cases = await synthetic_client.list_cases()
    ids = {c.case_id for c in cases}
    assert ids == REAL_CASE_IDS | SYNTHETIC_CASE_IDS


async def test_get_case_returns_parsed_fields_for_case_1001(client: FixtureClient):
    case = await client.get_case("CASE-1001")
    assert case.account_name == "Best Paw Nutrition"
    assert case.status == "New"
    assert case.sub_category == "Claim | Damaged in Transit"
    assert case.order_id == "334291211"
    assert case.shipment_id == "342578703"
    assert case.delivered_date == "2026-02-11T11:36:14.000+0000"


async def test_get_case_case_1004_is_closed_and_old():
    client = FixtureClient()
    case = await client.get_case("CASE-1004")
    assert case.status == "Closed"
    assert case.account_name == "Catalyze-X"


async def test_get_case_raises_not_found_for_unknown_id(client: FixtureClient):
    with pytest.raises(NotFoundError) as exc_info:
        await client.get_case("CASE-9999")
    assert exc_info.value.entity == "case"
    assert exc_info.value.entity_id == "CASE-9999"


async def test_get_case_synthetic_insured_case(synthetic_client: FixtureClient):
    case = await synthetic_client.get_case("CASE-9001-INSURED")
    assert case.account_name == "Synthetic Insured Co"
    assert case.shipment_id == "900001"


# --- list_attachments ---------------------------------------------------------


async def test_list_attachments_case_1001_has_three_photos(client: FixtureClient):
    attachments = await client.list_attachments("CASE-1001")
    assert len(attachments) == 3
    assert all(a.content_type == "image/png" for a in attachments)
    assert all(a.url and a.url.startswith(f"https://{settings.allowed_attachment_host}/") for a in attachments)


async def test_list_attachments_case_1005_is_empty_missing_evidence(client: FixtureClient):
    attachments = await client.list_attachments("CASE-1005")
    assert attachments == []


async def test_list_attachments_case_1003_includes_invoice_screenshot(client: FixtureClient):
    attachments = await client.list_attachments("CASE-1003")
    file_names = {a.file_name for a in attachments}
    assert "Inv.png" in file_names


async def test_list_attachments_raises_not_found_for_unknown_case(client: FixtureClient):
    with pytest.raises(NotFoundError):
        await client.list_attachments("CASE-9999")


async def test_list_attachments_returns_a_copy_not_the_cached_list(client: FixtureClient):
    """Mutating the returned list must not corrupt the fixture cache for
    subsequent calls (or subsequent cases, since the parser result is
    lru_cache'd for the process lifetime).
    """
    first_call = await client.list_attachments("CASE-1001")
    original_length = len(first_call)

    first_call.append("not-a-real-attachment")
    first_call.clear()

    second_call = await client.list_attachments("CASE-1001")
    assert len(second_call) == original_length == 3


# --- get_shipment --------------------------------------------------------------


async def test_get_shipment_all_real_shipments_are_uninsured(client: FixtureClient):
    for case_id, shipment_id in [
        ("CASE-1001", "342578703"),
        ("CASE-1002", "344745459"),
        ("CASE-1003", "346106093"),
        ("CASE-1004", "330936165"),
        ("CASE-1005", "349164073"),
    ]:
        shipment = await client.get_shipment(shipment_id)
        assert shipment.is_insured is False, case_id


async def test_get_shipment_synthetic_9001_is_insured(synthetic_client: FixtureClient):
    shipment = await synthetic_client.get_shipment("900001")
    assert shipment.is_insured is True


async def test_get_shipment_raises_not_found(client: FixtureClient):
    with pytest.raises(NotFoundError):
        await client.get_shipment("999999999")


# --- get_order -----------------------------------------------------------------


async def test_get_order_case_1001_has_two_line_items(client: FixtureClient):
    order = await client.get_order("334291211")
    assert len(order.line_items) == 2
    prices = sorted(li.unit_price for li in order.line_items)
    assert prices == [Decimal("38.00"), Decimal("52.00")]
    assert order.shipment_id == "342578703"


async def test_get_order_case_1005_has_zero_price_line_item(client: FixtureClient):
    order = await client.get_order("340775987")
    zero_price_items = [li for li in order.line_items if li.unit_price == Decimal("0.00")]
    assert len(zero_price_items) == 1
    assert zero_price_items[0].name == "Insert Card"


async def test_get_order_synthetic_cap_case_has_single_item_over_100(synthetic_client: FixtureClient):
    order = await synthetic_client.get_order("900002")
    assert len(order.line_items) == 1
    assert order.line_items[0].unit_price == Decimal("150.00")


async def test_get_order_raises_not_found(client: FixtureClient):
    with pytest.raises(NotFoundError):
        await client.get_order("999999999")


# --- generate_invoice ------------------------------------------------------------


async def test_generate_invoice_matches_order_line_items_for_case_1001(client: FixtureClient):
    invoice = await client.generate_invoice(shipment_id="342578703", user_id="334430")
    assert invoice.invoice_id == "INV-342578703"
    order = await client.get_order("334291211")
    assert [li.sku for li in invoice.line_items] == [li.sku for li in order.line_items]


async def test_generate_invoice_raises_not_found_for_unknown_shipment(client: FixtureClient):
    with pytest.raises(NotFoundError):
        await client.generate_invoice(shipment_id="999999999", user_id="000000")


async def test_generate_invoice_synthetic_cap_case(synthetic_client: FixtureClient):
    invoice = await synthetic_client.generate_invoice(shipment_id="900002", user_id="900002")
    assert invoice.line_items[0].unit_price == Decimal("150.00")


# --- attachment/case alignment guard (see attachment_guard for URL/size/
# --- content-type guardrail tests) ------------------------------------------


def test_assert_attachments_match_cases_passes_when_keys_align():
    cases = {"CASE-1001": object(), "CASE-1002": object()}
    attachments = {"CASE-1001": [], "CASE-1002": []}
    _assert_attachments_match_cases(cases, attachments)  # should not raise


def test_assert_attachments_match_cases_raises_on_mismatch():
    """Simulates a collection example being renamed out of sync with its
    case ID -- must fail loudly at load time, not silently look like a
    legitimate missing-evidence case.
    """
    cases = {"CASE-1001": object(), "CASE-1002": object()}
    attachments = {"CASE-1001": [], "CASE-1002-RENAMED": []}
    with pytest.raises(AssertionError, match="attachment/case name mismatch"):
        _assert_attachments_match_cases(cases, attachments)


async def test_get_attachment_bytes_downloads_and_caches_real_image(client: FixtureClient, tmp_path, monkeypatch):
    import claimpilot.clients.attachment_guard as attachment_guard_module

    monkeypatch.setattr(attachment_guard_module, "_IMAGE_CACHE_DIR", tmp_path)

    attachments = await client.list_attachments("CASE-1001")
    attachment = attachments[0]

    try:
        data = await client.get_attachment_bytes(attachment)
    except httpx.HTTPError as exc:
        pytest.skip(f"network unavailable in this environment: {exc}")

    assert len(data) > 0
    cached_path = tmp_path / attachment.attachment_id
    assert cached_path.exists()
    assert cached_path.read_bytes() == data


# --- send_email / submit_reimbursement (outbox) -----------------------------------


async def test_send_email_returns_collection_shaped_response_and_writes_outbox(tmp_path, monkeypatch):
    import claimpilot.clients.fixtures as fixtures_module

    monkeypatch.setattr(fixtures_module, "_OUTBOX_DIR", tmp_path)
    client = FixtureClient()

    response = await client.send_email(
        case_id="CASE-1001", to="sakukreja@shipbob.com", subject="Hello", body="Hello Case1001"
    )

    assert response == {"success": True, "message": "Email queued", "case_id": "CASE-1001"}

    outbox_file = tmp_path / "outbox.jsonl"
    lines = outbox_file.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["type"] == "email"
    assert record["case_id"] == "CASE-1001"
    assert record["to"] == "sakukreja@shipbob.com"


async def test_submit_reimbursement_returns_collection_shaped_response_and_writes_outbox(tmp_path, monkeypatch):
    import claimpilot.clients.fixtures as fixtures_module

    monkeypatch.setattr(fixtures_module, "_OUTBOX_DIR", tmp_path)
    client = FixtureClient()

    response = await client.submit_reimbursement(
        case_id="CASE-1001",
        order_id="334291211",
        user_id="334430",
        shipment_id="342578703",
        product_name="Additional Collagen Ampoule Duo",
        amount=Decimal("38.00"),
    )

    assert response["status"] == "submitted"
    assert response["reimbursement_id"].startswith("RMB-")
    assert "created_at" in response

    outbox_file = tmp_path / "outbox.jsonl"
    lines = outbox_file.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["type"] == "reimbursement"
    assert record["amount"] == "38.00"
    assert record["product_name"] == "Additional Collagen Ampoule Duo"


async def test_submit_reimbursement_ids_are_unique_across_calls(tmp_path, monkeypatch):
    import claimpilot.clients.fixtures as fixtures_module

    monkeypatch.setattr(fixtures_module, "_OUTBOX_DIR", tmp_path)
    client = FixtureClient()

    r1 = await client.submit_reimbursement(
        case_id="CASE-1001",
        order_id="334291211",
        user_id="334430",
        shipment_id="342578703",
        product_name="Additional Collagen Ampoule Duo",
        amount=Decimal("38.00"),
    )
    r2 = await client.submit_reimbursement(
        case_id="CASE-1002",
        order_id="336431771",
        user_id="283959",
        shipment_id="344745459",
        product_name="CleanBoss Botanical Disinfectant & Cleaner 24oz 2 Pack",
        amount=Decimal("24.99"),
    )

    assert r1["reimbursement_id"] != r2["reimbursement_id"]
