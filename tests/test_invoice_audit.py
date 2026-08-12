"""Tests for the retail-invoice audit -- reconciling ShipBob's invoice
against the merchant's own submitted invoice.

The pure reconciliation logic (`audit_invoice` and the individual checks) is
exercised directly with hand-built inputs, no LLM and no pipeline, so
"would a real discrepancy actually be caught" is a testable question rather
than something inferred from an end-to-end run. The extraction call itself
is covered through a `FakeTransport` so the real prompt file and image
wrapping are still exercised.

Several cases below use the actual figures found in this project's own
fixture data (CASE-1002's $24.99-vs-$19.99 gap, CASE-1003's $14.99
order-level discount, CASE-1001's GBP order) rather than invented round
numbers -- if the reconciliation ever stops catching the real discrepancies
that motivated it, these fail.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from claimpilot.config import settings
from claimpilot.gates.invoice_audit import (
    Discrepancy,
    ExtractedInvoice,
    ExtractedLine,
    InvoiceAudit,
    Severity,
    audit_invoice,
    check_currency,
    check_damaged_item_prices,
    effective_unit_price,
    extract_invoice,
)
from claimpilot.models import Invoice, LineItem
from tests.test_llm import FakeTransport, TransportResult

pytestmark = pytest.mark.anyio


def _invoice(*lines: LineItem) -> Invoice:
    return Invoice(
        invoice_id="INV-1",
        shipment_id="SHIP-1",
        line_items=list(lines),
        generated_at="2026-03-21T10:00:00.000+0000",
    )


def _line(sku: str, name: str, qty: int, unit_price: str) -> LineItem:
    return LineItem(product_id=f"P-{sku}", name=name, sku=sku, quantity=qty, unit_price=Decimal(unit_price))


def _extracted(
    *items: ExtractedLine,
    readable: bool = True,
    currency: str | None = "USD",
    order_discount_total: float = 0.0,
) -> ExtractedInvoice:
    return ExtractedInvoice(
        readable=readable,
        currency=currency,
        line_items=list(items),
        order_discount_total=order_discount_total,
    )


# --- the real CASE-1002 discrepancy ----------------------------------------


def test_real_case_1002_price_gap_is_caught():
    """ShipBob prices A00360 at $24.99; the merchant's own sales order shows
    the customer paid $19.99. Reimbursing from the API figure overpays by
    exactly $5.00 -- the discrepancy that motivated this whole check.
    """
    invoice = _invoice(_line("A00360", "CleanBoss Botanical Disinfectant & Cleaner 24oz 2 Pack", 1, "24.99"))
    extracted = _extracted(
        ExtractedLine(
            description="CleanBoss Clean Cleaner Botanical Disinfectant & Cleaner 24 Ounce Trigger Off 2 Pack",
            sku="A00360",
            quantity=1,
            unit_price=19.99,
            line_total=19.99,
        )
    )

    audit = audit_invoice(extracted, invoice, ["A00360"])

    assert audit.verified is True
    assert audit.should_escalate is True
    codes = [d.code for d in audit.discrepancies]
    assert codes == ["PRICE_MISMATCH"]
    detail = audit.discrepancies[0].detail
    assert "24.99" in detail and "19.99" in detail
    assert "5.00 more than" in detail


def test_matching_prices_produce_no_discrepancy():
    invoice = _invoice(_line("A00360", "Botanical Disinfectant 2 Pack", 1, "19.99"))
    extracted = _extracted(
        ExtractedLine(description="Botanical Disinfectant 2 Pack", sku="A00360", quantity=1, unit_price=19.99)
    )

    audit = audit_invoice(extracted, invoice, ["A00360"])

    assert audit.verified is True
    assert audit.discrepancies == []
    assert audit.should_escalate is False


def test_difference_within_tolerance_is_not_flagged(monkeypatch: pytest.MonkeyPatch):
    """Tolerance is a real threshold, not decoration -- a sub-tolerance gap
    stays out of the queue.
    """
    monkeypatch.setattr(settings, "invoice_price_tolerance", Decimal("0.50"))
    invoice = _invoice(_line("SKU1", "Thing", 1, "20.00"))
    extracted = _extracted(ExtractedLine(description="Thing", sku="SKU1", quantity=1, unit_price=19.75))

    assert audit_invoice(extracted, invoice, ["SKU1"]).discrepancies == []


def test_underpayment_direction_is_reported_distinctly():
    """A merchant-invoice price *higher* than ShipBob's is still a
    discrepancy, but the rep needs to know which way it cuts.
    """
    invoice = _invoice(_line("SKU1", "Thing", 1, "27.99"))
    extracted = _extracted(ExtractedLine(description="Thing", sku="SKU1", quantity=1, unit_price=30.36))

    audit = audit_invoice(extracted, invoice, ["SKU1"])

    assert [d.code for d in audit.discrepancies] == ["PRICE_MISMATCH"]
    assert "2.37 less than" in audit.discrepancies[0].detail


# --- order-level discount allocation (the real CASE-1003 shape) -------------


def test_order_level_discount_is_allocated_proportionally():
    """CASE-1003's real invoice: $149.98 of line items with a single $14.99
    order-level discount (10%). The damaged whey line's own share of that
    discount has to come off before it can be compared per-SKU, since
    reimbursement is scoped to the specific damaged item.
    """
    lines = [
        ExtractedLine(description="liquid carnitine 3000", quantity=1, unit_price=35.96, line_total=35.96),
        ExtractedLine(description="huge whey | protein powder", quantity=1, unit_price=42.45, line_total=42.45),
        ExtractedLine(description="wrecked pre workout", quantity=1, unit_price=41.21, line_total=41.21),
        ExtractedLine(description="liquid glycerol | pump supplement", quantity=1, unit_price=30.36, line_total=30.36),
    ]
    extracted = _extracted(*lines, order_discount_total=14.99)

    # 42.45 / 149.98 * 14.99 = 4.24 share -> 38.21 effective
    assert effective_unit_price(lines[1], extracted) == Decimal("38.21")
    # and the untouched-by-discount comparison would have been 42.45
    assert effective_unit_price(lines[1], _extracted(*lines)) == Decimal("42.45")


def test_discount_allocation_flows_into_the_reported_discrepancy():
    invoice = _invoice(_line("0159", "2.5LBS White Chocolate Raspberry Huge Whey", 1, "59.99"))
    lines = [
        ExtractedLine(description="huge whey | protein powder", quantity=1, unit_price=42.45, line_total=42.45),
        ExtractedLine(description="liquid carnitine 3000", quantity=1, unit_price=35.96, line_total=35.96),
        ExtractedLine(description="wrecked pre workout", quantity=1, unit_price=41.21, line_total=41.21),
        ExtractedLine(description="liquid glycerol | pump supplement", quantity=1, unit_price=30.36, line_total=30.36),
    ]
    extracted = _extracted(*lines, order_discount_total=14.99)

    audit = audit_invoice(extracted, invoice, ["0159"])

    assert [d.code for d in audit.discrepancies] == ["PRICE_MISMATCH"]
    detail = audit.discrepancies[0].detail
    assert "38.21" in detail
    assert "order-level discount" in detail
    assert "21.78 more than" in detail


def test_zero_discount_basis_cannot_divide_by_zero():
    """A malformed extraction (a discount but no priced rows to allocate it
    across) must degrade, not raise.
    """
    line = ExtractedLine(description="mystery", quantity=1, unit_price=None, line_total=None)
    extracted = _extracted(line, order_discount_total=10.0)
    assert effective_unit_price(line, extracted) is None


def test_discount_larger_than_the_line_does_not_produce_a_negative_price():
    line = ExtractedLine(description="thing", quantity=1, unit_price=5.0, line_total=5.0)
    extracted = _extracted(line, order_discount_total=50.0)
    assert effective_unit_price(line, extracted) is None


# --- currency (the real CASE-1001 shape) ------------------------------------


def test_gbp_invoice_is_flagged_against_a_usd_cap():
    """CASE-1001 is plainly a GBP order (Royal Mail, UK customer, GBP
    prices), but the mock API exposes no currency field at all -- so without
    this check the amount is computed and capped as though it were USD.
    """
    extracted = _extracted(currency="GBP")

    found = check_currency(extracted)

    assert [d.code for d in found] == ["CURRENCY_MISMATCH"]
    assert found[0].severity is Severity.ESCALATE
    assert "GBP" in found[0].detail and "USD" in found[0].detail


def test_matching_currency_is_silent():
    assert check_currency(_extracted(currency="USD")) == []
    assert check_currency(_extracted(currency=" usd ")) == []


def test_absent_currency_is_not_treated_as_a_mismatch():
    """No currency printed on the document is missing information, not
    evidence of a foreign currency -- don't manufacture a finding.
    """
    assert check_currency(_extracted(currency=None)) == []


# --- phantom / unmatched lines ----------------------------------------------


def test_damaged_sku_absent_from_the_retail_invoice_is_flagged():
    """CASE-1001's AMP1 is on ShipBob's order but nowhere on the merchant's
    invoice -- likely a free promo item the customer paid nothing for.
    """
    invoice = _invoice(
        _line("AMP1", "Additional Collagen Ampoule Duo", 1, "38.00"),
        _line("COLLAGEN1", "Liposomal Tripeptide Collagen", 1, "52.00"),
    )
    extracted = _extracted(
        ExtractedLine(description="Liposomal Tripeptide Collagen", quantity=1, unit_price=52.00)
    )

    audit = audit_invoice(extracted, invoice, ["AMP1"])

    assert [d.code for d in audit.discrepancies] == ["LINE_NOT_ON_RETAIL_INVOICE"]
    assert audit.should_escalate is True
    assert "AMP1" in audit.discrepancies[0].detail


def test_unmatched_is_worded_as_unmatched_not_as_absent():
    """Matching is best-effort, so the finding must not overclaim -- a rep
    has to be able to tell "we couldn't find it" from "it isn't there".
    """
    invoice = _invoice(_line("SKU1", "Some Product", 1, "10.00"))
    audit = audit_invoice(_extracted(ExtractedLine(description="Totally Different Thing")), invoice, ["SKU1"])

    detail = audit.discrepancies[0].detail
    assert "could not be matched" in detail
    assert "is not on" not in detail


def test_only_damaged_skus_are_compared():
    """A price disagreement on some other line of the order has no bearing
    on this payout and must not add noise to the queue.
    """
    invoice = _invoice(
        _line("SKU1", "Damaged Thing", 1, "10.00"),
        _line("SKU2", "Untouched Thing", 1, "99.00"),
    )
    extracted = _extracted(
        ExtractedLine(description="Damaged Thing", sku="SKU1", quantity=1, unit_price=10.00),
        ExtractedLine(description="Untouched Thing", sku="SKU2", quantity=1, unit_price=5.00),
    )

    assert audit_invoice(extracted, invoice, ["SKU1"]).discrepancies == []


def test_sku_not_on_shipbob_invoice_is_left_to_the_calc_gate():
    """`calc.reimbursement` already raises `ItemNotOnInvoice` and
    `pipeline.py` escalates on it -- this check must not double-report it.
    """
    invoice = _invoice(_line("SKU1", "Thing", 1, "10.00"))
    assert audit_invoice(_extracted(), invoice, ["GHOST-SKU"]).discrepancies == []


# --- SKU / name matching ----------------------------------------------------


def test_suffixed_channel_sku_still_matches_the_base_sku():
    """The real CASE-1002 invoice sells `A00299-LV-8-N` where ShipBob stores
    `A00299` -- same product, channel-suffixed SKU.
    """
    invoice = _invoice(_line("A00299", "CleanBoss Foaming Cleaning Wipes 70 pack", 1, "14.99"))
    extracted = _extracted(
        ExtractedLine(
            description="CleanBoss Clean Cleaner Foaming Cleaning Wipes (70 pack)",
            sku="A00299-LV-8-N",
            quantity=1,
            unit_price=9.95,
        )
    )

    audit = audit_invoice(extracted, invoice, ["A00299"])

    assert [d.code for d in audit.discrepancies] == ["PRICE_MISMATCH"]
    assert "5.04 more than" in audit.discrepancies[0].detail


def test_matches_on_product_name_when_the_invoice_prints_no_sku():
    """CASE-1003's real invoice prints descriptions only, no SKU column."""
    invoice = _invoice(_line("0179", "Unflavored Liquid Glycerol", 1, "27.99"))
    extracted = _extracted(
        ExtractedLine(description="liquid glycerol | pump supplement", quantity=1, unit_price=30.36)
    )

    audit = audit_invoice(extracted, invoice, ["0179"])

    assert [d.code for d in audit.discrepancies] == ["PRICE_MISMATCH"]


def test_name_matching_does_not_fire_on_shared_brand_filler_alone():
    """Two different products from the same brand share tokens like the
    brand name and "pack"/"oz" -- that must not be enough to call them the
    same line, or the wrong price gets compared.
    """
    invoice = _invoice(_line("A00299", "CleanBoss Foaming Cleaning Wipes 70 pack", 1, "14.99"))
    extracted = _extracted(
        ExtractedLine(description="CleanBoss Botanical Disinfectant 24 oz 2 Pack", quantity=1, unit_price=19.99)
    )

    audit = audit_invoice(extracted, invoice, ["A00299"])

    assert [d.code for d in audit.discrepancies] == ["LINE_NOT_ON_RETAIL_INVOICE"]


def test_exact_sku_match_wins_over_a_similar_name():
    invoice = _invoice(_line("SKU-B", "Blue Widget", 1, "10.00"))
    extracted = _extracted(
        ExtractedLine(description="Blue Widget", sku="SKU-A", quantity=1, unit_price=99.00),
        ExtractedLine(description="Totally Unrelated", sku="SKU-B", quantity=1, unit_price=10.00),
    )

    assert audit_invoice(extracted, invoice, ["SKU-B"]).discrepancies == []


# --- fail-open behavior ------------------------------------------------------


def test_unreadable_invoice_is_unverified_not_escalated():
    """Fail-open by design (module docstring point 1): a blurry phone photo
    must not escalate a case that would otherwise be fine.
    """
    invoice = _invoice(_line("SKU1", "Thing", 1, "10.00"))
    extracted = ExtractedInvoice(readable=False, note="too blurry to read the totals")

    audit = audit_invoice(extracted, invoice, ["SKU1"])

    assert audit.verified is False
    assert audit.should_escalate is False
    assert audit.discrepancies == []
    assert "blurry" in audit.reason


def test_missing_extraction_is_unverified_with_the_supplied_reason():
    invoice = _invoice(_line("SKU1", "Thing", 1, "10.00"))

    audit = audit_invoice(None, invoice, ["SKU1"], extraction_error="could not read the retail invoice (TimeoutError)")

    assert audit.verified is False
    assert audit.should_escalate is False
    assert "TimeoutError" in audit.reason


def test_matched_line_with_no_readable_price_warns_rather_than_escalating():
    invoice = _invoice(_line("SKU1", "Thing", 1, "10.00"))
    extracted = _extracted(ExtractedLine(description="Thing", sku="SKU1", quantity=1))

    audit = audit_invoice(extracted, invoice, ["SKU1"])

    assert [d.code for d in audit.discrepancies] == ["RETAIL_PRICE_UNREADABLE"]
    assert audit.should_escalate is False  # WARN severity does not escalate


# --- severity / summary plumbing --------------------------------------------


def test_should_escalate_only_counts_escalate_severity():
    audit = InvoiceAudit(
        verified=True,
        discrepancies=[Discrepancy(code="X", detail="warn only", severity=Severity.WARN)],
    )
    assert audit.should_escalate is False
    assert audit.escalating == []


def test_summary_joins_only_escalating_findings():
    audit = InvoiceAudit(
        verified=True,
        discrepancies=[
            Discrepancy(code="A", detail="first problem", severity=Severity.ESCALATE),
            Discrepancy(code="B", detail="just a note", severity=Severity.WARN),
            Discrepancy(code="C", detail="second problem", severity=Severity.ESCALATE),
        ],
    )
    assert audit.summary() == "first problem; second problem"


# --- quantity handling -------------------------------------------------------


def test_multi_quantity_line_is_compared_per_unit():
    """The invoice shows a line total for 3 units; ShipBob stores a per-unit
    price. Comparing the two requires dividing, not comparing totals.
    """
    invoice = _invoice(_line("SKU1", "Thing", 3, "10.00"))
    extracted = _extracted(ExtractedLine(description="Thing", sku="SKU1", quantity=3, line_total=27.00))

    audit = audit_invoice(extracted, invoice, ["SKU1"])

    assert [d.code for d in audit.discrepancies] == ["PRICE_MISMATCH"]
    assert "9.00" in audit.discrepancies[0].detail  # 27.00 / 3


def test_bundle_kit_quantity_disagreement_is_reported_as_incomparable():
    """The real CASE-1002 kit case, caught during a live run: ShipBob fulfils
    2 units of `A00300` at 12.99 each; the merchant sold one `A00384-KIT`
    2-pack at 16.99. Comparing 12.99 against 16.99 per-unit would assert a
    confidently-wrong "4.00 less" delta -- the check has to say the
    quantities disagree instead of inventing a comparison.
    """
    invoice = _invoice(_line("A00300", "CleanBoss Multi Surface Cleaner 24oz", 2, "12.99"))
    extracted = _extracted(
        ExtractedLine(
            description="CleanBoss Everyday Multi-Surface Cleaner (24 oz) 2 Pack Kit",
            sku="A00384-KIT",
            quantity=1,
            unit_price=16.99,
            line_total=16.99,
        )
    )

    audit = audit_invoice(extracted, invoice, ["A00300"])

    assert [d.code for d in audit.discrepancies] == ["QUANTITY_MISMATCH"]
    assert audit.should_escalate is True
    detail = audit.discrepancies[0].detail
    assert "2 unit(s)" in detail and "shows 1" in detail
    assert "not directly comparable" in detail
    assert "bundle/kit" in detail
    # crucially, it must NOT assert a per-unit delta it cannot justify
    assert "less than" not in detail
    assert "more than" not in detail


def test_matching_quantities_still_compare_per_unit_normally():
    """The quantity guard must not suppress a genuine same-quantity price
    gap -- that's the main check.
    """
    invoice = _invoice(_line("SKU1", "Thing", 2, "12.99"))
    extracted = _extracted(
        ExtractedLine(description="Thing", sku="SKU1", quantity=2, line_total=20.00)
    )

    audit = audit_invoice(extracted, invoice, ["SKU1"])

    assert [d.code for d in audit.discrepancies] == ["PRICE_MISMATCH"]
    assert "10.00" in audit.discrepancies[0].detail  # 20.00 / 2


def test_zero_quantity_is_treated_as_one_rather_than_dividing_by_zero():
    line = ExtractedLine(description="Thing", sku="SKU1", quantity=0, line_total=12.00)
    assert effective_unit_price(line, _extracted(line)) == Decimal("12.00")


def test_line_discount_is_applied_before_comparison():
    line = ExtractedLine(description="Thing", sku="SKU1", quantity=1, line_total=20.00, line_discount=5.00)
    assert effective_unit_price(line, _extracted(line)) == Decimal("15.00")


# --- the extraction call itself ----------------------------------------------


async def test_extract_invoice_calls_the_real_prompt_and_parses(tmp_path: Path):
    """Exercises the real prompt file, image wrapping, and schema parsing --
    only the network call is faked.
    """
    transport = FakeTransport(
        [
            TransportResult(
                tool_input={
                    "readable": True,
                    "currency": "GBP",
                    "line_items": [
                        {
                            "description": "Liposomal Tripeptide Collagen",
                            "sku": "COLLAGEN1",
                            "quantity": 1,
                            "unit_price": 55.95,
                            "line_total": 55.95,
                            "line_discount": 0.0,
                        }
                    ],
                    "order_discount_total": 0.0,
                    "subtotal": 55.95,
                    "tax_total": None,
                    "grand_total": 55.95,
                    "order_reference": "#140744",
                    "note": None,
                },
                input_tokens=30,
                output_tokens=20,
                raw_content=[],
            )
        ]
    )

    extracted = await extract_invoice(
        "CASE-1001", b"\x89PNG-fake-bytes", transport=transport, db_path=tmp_path / "t.db"
    )

    assert extracted.readable is True
    assert extracted.currency == "GBP"
    assert extracted.line_items[0].sku == "COLLAGEN1"
    assert extracted.order_reference == "#140744"

    # the image actually reached the transport, and no ShipBob-side pricing
    # was leaked into the prompt that would anchor the model's reading
    call = transport.calls[0]
    assert call["images"] == [b"\x89PNG-fake-bytes"]
    sent_text = " ".join(str(m.get("content", "")) for m in call["messages"])
    assert "55.95" not in sent_text
    assert "COLLAGEN1" not in sent_text


async def test_extract_invoice_wraps_a_gbp_case_end_to_end(tmp_path: Path):
    """The CASE-1001 shape all the way through: extract, then reconcile
    against ShipBob's USD-assumed $52.00, and catch both problems at once.
    """
    transport = FakeTransport(
        [
            TransportResult(
                tool_input={
                    "readable": True,
                    "currency": "GBP",
                    "line_items": [
                        {
                            "description": "Liposomal Tripeptide Collagen",
                            "sku": "COLLAGEN1",
                            "quantity": 1,
                            "unit_price": 55.95,
                            "line_total": 55.95,
                            "line_discount": 0.0,
                        }
                    ],
                    "order_discount_total": 0.0,
                    "subtotal": 55.95,
                    "tax_total": None,
                    "grand_total": 55.95,
                    "order_reference": "#140744",
                    "note": None,
                },
                input_tokens=30,
                output_tokens=20,
                raw_content=[],
            )
        ]
    )
    extracted = await extract_invoice(
        "CASE-1001", b"img", transport=transport, db_path=tmp_path / "t.db"
    )
    invoice = _invoice(
        _line("COLLAGEN1", "Liposomal Tripeptide Collagen", 1, "52.00"),
        _line("AMP1", "Additional Collagen Ampoule Duo", 1, "38.00"),
    )

    audit = audit_invoice(extracted, invoice, ["COLLAGEN1", "AMP1"])

    codes = sorted(d.code for d in audit.discrepancies)
    assert codes == ["CURRENCY_MISMATCH", "LINE_NOT_ON_RETAIL_INVOICE", "PRICE_MISMATCH"]
    assert audit.should_escalate is True


# --- check-level entry points -------------------------------------------------


def test_check_damaged_item_prices_is_callable_standalone():
    """The individual checks are part of the module's surface, not just
    internals of `audit_invoice` -- they're what a future gate would reuse.
    """
    invoice = _invoice(_line("SKU1", "Thing", 1, "10.00"))
    extracted = _extracted(ExtractedLine(description="Thing", sku="SKU1", quantity=1, unit_price=8.00))

    found = check_damaged_item_prices(extracted, invoice, ["SKU1"])

    assert [d.code for d in found] == ["PRICE_MISMATCH"]
