from decimal import Decimal

import pytest

from claimpilot.calc import (
    CalcResult,
    DamagedItem,
    ItemNotOnInvoice,
    QuantityExceedsInvoice,
    reimbursement,
)
from claimpilot.config import settings
from claimpilot.models import Invoice, LineItem


def _invoice(*line_items: LineItem) -> Invoice:
    return Invoice(invoice_id="inv1", shipment_id="s1", line_items=list(line_items))


def _line(sku: str, quantity: int, unit_price: str) -> LineItem:
    return LineItem(product_id=sku, name=sku, sku=sku, quantity=quantity, unit_price=Decimal(unit_price))


def test_single_item_under_cap():
    invoice = _invoice(_line("SKU1", 5, "10.00"))
    damaged = [DamagedItem(sku="SKU1", quantity=2)]

    result = reimbursement(invoice, damaged)

    assert isinstance(result, CalcResult)
    assert result.amount == Decimal("20.00")
    assert result.capped is False
    assert len(result.line_items) == 1
    assert result.line_items[0].sku == "SKU1"
    assert result.line_items[0].quantity == 2
    assert result.line_items[0].unit_price == Decimal("10.00")
    assert result.line_items[0].subtotal == Decimal("20.00")


def test_single_item_over_cap_is_exactly_capped():
    invoice = _invoice(_line("SKU1", 10, "50.00"))
    damaged = [DamagedItem(sku="SKU1", quantity=5)]  # raw total = 250.00

    result = reimbursement(invoice, damaged)

    assert result.amount == Decimal("100.00")
    assert result.capped is True
    assert len(result.line_items) == 1
    assert result.line_items[0].subtotal == Decimal("100.00")


def test_multi_item_sum_capped_distributes_proportionally_and_sums_exactly():
    # Raw subtotals: SKU1 = 80.00, SKU2 = 40.00 -> raw total 120.00, over CAP.
    # Scale = 100/120 = 0.8333... -> SKU1 scaled = 66.666.. -> floor to 66.66
    # SKU2 (last) absorbs remainder: 100.00 - 66.66 = 33.34
    invoice = _invoice(
        _line("SKU1", 10, "8.00"),
        _line("SKU2", 10, "4.00"),
    )
    damaged = [DamagedItem(sku="SKU1", quantity=10), DamagedItem(sku="SKU2", quantity=10)]

    result = reimbursement(invoice, damaged)

    assert result.capped is True
    assert result.amount == Decimal("100.00")
    assert len(result.line_items) == 2

    sku1_item = next(li for li in result.line_items if li.sku == "SKU1")
    sku2_item = next(li for li in result.line_items if li.sku == "SKU2")

    assert sku1_item.subtotal == Decimal("66.66")
    assert sku2_item.subtotal == Decimal("33.34")

    # Exact-sum-to-the-cent invariant: naive independent rounding of each
    # item would NOT necessarily hit this exactly, which is exactly what
    # this test guards against.
    assert sku1_item.subtotal + sku2_item.subtotal == settings.cap.quantize(Decimal("0.01"))


def test_multi_item_naive_rounding_would_undershoot_but_exact_strategy_hits_cap():
    # Three equal items whose proportional shares are all 33.33... :
    # naive per-item rounding (33.33 * 3 = 99.99) would miss CAP by a cent.
    invoice = _invoice(
        _line("SKU1", 10, "10.00"),
        _line("SKU2", 10, "10.00"),
        _line("SKU3", 10, "10.00"),
    )
    damaged = [
        DamagedItem(sku="SKU1", quantity=10),
        DamagedItem(sku="SKU2", quantity=10),
        DamagedItem(sku="SKU3", quantity=10),
    ]
    # raw total = 300.00, scale = 1/3 each -> naive rounding of 33.333... to
    # 33.33 for all three sums to 99.99, not 100.00.

    result = reimbursement(invoice, damaged)

    total = sum((li.subtotal for li in result.line_items), Decimal("0"))
    assert total == Decimal("100.00")
    # First two items rounded down to 33.33, last absorbs the remainder (33.34).
    assert result.line_items[0].subtotal == Decimal("33.33")
    assert result.line_items[1].subtotal == Decimal("33.33")
    assert result.line_items[2].subtotal == Decimal("33.34")


def test_damaged_quantity_exceeding_invoiced_quantity_is_rejected():
    invoice = _invoice(_line("SKU1", 3, "10.00"))
    damaged = [DamagedItem(sku="SKU1", quantity=4)]

    with pytest.raises(QuantityExceedsInvoice) as exc_info:
        reimbursement(invoice, damaged)

    assert exc_info.value.sku == "SKU1"
    assert exc_info.value.damaged_qty == 4
    assert exc_info.value.invoiced_qty == 3


def test_sku_not_on_invoice_is_rejected():
    invoice = _invoice(_line("SKU1", 3, "10.00"))
    damaged = [DamagedItem(sku="SKU_MISSING", quantity=1)]

    with pytest.raises(ItemNotOnInvoice) as exc_info:
        reimbursement(invoice, damaged)

    assert exc_info.value.sku == "SKU_MISSING"


def test_undamaged_lines_are_ignored():
    invoice = _invoice(
        _line("SKU1", 5, "10.00"),
        _line("SKU2", 5, "20.00"),
    )
    damaged = [DamagedItem(sku="SKU1", quantity=1)]

    result = reimbursement(invoice, damaged)

    assert result.amount == Decimal("10.00")
    assert len(result.line_items) == 1
    assert result.line_items[0].sku == "SKU1"


def test_decimal_precision_no_float_contamination():
    # 0.1 + 0.2 != 0.3 in float; Decimal must not exhibit this artifact.
    invoice = _invoice(_line("SKU1", 10, "0.10"))
    damaged = [DamagedItem(sku="SKU1", quantity=3)]  # 0.10 * 3 = 0.30 exactly in Decimal

    result = reimbursement(invoice, damaged)

    assert isinstance(result.amount, Decimal)
    assert result.amount == Decimal("0.30")
    for li in result.line_items:
        assert isinstance(li.subtotal, Decimal)
        assert isinstance(li.unit_price, Decimal)


def test_duplicate_damaged_entries_for_same_sku_are_validated_cumulatively():
    # Two separate damaged entries for the same SKU, each individually
    # within the invoiced quantity, but exceeding it in aggregate.
    invoice = _invoice(_line("SKU1", 3, "10.00"))
    damaged = [DamagedItem(sku="SKU1", quantity=2), DamagedItem(sku="SKU1", quantity=2)]

    with pytest.raises(QuantityExceedsInvoice) as exc_info:
        reimbursement(invoice, damaged)

    assert exc_info.value.sku == "SKU1"
    assert exc_info.value.invoiced_qty == 3


def test_multi_item_under_cap_not_marked_capped():
    invoice = _invoice(
        _line("SKU1", 5, "10.00"),
        _line("SKU2", 5, "20.00"),
    )
    damaged = [DamagedItem(sku="SKU1", quantity=1), DamagedItem(sku="SKU2", quantity=1)]

    result = reimbursement(invoice, damaged)

    assert result.amount == Decimal("30.00")
    assert result.capped is False


def test_reimbursement_reads_settings_cap_live_not_at_import_time(monkeypatch):
    """`reimbursement()` must read `settings.cap` at call time, not cache it
    into a module-level name at import time -- a stale import-time snapshot
    would silently ignore both a `monkeypatch.setattr(settings, "cap", ...)`
    test override (breaking test isolation) and a real `CAP` env var change
    in production. A raw total of $150 is over the real default cap ($100)
    but under this overridden one ($250), so an uncapped result here is only
    possible if the override actually took effect.
    """
    monkeypatch.setattr(settings, "cap", Decimal("250.00"))

    invoice = _invoice(_line("SKU1", 10, "15.00"))
    damaged = [DamagedItem(sku="SKU1", quantity=10)]  # raw total = 150.00

    result = reimbursement(invoice, damaged)

    assert result.capped is False
    assert result.amount == Decimal("150.00")
