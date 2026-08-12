"""Reimbursement calculator.

Pure, deterministic computation of the reimbursement amount for a set of
damaged line items against an invoice. No I/O, no LLM calls -- everything
the function needs is passed in as plain arguments, matching the style of
`gates/eligibility.py`.

Business rules (see the task write-up for the full ambiguity discussion):

1. For each `DamagedItem`, look up the matching `LineItem` on the invoice by
   `sku`. A damaged SKU that isn't on the invoice raises `ItemNotOnInvoice`
   -- we never silently ignore or zero it out, since that would understate
   or fabricate a claim without a human noticing.
2. A damaged quantity greater than the invoiced quantity for that SKU raises
   `QuantityExceedsInvoice` rather than being silently clamped to the
   invoiced quantity. Silently clamping would hide a data-quality problem
   (e.g. the damage-assessment step over-counting) behind a plausible-looking
   number; the caller needs to see the failure and reconcile it.
3. Lines not present in `damaged` are simply not reimbursed -- only damaged
   lines contribute to the total.
4. The raw total (`Σ unit_price × damaged qty`) is capped at `settings.cap`
   (`claimpilot.config.Settings.cap`, currently $100.00, env-overridable via
   `CAP`). It lives in `config.py` alongside `claim_window_days` /
   `eligible_sub_category` for the same reason documented in
   `gates/eligibility.py`: it's a business-policy value, not local
   implementation detail --
   read from `settings` at call time here, never cached at import time.

Cap-distribution design (the ambiguous point the plan leaves open):

The plan notes "POST /reimbursements takes one product per call," meaning
each line item's amount in `CalcResult.line_items` will later be submitted
as an independent reimbursement request. When the raw total is under the
cap, no distribution question arises -- each item is reimbursed at its full
subtotal. When the raw total exceeds the cap, we chose to **proportionally
scale down every line item's subtotal** so the per-item amounts sum to
exactly `CAP`, rather than e.g. fully paying items in list order until the
cap is exhausted ("first-come-first-served") or splitting the cap evenly
per item regardless of value. Proportional scaling was chosen because it
preserves the *relative* weighting between items -- a $80 item and a $20
item both lose 20% of their claim if the cap forces an overall 20% cut,
rather than one item being paid in full and the other zeroed out (which a
first-come-first-served split would do based on arbitrary list order).

Rounding: naively rounding each scaled item to the nearest cent independently
can leave the per-item amounts summing to one cent above or below `CAP`
(e.g. three items scaling to 33.33/33.33/33.33 sum to 99.99, not 100.00).
To guarantee an *exact* sum-to-the-cent, every item except the last is
rounded down (`ROUND_DOWN`) to the nearest cent, and the last item absorbs
whatever remainder is needed to make the total land on exactly
`CAP.quantize(Decimal("0.01"))`. "Last" here means the last item in the
order the caller passed `damaged` -- this is an arbitrary but deterministic
tie-break (documented so it doesn't drift), and moving to e.g. largest-
remainder allocation would be a reasonable alternative if the "which item
eats the rounding cent" choice ever needs to be fairer.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

from claimpilot.config import settings
from claimpilot.models import Invoice, RecommendationLineItem

CENT = Decimal("0.01")


@dataclass(frozen=True)
class DamagedItem:
    """A single damaged SKU + quantity, as found by the (not-yet-built)
    damage-assessment step. Plain dataclass (not pydantic) since it's an
    internal, already-trusted intermediate value -- consistent with
    `EligibilityResult`'s style in `gates/eligibility.py`, and there's no external
    payload to validate here (unlike `LineItem`/`Invoice`, which come
    straight off the wire).
    """

    sku: str
    quantity: int


@dataclass(frozen=True)
class CalcResult:
    """Output of `reimbursement`: the final capped total plus a per-item
    breakdown suitable for submitting one `POST /reimbursements` call per
    line item later.
    """

    amount: Decimal
    line_items: list[RecommendationLineItem]
    # Whether the raw (pre-cap) total exceeded CAP and was truncated.
    # Kept for audit-trail/UI purposes -- lets a reviewer or the eventual
    # UI distinguish "this claim was paid in full" from "this claim hit the
    # policy cap and was scaled down" without recomputing the raw total.
    capped: bool


class ItemNotOnInvoice(Exception):
    """Raised when a damaged SKU has no matching line item on the invoice."""

    def __init__(self, sku: str) -> None:
        self.sku = sku
        super().__init__(f"SKU {sku!r} not found on invoice")


class QuantityExceedsInvoice(Exception):
    """Raised when a damaged item's quantity exceeds the invoiced quantity
    for that SKU. We never silently clamp -- the caller must see and
    reconcile the discrepancy.
    """

    def __init__(self, sku: str, damaged_qty: int, invoiced_qty: int) -> None:
        self.sku = sku
        self.damaged_qty = damaged_qty
        self.invoiced_qty = invoiced_qty
        super().__init__(
            f"Damaged quantity {damaged_qty} for SKU {sku!r} exceeds "
            f"invoiced quantity {invoiced_qty}"
        )


def reimbursement(invoice: Invoice, damaged: list[DamagedItem]) -> CalcResult:
    """Compute the capped reimbursement amount and per-item breakdown for
    `damaged` lines against `invoice`.

    Raises:
        ItemNotOnInvoice: a damaged SKU is missing from `invoice.line_items`.
        QuantityExceedsInvoice: a damaged quantity exceeds the invoiced
            quantity for that SKU.
    """
    # Read at call time (not snapshotted into a module-level name), so a
    # `monkeypatch.setattr(settings, "cap", ...)` override in a test -- or a
    # real `CAP` env var change -- is honored immediately.
    cap = settings.cap
    invoice_by_sku = {line.sku: line for line in invoice.line_items}

    subtotals: list[tuple[str, int, Decimal, Decimal]] = []  # sku, qty, unit_price, subtotal
    raw_total = Decimal("0")
    # Track cumulative damaged quantity per SKU: `damaged` could (in theory)
    # list the same SKU more than once, and each entry's quantity must be
    # validated against what's *left* of the invoiced quantity, not checked
    # independently -- otherwise two damaged entries of qty 3 each against
    # an invoiced qty of 3 would each individually pass, silently claiming
    # double the invoiced quantity.
    seen_quantity_by_sku: dict[str, int] = {}
    for item in damaged:
        line = invoice_by_sku.get(item.sku)
        if line is None:
            raise ItemNotOnInvoice(item.sku)

        cumulative_quantity = seen_quantity_by_sku.get(item.sku, 0) + item.quantity
        if cumulative_quantity > line.quantity:
            raise QuantityExceedsInvoice(item.sku, cumulative_quantity, line.quantity)
        seen_quantity_by_sku[item.sku] = cumulative_quantity

        subtotal = line.unit_price * item.quantity
        subtotals.append((item.sku, item.quantity, line.unit_price, subtotal))
        raw_total += subtotal

    if raw_total <= cap:
        line_items = [
            RecommendationLineItem(sku=sku, quantity=qty, unit_price=unit_price, subtotal=subtotal)
            for sku, qty, unit_price, subtotal in subtotals
        ]
        return CalcResult(amount=raw_total.quantize(CENT), line_items=line_items, capped=False)

    # Over cap: proportionally scale each subtotal down so the per-item
    # amounts sum to exactly the cap. All items but the last are rounded
    # down to the cent; the last item absorbs the remainder so the sum is
    # exact.
    scale = cap / raw_total
    target_total = cap.quantize(CENT)

    line_items = []
    allocated = Decimal("0")
    last_index = len(subtotals) - 1
    for index, (sku, qty, unit_price, subtotal) in enumerate(subtotals):
        if index == last_index:
            scaled = target_total - allocated
        else:
            scaled = (subtotal * scale).quantize(CENT, rounding=ROUND_DOWN)
            allocated += scaled
        line_items.append(RecommendationLineItem(sku=sku, quantity=qty, unit_price=unit_price, subtotal=scaled))

    return CalcResult(amount=target_total, line_items=line_items, capped=True)
