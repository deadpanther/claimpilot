"""Retail-invoice audit: reconcile ShipBob's order data against the
merchant's own invoice.

The brief says reimbursement is "based on the invoice -- the price at time
of fulfillment, after discounts". There are two documents that could
plausibly mean, and they are not the same thing:

1. `POST /invoices/generate` (ShipBob's API). Structured, deterministic,
   already used by `calc.reimbursement` as the authoritative payout basis.
2. The **retail invoice the merchant submits as evidence** -- the
   `EvidenceItem.ORDER_PROOF` attachment, described in the brief's own
   evidence checklist as "proof of what was ordered and at what price".

Auditing the real fixture data showed the two disagree on every priced case
in the sample set, and the API is the weaker candidate for what the brief
actually describes:

- **"after discounts" is not expressible in the API data at all.** There is
  no discount field anywhere in the mock's schema, and `generate_invoice`
  returns byte-identical line items to `get_order` -- it applies no
  transformation. The merchant invoices *do* carry discounts (CASE-1003's
  has an explicit discount column reducing a 149.98 subtotal to 134.99).
- **"at time of fulfillment" is not what the API returns.** Every generated
  invoice in the fixture set carries the same `generated_at` timestamp, and
  it postdates every delivery date in the set (one by nearly three months).
  It is a claim-time snapshot of current catalog data, not a record frozen
  at fulfillment. The merchant documents are dated at/near their orders.
- **Currency only exists on the merchant document.** The API has no currency
  field, yet one fixture case is plainly a GBP order. `settings.cap` is
  denominated in one currency; applying it to an amount in another is
  silently wrong, and the API alone gives no way to notice.

So: the merchant invoice is the better reading of the brief's intent, but a
price read off a customer-supplied image by a vision model is untrusted data
of exactly the kind this codebase refuses to let drive money (same rule as
`gates/validation.py`'s affected-count check and `guard.py`'s amount
cross-check). Both readings can't be the payout basis, and neither one alone
is safe.

**This module's resolution: the API stays the payout basis; the retail
invoice becomes an independent verification source.** `calc.reimbursement`
is untouched and still computes from `generate_invoice`. What's added is a
reconciliation pass that reads the merchant's invoice and compares it, then
routes any material disagreement to a human. The extracted figure never
adjusts an amount -- it can only ask for a person.

Design decisions:

1. **Fails open, not closed.** An unreadable/unparseable invoice, a failed
   extraction call, or a missing order-proof attachment produces
   `verified=False` with a reason -- not an escalation. This check is purely
   additive to an already-shipped pipeline: it can move a case toward more
   human attention, never toward less. Failing closed would escalate every
   case whose invoice happens to be a low-quality phone photo, which is most
   of them, and would make the queue useless. `guard.py` fails *closed* on
   missing data because it guards a send that is about to happen with money
   attached; this runs earlier, on a case that still has a human review step
   ahead of it either way, and the API-derived amount it would be checking
   is exactly as trustworthy with or without the check. The unverified state
   is surfaced prominently in the review UI rather than hidden.

2. **Matching is conservative and reports its own uncertainty.** Retail
   invoices frequently don't print SKUs (CASE-1003's doesn't) and, when they
   do, they may not be the same SKU strings ShipBob stores -- one fixture
   case sells a bundle SKU that ShipBob's record explodes into two component
   units under a different SKU entirely. `_match_line` therefore tries exact
   SKU, then SKU-prefix, then a normalized-token name overlap, and when it
   still can't find a line it reports "could not be matched to any line"
   rather than asserting "is not on the invoice". Those are genuinely
   different claims and only a human can tell them apart.

3. **Order-level discounts are allocated proportionally by line total.** A
   discount applied to the order as a whole (CASE-1003: 14.99 off the whole
   invoice) has to be attributed to the specific damaged item before it can
   be compared per-SKU, since the brief scopes reimbursement to "the
   specific damaged item only". Proportional-by-value is the standard
   allocation and matches what `calc.py` already does when scaling a capped
   payout across lines. This allocated figure is reported for comparison
   only -- see the module-level rule above.

4. **Severity is a property of the finding, not the caller.** Each
   `Discrepancy` carries its own `severity`; `InvoiceAudit.should_escalate`
   is just "any finding at ESCALATE severity". This keeps the
   escalate-or-not policy in one place instead of scattered through
   `pipeline.py`, and lets the review UI show WARN-level findings (a
   currency note, an unmatched extra line) without forcing every one of them
   to interrupt a rep.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from claimpilot.config import settings
from claimpilot.llm import Transport, structured_call
from claimpilot.models import Invoice

PROMPT_NAME = "extract_invoice"


class ExtractedLine(BaseModel):
    """One product row transcribed off the merchant's retail invoice.

    Every monetary field is a `float` rather than a `Decimal` because this is
    an LLM tool-call schema (JSON Schema has no decimal type). They are
    converted to `Decimal` at the boundary in `_as_decimal` before any
    comparison, so no float arithmetic ever reaches a reported figure.
    """

    description: str
    sku: str | None = None
    quantity: int = 1
    unit_price: float | None = None
    line_total: float | None = None
    line_discount: float = 0.0


class ExtractedInvoice(BaseModel):
    """The financial contents of an `ORDER_PROOF` attachment, as transcribed
    by the vision model. Untrusted throughout -- see module docstring.
    """

    readable: bool
    currency: str | None = None
    line_items: list[ExtractedLine] = Field(default_factory=list)
    order_discount_total: float = 0.0
    subtotal: float | None = None
    tax_total: float | None = None
    grand_total: float | None = None
    order_reference: str | None = None
    note: str | None = None


class Severity(str, Enum):
    """How hard a finding pushes. `ESCALATE` routes the case to a human via
    `pipeline.py`; `WARN` is surfaced in the review UI but does not by itself
    change the case's path (module docstring point 4).
    """

    WARN = "warn"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class Discrepancy:
    """One reconciliation finding between ShipBob's invoice and the
    merchant's. `code` is a short stable identifier a UI or test can branch
    on; `detail` is written to be read by a claims rep with no context, so it
    always names both figures and where each came from.
    """

    code: str
    detail: str
    severity: Severity = Severity.ESCALATE


@dataclass(frozen=True)
class InvoiceAudit:
    """Result of reconciling one case's two invoices.

    `verified=False` means the comparison could not be performed at all
    (no order-proof attachment, extraction failed, or the document was
    unreadable) -- `reason` says which. It is deliberately NOT an
    escalation (module docstring point 1), but the review UI surfaces it so
    a rep can see the amount went unverified rather than silently assuming
    it was checked.
    """

    verified: bool
    reason: str | None = None
    extracted: ExtractedInvoice | None = None
    discrepancies: list[Discrepancy] = field(default_factory=list)

    @property
    def should_escalate(self) -> bool:
        return any(d.severity is Severity.ESCALATE for d in self.discrepancies)

    @property
    def escalating(self) -> list[Discrepancy]:
        return [d for d in self.discrepancies if d.severity is Severity.ESCALATE]

    def summary(self) -> str:
        """One-line reason string for the audit-log payload `pipeline.py`
        records when this audit escalates a case.
        """
        return "; ".join(d.detail for d in self.escalating)


async def extract_invoice(
    case_id: str,
    order_proof_image: bytes,
    *,
    transport: Transport | None = None,
    db_path: Path | str | None = None,
) -> ExtractedInvoice:
    """Transcribe the financial contents of one `ORDER_PROOF` attachment.

    Deliberately passes no ShipBob-side data (no SKUs, no expected prices)
    into the prompt: the model must read what is actually printed, not be
    primed with the very figures this audit is trying to independently
    verify against. Anchoring it on the API's numbers would defeat the whole
    point of a second source.
    """
    messages = [
        {
            "role": "user",
            "content": (
                "Transcribe the financial contents of the attached document, "
                "which was submitted as proof of purchase for a damage claim."
            ),
        }
    ]
    return await structured_call(
        case_id=case_id,
        prompt_name=PROMPT_NAME,
        messages=messages,
        schema=ExtractedInvoice,
        images=[order_proof_image],
        transport=transport,
        db_path=db_path,
    )


def _as_decimal(value: float | None) -> Decimal | None:
    """Convert an LLM-supplied number to `Decimal` via `str` so the value
    compares as the model actually reported it, without inheriting binary
    float representation error.
    """
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Words that carry no identifying signal when matching a merchant's product
# description against ShipBob's product name -- dropping them stops
# "CleanBoss Multi Surface Cleaner 24oz" and "CleanBoss Foaming Wipes 70
# pack" from looking similar purely on shared packaging/brand filler.
_STOPWORDS = frozenset(
    {"the", "a", "an", "of", "and", "with", "for", "pack", "kit", "oz", "ounce", "count", "ct", "size"}
)


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1}


def _normalize_sku(sku: str | None) -> str:
    return (sku or "").strip().upper()


def _match_line(
    sku: str,
    product_name: str,
    lines: list[ExtractedLine],
) -> ExtractedLine | None:
    """Find the retail-invoice row corresponding to a ShipBob SKU, or `None`.

    Three passes, most-reliable first (module docstring point 2):
      1. exact SKU equality (normalized case/whitespace),
      2. SKU prefix in either direction -- covers a channel-suffixed variant
         of the same base SKU (`A00299` vs `A00299-LV-8-N`),
      3. token-overlap on the product name, requiring at least half of the
         shorter token set to match, for the common case of an invoice that
         prints no SKU at all.
    """
    target_sku = _normalize_sku(sku)

    if target_sku:
        for line in lines:
            if _normalize_sku(line.sku) == target_sku:
                return line
        for line in lines:
            line_sku = _normalize_sku(line.sku)
            if not line_sku:
                continue
            if line_sku.startswith(target_sku) or target_sku.startswith(line_sku):
                return line

    name_tokens = _tokens(product_name)
    if not name_tokens:
        return None
    best: tuple[float, ExtractedLine] | None = None
    for line in lines:
        line_tokens = _tokens(line.description)
        if not line_tokens:
            continue
        overlap = len(name_tokens & line_tokens)
        if not overlap:
            continue
        score = overlap / min(len(name_tokens), len(line_tokens))
        if score >= 0.5 and (best is None or score > best[0]):
            best = (score, line)
    return best[1] if best else None


def effective_unit_price(line: ExtractedLine, extracted: ExtractedInvoice) -> Decimal | None:
    """The per-unit price the customer actually paid for `line`, after both
    its own row discount and its proportional share of any order-level
    discount (module docstring point 3).

    Returns `None` when the row carries no usable price at all -- callers
    treat that as "couldn't verify this line" rather than as a zero.
    """
    quantity = line.quantity if line.quantity and line.quantity > 0 else 1

    gross = _as_decimal(line.line_total)
    if gross is None:
        unit = _as_decimal(line.unit_price)
        if unit is None:
            return None
        gross = unit * quantity

    net = gross - (_as_decimal(line.line_discount) or Decimal("0"))

    order_discount = _as_decimal(extracted.order_discount_total) or Decimal("0")
    if order_discount > 0:
        # Allocate by share of gross value across all priced rows. Guarded on
        # a zero/absent basis so a malformed extraction can't divide by zero
        # or hand back a negative price.
        basis = Decimal("0")
        for other in extracted.line_items:
            other_qty = other.quantity if other.quantity and other.quantity > 0 else 1
            other_gross = _as_decimal(other.line_total)
            if other_gross is None:
                other_unit = _as_decimal(other.unit_price)
                other_gross = (other_unit * other_qty) if other_unit is not None else None
            if other_gross is not None and other_gross > 0:
                basis += other_gross
        if basis > 0:
            net -= (gross / basis) * order_discount

    if net < 0:
        return None
    return (net / quantity).quantize(Decimal("0.01"))


def check_currency(extracted: ExtractedInvoice) -> list[Discrepancy]:
    """Flag a retail invoice denominated in a currency other than the one
    `settings.cap` and every stored `unit_price` are assumed to be in.

    This is the one check with no API-side counterpart to compare against:
    the mock exposes no currency field anywhere, so a foreign-currency order
    is otherwise completely invisible to this system -- the amount would be
    computed and capped as though it were `settings.expected_currency`.
    Escalates rather than warns because the resulting payout is wrong by an
    exchange rate, not by rounding, and nothing downstream can detect it.
    """
    if not extracted.currency:
        return []
    found = extracted.currency.strip().upper()
    expected = settings.expected_currency.strip().upper()
    if found == expected:
        return []
    return [
        Discrepancy(
            code="CURRENCY_MISMATCH",
            detail=(
                f"Retail invoice is denominated in {found}, but reimbursement amounts and the "
                f"{expected} {settings.cap:.2f} cap are computed as {expected}. The payout figure is "
                f"not currency-converted and the cap does not apply as written."
            ),
            severity=Severity.ESCALATE,
        )
    ]


def check_damaged_item_prices(
    extracted: ExtractedInvoice,
    invoice: Invoice,
    matched_skus: list[str],
) -> list[Discrepancy]:
    """Compare, for each confirmed damaged SKU, ShipBob's invoice price
    against the price actually printed on the merchant's retail invoice.

    Only damaged SKUs are checked -- reimbursement is scoped to "the specific
    damaged item only", so a price disagreement on some other line of the
    order has no bearing on this payout and would be noise in the queue.

    Emits `LINE_NOT_ON_RETAIL_INVOICE` when a damaged SKU can't be matched to
    any row (module docstring point 2 -- phrased as unmatched, not absent),
    and `PRICE_MISMATCH` when both figures exist and differ by more than
    `settings.invoice_price_tolerance`.
    """
    discrepancies: list[Discrepancy] = []
    by_sku = {line.sku: line for line in invoice.line_items}
    tolerance = settings.invoice_price_tolerance

    for sku in sorted(set(matched_skus)):
        api_line = by_sku.get(sku)
        if api_line is None:
            # `calc.reimbursement` raises `ItemNotOnInvoice` for this case and
            # `pipeline.py` already escalates on it -- nothing to add here.
            continue

        match = _match_line(sku, api_line.name, extracted.line_items)
        if match is None:
            discrepancies.append(
                Discrepancy(
                    code="LINE_NOT_ON_RETAIL_INVOICE",
                    detail=(
                        f"Damaged item {sku} ({api_line.name}) is on ShipBob's invoice at "
                        f"{api_line.unit_price:.2f}, but could not be matched to any line on the "
                        f"merchant's retail invoice. It may be a promotional/free item the customer "
                        f"was never charged for, or the invoice may cover a different order."
                    ),
                    severity=Severity.ESCALATE,
                )
            )
            continue

        paid = effective_unit_price(match, extracted)
        if paid is None:
            discrepancies.append(
                Discrepancy(
                    code="RETAIL_PRICE_UNREADABLE",
                    detail=(
                        f"Matched {sku} to retail-invoice line {match.description!r}, but that line "
                        f"carries no readable price, so ShipBob's {api_line.unit_price:.2f} could not "
                        f"be independently verified."
                    ),
                    severity=Severity.WARN,
                )
            )
            continue

        # A quantity disagreement makes the per-unit figures incomparable, so
        # report that rather than a confidently-wrong delta. This is the real
        # bundle/kit case, caught live on CASE-1002: ShipBob fulfils 2 units
        # of `A00300` at 12.99 each, while the merchant sold one
        # `A00384-KIT` 2-pack at 16.99 -- the kit's line price covers both
        # units, so "12.99 vs 16.99" would be comparing a unit against a
        # pair. Only a human can confirm the bundle relationship, which is
        # exactly what this hands them.
        match_quantity = match.quantity if match.quantity and match.quantity > 0 else 1
        if match_quantity != api_line.quantity:
            discrepancies.append(
                Discrepancy(
                    code="QUANTITY_MISMATCH",
                    detail=(
                        f"Damaged item {sku} ({api_line.name}): ShipBob's invoice lists "
                        f"{api_line.quantity} unit(s) at {api_line.unit_price:.2f} each, but the closest "
                        f"matching retail-invoice line ({match.description!r}) shows {match_quantity} at "
                        f"{paid:.2f}. The quantities disagree, so the per-unit prices are not directly "
                        f"comparable -- a bundle/kit sold as a single line commonly maps to several "
                        f"fulfillment units. Confirm what the customer was actually charged per unit "
                        f"before approving."
                    ),
                    severity=Severity.ESCALATE,
                )
            )
            continue

        delta = api_line.unit_price - paid
        if abs(delta) <= tolerance:
            continue

        direction = "more than" if delta > 0 else "less than"
        discrepancies.append(
            Discrepancy(
                code="PRICE_MISMATCH",
                detail=(
                    f"Damaged item {sku} ({api_line.name}): ShipBob's invoice prices it at "
                    f"{api_line.unit_price:.2f}, but the merchant's retail invoice shows the customer "
                    f"paid {paid:.2f}"
                    + (
                        f" (after a {_as_decimal(extracted.order_discount_total)} order-level discount "
                        f"allocated across lines)"
                        if (_as_decimal(extracted.order_discount_total) or Decimal("0")) > 0
                        else ""
                    )
                    + f". Reimbursing from ShipBob's figure pays {abs(delta):.2f} {direction} the "
                    f"customer actually paid."
                ),
                severity=Severity.ESCALATE,
            )
        )

    return discrepancies


def audit_invoice(
    extracted: ExtractedInvoice | None,
    invoice: Invoice,
    matched_skus: list[str],
    *,
    extraction_error: str | None = None,
) -> InvoiceAudit:
    """Reconcile the two invoices for one case. Pure -- no I/O, no LLM call.

    `extracted=None` (with `extraction_error` explaining why) and
    `extracted.readable=False` both produce an unverified result rather than
    an escalation, per module docstring point 1.
    """
    if extracted is None:
        return InvoiceAudit(
            verified=False,
            reason=extraction_error or "no order-proof attachment was available to read",
        )
    if not extracted.readable:
        return InvoiceAudit(
            verified=False,
            reason=extracted.note or "the retail invoice could not be read",
            extracted=extracted,
        )

    discrepancies = check_currency(extracted)
    discrepancies += check_damaged_item_prices(extracted, invoice, matched_skus)
    return InvoiceAudit(verified=True, extracted=extracted, discrepancies=discrepancies)
