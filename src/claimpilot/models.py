"""Pydantic domain models mirroring the real ShipBob claims API shapes."""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class CaseState(str, Enum):
    """Internal pipeline lifecycle states, persisted to SQLite as strings."""

    INTAKE = "intake"
    ELIGIBILITY = "eligibility"
    EVIDENCE = "evidence"
    VALIDATION = "validation"
    CALC = "calc"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    NEEDS_INFO = "needs_info"
    DENIED = "denied"
    SENT = "sent"
    CLOSED = "closed"
    ESCALATED = "escalated"


class EvidenceItem(str, Enum):
    """Classification of what an attachment/evidence item is."""

    ORDER_PROOF = "ORDER_PROOF"
    CUSTOMER_CONFIRMATION = "CUSTOMER_CONFIRMATION"
    PRODUCT_PHOTO = "PRODUCT_PHOTO"
    PACKAGING_PHOTO = "PACKAGING_PHOTO"


class LineItem(BaseModel):
    product_id: str
    name: str
    sku: str
    quantity: int
    unit_price: Decimal

    @field_validator("unit_price")
    @classmethod
    def unit_price_must_be_non_negative(cls, v: Decimal) -> Decimal:
        if v < 0:
            raise ValueError("unit_price must not be negative")
        return v


class Case(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    case_number: str | None = None
    # Real API status values include "New", "Closed", "Waiting on Client";
    # kept as a plain str (not an enum) so the eligibility gate can judge
    # unrecognized values instead of validation rejecting them.
    status: str
    sub_category: str | None = None
    description: str | None = None
    order_id: str | None = None
    user_id: str | None = None
    shipment_id: str | None = None
    delivered_date: str | None = None
    contact_email: str | None = None
    account_name: str | None = None
    created_date: str | None = None
    origin: str | None = None


class Shipment(BaseModel):
    shipment_id: str
    order_id: str | None = None
    carrier: str | None = None
    tracking_number: str | None = None
    status: str | None = None
    delivered_date: str | None = None
    is_insured: bool | None = None
    # The value the merchant declared for insurance/shipping purposes. Real
    # ShipBob concept, but real API responses may omit it -- kept optional so
    # a missing value is treated as "doesn't trigger the declared-value risk
    # factor" in `risk.py`, not a validation error.
    declared_value: Decimal | None = None


class Order(BaseModel):
    order_id: str
    shipment_id: str | None = None
    line_items: list[LineItem] = []


class Invoice(BaseModel):
    invoice_id: str
    shipment_id: str | None = None
    line_items: list[LineItem] = []
    generated_at: str | None = None


class Attachment(BaseModel):
    attachment_id: str
    file_name: str
    content_type: str | None = None
    url: str | None = None


class RecommendationLineItem(BaseModel):
    sku: str
    quantity: int
    unit_price: Decimal
    subtotal: Decimal


class Recommendation(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Produced deterministically by our own pipeline logic, so the
    # vocabulary is fixed (unlike Case.status, which reflects external data).
    decision: Literal["approve", "deny", "request_info"]
    amount: Decimal
    line_items: list[RecommendationLineItem] = []
    rationale: str
    email_draft: str
    confidence: float
    risk_tier: str
