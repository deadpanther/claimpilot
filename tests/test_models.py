from decimal import Decimal

import pytest
from pydantic import ValidationError

from claimpilot.models import (
    Attachment,
    Case,
    CaseState,
    EvidenceItem,
    Invoice,
    LineItem,
    Order,
    Recommendation,
    RecommendationLineItem,
    Shipment,
)


def test_line_item_rejects_negative_unit_price():
    with pytest.raises(ValidationError):
        LineItem(product_id="p1", name="Widget", sku="SKU1", quantity=1, unit_price=Decimal("-1.00"))


def test_line_item_accepts_valid_unit_price():
    item = LineItem(product_id="p1", name="Widget", sku="SKU1", quantity=2, unit_price=Decimal("9.99"))
    assert item.unit_price == Decimal("9.99")


def test_case_status_preserves_unrecognized_string():
    case = Case(case_id="c1", case_number="CN-1", status="Some Weird Status")
    assert case.status == "Some Weird Status"


def test_case_status_known_values():
    for status in ["New", "Closed", "Waiting on Client"]:
        case = Case(case_id="c1", status=status)
        assert case.status == status


def test_case_full_shape_including_origin():
    case = Case(
        case_id="c1",
        case_number="CN-1",
        status="New",
        sub_category="Claim | Damaged in Transit",
        description="Box arrived crushed",
        order_id="o1",
        user_id="u1",
        shipment_id="s1",
        delivered_date="2026-01-01",
        contact_email="a@b.com",
        account_name="Acme",
        created_date="2026-01-01",
        origin="web",
    )
    assert case.origin == "web"
    assert case.sub_category == "Claim | Damaged in Transit"


def test_shipment_valid_instance():
    shipment = Shipment(
        shipment_id="s1",
        order_id="o1",
        carrier="UPS",
        tracking_number="1Z999",
        status="Delivered",
        delivered_date="2026-01-01",
        is_insured=True,
    )
    assert shipment.is_insured is True


def test_order_valid_instance():
    line_item = LineItem(product_id="p1", name="Widget", sku="SKU1", quantity=1, unit_price=Decimal("5.00"))
    order = Order(order_id="o1", shipment_id="s1", line_items=[line_item])
    assert order.line_items[0].sku == "SKU1"


def test_invoice_valid_instance():
    line_item = LineItem(product_id="p1", name="Widget", sku="SKU1", quantity=1, unit_price=Decimal("5.00"))
    invoice = Invoice(invoice_id="inv1", shipment_id="s1", line_items=[line_item], generated_at="2026-01-01")
    assert invoice.line_items[0].name == "Widget"


def test_attachment_valid_instance():
    attachment = Attachment(
        attachment_id="a1",
        file_name="photo.jpg",
        content_type="image/jpeg",
        url="https://example.com/photo.jpg",
    )
    assert attachment.file_name == "photo.jpg"


def test_evidence_item_enum_values():
    assert EvidenceItem.ORDER_PROOF.value == "ORDER_PROOF"
    assert EvidenceItem.CUSTOMER_CONFIRMATION.value == "CUSTOMER_CONFIRMATION"
    assert EvidenceItem.PRODUCT_PHOTO.value == "PRODUCT_PHOTO"
    assert EvidenceItem.PACKAGING_PHOTO.value == "PACKAGING_PHOTO"


def test_case_state_enum_values():
    assert CaseState.INTAKE.value == "intake"
    assert CaseState.ELIGIBILITY.value == "eligibility"
    assert CaseState.EVIDENCE.value == "evidence"
    assert CaseState.VALIDATION.value == "validation"
    assert CaseState.CALC.value == "calc"
    assert CaseState.PENDING_REVIEW.value == "pending_review"
    assert CaseState.APPROVED.value == "approved"
    assert CaseState.NEEDS_INFO.value == "needs_info"
    assert CaseState.DENIED.value == "denied"
    assert CaseState.SENT.value == "sent"
    assert CaseState.CLOSED.value == "closed"
    assert CaseState.ESCALATED.value == "escalated"


def test_recommendation_valid_instance():
    rec_line_item = RecommendationLineItem(
        sku="SKU1", quantity=2, unit_price=Decimal("9.99"), subtotal=Decimal("19.98")
    )
    recommendation = Recommendation(
        decision="approve",
        amount=Decimal("19.98"),
        line_items=[rec_line_item],
        rationale="Item confirmed damaged in transit with sufficient evidence.",
        email_draft="Dear customer, your claim has been approved.",
        confidence=0.92,
        risk_tier="low",
    )
    assert recommendation.decision == "approve"
    assert recommendation.line_items[0].subtotal == Decimal("19.98")


def test_recommendation_decision_rejects_invalid_value():
    with pytest.raises(ValidationError):
        Recommendation(
            decision="approved",
            amount=Decimal("0"),
            rationale="",
            email_draft="",
            confidence=0.0,
            risk_tier="unknown",
        )


def test_case_and_recommendation_are_frozen():
    case = Case(case_id="c1", status="New")
    with pytest.raises(ValidationError):
        case.status = "Closed"

    rec = Recommendation(
        decision="deny",
        amount=Decimal("0"),
        rationale="",
        email_draft="",
        confidence=0.0,
        risk_tier="unknown",
    )
    with pytest.raises(ValidationError):
        rec.decision = "approve"
