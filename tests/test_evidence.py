"""Tests for the evidence classifier.

`classify_attachment` is exercised with `tests/test_llm.py`'s own
`FakeTransport` pattern, injected via `structured_call`'s `transport=`
override -- this exercises the *real* `classify_attachment.md` prompt file
(via `structured_call`'s real `_load_prompt`) and the real untrusted-data
wrapping, not a mocked stand-in for this module's internals. No real
network/LLM/Anthropic SDK call happens anywhere in this file.

Note: since the post-demo-v1 swappable-provider feature, `structured_call`
no longer embeds `images` into provider-specific content blocks itself --
each concrete `Transport` (`AnthropicTransport`/`OpenAITransport`) does that
on its own (see `tests/test_llm.py`). `FakeTransport` is not a real
transport, so it never performs that embedding either -- tests here that
exercise the image path assert on the raw `images` list `FakeTransport`
recorded, not on embedded content blocks.

`evidence_gaps` is pure (no I/O), so its tests construct
`AttachmentClassification` instances directly with no LLM involved at all.
"""

from __future__ import annotations

from pathlib import Path

from claimpilot.config import settings
from claimpilot.gates.evidence import (
    AttachmentClassification,
    Gap,
    classify_attachment,
    evidence_gaps,
)
from claimpilot.llm import TransportResult
from claimpilot.models import Attachment, EvidenceItem
from tests.test_llm import FakeTransport

# --- classify_attachment: image path ----------------------------------------


def _attachment(content_type: str | None, file_name: str = "photo.png") -> Attachment:
    return Attachment(attachment_id="att-1", file_name=file_name, content_type=content_type, url=None)


async def test_classify_attachment_image_path_sends_images_and_returns_classification(tmp_path: Path):
    transport = FakeTransport(
        [
            TransportResult(
                tool_input={
                    "category": "PRODUCT_PHOTO",
                    "confidence": 0.92,
                    "usable": True,
                    "quality_issue": None,
                },
                input_tokens=10,
                output_tokens=5,
                raw_content=[],
            )
        ]
    )
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"fake-png-data"

    result = await classify_attachment(
        "CASE-1",
        _attachment("image/png"),
        png_bytes,
        transport=transport,
        db_path=tmp_path / "t.db",
    )

    assert isinstance(result, AttachmentClassification)
    assert result.category == EvidenceItem.PRODUCT_PHOTO
    assert result.confidence == 0.92
    assert result.usable is True

    assert transport.calls[0]["images"] == [png_bytes]

    # The real classify_attachment.md prompt was actually loaded and hashed
    # (not a mocked-out `structured_call`) -- confirm the system prompt
    # contains content specific to that file, not the `_example` placeholder.
    assert "ORDER_PROOF" in transport.calls[0]["system"]
    assert "PACKAGING_PHOTO" in transport.calls[0]["system"]


async def test_classify_attachment_missing_content_type_takes_image_path(tmp_path: Path):
    """Per module docstring point 1: missing/empty content_type defaults to
    the image path, since every real fetched attachment today is an image.
    """
    transport = FakeTransport(
        [
            TransportResult(
                tool_input={
                    "category": "PACKAGING_PHOTO",
                    "confidence": 0.8,
                    "usable": True,
                    "quality_issue": None,
                },
                input_tokens=1,
                output_tokens=1,
                raw_content=[],
            )
        ]
    )

    result = await classify_attachment(
        "CASE-1b",
        _attachment(None),
        b"\x89PNG\r\n\x1a\nfake",
        transport=transport,
        db_path=tmp_path / "t.db",
    )

    assert result.category == EvidenceItem.PACKAGING_PHOTO
    assert transport.calls[0]["images"] == [b"\x89PNG\r\n\x1a\nfake"]


async def test_classify_attachment_covers_all_four_categories(tmp_path: Path):
    for category in EvidenceItem:
        transport = FakeTransport(
            [
                TransportResult(
                    tool_input={
                        "category": category.value,
                        "confidence": 0.85,
                        "usable": True,
                        "quality_issue": None,
                    },
                    input_tokens=1,
                    output_tokens=1,
                    raw_content=[],
                )
            ]
        )
        result = await classify_attachment(
            "CASE-cat",
            _attachment("image/jpeg"),
            b"\xff\xd8\xfffake",
            transport=transport,
            db_path=tmp_path / "t.db",
        )
        assert result.category == category


# --- classify_attachment: text path ------------------------------------------


async def test_classify_attachment_text_content_type_takes_text_path_not_images(tmp_path: Path):
    """Synthetic case: no real fixture has a non-image content_type today
    (attachment_guard.py only allows image/*), but the plan's literal
    wording calls for a text path, so this is future-proofed and exercised
    here directly.
    """
    transport = FakeTransport(
        [
            TransportResult(
                tool_input={
                    "category": "CUSTOMER_CONFIRMATION",
                    "confidence": 0.75,
                    "usable": True,
                    "quality_issue": None,
                },
                input_tokens=1,
                output_tokens=1,
                raw_content=[],
            )
        ]
    )
    text_content = b"Customer email: the item arrived shattered, please help."

    result = await classify_attachment(
        "CASE-2",
        _attachment("text/plain", file_name="customer_email.txt"),
        text_content,
        transport=transport,
        db_path=tmp_path / "t.db",
    )

    assert result.category == EvidenceItem.CUSTOMER_CONFIRMATION

    sent_messages = transport.calls[0]["messages"]
    last_content = sent_messages[-1]["content"]
    # Text path: content is a plain string (no image content blocks at all),
    # and the decoded text appears inline, wrapped in untrusted_data tags.
    assert isinstance(last_content, str)
    assert "the item arrived shattered" in last_content
    assert "<untrusted_data>" in last_content


async def test_classify_attachment_normalizes_content_type_case_and_parameters(tmp_path: Path):
    """`_is_image_content_type` must match despite uppercase content-types
    and trailing `;`-parameters (e.g. `image/jpeg; charset=binary`), both of
    which real HTTP responses can send.
    """
    transport = FakeTransport(
        [
            TransportResult(
                tool_input={
                    "category": "PRODUCT_PHOTO",
                    "confidence": 0.9,
                    "usable": True,
                    "quality_issue": None,
                },
                input_tokens=1,
                output_tokens=1,
                raw_content=[],
            )
        ]
    )

    result = await classify_attachment(
        "CASE-ct",
        _attachment("IMAGE/JPEG; charset=binary"),
        b"\xff\xd8\xfffake",
        transport=transport,
        db_path=tmp_path / "t.db",
    )

    assert result.category == EvidenceItem.PRODUCT_PHOTO
    assert transport.calls[0]["images"] == [b"\xff\xd8\xfffake"]


async def test_classify_attachment_text_path_never_raises_on_non_utf8_bytes(tmp_path: Path):
    """Text-path decoding uses errors="replace", not strict UTF-8, so a
    mislabeled/corrupted attachment degrades instead of crashing the
    pipeline (module docstring point 3).
    """
    transport = FakeTransport(
        [
            TransportResult(
                tool_input={
                    "category": "ORDER_PROOF",
                    "confidence": 0.7,
                    "usable": False,
                    "quality_issue": "content unreadable",
                },
                input_tokens=1,
                output_tokens=1,
                raw_content=[],
            )
        ]
    )
    invalid_utf8 = b"\xff\xfe\x00bad-bytes-not-utf8"

    result = await classify_attachment(
        "CASE-badbytes",
        _attachment("application/pdf", file_name="corrupt.pdf"),
        invalid_utf8,
        transport=transport,
        db_path=tmp_path / "t.db",
    )

    assert result.category == EvidenceItem.ORDER_PROOF
    # Didn't raise, and the replacement-decoded text made it into the
    # message sent to the transport.
    last_content = transport.calls[0]["messages"][-1]["content"]
    assert isinstance(last_content, str)
    assert "bad-bytes-not-utf8" in last_content


async def test_classify_attachment_pdf_content_type_also_takes_text_path(tmp_path: Path):
    transport = FakeTransport(
        [
            TransportResult(
                tool_input={
                    "category": "ORDER_PROOF",
                    "confidence": 0.8,
                    "usable": True,
                    "quality_issue": None,
                },
                input_tokens=1,
                output_tokens=1,
                raw_content=[],
            )
        ]
    )

    result = await classify_attachment(
        "CASE-3",
        _attachment("application/pdf", file_name="invoice.pdf"),
        b"Invoice #12345 total $42.00",
        transport=transport,
        db_path=tmp_path / "t.db",
    )

    assert result.category == EvidenceItem.ORDER_PROOF
    last_content = transport.calls[0]["messages"][-1]["content"]
    assert isinstance(last_content, str)
    assert "Invoice #12345" in last_content


# --- evidence_gaps: pure reconciliation logic --------------------------------


def _classification(
    category: EvidenceItem,
    *,
    confidence: float = 0.9,
    usable: bool = True,
    quality_issue: str | None = None,
) -> AttachmentClassification:
    return AttachmentClassification(
        category=category, confidence=confidence, usable=usable, quality_issue=quality_issue
    )


def _all_good() -> list[AttachmentClassification]:
    return [_classification(item) for item in EvidenceItem]


def test_evidence_gaps_no_gaps_when_all_four_present_usable_and_confident():
    assert evidence_gaps(_all_good()) == []


def test_evidence_gaps_missing_category_entirely():
    classified = [c for c in _all_good() if c.category != EvidenceItem.PACKAGING_PHOTO]

    gaps = evidence_gaps(classified)

    assert gaps == [Gap(item=EvidenceItem.PACKAGING_PHOTO, reason="MISSING", detail=None)]


def test_evidence_gaps_unusable_instance_surfaces_quality_issue():
    classified = [
        c for c in _all_good() if c.category != EvidenceItem.PRODUCT_PHOTO
    ] + [
        _classification(EvidenceItem.PRODUCT_PHOTO, usable=False, quality_issue="blurry")
    ]

    gaps = evidence_gaps(classified)

    assert gaps == [Gap(item=EvidenceItem.PRODUCT_PHOTO, reason="UNUSABLE", detail="blurry")]


def test_evidence_gaps_unusable_without_quality_issue_gets_fallback_detail():
    classified = [
        c for c in _all_good() if c.category != EvidenceItem.PRODUCT_PHOTO
    ] + [
        _classification(EvidenceItem.PRODUCT_PHOTO, usable=False, quality_issue=None)
    ]

    gaps = evidence_gaps(classified)

    assert len(gaps) == 1
    assert gaps[0].item == EvidenceItem.PRODUCT_PHOTO
    assert gaps[0].reason == "UNUSABLE"
    assert gaps[0].detail  # non-empty fallback string, not None


def test_evidence_gaps_low_confidence_below_threshold_is_a_gap():
    classified = [
        c for c in _all_good() if c.category != EvidenceItem.ORDER_PROOF
    ] + [
        _classification(EvidenceItem.ORDER_PROOF, confidence=0.5, usable=True)
    ]

    gaps = evidence_gaps(classified)

    assert len(gaps) == 1
    assert gaps[0].item == EvidenceItem.ORDER_PROOF
    assert gaps[0].reason == "LOW_CONFIDENCE"
    # `detail` must be customer-safe email text, not an internal diagnostic
    # (e.g. must NOT leak the raw "0.50" confidence number into an email).
    assert gaps[0].detail
    assert "0.5" not in gaps[0].detail
    assert "confirm" in gaps[0].detail.lower()


def test_evidence_gaps_unusable_takes_precedence_over_low_confidence():
    """When both conditions are true (usable=False AND confidence below
    threshold), the reported reason is UNUSABLE, not LOW_CONFIDENCE -- module
    docstring point 4's documented precedence.
    """
    classified = [
        c for c in _all_good() if c.category != EvidenceItem.ORDER_PROOF
    ] + [
        _classification(
            EvidenceItem.ORDER_PROOF, confidence=0.2, usable=False, quality_issue="cropped"
        )
    ]

    gaps = evidence_gaps(classified)

    assert len(gaps) == 1
    assert gaps[0].reason == "UNUSABLE"
    assert gaps[0].detail == "cropped"


def test_evidence_gaps_confidence_exactly_at_threshold_is_not_a_gap():
    classified = [
        c for c in _all_good() if c.category != EvidenceItem.ORDER_PROOF
    ] + [
        _classification(EvidenceItem.ORDER_PROOF, confidence=settings.evidence_min_conf, usable=True)
    ]

    assert evidence_gaps(classified) == []


def test_evidence_gaps_reads_settings_evidence_min_conf_live_not_at_import_time(monkeypatch):
    """`evidence_gaps` must read `settings.evidence_min_conf` at call time,
    not cache it into a module-level name at import time -- a stale
    import-time snapshot would silently ignore both a
    `monkeypatch.setattr(settings, ...)` test override and a real env var
    change in production. Confidence 0.5 is below the real default (0.7) but
    at/above this overridden one (0.3) -- no gap here is only possible if
    `evidence_gaps` actually read the live override.
    """
    monkeypatch.setattr(settings, "evidence_min_conf", 0.3)
    classified = [
        c for c in _all_good() if c.category != EvidenceItem.ORDER_PROOF
    ] + [
        _classification(EvidenceItem.ORDER_PROOF, confidence=0.5, usable=True)
    ]

    assert evidence_gaps(classified) == []


def test_evidence_gaps_duplicate_category_one_bad_one_good_is_not_a_gap():
    """Reconciliation rule: any usable+confident instance satisfies the
    category, even if another attachment classified into the same category
    was bad.
    """
    classified = [
        c for c in _all_good() if c.category != EvidenceItem.CUSTOMER_CONFIRMATION
    ] + [
        _classification(EvidenceItem.CUSTOMER_CONFIRMATION, usable=False, quality_issue="too dark"),
        _classification(EvidenceItem.CUSTOMER_CONFIRMATION, usable=True, confidence=0.95),
    ]

    assert evidence_gaps(classified) == []


def test_evidence_gaps_duplicate_category_all_bad_uses_first_in_order_for_detail():
    classified = [
        c for c in _all_good() if c.category != EvidenceItem.CUSTOMER_CONFIRMATION
    ] + [
        _classification(EvidenceItem.CUSTOMER_CONFIRMATION, usable=False, quality_issue="too dark"),
        _classification(EvidenceItem.CUSTOMER_CONFIRMATION, usable=False, quality_issue="cropped"),
    ]

    gaps = evidence_gaps(classified)

    assert len(gaps) == 1
    assert gaps[0].detail == "too dark"


def test_evidence_gaps_empty_input_reports_all_four_missing():
    gaps = evidence_gaps([])

    assert [g.item for g in gaps] == list(EvidenceItem)
    assert all(g.reason == "MISSING" for g in gaps)


def test_evidence_gaps_output_order_matches_enum_declaration_order():
    # Deliberately supply classifications in a different order than the enum
    # declares them, to prove output order tracks EvidenceItem, not input.
    classified = [
        _classification(EvidenceItem.PACKAGING_PHOTO, usable=False, quality_issue="x"),
        _classification(EvidenceItem.ORDER_PROOF, usable=False, quality_issue="y"),
    ]

    gaps = evidence_gaps(classified)

    assert [g.item for g in gaps] == [
        EvidenceItem.ORDER_PROOF,
        EvidenceItem.CUSTOMER_CONFIRMATION,
        EvidenceItem.PRODUCT_PHOTO,
        EvidenceItem.PACKAGING_PHOTO,
    ]
