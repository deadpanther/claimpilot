"""Tests for the review UI + approval endpoints.

Drives real cases through `pipeline.process_case` first (same scripted
transport shapes as `tests/test_pipeline.py`) to get a realistic
`Recommendation` + audit trail into a `tmp_path` SQLite database, then hits
the FastAPI app built by `claimpilot.web.app.create_app(...)` via
`TestClient` -- same test-injection convention as `process_case` itself
(explicit `client`/`transport`/`db_path` overrides, never the real
network/LLM/on-disk database).

The two explicit plan requirements this file proves:
- "nothing hits outbox without approve" (`test_nothing_sent_before_approve`).
- "edited body is what gets sent"
  (`test_edited_body_is_what_gets_sent_and_correction_recorded`).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from claimpilot.clients.fixtures import FixtureClient
from claimpilot.db import get_connection
from claimpilot.llm import TransportResult
from claimpilot.memory import global_policies, merchant_context, record_policy_note
from claimpilot.models import CaseState
from claimpilot.pipeline import process_case
from claimpilot.clients.synthetic import synthetic_case_ids
from claimpilot.config import settings
from claimpilot.store import (
    create_case,
    get_audit_log,
    get_case,
    get_recommendation,
    get_status,
    list_cases_by_status,
    transition,
)
from claimpilot.web.app import create_app
from tests.test_llm import FakeTransport

NOW = datetime(2026, 3, 25, tzinfo=timezone.utc)


def _memory_correction_rows(db_path: Path, case_id: str) -> list[dict]:
    """Raw query against the `memory` table for `kind="correction"` rows
    written about `case_id` -- there's no `claimpilot.memory` reader keyed
    on `source_case_id` yet (only `merchant_id`), so this mirrors
    `test_store.py`'s convention of querying tables directly for assertions
    the public API doesn't expose a getter for.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM memory WHERE kind = 'correction' AND source_case_id = ? ORDER BY id",
            (case_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

ORIGINAL_DRAFT = "Dear customer, original draft."


class _RecordingClient:
    """Wraps a real `ShipBobClient`, delegating everything except:

    - `get_attachment_bytes`: deterministic canned bytes, no real network
      call (same reasoning as `test_pipeline.py`'s `_NoNetworkBytesClient`).
    - `send_email` / `submit_reimbursement`: recorded in-memory instead of
      writing to `FixtureClient`'s real `outbox/outbox.jsonl` file, so these
      tests never touch the filesystem outbox and can assert exactly what
      was "sent" without parsing JSONL off disk.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.sent_emails: list[dict] = []
        self.reimbursements: list[dict] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def get_attachment_bytes(self, attachment) -> bytes:
        return b"\x89PNG\r\n\x1a\n" + attachment.attachment_id.encode()

    async def send_email(self, case_id, to, subject, body) -> dict:
        self.sent_emails.append({"case_id": case_id, "to": to, "subject": subject, "body": body})
        return {"success": True, "case_id": case_id}

    async def submit_reimbursement(self, case_id, order_id, user_id, shipment_id, product_name, amount) -> dict:
        self.reimbursements.append(
            {
                "case_id": case_id,
                "order_id": order_id,
                "user_id": user_id,
                "shipment_id": shipment_id,
                "product_name": product_name,
                "amount": amount,
            }
        )
        return {"status": "submitted"}


@pytest.fixture
def client() -> _RecordingClient:
    return _RecordingClient(FixtureClient(include_synthetic=True))


def _classification(category: str, *, confidence: float = 0.95, usable: bool = True) -> TransportResult:
    return TransportResult(
        tool_input={"category": category, "confidence": confidence, "usable": usable, "quality_issue": None},
        input_tokens=1,
        output_tokens=1,
        raw_content=[],
    )


def _validation(*, matched_skus: list[str], confidence: float = 0.9) -> TransportResult:
    judgment = {"passed": True, "confidence": confidence, "note": "looks consistent with the claim"}
    return TransportResult(
        tool_input={
            "damage_visible": judgment,
            "product_identifiable": judgment,
            "product_on_invoice": judgment,
            "packaging_documented": judgment,
            "matched_skus": matched_skus,
        },
        input_tokens=1,
        output_tokens=1,
        raw_content=[],
    )


def _draft(email_draft: str = ORIGINAL_DRAFT, rationale: str = "Deterministic pipeline decision.") -> TransportResult:
    return TransportResult(
        tool_input={"rationale": rationale, "email_draft": email_draft},
        input_tokens=1,
        output_tokens=1,
        raw_content=[],
    )


def _extraction(
    *,
    readable: bool = False,
    currency: str | None = None,
    line_items: list[dict] | None = None,
    order_discount_total: float = 0.0,
    note: str | None = "the uploaded document is not legible enough to read prices from",
) -> TransportResult:
    """Canned `gates.invoice_audit.extract_invoice` response.

    Defaults to `readable=False`, which `audit_invoice` turns into an
    unverified audit with zero discrepancies -- so a test that isn't
    specifically about retail-invoice reconciliation keeps exactly the
    pipeline outcome it had before that check was added, while still
    supplying the extra scripted LLM response the call now consumes.
    Tests that *do* exercise the audit pass real `line_items`/`currency`.
    """
    return TransportResult(
        tool_input={
            "readable": readable,
            "currency": currency,
            "line_items": line_items or [],
            "order_discount_total": order_discount_total,
            "subtotal": None,
            "tax_total": None,
            "grand_total": None,
            "order_reference": None,
            "note": note,
        },
        input_tokens=15,
        output_tokens=8,
        raw_content=[],
    )


def _app(client: _RecordingClient, db_path: Path, transport: FakeTransport | None = None):
    return create_app(client=client, transport=transport or FakeTransport([]), db_path=db_path)


# --- access control (found missing entirely in a full-codebase security audit) --


def test_no_auth_required_when_credentials_unconfigured(client: _RecordingClient, tmp_path: Path):
    """Default state (settings.review_ui_username/password both blank) --
    every other test in this file relies on this working, but this test
    makes it an explicit, named assertion rather than an implicit side
    effect of 425 other tests happening to pass.
    """
    db_path = tmp_path / "t.db"
    with TestClient(_app(client, db_path)) as tc:
        resp = tc.get("/cases")
    assert resp.status_code == 200


def test_health_endpoint_never_requires_auth(client: _RecordingClient, tmp_path: Path, monkeypatch):
    """`/health` must stay reachable with no credentials even when Basic
    Auth is fully configured -- Docker's HEALTHCHECK (and any load
    balancer) has no way to supply credentials, and a liveness probe must
    never depend on them.
    """
    from claimpilot.config import settings

    monkeypatch.setattr(settings, "review_ui_username", "rep")
    monkeypatch.setattr(settings, "review_ui_password", "secret")
    db_path = tmp_path / "t.db"
    with TestClient(_app(client, db_path)) as tc:
        resp = tc.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_configured_auth_rejects_missing_credentials(
    client: _RecordingClient, tmp_path: Path, monkeypatch
):
    from claimpilot.config import settings

    monkeypatch.setattr(settings, "review_ui_username", "rep")
    monkeypatch.setattr(settings, "review_ui_password", "secret")
    db_path = tmp_path / "t.db"
    with TestClient(_app(client, db_path)) as tc:
        resp = tc.get("/cases")
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Basic"


def test_configured_auth_rejects_wrong_credentials(
    client: _RecordingClient, tmp_path: Path, monkeypatch
):
    from claimpilot.config import settings

    monkeypatch.setattr(settings, "review_ui_username", "rep")
    monkeypatch.setattr(settings, "review_ui_password", "secret")
    db_path = tmp_path / "t.db"
    with TestClient(_app(client, db_path)) as tc:
        resp = tc.get("/cases", auth=("rep", "wrong-password"))
    assert resp.status_code == 401


def test_configured_auth_accepts_correct_credentials(
    client: _RecordingClient, tmp_path: Path, monkeypatch
):
    from claimpilot.config import settings

    monkeypatch.setattr(settings, "review_ui_username", "rep")
    monkeypatch.setattr(settings, "review_ui_password", "secret")
    db_path = tmp_path / "t.db"
    with TestClient(_app(client, db_path)) as tc:
        resp = tc.get("/cases", auth=("rep", "secret"))
    assert resp.status_code == 200


async def test_configured_auth_protects_post_routes_too(
    client: _RecordingClient, tmp_path: Path, monkeypatch
):
    """Not just the queue GET -- a POST action route (approve/pushback/etc.)
    must be equally protected, since those are exactly the routes that can
    send a real email or submit a real reimbursement. Confirms the blocked
    request never reaches the handler at all (case stays untouched in
    PENDING_REVIEW, nothing hits the outbox) -- not just that the HTTP
    response code is 401.
    """
    from claimpilot.config import settings

    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)  # -> PENDING_REVIEW, no auth configured yet

    monkeypatch.setattr(settings, "review_ui_username", "rep")
    monkeypatch.setattr(settings, "review_ui_password", "secret")
    with TestClient(_app(client, db_path)) as tc:
        resp = tc.post(
            "/cases/CASE-1002/approve", data={"action": "approve"}, follow_redirects=False
        )

    assert resp.status_code == 401
    assert get_status("CASE-1002", db_path=db_path) == CaseState.PENDING_REVIEW
    assert client.sent_emails == []


# --- fixture-case setup helpers (mirrors tests/test_pipeline.py's scripting) -


async def _process_1002_to_approve(client: _RecordingClient, db_path: Path) -> None:
    """CASE-1002 (4 attachments) -> full happy path -> `approve`,
    `PENDING_REVIEW`. Single matched SKU (A00360, $24.99, under cap).
    """
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["A00360"]),
            _extraction(),
            _draft(),
        ]
    )
    await process_case("CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW)


async def _process_1001_to_request_info(client: _RecordingClient, db_path: Path) -> None:
    """CASE-1001 (3 attachments) -> one evidence gap -> `request_info`,
    `PENDING_REVIEW`.
    """
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _draft(),
        ]
    )
    await process_case("CASE-1001", client=client, transport=transport, db_path=db_path, now=NOW)


async def _process_9001_to_escalated(client: _RecordingClient, db_path: Path) -> None:
    """CASE-9001-INSURED (synthetic) -> insured routing -> `ESCALATED`."""
    transport = FakeTransport([_draft()])
    await process_case("CASE-9001-INSURED", client=client, transport=transport, db_path=db_path, now=NOW)


def _make_hidden_closed_case(db_path: Path) -> None:
    """Walks a case straight to `CLOSED` via the store API directly
    (bypassing the client/pipeline entirely -- `process_case` never reaches
    `CLOSED` on its own, per its "always ends in pending_review or
    escalated" invariant). Used as the negative case for the queue-scope
    test: a `CLOSED` case must never appear in `GET /cases`.
    """
    create_case("CASE-HIDDEN", db_path=db_path)
    for to_state in (
        CaseState.ELIGIBILITY,
        CaseState.EVIDENCE,
        CaseState.VALIDATION,
        CaseState.CALC,
        CaseState.PENDING_REVIEW,
        CaseState.APPROVED,
        CaseState.SENT,
        CaseState.CLOSED,
    ):
        transition("CASE-HIDDEN", to_state, actor="system", event="advance", db_path=db_path)


# --- queue page ----------------------------------------------------------


async def test_queue_lists_only_pending_review_and_escalated_with_risk_badges(
    client: _RecordingClient, tmp_path: Path
):
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)  # -> PENDING_REVIEW
    await _process_9001_to_escalated(client, db_path)  # -> ESCALATED
    _make_hidden_closed_case(db_path)  # -> CLOSED, must never show up

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.get("/cases")

    assert resp.status_code == 200
    assert "CASE-1002" in resp.text
    assert "CASE-9001-INSURED" in resp.text
    assert "CASE-HIDDEN" not in resp.text
    # Risk-tier badge rendered for at least one case (exact tier depends on
    # fixture data, so just confirm one of the three badge classes appears).
    assert any(f"badge-{tier}" in resp.text for tier in ("LOW", "ELEVATED", "HIGH"))


async def test_queue_stats_bar_renders_real_computed_numbers(client: _RecordingClient, tmp_path: Path):
    """The queue header's stats bar must render actual computed
    numbers, not just labels -- drives one case to a clean approve (no
    edit), one case to a pushback (never approved), and one case straight
    to ESCALATED, then checks the rendered percentages match hand-derived
    values for that exact scenario.

    - CASE-1002: happy path -> approve (action="approve", no edit).
    - CASE-1001: request_info path -> pushed back once, never approved.
    - CASE-9001-INSURED: escalates directly from ELIGIBILITY (never visits
      PENDING_REVIEW), so it must NOT count in the reviewed/pushback
      denominator, only in the escalation-rate denominator+numerator.

    Hand-derived expectations:
      total_case_count = 3, escalated = 1               -> escalation 33%
      reviewed_case_count = 2 (1002, 1001)               -> pushback 1/2 = 50%
      approved_count = 1 (1002 only), 0 corrections      -> approve-as-is 100%, edit 0%
    """
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)
    await _process_1001_to_request_info(client, db_path)
    await _process_9001_to_escalated(client, db_path)

    pushback_transport = FakeTransport([_draft(email_draft="Dear customer, revised.", rationale="Updated.")])
    with TestClient(_app(client, db_path, transport=pushback_transport)) as tc:
        tc.post(
            "/cases/CASE-1001/pushback",
            data={"feedback": "Please double-check the SKU."},
            follow_redirects=False,
        )
        tc.post("/cases/CASE-1002/approve", data={"action": "approve"}, follow_redirects=False)

        resp = tc.get("/cases")

    assert resp.status_code == 200
    body = resp.text

    def stat_value(label: str) -> str:
        # Anchor to the specific label's adjacent value span -- a plain
        # substring check (e.g. `"0%" in body`) would pass vacuously since
        # "0%" is a substring of "100%"/"50%", defeating the point of this
        # test (real, per-metric numbers, not just any digit on the page).
        match = re.search(
            re.escape(label) + r"</span>\s*<span class=\"stat-value\">([^<]+)</span>", body
        )
        assert match, f"stat {label!r} not found in rendered queue page"
        return match.group(1).strip()

    assert stat_value("Approve as-is") == "100%"
    assert stat_value("Edit rate") == "0%"
    assert stat_value("Pushback rate") == "50%"
    assert stat_value("Escalation rate") == "33%"
    # Cost/latency are real (non-placeholder) numbers -- the pipeline made
    # actual LLM calls via FakeTransport, so these must not render as the
    # "—" no-data placeholder.
    assert re.match(r"^\$\d+\.\d+$", stat_value("Mean LLM cost / claim"))
    assert re.match(r"^\d+(\.\d+)?(ms|s)$", stat_value("Mean LLM latency / claim"))


# --- case history page ------------------------------------------------------


async def test_history_shows_sent_and_closed_cases_hidden_from_queue(
    client: _RecordingClient, tmp_path: Path
):
    """`GET /cases/history` is the complement of `GET /cases`: cases that
    have left the active queue (approved/denied -> SENT, or a fully wrapped
    up CLOSED case) must still be reachable somewhere in the UI rather than
    silently disappearing the moment a rep acts on them. A still-pending
    case must NOT show up here (it belongs on the queue, not history).
    """
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)  # -> PENDING_REVIEW
    _make_hidden_closed_case(db_path)  # -> CLOSED

    with TestClient(_app(client, db_path)) as tc:
        tc.post("/cases/CASE-1002/approve", data={"action": "approve"}, follow_redirects=False)
        resp = tc.get("/cases/history")

    assert resp.status_code == 200
    body = resp.text
    assert "CASE-1002" in body
    assert "status-badge-sent" in body
    assert "CASE-HIDDEN" in body
    assert "status-badge-closed" in body
    assert 'href="/cases/CASE-1002"' in body  # links back into the case detail page


async def test_history_excludes_still_pending_cases(client: _RecordingClient, tmp_path: Path):
    db_path = tmp_path / "t.db"
    await _process_1001_to_request_info(client, db_path)  # stays PENDING_REVIEW

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.get("/cases/history")

    assert resp.status_code == 200
    assert "CASE-1001" not in resp.text
    assert "No cases have been sent, denied, or closed yet." in resp.text


# --- case detail page ------------------------------------------------------


async def test_case_detail_renders_draft_evidence_and_calc_breakdown(
    client: _RecordingClient, tmp_path: Path
):
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.get("/cases/CASE-1002")

    assert resp.status_code == 200
    body = resp.text
    assert ORIGINAL_DRAFT in body  # editable textarea pre-filled with the stored draft
    assert "A00360" in body  # calc breakdown line item
    assert "24.99" in body
    assert "IMG_9726.jpeg" in body  # evidence attachment filename
    assert "/cases/CASE-1002/attachments/ATT-CASE-1002-01" in body  # thumbnail src
    # Lightbox: thumbnails are clickable into an enlarged prev/next viewer,
    # and the JSON payload driving it is embedded with real attachment data
    # (not an empty/placeholder array).
    assert 'onclick="openLightbox(0)"' in body
    assert 'id="lightbox-overlay"' in body
    assert '"url": "/cases/CASE-1002/attachments/ATT-CASE-1002-01"' in body
    assert '"name": "IMG_9726.jpeg"' in body
    # Original-claim section: the merchant's own submitted description and
    # claim metadata, not just our pipeline's derived recommendation.
    assert "Damage due to poor/bad packaging" in body  # case.description, verbatim
    assert "01838273" in body  # case_number
    assert "Claim | Damaged in Transit" in body  # sub_category
    assert "Case Portal - Claim" in body  # origin
    assert "344745459" in body  # shipment_id
    # Email draft header preview (UI-polish audit finding): the same
    # recipient/subject approve() actually sends with, shown above the
    # draft textarea -- not just a bare, headerless body.
    assert "mtaparia@shipbob.com" in body
    assert "Update on your ShipBob claim CASE-1002" in body
    # Merchant memory section renders real data (Task: wire up case-page memory panel),
    # not the old static disclaimer -- empty-state text since this tmp_path db has no
    # memory rows recorded yet for CASE-1002's merchant (283959).
    assert "No policy notes recorded for this merchant or globally yet." in body
    assert "No recent notes or corrections recorded for this merchant." in body
    assert "Claim frequency (last 90 days" in body
    # PENDING_REVIEW, not ESCALATED -- no escalation banner should render.
    assert "status-badge-pending_review" in body
    assert 'class="escalation-banner"' not in body


async def test_case_detail_shows_merchant_and_global_policy_notes_filtered_to_case(
    client: _RecordingClient, tmp_path: Path
):
    """Wires up the case detail page's merchant memory panel (previously a
    static disclaimer). CASE-1002's real
    fixture `user_id` is `283959`. A note scoped to a *different* merchant
    must NOT leak onto this page -- that's the entire point of this task
    (before it, a rep had to scan the queue's fully unfiltered notes list
    to find ones relevant to one case's merchant).
    """
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)  # merchant_id=283959, PENDING_REVIEW

    record_policy_note(
        "Mention the account manager by name.",
        scope="merchant",
        merchant_id="283959",
        source_case_id="CASE-1002",
        db_path=db_path,
    )
    record_policy_note(
        "Limit expressions of regret to a single instance per email.",
        scope="global",
        db_path=db_path,
    )
    record_policy_note(
        "This note belongs to a totally different merchant.",
        scope="merchant",
        merchant_id="334430",
        db_path=db_path,
    )

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.get("/cases/CASE-1002")

    assert resp.status_code == 200
    body = resp.text
    assert "Mention the account manager by name." in body
    assert "Limit expressions of regret to a single instance per email." in body
    assert "This note belongs to a totally different merchant." not in body


async def test_case_detail_delete_note_redirects_back_to_case_page(
    client: _RecordingClient, tmp_path: Path
):
    """The case page's delete form sends `next=/cases/{case_id}` so deleting
    a note from there returns the rep to the case they were reviewing,
    instead of bouncing them to the queue. The queue page's own delete
    forms never send `next` and must keep defaulting to `/cases`,
    unaffected by this change. Also checks the open-redirect guard: a
    `next` value that doesn't start with a single leading `/` (e.g. a
    protocol-relative `//evil.example.com`) falls back to `/cases` rather
    than being honored verbatim.
    """
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)
    record_policy_note(
        "Mention the account manager by name.",
        scope="merchant",
        merchant_id="283959",
        source_case_id="CASE-1002",
        db_path=db_path,
    )
    conn = get_connection(db_path)
    try:
        note_id = conn.execute("SELECT id FROM memory WHERE kind = 'policy'").fetchone()["id"]
    finally:
        conn.close()

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.post(
            f"/memory/notes/{note_id}/delete",
            data={"next": "/cases/CASE-1002"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/cases/CASE-1002"

        after = tc.get("/cases/CASE-1002")
        assert "Mention the account manager by name." not in after.text

        # Queue page's own delete forms never send `next` -- must still
        # default to /cases, unaffected by this change.
        record_policy_note(
            "Another note.", scope="merchant", merchant_id="283959", db_path=db_path
        )
        note_id_2 = conn2 = get_connection(db_path)
        try:
            note_id_2 = conn2.execute(
                "SELECT id FROM memory WHERE kind = 'policy' ORDER BY id DESC LIMIT 1"
            ).fetchone()["id"]
        finally:
            conn2.close()
        resp2 = tc.post(f"/memory/notes/{note_id_2}/delete", follow_redirects=False)
        assert resp2.status_code == 303
        assert resp2.headers["location"] == "/cases"

        # Open-redirect guard: a protocol-relative `next` is rejected, not honored.
        record_policy_note(
            "Yet another note.", scope="merchant", merchant_id="283959", db_path=db_path
        )
        conn3 = get_connection(db_path)
        try:
            note_id_3 = conn3.execute(
                "SELECT id FROM memory WHERE kind = 'policy' ORDER BY id DESC LIMIT 1"
            ).fetchone()["id"]
        finally:
            conn3.close()
        resp3 = tc.post(
            f"/memory/notes/{note_id_3}/delete",
            data={"next": "//evil.example.com"},
            follow_redirects=False,
        )
        assert resp3.status_code == 303
        assert resp3.headers["location"] == "/cases"


async def test_case_detail_email_header_falls_back_when_no_contact_email(
    client: _RecordingClient, tmp_path: Path
):
    """`Case.contact_email` is optional on the model -- the email-header
    preview must degrade to a clear placeholder, not a blank/crashing
    template, when it's missing.
    """

    class _NoContactEmailClient:
        def __init__(self, inner) -> None:
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        async def get_case(self, case_id: str):
            case = await self._inner.get_case(case_id)
            return case.model_copy(update={"contact_email": None})

    db_path = tmp_path / "t.db"
    no_contact_client = _NoContactEmailClient(client)
    await _process_1002_to_approve(no_contact_client, db_path)

    with TestClient(_app(no_contact_client, db_path)) as tc:
        resp = tc.get("/cases/CASE-1002")

    assert resp.status_code == 200
    assert "(no contact email on file)" in resp.text
    assert "Update on your ShipBob claim CASE-1002" in resp.text


async def test_escalated_status_renders_distinctly_from_pending_review(
    client: _RecordingClient, tmp_path: Path
):
    """`ESCALATED` must render with a visually distinct,
    clearly-labeled indicator on both the queue and case detail pages, not
    just the bare word blending in next to `PENDING_REVIEW` rows/pages.
    Checks a CSS class distinction (`status-badge-escalated` vs.
    `status-badge-pending_review`), not merely a substring match on
    "ESCALATED" (which the plain status column already satisfied before this
    task's fix, without being visually distinct in any way).

    CASE-9001-INSURED escalates directly from ELIGIBILITY (insured routing,
    `gate:eligibility_insured`); CASE-1002 stays PENDING_REVIEW.
    """
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)  # -> PENDING_REVIEW
    await _process_9001_to_escalated(client, db_path)  # -> ESCALATED

    with TestClient(_app(client, db_path)) as tc:
        queue_resp = tc.get("/cases")
        pending_case_resp = tc.get("/cases/CASE-1002")
        escalated_case_resp = tc.get("/cases/CASE-9001-INSURED")

    # Queue page: anchor each status badge to its own row's <td> -- a plain
    # substring check would pass even if the template applied both classes
    # to the same row (e.g. always rendering "escalated"), since both class
    # names appear somewhere on a two-row page either way.
    def row_status_badge(case_id: str) -> str:
        match = re.search(
            re.escape(f"<td>{case_id}</td>") + r".*?class=\"status-badge (status-badge-\S+)\"",
            queue_resp.text,
            re.DOTALL,
        )
        assert match, f"no status badge found for {case_id!r} row"
        return match.group(1)

    assert row_status_badge("CASE-1002") == "status-badge-pending_review"
    assert row_status_badge("CASE-9001-INSURED") == "status-badge-escalated"

    # Pending-review case detail: no escalation banner, pending-review badge.
    assert "status-badge-pending_review" in pending_case_resp.text
    assert 'class="escalation-banner"' not in pending_case_resp.text

    # Escalated case detail: distinct badge class AND an explanatory banner
    # with a real reason (not just the bare status word).
    assert "status-badge-escalated" in escalated_case_resp.text
    assert 'class="escalation-banner"' in escalated_case_resp.text
    assert "insured-claims process" in escalated_case_resp.text

    # The audit timeline <details> auto-expands for an escalated case (so a
    # rep sees the full event history immediately, without an extra click)
    # but stays collapsed by default for a plain pending-review case.
    escalated_audit_tag = re.search(
        r"<details([^>]*)>\s*<summary>Audit timeline", escalated_case_resp.text
    )
    pending_audit_tag = re.search(
        r"<details([^>]*)>\s*<summary>Audit timeline", pending_case_resp.text
    )
    assert escalated_audit_tag and "open" in escalated_audit_tag.group(1)
    assert pending_audit_tag and "open" not in pending_audit_tag.group(1)


async def _process_1002_with_invoice_discrepancy(client: _RecordingClient, db_path: Path) -> None:
    """CASE-1002 with the real retail-invoice gap present: ShipBob's invoice
    says $24.99, the merchant's own sales order says the customer paid
    $19.99 -> escalates on `gate:invoice_audit_discrepancy`.
    """
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["A00360"]),
            _extraction(
                readable=True,
                currency="USD",
                line_items=[
                    {
                        "description": "CleanBoss Botanical Disinfectant & Cleaner 24 Ounce 2 Pack",
                        "sku": "A00360",
                        "quantity": 1,
                        "unit_price": 19.99,
                        "line_total": 19.99,
                        "line_discount": 0.0,
                    }
                ],
                note=None,
            ),
            _draft(),
        ]
    )
    await process_case("CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW)


async def test_invoice_discrepancy_is_surfaced_on_the_case_page(
    client: _RecordingClient, tmp_path: Path
):
    """The whole point of the audit is that a rep can *see* the conflict --
    both figures, which source each came from, and which way it cuts. A
    finding buried only in the audit-log JSON would not be surfaced.
    """
    db_path = tmp_path / "t.db"
    await _process_1002_with_invoice_discrepancy(client, db_path)

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.get("/cases/CASE-1002")

    assert resp.status_code == 200
    body = resp.text

    # The panel itself, flagged and auto-expanded rather than collapsed.
    assert "Retail-invoice reconciliation" in body
    assert 'class="audit-badge audit-badge-flag"' in body
    panel = re.search(r"<details([^>]*)class=\"audit-panel\"", body) or re.search(
        r"<details([^>]*)>\s*<summary>\s*Retail-invoice reconciliation", body
    )
    assert panel and "open" in panel.group(1)

    # Both figures and the direction of the error, in the rep's own words.
    assert "PRICE_MISMATCH" in body
    assert "24.99" in body
    assert "19.99" in body
    assert "5.00 more than" in body

    # The extracted document is shown, so the rep can check the read.
    assert "Read from the merchant" in body
    assert "audit-escalate" in body

    # And the case really did escalate with a reason on the banner.
    assert "status-badge-escalated" in body
    assert "Retail-invoice discrepancy" in body


async def test_unverified_invoice_audit_says_so_rather_than_looking_clean(
    client: _RecordingClient, tmp_path: Path
):
    """"Couldn't verify" must not render identically to "verified and
    matching" -- that is the whole risk of a fail-open check, so the UI has
    to distinguish them explicitly.
    """
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)  # `_extraction()` default: unreadable

    with TestClient(_app(client, db_path)) as tc:
        body = tc.get("/cases/CASE-1002").text

    # Assert on the rendered badge element, not the bare class name -- every
    # badge class also appears in the page's own <style> block, so a plain
    # substring check would pass no matter what the template rendered.
    assert "Retail-invoice reconciliation" in body
    assert 'class="audit-badge audit-badge-unverified"' in body
    assert "Amount not independently verified" in body
    # ...and it must NOT claim a clean match
    assert 'class="audit-badge audit-badge-ok"' not in body
    assert "matches the price on the merchant" not in body


async def test_clean_invoice_audit_renders_as_matching(client: _RecordingClient, tmp_path: Path):
    db_path = tmp_path / "t.db"
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["A00360"]),
            _extraction(
                readable=True,
                currency="USD",
                line_items=[
                    {
                        "description": "CleanBoss Botanical Disinfectant & Cleaner 24oz 2 Pack",
                        "sku": "A00360",
                        "quantity": 1,
                        "unit_price": 24.99,
                        "line_total": 24.99,
                        "line_discount": 0.0,
                    }
                ],
                note=None,
            ),
            _draft(),
        ]
    )
    await process_case("CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW)

    with TestClient(_app(client, db_path)) as tc:
        body = tc.get("/cases/CASE-1002").text

    assert 'class="audit-badge audit-badge-ok"' in body
    assert 'class="audit-badge audit-badge-flag"' not in body
    assert "status-badge-pending_review" in body


# --- demo controls: reset / fetch / process ---------------------------------


async def test_fetch_cases_pulls_the_api_list_into_the_queue_without_running_the_pipeline(
    client: _RecordingClient, tmp_path: Path
):
    """Fetch has to be instant and LLM-free -- the transport below is empty
    and would raise "FakeTransport exhausted" if the pipeline ran.
    """
    db_path = tmp_path / "t.db"

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.post("/admin/fetch-cases", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/cases"
        queue = tc.get("/cases").text

    rows = list_cases_by_status(CaseState.INTAKE, db_path=db_path)
    fetched = {row["case_id"] for row in rows}
    assert {"CASE-1001", "CASE-1002", "CASE-1003", "CASE-1004", "CASE-1005"} <= fetched

    # ...and they're visible as their own "not yet processed" section, not
    # padding the review-queue count with rows nothing is waiting on.
    assert "Fetched, not yet processed" in queue
    assert "CASE-1001" in queue
    assert "0 case(s) awaiting review" in queue

    # merchant attribution survives the fetch (needed for memory later)
    by_id = {row["case_id"]: row for row in rows}
    assert by_id["CASE-1001"]["merchant_id"] == "334430"


async def test_fetch_cases_is_idempotent(client: _RecordingClient, tmp_path: Path):
    """A double-clicked button must not blow up on the duplicate primary
    key or create ghost rows.
    """
    db_path = tmp_path / "t.db"

    with TestClient(_app(client, db_path)) as tc:
        tc.post("/admin/fetch-cases", follow_redirects=False)
        first = len(list_cases_by_status(CaseState.INTAKE, db_path=db_path))
        resp = tc.post("/admin/fetch-cases", follow_redirects=False)

    assert resp.status_code == 303
    assert len(list_cases_by_status(CaseState.INTAKE, db_path=db_path)) == first


async def test_process_button_runs_one_case_through_the_pipeline(
    client: _RecordingClient, tmp_path: Path
):
    db_path = tmp_path / "t.db"
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["A00360"]),
            _extraction(),
            _draft(),
        ]
    )

    with TestClient(_app(client, db_path, transport)) as tc:
        tc.post("/admin/fetch-cases", follow_redirects=False)
        assert get_status("CASE-1002", db_path=db_path) == CaseState.INTAKE

        resp = tc.post("/cases/CASE-1002/process", follow_redirects=False)

    assert resp.status_code == 303
    assert resp.headers["location"] == "/cases/CASE-1002"
    assert get_status("CASE-1002", db_path=db_path) == CaseState.PENDING_REVIEW
    assert get_recommendation("CASE-1002", db_path=db_path) is not None


async def test_process_refuses_a_case_a_rep_has_already_acted_on(
    client: _RecordingClient, tmp_path: Path
):
    """Re-processing would overwrite a decision a human already made."""
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)  # -> PENDING_REVIEW

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.post("/cases/CASE-1002/process", follow_redirects=False)

    assert resp.status_code == 409
    assert "already pending_review" in resp.json()["detail"]


async def test_reset_wipes_cases_audit_actions_and_memory(
    client: _RecordingClient, tmp_path: Path
):
    """Reset has to mean reset -- including memory. A reset that kept policy
    notes would silently break the memory carry-forward demo by replaying it
    with the note already present.
    """
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)
    record_policy_note("Mention the account manager by name.", scope="global", db_path=db_path)

    assert list_cases_by_status(CaseState.PENDING_REVIEW, db_path=db_path)
    assert global_policies(db_path=db_path)

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.post("/admin/reset", follow_redirects=False)
        assert resp.status_code == 303
        queue = tc.get("/cases").text

    assert list_cases_by_status(CaseState.PENDING_REVIEW, db_path=db_path) == []
    assert list_cases_by_status(CaseState.ESCALATED, db_path=db_path) == []
    assert global_policies(db_path=db_path) == []
    assert get_case("CASE-1002", db_path=db_path) is None
    assert "Nothing waiting on a rep right now." in queue


async def test_reset_then_fetch_then_process_is_a_clean_round_trip(
    client: _RecordingClient, tmp_path: Path
):
    """The actual demo sequence, end to end."""
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)

    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["A00360"]),
            _extraction(),
            _draft(),
        ]
    )

    with TestClient(_app(client, db_path, transport)) as tc:
        tc.post("/admin/reset", follow_redirects=False)
        assert get_case("CASE-1002", db_path=db_path) is None

        tc.post("/admin/fetch-cases", follow_redirects=False)
        assert get_status("CASE-1002", db_path=db_path) == CaseState.INTAKE

        tc.post("/cases/CASE-1002/process", follow_redirects=False)

    assert get_status("CASE-1002", db_path=db_path) == CaseState.PENDING_REVIEW


async def test_process_all_runs_every_fetched_case(client: _RecordingClient, tmp_path: Path):
    """The second demo button: one click, every fetched case through the real
    pipeline. Runs as a background task, which Starlette's TestClient awaits
    before returning -- so this asserts on the finished state.
    """
    db_path = tmp_path / "t.db"

    def full_path(sku: str) -> list:
        """4 usable attachments -> validation -> invoice audit -> draft."""
        return [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=[sku]),
            _extraction(),
            _draft(),
        ]

    def evidence_gap(n_attachments: int) -> list:
        """Short of the 4 required categories -> exits at the evidence gate."""
        return [_classification("ORDER_PROOF") for _ in range(n_attachments)] + [_draft()]

    # Processed in sorted case_id order. CASE-1004 (too old) and
    # CASE-9001-INSURED (insured routing) both exit at eligibility, before
    # any attachment is classified, so they only spend the drafter call.
    transport = FakeTransport(
        [
            *evidence_gap(3),                        # CASE-1001
            *full_path("A00360"),                    # CASE-1002
            *evidence_gap(3),                        # CASE-1003
            _draft(),                                # CASE-1004  (denied: too old)
            _draft(),                                # CASE-1005  (no attachments)
            _draft(),                                # CASE-9001-INSURED (escalated)
            *full_path("A00300-CASE12"),             # CASE-9002-CAP
            # CASE-9003-REPEAT is deliberately held back from bulk processing
            # (BULK_PROCESS_HOLDBACK) so the memory beat has something to show.
        ]
    )

    with TestClient(_app(client, db_path, transport)) as tc:
        tc.post("/admin/fetch-cases", follow_redirects=False)
        tc.post("/admin/add-demo-cases", follow_redirects=False)
        assert len(list_cases_by_status(CaseState.INTAKE, db_path=db_path)) == 8

        resp = tc.post("/admin/process-all", follow_redirects=False)
        assert resp.status_code == 303
        queue = tc.get("/cases").text
        state = tc.app.state.demo_run

    assert state.running is False
    assert state.done == state.total == 7  # 8 fetched, 1 held back
    assert state.errors == []

    for case_id in (
        "CASE-1001",
        "CASE-1002",
        "CASE-1003",
        "CASE-1004",
        "CASE-1005",
        "CASE-9001-INSURED",
        "CASE-9002-CAP",
    ):
        assert get_recommendation(case_id, db_path=db_path) is not None, case_id

    # only the held-back case remains unprocessed
    remaining = [r["case_id"] for r in list_cases_by_status(CaseState.INTAKE, db_path=db_path)]
    assert remaining == ["CASE-9003-REPEAT"]
    assert "Fetched, not yet processed" in queue


async def test_fetch_returns_only_real_api_cases_not_demo_scenarios(
    client: _RecordingClient, tmp_path: Path
):
    """A fetch must show exactly what ShipBob's API has. The synthetic demo
    scenarios are opt-in -- they should never appear just because someone
    clicked Fetch, or the queue silently mixes invented cases into real data.

    (The `client` fixture wraps `FixtureClient(include_synthetic=True)`, so
    the scenarios ARE resolvable here -- this asserts they're excluded by
    the fetch endpoint's own list, not merely absent from the data.)
    """
    db_path = tmp_path / "t.db"

    with TestClient(_app(client, db_path)) as tc:
        tc.post("/admin/fetch-cases", follow_redirects=False)

    fetched = {r["case_id"] for r in list_cases_by_status(CaseState.INTAKE, db_path=db_path)}
    assert fetched == {"CASE-1001", "CASE-1002", "CASE-1003", "CASE-1004", "CASE-1005"}
    for scenario in synthetic_case_ids():
        assert scenario not in fetched


async def test_add_demo_cases_button_adds_the_scenarios(
    client: _RecordingClient, tmp_path: Path
):
    db_path = tmp_path / "t.db"

    with TestClient(_app(client, db_path)) as tc:
        tc.post("/admin/fetch-cases", follow_redirects=False)
        resp = tc.post("/admin/add-demo-cases", follow_redirects=False)
        queue = tc.get("/cases").text

    assert resp.status_code == 303
    present = {r["case_id"] for r in list_cases_by_status(CaseState.INTAKE, db_path=db_path)}
    for scenario in synthetic_case_ids():
        assert scenario in present
    assert len(present) == 8  # 5 real + 3 scenarios

    # merchant attribution comes through, which the memory beat depends on
    rows = {r["case_id"]: r for r in list_cases_by_status(CaseState.INTAKE, db_path=db_path)}
    assert rows["CASE-9003-REPEAT"]["merchant_id"] == "334430"  # same as CASE-1001
    assert "+ Add demo scenarios (3)" in queue


async def test_add_demo_cases_is_idempotent(client: _RecordingClient, tmp_path: Path):
    db_path = tmp_path / "t.db"

    with TestClient(_app(client, db_path)) as tc:
        tc.post("/admin/add-demo-cases", follow_redirects=False)
        first = len(list_cases_by_status(CaseState.INTAKE, db_path=db_path))
        resp = tc.post("/admin/add-demo-cases", follow_redirects=False)

    assert resp.status_code == 303
    assert len(list_cases_by_status(CaseState.INTAKE, db_path=db_path)) == first == 3


async def test_add_demo_cases_skips_scenarios_the_client_cannot_resolve(
    tmp_path: Path,
):
    """Against a client with no synthetic data (e.g. the bare live HTTP
    client), the button must not create rows that would 404 the moment
    anyone pressed Process on them.
    """
    db_path = tmp_path / "t.db"
    real_only = _RecordingClient(FixtureClient())  # no include_synthetic

    with TestClient(_app(real_only, db_path)) as tc:
        resp = tc.post("/admin/add-demo-cases", follow_redirects=False)

    assert resp.status_code == 303
    assert list_cases_by_status(CaseState.INTAKE, db_path=db_path) == []


async def test_process_all_holds_back_the_memory_demo_case(
    client: _RecordingClient, tmp_path: Path
):
    """`CASE-9003-REPEAT` must survive "Process all" unprocessed.

    It's the second case from `CASE-1001`'s merchant, and its whole purpose
    is showing a rep's correction carry into a *later* draft. Processing it
    in the same batch would run it before any pushback exists, so the beat
    would silently show nothing -- the failure mode is an empty demo, not an
    error, which is exactly the kind that survives to the live run.
    """
    db_path = tmp_path / "t.db"
    transport = FakeTransport([_draft() for _ in range(40)])

    with TestClient(_app(client, db_path, transport)) as tc:
        tc.post("/admin/fetch-cases", follow_redirects=False)
        tc.post("/admin/add-demo-cases", follow_redirects=False)
        tc.post("/admin/process-all", follow_redirects=False)
        queue = tc.get("/cases").text
        state = tc.app.state.demo_run

    # 8 in the queue, 7 processed -- the repeat case is deliberately skipped
    assert state.total == 7
    assert get_status("CASE-9003-REPEAT", db_path=db_path) == CaseState.INTAKE
    assert get_recommendation("CASE-9003-REPEAT", db_path=db_path) is None

    # ...and the UI says so rather than looking like a bug
    assert "held back" in queue
    assert "Held back" in queue


async def test_held_back_case_can_still_be_processed_individually(
    client: _RecordingClient, tmp_path: Path
):
    """The holdback only applies to the bulk button -- the per-case Process
    button is exactly how the memory beat is meant to run it.
    """
    db_path = tmp_path / "t.db"
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["HG-FRCAST-KITTEDROLL-3PK"]),
            _extraction(),
            _draft(),
        ]
    )

    with TestClient(_app(client, db_path, transport)) as tc:
        tc.post("/admin/add-demo-cases", follow_redirects=False)
        resp = tc.post("/cases/CASE-9003-REPEAT/process", follow_redirects=False)

    assert resp.status_code == 303
    assert get_recommendation("CASE-9003-REPEAT", db_path=db_path) is not None


async def test_process_all_survives_one_failing_case(client: _RecordingClient, tmp_path: Path):
    """One bad case must not abandon the batch -- the failure is recorded
    against that case and the run continues.
    """
    db_path = tmp_path / "t.db"
    # Exhausts after CASE-1001's run, so every later case raises.
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("PRODUCT_PHOTO"),
            _classification("CUSTOMER_CONFIRMATION"),
            _draft(),
        ]
    )

    with TestClient(_app(client, db_path, transport)) as tc:
        tc.post("/admin/fetch-cases", follow_redirects=False)
        tc.post("/admin/process-all", follow_redirects=False)
        queue = tc.get("/cases").text
        state = tc.app.state.demo_run

    # the first case still got through despite the later ones failing
    assert get_recommendation("CASE-1001", db_path=db_path) is not None
    # the run completed rather than hanging, and reported the failures
    assert state.running is False
    assert state.done == state.total == 5  # the 5 real API cases
    assert len(state.errors) == 4
    assert "Last run finished with 4 error(s)" in queue


async def test_process_all_refuses_to_start_a_second_concurrent_run(
    client: _RecordingClient, tmp_path: Path
):
    """A double-click must not process every case twice at once."""
    db_path = tmp_path / "t.db"

    with TestClient(_app(client, db_path)) as tc:
        tc.post("/admin/fetch-cases", follow_redirects=False)
        tc.app.state.demo_run.reset(total=5)  # simulate a run already in flight

        resp = tc.post("/admin/process-all", follow_redirects=False)
        queue = tc.get("/cases").text

    assert resp.status_code == 303
    # still untouched -- the second run never started
    assert len(list_cases_by_status(CaseState.INTAKE, db_path=db_path)) == 5
    # ...and the UI shows progress + disables the buttons while running
    assert "Processing 0 of 5" in queue
    assert "disabled" in queue


async def test_process_all_with_nothing_fetched_is_a_no_op(
    client: _RecordingClient, tmp_path: Path
):
    db_path = tmp_path / "t.db"

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.post("/admin/process-all", follow_redirects=False)
        state = tc.app.state.demo_run

    assert resp.status_code == 303
    assert state.running is False
    assert state.total == 0


async def test_demo_endpoints_404_when_controls_are_disabled(
    client: _RecordingClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The off switch has to close the door, not just remove the handle --
    hiding the buttons while leaving the endpoints live would mean a stray
    POST could still empty a real claims database.
    """
    monkeypatch.setattr(settings, "demo_controls_enabled", False)
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)

    with TestClient(_app(client, db_path)) as tc:
        assert tc.post("/admin/reset", follow_redirects=False).status_code == 404
        assert tc.post("/admin/fetch-cases", follow_redirects=False).status_code == 404
        assert tc.post("/admin/process-all", follow_redirects=False).status_code == 404
        assert tc.post("/cases/CASE-1002/process", follow_redirects=False).status_code == 404
        queue = tc.get("/cases").text

    # buttons gone from the UI too
    assert "/admin/reset" not in queue
    assert "Fetch cases from ShipBob" not in queue
    # ...and the reset genuinely didn't happen
    assert get_case("CASE-1002", db_path=db_path) is not None


async def test_reset_button_is_guarded_by_a_confirmation(
    client: _RecordingClient, tmp_path: Path
):
    """A one-click irreversible wipe next to ordinary rep actions is an
    accident waiting to happen during a live demo.
    """
    db_path = tmp_path / "t.db"

    with TestClient(_app(client, db_path)) as tc:
        queue = tc.get("/cases").text

    assert "onsubmit=\"return confirm(" in queue
    assert "cannot be undone" in queue


async def test_case_detail_renders_cleanly_with_no_recommendation_and_no_evidence(
    client: _RecordingClient, tmp_path: Path
):
    """CASE-1005 has zero attachments in the real fixture data, and a case
    can legitimately sit in `PENDING_REVIEW` with no recommendation saved
    yet (e.g. a `create_case` + `transition` sequence run directly, before
    `pipeline.process_case` gets to the final `_exit()` that saves one).
    The template must degrade gracefully in both "empty" cases rather than
    raising a Jinja `UndefinedError`/crashing the request.
    """
    db_path = tmp_path / "t.db"
    create_case("CASE-1005", db_path=db_path)
    for state in (CaseState.ELIGIBILITY, CaseState.EVIDENCE, CaseState.PENDING_REVIEW):
        transition("CASE-1005", state, actor="system", event="advance", db_path=db_path)

    with TestClient(_app(client, db_path)) as tc:
        case_resp = tc.get("/cases/CASE-1005")
        queue_resp = tc.get("/cases")

    assert case_resp.status_code == 200
    assert "No recommendation has been generated" in case_resp.text
    assert "No evidence attachments on file" in case_resp.text
    assert "No policy notes recorded for this merchant or globally yet." in case_resp.text
    # No attachments -> no lightbox overlay element/script rendered (only the
    # static CSS rule for the class stays, since that's outside the `{% if
    # attachments %}` block; the actual element and its driving script are
    # inside it, so check for the element itself, not the bare class name).
    assert 'id="lightbox-overlay"' not in case_resp.text
    assert "LIGHTBOX_ATTACHMENTS" not in case_resp.text
    # No draft/pushback forms when there's nothing to approve/push back on.
    assert "Approve &amp; Send" not in case_resp.text

    assert queue_resp.status_code == 200
    assert "badge-UNKNOWN" in queue_resp.text  # no recommendation -> unknown risk tier


async def test_case_detail_missing_case_is_404(client: _RecordingClient, tmp_path: Path):
    db_path = tmp_path / "t.db"
    with TestClient(_app(client, db_path)) as tc:
        resp = tc.get("/cases/NO-SUCH-CASE")
    assert resp.status_code == 404


async def test_attachment_endpoint_streams_bytes(client: _RecordingClient, tmp_path: Path):
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.get("/cases/CASE-1002/attachments/ATT-CASE-1002-01")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content.startswith(b"\x89PNG\r\n\x1a\n")


# --- nothing hits outbox without approve (explicit plan requirement) -------


async def test_nothing_sent_before_approve(client: _RecordingClient, tmp_path: Path):
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)

    with TestClient(_app(client, db_path)) as tc:
        tc.get("/cases")
        tc.get("/cases/CASE-1002")

    assert client.sent_emails == []
    assert client.reimbursements == []


# --- approve: sends + submits reimbursement, transitions to SENT ----------


async def test_approve_sends_original_draft_and_submits_reimbursement(
    client: _RecordingClient, tmp_path: Path
):
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.post("/cases/CASE-1002/approve", data={"action": "approve"}, follow_redirects=False)

    assert resp.status_code == 303

    assert len(client.sent_emails) == 1
    sent = client.sent_emails[0]
    assert sent["to"] == "mtaparia@shipbob.com"  # case.contact_email, fetched fresh
    assert sent["body"] == ORIGINAL_DRAFT

    assert len(client.reimbursements) == 1
    reimb = client.reimbursements[0]
    assert reimb["amount"] == Decimal("24.99")
    assert reimb["order_id"] == "336431771"
    assert reimb["product_name"]  # resolved via order line items, non-empty

    assert get_status("CASE-1002", db_path=db_path) == CaseState.SENT
    events = [r["event"] for r in get_audit_log("CASE-1002", db_path=db_path)]
    assert "transition:pending_review->approved" in events
    assert events[-1] == "approved"


async def test_approving_request_info_sends_email_but_no_reimbursement(
    client: _RecordingClient, tmp_path: Path
):
    db_path = tmp_path / "t.db"
    await _process_1001_to_request_info(client, db_path)

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.post("/cases/CASE-1001/approve", data={"action": "approve"}, follow_redirects=False)

    assert resp.status_code == 303
    assert len(client.sent_emails) == 1
    assert client.reimbursements == []  # decision != "approve" -> no payout submission

    assert get_status("CASE-1001", db_path=db_path) == CaseState.SENT
    events = [r["event"] for r in get_audit_log("CASE-1001", db_path=db_path)]
    assert "transition:pending_review->needs_info" in events


async def test_approving_escalated_case_transitions_straight_to_sent(
    client: _RecordingClient, tmp_path: Path
):
    """ESCALATED -> SENT is a single legal hop (unlike PENDING_REVIEW, which
    must pass through an intermediate decision state first) -- see
    `store.LEGAL_TRANSITIONS` and `web/app.py` module docstring point 7.
    """
    db_path = tmp_path / "t.db"
    await _process_9001_to_escalated(client, db_path)

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.post("/cases/CASE-9001-INSURED/approve", data={"action": "approve"}, follow_redirects=False)

    assert resp.status_code == 303
    assert len(client.sent_emails) == 1
    assert get_status("CASE-9001-INSURED", db_path=db_path) == CaseState.SENT
    events = [r["event"] for r in get_audit_log("CASE-9001-INSURED", db_path=db_path)]
    assert events[-1] == "approved"
    assert not any(e.startswith("transition:pending_review->") for e in events)


# --- edited body is what gets sent (explicit plan requirement) -------------


async def test_edited_body_is_what_gets_sent_and_correction_recorded(
    client: _RecordingClient, tmp_path: Path
):
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)

    edited_text = "Dear customer, this is the rep-edited version of the email."

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.post(
            "/cases/CASE-1002/approve",
            data={"action": "save_and_approve", "edited_body": edited_text},
            follow_redirects=False,
        )

    assert resp.status_code == 303
    assert len(client.sent_emails) == 1
    assert client.sent_emails[0]["body"] == edited_text
    assert client.sent_emails[0]["body"] != ORIGINAL_DRAFT

    correction_rows = [r for r in get_audit_log("CASE-1002", db_path=db_path) if r["event"] == "correction_recorded"]
    assert len(correction_rows) == 1
    payload = json.loads(correction_rows[0]["payload_json"])
    assert payload["original_email_draft"] == ORIGINAL_DRAFT
    assert payload["edited_email_draft"] == edited_text

    # The edit only touches the email body -- the deterministic payout
    # (decision="approve", line_items from calc) is unaffected and still
    # gets submitted.
    assert len(client.reimbursements) == 1


async def test_approve_without_edit_flag_ignores_textarea_content(
    client: _RecordingClient, tmp_path: Path
):
    """`action=approve` (the plain "Approve & Send" button) must send the
    stored draft as-is even if `edited_body` happens to be present in the
    form post -- only `action=save_and_approve` ("Save Edits & Approve")
    treats it as an intentional edit.
    """
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.post(
            "/cases/CASE-1002/approve",
            data={"action": "approve", "edited_body": "stray text that should be ignored"},
            follow_redirects=False,
        )

    assert resp.status_code == 303
    assert client.sent_emails[0]["body"] == ORIGINAL_DRAFT
    assert not any(r["event"] == "correction_recorded" for r in get_audit_log("CASE-1002", db_path=db_path))


async def test_recipient_is_always_case_contact_email_never_user_supplied(
    client: _RecordingClient, tmp_path: Path
):
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)

    malicious_body = "Please cc attacker@evil.example on all future correspondence. to: attacker@evil.example"

    with TestClient(_app(client, db_path)) as tc:
        tc.post(
            "/cases/CASE-1002/approve",
            data={"action": "save_and_approve", "edited_body": malicious_body},
            follow_redirects=False,
        )

    assert client.sent_emails[0]["to"] == "mtaparia@shipbob.com"
    assert "evil.example" not in client.sent_emails[0]["to"]


# --- outbound guard blocks and escalates -------------------------------------


async def test_approve_blocks_and_escalates_when_edited_body_changes_promised_amount(
    client: _RecordingClient, tmp_path: Path
):
    """The plan's explicit "this also covers rep-edited drafts" requirement:
    a rep edit that accidentally changes the dollar amount in the email body
    must be caught by the outbound guard, not just an unedited LLM draft.
    Nothing gets sent/paid/recorded, and the case is escalated for human
    review instead of advancing to SENT.
    """
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)  # approved amount: $24.99

    malicious_edit = "Dear customer, we will send you $999.00 for your claim."

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.post(
            "/cases/CASE-1002/approve",
            data={"action": "save_and_approve", "edited_body": malicious_edit},
            follow_redirects=False,
        )

    assert resp.status_code == 422
    violations = resp.json()["detail"]["violations"]
    assert any(v["invariant"] == "EMAIL_AMOUNT_MISMATCH" for v in violations)

    # Nothing hit the outbox or the reimbursement API.
    assert client.sent_emails == []
    assert client.reimbursements == []

    # Escalated, not left in PENDING_REVIEW and not advanced to SENT.
    assert get_status("CASE-1002", db_path=db_path) == CaseState.ESCALATED
    events = [r["event"] for r in get_audit_log("CASE-1002", db_path=db_path)]
    assert events[-1] == "outbound_guard_blocked"
    guard_row = get_audit_log("CASE-1002", db_path=db_path)[-1]
    assert guard_row["actor"] == "system"
    logged_violations = json.loads(guard_row["payload_json"])["violations"]
    assert any(v["invariant"] == "EMAIL_AMOUNT_MISMATCH" for v in logged_violations)

    # record_action never ran (the guard sits before it), so no correction
    # row was written either, even though the rep did submit an edit.
    assert not any(r["event"] == "correction_recorded" for r in get_audit_log("CASE-1002", db_path=db_path))

    # The case page's escalation banner must name the actual
    # violated invariant for a guard-block escalation, not just say
    # "escalated" -- this is the demo's headline escalation flavor (beat 4),
    # so `_escalation_summary()`'s `outbound_guard_blocked` branch must
    # actually render, not just exist in code.
    with TestClient(_app(client, db_path)) as tc:
        case_resp = tc.get("/cases/CASE-1002")
    assert 'class="escalation-banner"' in case_resp.text
    assert "EMAIL_AMOUNT_MISMATCH" in case_resp.text


async def test_approve_blocked_case_can_be_re_approved_after_the_draft_is_fixed(
    client: _RecordingClient, tmp_path: Path
):
    """A blocked send must not permanently strand the case: once it's back
    in front of a rep (ESCALATED is a reviewable status) with a clean draft,
    approving again must succeed normally.
    """
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)

    with TestClient(_app(client, db_path)) as tc:
        blocked = tc.post(
            "/cases/CASE-1002/approve",
            data={"action": "save_and_approve", "edited_body": "Dear customer, we will send you $999.00."},
            follow_redirects=False,
        )
        assert blocked.status_code == 422
        assert get_status("CASE-1002", db_path=db_path) == CaseState.ESCALATED

        retried = tc.post("/cases/CASE-1002/approve", data={"action": "approve"}, follow_redirects=False)

    assert retried.status_code == 303
    assert len(client.sent_emails) == 1
    assert client.sent_emails[0]["body"] == ORIGINAL_DRAFT
    assert len(client.reimbursements) == 1
    assert get_status("CASE-1002", db_path=db_path) == CaseState.SENT


async def test_approve_blocks_already_escalated_case_without_illegal_transition(
    client: _RecordingClient, tmp_path: Path
):
    """`ESCALATED` has no legal self-loop transition -- a guard block on a
    case that's already `ESCALATED` must log the block via `log_event`
    (audit-only) rather than attempt `transition(..., ESCALATED)`, which
    would raise `IllegalTransitionError` and turn a safety block into a
    server error.
    """
    db_path = tmp_path / "t.db"
    await _process_9001_to_escalated(client, db_path)  # request_info, amount=$0

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.post(
            "/cases/CASE-9001-INSURED/approve",
            data={"action": "save_and_approve", "edited_body": "Dear customer, you will receive $50.00."},
            follow_redirects=False,
        )

    assert resp.status_code == 422
    assert client.sent_emails == []
    assert client.reimbursements == []

    assert get_status("CASE-9001-INSURED", db_path=db_path) == CaseState.ESCALATED
    events = [r["event"] for r in get_audit_log("CASE-9001-INSURED", db_path=db_path)]
    assert events[-1] == "outbound_guard_blocked"


# --- double-approve is rejected (idempotency, DuplicateActionError)


async def test_double_approve_is_rejected_and_does_not_double_send(
    client: _RecordingClient, tmp_path: Path
):
    """The second click lands after the case has already moved to `SENT`,
    so this exercises the `_require_reviewable` state check, not
    `record_action`'s `DuplicateActionError` path directly -- see
    `test_repeat_approve_while_still_reviewable_hits_duplicate_action_guard`
    below for a test that isolates the idempotency backstop itself (the
    plan's explicit "double-clicked approve button" scenario, e.g. a client
    that fires the POST twice before either response comes back).
    """
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)

    with TestClient(_app(client, db_path)) as tc:
        first = tc.post("/cases/CASE-1002/approve", data={"action": "approve"}, follow_redirects=False)
        second = tc.post("/cases/CASE-1002/approve", data={"action": "approve"}, follow_redirects=False)

    assert first.status_code == 303
    assert second.status_code == 409  # rejected, not silently re-sent
    assert len(client.sent_emails) == 1
    assert len(client.reimbursements) == 1


async def test_repeat_approve_while_still_reviewable_hits_duplicate_action_guard(
    client: _RecordingClient, tmp_path: Path
):
    """Isolates `record_action`'s `DuplicateActionError` idempotency
    backstop itself, as distinct from the state-machine check
    above: pre-record the "email" action for a case that's still sitting in
    `PENDING_REVIEW` (simulating two in-flight approve requests racing each
    other before either one's `transition()` call has landed), then hit the
    endpoint once. It must be rejected with nothing sent, via the
    `DuplicateActionError` branch specifically -- not via
    `_require_reviewable`, since the case's status hasn't changed yet.
    """
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)
    assert get_status("CASE-1002", db_path=db_path) == CaseState.PENDING_REVIEW

    from claimpilot.store import record_action

    record_action(
        "CASE-1002",
        "email",
        {"to": "mtaparia@shipbob.com", "subject": "already claimed", "body": "..."},
        db_path=db_path,
    )

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.post("/cases/CASE-1002/approve", data={"action": "approve"}, follow_redirects=False)

    assert resp.status_code == 409
    assert client.sent_emails == []
    assert client.reimbursements == []
    # No phantom correction row for an email that was never actually sent.
    assert not any(r["event"] == "correction_recorded" for r in get_audit_log("CASE-1002", db_path=db_path))


# --- pushback ---------------------------------------------------------------


async def test_pushback_stores_feedback_and_redrafts_without_leaving_the_queue(
    client: _RecordingClient, tmp_path: Path
):
    db_path = tmp_path / "t.db"
    await _process_1001_to_request_info(client, db_path)

    revised_draft = "Dear customer, revised per rep feedback."
    pushback_transport = FakeTransport([_draft(email_draft=revised_draft, rationale="Updated per rep feedback.")])

    with TestClient(_app(client, db_path, transport=pushback_transport)) as tc:
        resp = tc.post(
            "/cases/CASE-1001/pushback",
            data={"feedback": "Please mention their account manager Dana."},
            follow_redirects=False,
        )
        assert resp.status_code == 303

        queue_resp = tc.get("/cases")
        case_resp = tc.get("/cases/CASE-1001")

    # Still reachable from the queue -- pushback doesn't remove it, and
    # doesn't advance the state machine (still PENDING_REVIEW).
    assert "CASE-1001" in queue_resp.text
    assert get_status("CASE-1001", db_path=db_path) == CaseState.PENDING_REVIEW

    # A new, differently-scripted Recommendation was produced and persisted.
    assert revised_draft in case_resp.text
    assert ORIGINAL_DRAFT not in case_resp.text

    pushback_rows = [r for r in get_audit_log("CASE-1001", db_path=db_path) if r["event"] == "pushback"]
    assert len(pushback_rows) == 1
    payload = json.loads(pushback_rows[0]["payload_json"])
    assert payload["feedback"] == "Please mention their account manager Dana."

    # Pushback never sends/pays anything.
    assert client.sent_emails == []
    assert client.reimbursements == []


async def test_pushback_then_approve_preserves_line_items_and_still_pays(
    client: _RecordingClient, tmp_path: Path
):
    """Regression test for `_pushback_calc_result()`: `draft()` copies
    `Recommendation.line_items` from `inputs.calc_result.line_items`
    verbatim, so a naive pushback reconstruction that always passes
    `calc_result=None` would silently produce a `[]`-line-item
    Recommendation for an `approve`-decision case -- which would then make
    the approve endpoint's `submit_reimbursement` loop iterate zero times,
    a silent zero-payout bug on an already-approved claim. Uses CASE-1002
    (single line item, A00360 @ $24.99) end-to-end: process -> pushback
    (differently-scripted redraft) -> approve, and asserts the reimbursement
    still goes out for the original SKU/amount.
    """
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)

    revised_draft = "Dear customer, approved and revised per rep feedback."
    pushback_transport = FakeTransport(
        [_draft(email_draft=revised_draft, rationale="Approved -- revised wording per rep feedback.")]
    )

    with TestClient(_app(client, db_path, transport=pushback_transport)) as tc:
        pushback_resp = tc.post(
            "/cases/CASE-1002/pushback",
            data={"feedback": "Tone it down a little."},
            follow_redirects=False,
        )
        assert pushback_resp.status_code == 303

        approve_resp = tc.post("/cases/CASE-1002/approve", data={"action": "approve"}, follow_redirects=False)

    assert approve_resp.status_code == 303
    assert client.sent_emails[0]["body"] == revised_draft

    assert len(client.reimbursements) == 1
    reimb = client.reimbursements[0]
    assert reimb["amount"] == Decimal("24.99")
    assert reimb["order_id"] == "336431771"


async def test_pushback_rejects_blank_feedback(client: _RecordingClient, tmp_path: Path):
    db_path = tmp_path / "t.db"
    await _process_1001_to_request_info(client, db_path)

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.post("/cases/CASE-1001/pushback", data={"feedback": "   "}, follow_redirects=False)

    assert resp.status_code == 422
    assert get_status("CASE-1001", db_path=db_path) == CaseState.PENDING_REVIEW


# --- memory wiring (record_correction + pushback composition) -------------


async def test_edited_body_approve_also_records_memory_correction(
    client: _RecordingClient, tmp_path: Path
):
    """The edit-detection branch of the approve endpoint writes a
    `claimpilot.memory` `kind="correction"` row ALONGSIDE (not instead of)
    the existing `audit_log` `correction_recorded` row -- the
    feedback distiller reads from `memory`, not `audit_log`.
    """
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)  # CASE-1002, user_id="283959"

    edited_text = "Dear customer, this is the rep-edited version of the email."

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.post(
            "/cases/CASE-1002/approve",
            data={"action": "save_and_approve", "edited_body": edited_text},
            follow_redirects=False,
        )

    assert resp.status_code == 303

    rows = _memory_correction_rows(db_path, "CASE-1002")
    assert len(rows) == 1
    assert rows[0]["scope"] == "merchant"
    assert rows[0]["merchant_id"] == "283959"
    payload = json.loads(rows[0]["content"])
    assert payload["original_draft"] == ORIGINAL_DRAFT
    assert payload["final_draft"] == edited_text
    assert payload["feedback"] is None  # approve-endpoint edits carry no separate feedback text

    # Both the audit trail AND the memory row exist -- neither replaces the
    # other (store.log_event's docstring covers why both are kept).
    assert any(r["event"] == "correction_recorded" for r in get_audit_log("CASE-1002", db_path=db_path))


async def test_approve_without_edit_records_no_memory_correction(
    client: _RecordingClient, tmp_path: Path
):
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)

    with TestClient(_app(client, db_path)) as tc:
        tc.post("/cases/CASE-1002/approve", data={"action": "approve"}, follow_redirects=False)

    assert _memory_correction_rows(db_path, "CASE-1002") == []


async def test_guard_blocked_approve_records_no_memory_correction(
    client: _RecordingClient, tmp_path: Path
):
    """A rep edit that gets blocked by the outbound guard must leave no
    memory correction row behind, same as it leaves no `audit_log`
    `correction_recorded` row -- an email that was never sent must never
    look like a recorded correction to the future feedback distiller.
    """
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)
    malicious_edit = "Dear customer, we will send you $999.00 for your claim."

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.post(
            "/cases/CASE-1002/approve",
            data={"action": "save_and_approve", "edited_body": malicious_edit},
            follow_redirects=False,
        )

    assert resp.status_code == 422
    assert _memory_correction_rows(db_path, "CASE-1002") == []


async def test_duplicate_action_approve_records_no_memory_correction(
    client: _RecordingClient, tmp_path: Path
):
    """Mirrors `test_repeat_approve_while_still_reviewable_hits_duplicate_
    action_guard`: a pre-recorded `email` action must reject the approve
    attempt before any memory correction row is written, exactly as it
    writes no `audit_log` correction row.
    """
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)

    from claimpilot.store import record_action

    record_action(
        "CASE-1002",
        "email",
        {"to": "mtaparia@shipbob.com", "subject": "already claimed", "body": "..."},
        db_path=db_path,
    )

    with TestClient(_app(client, db_path)) as tc:
        resp = tc.post(
            "/cases/CASE-1002/approve",
            data={"action": "save_and_approve", "edited_body": "Dear customer, edited."},
            follow_redirects=False,
        )

    assert resp.status_code == 409
    assert _memory_correction_rows(db_path, "CASE-1002") == []


async def test_pushback_records_memory_correction_with_feedback(
    client: _RecordingClient, tmp_path: Path
):
    db_path = tmp_path / "t.db"
    await _process_1001_to_request_info(client, db_path)  # CASE-1001, user_id="334430"

    revised_draft = "Dear customer, revised per rep feedback."
    pushback_transport = FakeTransport([_draft(email_draft=revised_draft, rationale="Updated.")])

    with TestClient(_app(client, db_path, transport=pushback_transport)) as tc:
        resp = tc.post(
            "/cases/CASE-1001/pushback",
            data={"feedback": "Please mention their account manager Dana."},
            follow_redirects=False,
        )

    assert resp.status_code == 303

    rows = _memory_correction_rows(db_path, "CASE-1001")
    assert len(rows) == 1
    assert rows[0]["merchant_id"] == "334430"
    payload = json.loads(rows[0]["content"])
    assert payload["original_draft"] == ORIGINAL_DRAFT
    assert payload["final_draft"] == revised_draft
    assert payload["feedback"] == "Please mention their account manager Dana."


async def test_pushback_redraft_prompt_composes_merchant_memory_and_immediate_feedback(
    client: _RecordingClient, tmp_path: Path
):
    """The pushback redraft's `memory_context` is durable merchant/policy
    memory PLUS the rep's immediate feedback as its own labeled section --
    not one overloaded string (web/app.py module docstring point 9).
    """
    from claimpilot import memory as memory_module

    db_path = tmp_path / "t.db"
    await _process_1001_to_request_info(client, db_path)  # CASE-1001, user_id="334430"
    memory_module.record_policy_note(
        "This merchant prefers formal language.", scope="merchant", merchant_id="334430", db_path=db_path
    )

    pushback_transport = FakeTransport([_draft(email_draft="revised", rationale="Updated.")])
    feedback_text = "Please mention their account manager Dana."

    with TestClient(_app(client, db_path, transport=pushback_transport)) as tc:
        resp = tc.post(
            "/cases/CASE-1001/pushback", data={"feedback": feedback_text}, follow_redirects=False
        )

    assert resp.status_code == 303
    # calls[0], not calls[-1]: the draft() redraft call is always first;
    # The feedback distiller (evolve.distill_feedback) makes its own,
    # separate structured_call right after, so `calls` may have a second
    # entry here that has nothing to do with this prompt-composition
    # assertion.
    prompt = pushback_transport.calls[0]["messages"][-1]["content"]
    # Durable merchant memory reaches the prompt...
    assert "This merchant prefers formal language." in prompt
    # ...and the immediate feedback reaches the prompt too, as a distinct,
    # clearly labeled section (not silently dropped or merged away).
    assert "Immediate rep feedback" in prompt
    assert feedback_text in prompt


async def test_pushback_redraft_prompt_includes_real_persisted_gate_results(
    client: _RecordingClient, tmp_path: Path
):
    """The pushback redraft must reconstruct `DraftInputs`
    from `store.load_gate_results`, not just the
    `decision`/`amount`/`confidence`/`risk_tier` that survive on the
    `Recommendation` alone (the "known gap" `web/app.py`'s module docstring
    used to describe).

    CASE-1001 (`_process_1001_to_request_info`) has 3 attachments -- no
    `CUSTOMER_CONFIRMATION` -- so it hits exactly one evidence gap
    (`evidence_gaps()` -> `Gap(item=CUSTOMER_CONFIRMATION, reason="MISSING")`)
    and never reaches validation. Before the fix, a pushback redraft always
    passed `evidence_gaps=[]`/`eligibility_result=None` regardless of what
    the case's real gate run found, so `draft()`'s prompt would have
    (wrongly) said "no unresolved evidence gaps" and "Eligibility gate: not
    run for this case" even though both gates DID run and DID have real,
    persisted results.
    """
    db_path = tmp_path / "t.db"
    await _process_1001_to_request_info(client, db_path)

    pushback_transport = FakeTransport([_draft(email_draft="revised", rationale="Updated.")])

    with TestClient(_app(client, db_path, transport=pushback_transport)) as tc:
        resp = tc.post(
            "/cases/CASE-1001/pushback",
            data={"feedback": "Keep it brief."},
            follow_redirects=False,
        )

    assert resp.status_code == 303
    prompt = pushback_transport.calls[0]["messages"][-1]["content"]

    # Real eligibility gate output reached the prompt (not the "not run for
    # this case" placeholder a `None` eligibility_result would render).
    assert "Eligibility gate: eligible=True" in prompt
    assert "Eligibility gate: not run for this case" not in prompt

    # Real evidence gap reached the prompt (not the "no unresolved evidence
    # gaps" placeholder an empty list would render).
    assert "CUSTOMER_CONFIRMATION" in prompt
    assert "MISSING" in prompt
    assert "Evidence gate: no unresolved evidence gaps." not in prompt


async def test_pushback_case_with_no_user_id_still_redrafts_without_crashing(
    client: _RecordingClient, tmp_path: Path
):
    """A case with no `user_id` must not crash the pushback endpoint --
    memory lookup/record are both skipped gracefully (module docstring
    point 11), same defensive guard as the approve endpoint's edit branch.
    """

    class _NoUserIdRecordingClient(_RecordingClient):
        async def get_case(self, case_id: str):
            # `_RecordingClient.get_case` is served via `__getattr__`
            # delegation to `self._inner`, not a real inherited method, so
            # `super().get_case(...)` isn't reachable -- go straight to
            # `self._inner` instead.
            case = await self._inner.get_case(case_id)
            return case.model_copy(update={"user_id": None})

    no_user_id_client = _NoUserIdRecordingClient(FixtureClient(include_synthetic=True))
    db_path = tmp_path / "t.db"
    await _process_1001_to_request_info(no_user_id_client, db_path)

    pushback_transport = FakeTransport([_draft(email_draft="revised", rationale="Updated.")])

    with TestClient(_app(no_user_id_client, db_path, transport=pushback_transport)) as tc:
        resp = tc.post(
            "/cases/CASE-1001/pushback", data={"feedback": "Tone it down."}, follow_redirects=False
        )

    assert resp.status_code == 303
    assert _memory_correction_rows(db_path, "CASE-1001") == []
    # calls[0]: see comment in the prompt-composition test above -- the
    # distiller's own structured_call may append a second entry.
    prompt = pushback_transport.calls[0]["messages"][-1]["content"]
    assert "No merchant identifier available" in prompt


# --- feedback distiller wiring (approve-edit-branch + pushback) -----------


def _distill_notes(notes: list[dict]) -> TransportResult:
    return TransportResult(tool_input={"notes": notes}, input_tokens=1, output_tokens=1, raw_content=[])


class _ScriptedThenRaisingTransport:
    """Serves `results` in order for the first `len(results)` calls (e.g.
    the pushback endpoint's `draft()` redraft call), then raises on every
    call after that -- unlike `FakeTransport`'s plain "exhausted" assertion
    error, this deterministically isolates a failure to a *specific*, later
    call (the distiller's) without touching the calls before it, so a test
    using this can assert the earlier call(s) still succeeded normally.
    """

    def __init__(self, results: list[TransportResult]) -> None:
        self._results = list(results)
        self.calls: list[dict] = []

    async def create(self, **kwargs) -> TransportResult:
        self.calls.append(kwargs)
        if self._results:
            return self._results.pop(0)
        raise RuntimeError("distiller boom -- simulated LLM failure")


async def test_edited_approve_triggers_distiller_and_note_becomes_available(
    client: _RecordingClient, tmp_path: Path
):
    """The approve endpoint's edit-detection branch triggers
    `evolve.distill_feedback` (not just `memory.record_correction`) -- the
    distilled note actually gets persisted and is visible via
    `memory.merchant_context()`/`global_policies()` afterward, not just
    "some call happened."
    """
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)  # CASE-1002, user_id="283959"

    transport = FakeTransport(
        [_distill_notes([{"content": "Mention the account manager by name.", "scope": "merchant"}])]
    )
    edited_text = "Dear customer, this is the rep-edited version of the email."

    with TestClient(_app(client, db_path, transport=transport)) as tc:
        resp = tc.post(
            "/cases/CASE-1002/approve",
            data={"action": "save_and_approve", "edited_body": edited_text},
            follow_redirects=False,
        )

    assert resp.status_code == 303
    # The send/approve itself completed fully -- distillation runs after.
    assert len(client.sent_emails) == 1
    assert merchant_context("283959", db_path=db_path).policy_notes == [
        "Mention the account manager by name."
    ]


async def test_approve_distiller_failure_does_not_block_the_send(
    client: _RecordingClient, tmp_path: Path
):
    """A distiller exception must never prevent the actual send from
    completing -- the approve request still succeeds (303, email sent) even
    though the injected transport always raises.
    """
    db_path = tmp_path / "t.db"
    await _process_1002_to_approve(client, db_path)

    always_raising = _ScriptedThenRaisingTransport([])  # raises on the very first call
    edited_text = "Dear customer, this is the rep-edited version of the email."

    with TestClient(_app(client, db_path, transport=always_raising)) as tc:
        resp = tc.post(
            "/cases/CASE-1002/approve",
            data={"action": "save_and_approve", "edited_body": edited_text},
            follow_redirects=False,
        )

    assert resp.status_code == 303
    assert len(client.sent_emails) == 1
    assert client.sent_emails[0]["body"] == edited_text
    assert len(client.reimbursements) == 1
    assert get_status("CASE-1002", db_path=db_path) == CaseState.SENT
    # The distiller's call was attempted (and raised) -- confirms this test
    # actually exercised the failure path rather than the distiller quietly
    # never being invoked.
    assert len(always_raising.calls) == 1
    assert merchant_context("283959", db_path=db_path).policy_notes == []


async def test_pushback_triggers_distiller_and_note_becomes_available(
    client: _RecordingClient, tmp_path: Path
):
    db_path = tmp_path / "t.db"
    await _process_1001_to_request_info(client, db_path)  # CASE-1001, user_id="334430"

    pushback_transport = FakeTransport(
        [
            _draft(email_draft="Dear customer, revised per rep feedback.", rationale="Updated."),
            _distill_notes([{"content": "Keep paragraphs short in every email.", "scope": "global"}]),
        ]
    )

    with TestClient(_app(client, db_path, transport=pushback_transport)) as tc:
        resp = tc.post(
            "/cases/CASE-1001/pushback",
            data={"feedback": "Please mention their account manager Dana."},
            follow_redirects=False,
        )

    assert resp.status_code == 303
    assert global_policies(db_path=db_path) == ["Keep paragraphs short in every email."]


async def test_pushback_distiller_failure_does_not_block_the_redraft(
    client: _RecordingClient, tmp_path: Path
):
    """A distiller exception must never prevent the pushback redraft itself
    from completing -- the request still succeeds (303, new recommendation
    persisted) even though the distiller call (the second `create()` call,
    after the draft redraft) always raises.
    """
    db_path = tmp_path / "t.db"
    await _process_1001_to_request_info(client, db_path)

    revised_draft = "Dear customer, revised per rep feedback."
    scripted = _ScriptedThenRaisingTransport([_draft(email_draft=revised_draft, rationale="Updated.")])

    with TestClient(_app(client, db_path, transport=scripted)) as tc:
        resp = tc.post(
            "/cases/CASE-1001/pushback",
            data={"feedback": "Tone it down."},
            follow_redirects=False,
        )
        case_resp = tc.get("/cases/CASE-1001")

    assert resp.status_code == 303
    # The redraft itself succeeded and was persisted, despite the distiller
    # call (scripted.calls[1]) raising.
    assert revised_draft in case_resp.text
    assert len(scripted.calls) == 2
    assert global_policies(db_path=db_path) == []


# --- missing-API-key experience ----------------------------------------------
#
# Found by cloning the repo fresh and following the README as a reviewer
# would: with no key in .env, clicking "Process all" produced five copies of
# a raw SDK error ("Could not resolve authentication method. Expected one of
# api_key, auth_token, or credentials to be set") and nothing telling you to
# add a key. scripts/seed.py already failed fast with a clear message; the UI
# was the one path that didn't.


async def test_process_all_refuses_clearly_when_no_api_key_is_configured(
    client: _RecordingClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    db_path = tmp_path / "t.db"

    with TestClient(_app(client, db_path)) as tc:
        tc.post("/admin/fetch-cases", follow_redirects=False)
        resp = tc.post("/admin/process-all", follow_redirects=False)

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "ANTHROPIC_API_KEY" in detail          # names the exact variable
    assert ".env" in detail                        # and where to put it
    # nothing was attempted, so no half-processed cases left behind
    assert len(list_cases_by_status(CaseState.INTAKE, db_path=db_path)) == 5


async def test_per_case_process_refuses_clearly_when_no_api_key(
    client: _RecordingClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "")
    db_path = tmp_path / "t.db"

    with TestClient(_app(client, db_path)) as tc:
        tc.post("/admin/fetch-cases", follow_redirects=False)
        resp = tc.post("/cases/CASE-1002/process", follow_redirects=False)

    assert resp.status_code == 409
    assert "OPENAI_API_KEY" in resp.json()["detail"]  # provider-aware, not hardcoded


async def test_queue_warns_up_front_when_no_api_key_is_configured(
    client: _RecordingClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Proactive, not just on failure -- someone who just cloned this should
    learn a key is needed before clicking a button and waiting.
    """
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "")

    with TestClient(_app(client, tmp_path / "t.db")) as tc:
        body = tc.get("/cases").text

    assert 'class="setup-banner"' in body
    assert "No Anthropic API key configured" in body
    assert "ANTHROPIC_API_KEY" in body
    # ...and it's honest about what still works without one
    assert "test suite all work without" in body


async def test_no_setup_banner_once_a_key_is_configured(
    client: _RecordingClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-not-a-real-key")

    with TestClient(_app(client, tmp_path / "t.db")) as tc:
        body = tc.get("/cases").text

    assert 'class="setup-banner"' not in body


def test_configured_api_key_is_provider_aware(monkeypatch: pytest.MonkeyPatch):
    """One shared helper -- seed.py, the eval harness and the web app all read
    it, so a provider added later can't leave one of them silently wrong.
    """
    from claimpilot.config import configured_api_key

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-x")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    assert configured_api_key() == (True, "OPENAI_API_KEY", "OpenAI")

    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    assert configured_api_key() == (False, "ANTHROPIC_API_KEY", "Anthropic")
