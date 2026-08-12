"""Review UI + approval endpoints.

This is the human-in-the-loop layer -- the whole point of the project's
safety model is that nothing reaches the merchant/reimbursement API without
a rep clicking a button here. `pipeline.process_case` never sends
anything; this module is where a `Recommendation` sitting in
`PENDING_REVIEW`/`ESCALATED` finally turns into an outbound email (and,
for approved payouts, reimbursement submissions).

**Non-negotiable invariant, restated because it must never regress:**
nothing in this module calls `client.send_email` or
`client.submit_reimbursement` except inside `POST /cases/{id}/approve`, and
never with a recipient other than the case's own `contact_email` fetched
fresh from the client -- never from user-supplied form data or draft text.
`tests/test_web.py`'s "nothing hits outbox without approve" / "edited body
is what gets sent" tests guard exactly this.

Design decisions / judgment calls (this area has substantial freedom --
documenting the choices made here):

1. **Routes.** `GET /cases` is the queue (not `/`) -- explicit and
   RESTful, leaves `/` free for a future landing page/redirect without
   reshuffling this module. `GET /cases/{case_id}` is the case detail page.
   `GET /cases/{case_id}/attachments/{attachment_id}` is a small helper
   endpoint that streams attachment bytes for `<img>` tags on the case page
   (chosen over inlining base64 in the HTML -- keeps the page HTML small and
   is trivially testable on its own via a direct GET).
2. **Queue scope.** Shows both `PENDING_REVIEW` and `ESCALATED` cases --
   both need a rep's eyes, even though `ESCALATED` cases carry a
   `decision="request_info"` label that's a stretch (see `pipeline.py`
   module docstring point 8). `CLOSED`/`SENT`/other terminal-ish states are
   never listed -- the queue skips anything not awaiting a rep.
3. **Plain HTML forms, not HTMX.** A plain `<form method="post">` per
   button is sufficient for a demo review UI and is far simpler to drive
   from `TestClient` (no JS runtime, no partial-page-swap semantics to fake
   in tests). If a future need calls for a snappier UX, HTMX can be layered
   on top of these same endpoints without changing their contract.
4. **Gate-result detail, given the persistence gap.** `pipeline.py` only
   persists the final `Recommendation` (decision/amount/line_items/
   rationale/email_draft/confidence/risk_tier) -- the intermediate
   `EligibilityResult`/`evidence_gaps`/`ValidationDecision`/`CalcResult`
   objects are never written anywhere queryable. So the case page shows:
   `Recommendation.rationale` (LLM prose that already cites which gate
   produced each point), `Recommendation.confidence`
   and `.risk_tier` (the two structured numbers that *are* persisted), and
   the full `audit_log` timeline (`store.get_audit_log`) whose payloads
   carry gate-specific detail for the branches that recorded one -- e.g.
   `{"reason": "TOO_OLD"}` for an eligibility deny, `{"gaps": [...]}` for an
   evidence gap, `{"reason": "..."}` for a validation escalate/request-info.
   This is what's actually available; no new persistence was invented just
   for this page.
   **Update:** `store.save_gate_results`/`load_gate_results`
   now persist the real intermediate gate objects after all (added for the
   outbound guard's re-verification, see `store.GateResults`, and reused by
   `pushback()` below per module docstring point 9) -- so the case detail
   page's continued reliance on `Recommendation.rationale` prose plus the
   audit log, rather than also rendering `store.load_gate_results(case_id)`
   directly, is now a deliberate UI-simplicity choice, not a hard
   persistence limitation. Surfacing the structured gate objects on
   `case.html` directly (rather than via LLM-authored rationale prose) would
   be a reasonable follow-up.
5. **Reimbursement `product_name`.** `RecommendationLineItem` has no
   product-name field (only `sku`/`quantity`/`unit_price`/`subtotal`), but
   `submit_reimbursement` requires one. `_sku_product_names()` fetches the
   case's order and maps `sku -> LineItem.name`; falls back to the bare SKU
   string if the order can't be fetched or doesn't have a match. A missing
   product name is a cosmetic downgrade, not a reason to fail the approve
   flow.
6. **Idempotency / double-send prevention.** `store.record_action` is
   called *before* the corresponding `client` call, per the task's own
   suggested ordering: attempt-record-first means a duplicate raises
   `DuplicateActionError` and the endpoint returns `409` with nothing sent,
   rather than sending first and racing the record. The accepted tradeoff
   (documented, not hidden): if the process crashes between a successful
   `record_action` and the `client.send_email`/`submit_reimbursement` call
   actually completing, the action is marked "done" but wasn't -- this
   trades a rare "recorded but not delivered" edge case for the guarantee
   that actually matters for a payments-adjacent system: a double-clicked
   button or a retried request can NEVER double-send/double-pay. The email
   and reimbursement actions are recorded/guarded independently (two
   separate `(case_id, action)` keys), so retrying an approve after a
   mid-flight failure can still complete whichever half didn't finish --
   except the half that already recorded successfully, which is treated as
   done and skipped.
7. **State-machine hop for approve.** `store.LEGAL_TRANSITIONS` does not
   allow `PENDING_REVIEW -> SENT` directly -- only via `APPROVED`/`DENIED`/
   `NEEDS_INFO` (each of which *does* legally lead to `SENT`). So approving
   a `PENDING_REVIEW` case writes two transitions (decision-labeled state,
   then `SENT`), both `actor="rep"`. `ESCALATED` cases, by contrast, are
   legally allowed to go straight to `SENT` (see `store.py`'s transition
   map), so that path is a single hop. The plan's literal wording ("transition
   the case to SENT (actor='rep', event='approved')") is honored for the
   final hop's event name; the intermediate hop (when needed) gets its own
   `transition:pending_review-><state>` event, consistent with
   `pipeline.py`'s existing transition-naming convention.
8. **Pushback never calls `transition()`.** See `store.py`'s module-level
   comment above `LEGAL_TRANSITIONS` for why: a pushback doesn't advance the
   case anywhere, it just produces a fresher draft for the same review
   queue. `store.log_event()` (new in this task) writes the audit row;
   `store.save_recommendation()` updates the persisted `Recommendation`;
   `cases.status` never changes.
9. **Pushback's `DraftInputs` reconstruction uses `store.load_gate_results`.**
   `cases.gate_results_json` persists the full
   intermediate gate objects (`eligibility`/`evidence_gaps`/`validation`/
   `calc`) so the outbound guard could
   re-verify against them (see `store.GateResults`) -- an earlier version
   of this endpoint instead reconstructed `DraftInputs` from only the
   persisted `Recommendation` (`decision`/`amount`/`confidence`/`risk_tier`),
   leaving that richer data unused. `pushback()`
   calls `store.load_gate_results(case_id, db_path=db_path)` and passes
   `eligibility_result`/`evidence_gaps`/`validation_decision` (the last via
   `combine_validation(gate_results.validation)`, since `GateResults` stores
   the raw LLM `ValidationResult`, not the derived `ValidationDecision`
   `DraftInputs` expects) straight through to the redraft. A redraft after
   pushback now sees the same gate-result context the original draft saw,
   not a strictly smaller one.
   **`memory_context` composition:**
   `memory_context` is built from two concatenated parts, each its own
   clearly labeled section, since they're different kinds of information
   that both need to reach the prompt: (1) `claimpilot.memory.
   merchant_context(case.user_id, ...).to_prompt_text()` -- the durable,
   cross-case merchant/policy memory (recent notes/corrections, claim
   frequency, merchant + global policy notes), same call `pipeline.py`
   makes; (2) the rep's immediate feedback text on *this* pushback,
   labeled as a one-off instruction rather than a durable fact about the
   merchant ("fix this now" vs. "always do X for this merchant" -- see
   `claimpilot.memory`'s module docstring for why raw feedback is never
   itself treated as a policy note). Part (2) is appended after part (1) so
   the model sees durable context first and the immediate ask last, closest
   to where it starts writing.
   `calc_result` uses `gate_results.calc` directly when present (it now
   carries the real `capped` flag, unlike the old synthetic
   reconstruction), falling back to `_pushback_calc_result()`'s synthetic
   `CalcResult` (built from `Recommendation.line_items`/`.amount`,
   `capped=False`) only when `gate_results.calc` is `None` -- i.e. a case
   whose gate results were never persisted at all (built directly via
   `store.create_case`/`transition` in a test, bypassing the pipeline) but
   somehow still carries line items on its `Recommendation`. This fallback
   should be unreachable for any case that went through the real pipeline
   (`pipeline._exit()` always calls `save_gate_results`), so it exists only
   as a defensive floor, not an expected code path.
10. **The outbound guard (`check_outbound`) runs in the approve
    endpoint**, after the rep clicks approve and the final email body
    (edited or not) is assembled, but BEFORE `record_action`/`send_email`/
    `submit_reimbursement` -- see the inline comments at the call site for
    why it must sit before `record_action` specifically. A non-empty
    `violations` list blocks all three outbound calls, writes an
    `outbound_guard_blocked` audit event (a real `transition()` to
    `ESCALATED` for a `PENDING_REVIEW` case; a `log_event()` for an
    already-`ESCALATED` case, since `ESCALATED -> ESCALATED` has no legal
    self-loop), and returns `422` with the violations in the response body
    so the rep sees exactly which invariant fired. `guard.py`'s own
    docstring covers the invariants and their scope in full.
11. **`claimpilot.memory.record_correction` is called from both the
    approve endpoint's edit-detection branch and the pushback endpoint**,
    ALONGSIDE (not instead of) their existing `store.log_event` calls --
    see `store.log_event`'s docstring for why both are kept. Both call
    sites guard on `case.user_id is not None` first (mirroring how
    `_sku_product_names()` above already treats a missing lookup as
    cosmetic rather than fatal): `record_correction` raises if `user_id` is
    `None`, and letting that propagate from inside the approve endpoint --
    specifically from the `if edited:` block, which runs AFTER
    `record_action` has already claimed the "email" action and BEFORE
    `send_email` actually runs -- would 500 there and strand the case with
    an action permanently marked sent but never actually delivered. No
    fixture/demo case is missing a `user_id` today, so this guard is
    defense against future/real data, not something the test suite can
    exercise via a fixture case.
12. **The feedback distiller (`evolve.distill_feedback`) is called
    from both the same two spots, but NOT at the same place in the approve
    endpoint's control flow as `record_correction`.** In the pushback
    endpoint, nothing outbound has been claimed yet, so the distiller call
    sits right after `record_correction`. In the approve endpoint, by contrast, `record_correction`
    lives inside the `if edited:` block that runs AFTER `record_action` has
    claimed the "email" action and BEFORE `send_email` actually runs (see
    point 6's documented "recorded but not delivered" crash window) --
    deliberately kept as short as possible today. Adding an LLM round-trip
    (up to `LLM_TIMEOUT_SECONDS`) into that exact window would materially
    widen it: a slow or hanging distiller call could strand a case with its
    email action permanently marked sent but nothing actually delivered.
    So the approve endpoint's distiller call is deferred to the very end of
    the handler, immediately before the final `RedirectResponse` -- after
    every outbound call and state transition has already completed, where a
    slow/failing distiller can no longer strand anything. Both call sites
    wrap `distill_feedback(...)` in a narrow `try/except Exception` that
    logs a warning and continues: distillation is a best-effort
    self-improvement side effect, never a safety-critical path, and an LLM
    call failing/erroring must never prevent the actual send/pushback
    (which has, in the approve case, already fully completed by the time
    the distiller runs) from being reported as successful to the rep.
"""

from __future__ import annotations

import json
import logging
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from claimpilot import evolve, memory, store
from claimpilot.calc import CalcResult
from claimpilot.clients.base import NotFoundError, ShipBobClient, get_client
from claimpilot.clients.synthetic import synthetic_case_ids
from claimpilot.config import configured_api_key, settings
from claimpilot.draft import DraftInputs, draft
from claimpilot.gates.validation import combine_validation
from claimpilot.guard import EmailToSend, check_outbound
from claimpilot.llm import Transport
from claimpilot.models import Case, CaseState, Recommendation
from claimpilot.pipeline import process_case
from claimpilot.risk import RiskAssessment, RiskTier
from claimpilot.store import DuplicateActionError

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# --- Access control (found missing entirely in a full-codebase security
# audit -- see config.py's `review_ui_username`/`review_ui_password`
# comment). `auto_error=False` so a request with NO Authorization header at
# all reaches `require_auth` itself (as `credentials=None`) instead of
# FastAPI's `HTTPBasic` short-circuiting straight to a 401 -- needed so
# `require_auth` can tell "no auth configured, let it through" apart from
# "auth configured but the client didn't send any" (both start as "no
# credentials on the request", but only the second should ever 401).
_basic_auth = HTTPBasic(auto_error=False)


def require_auth(credentials: HTTPBasicCredentials | None = Depends(_basic_auth)) -> None:
    """Global auth gate, applied to every route via `FastAPI(dependencies=[...])`
    below -- opt-in HTTP Basic Auth, active only when both
    `settings.review_ui_username` and `.review_ui_password` are configured
    (both blank, the default, preserves today's no-auth behavior exactly, so
    this is additive and never breaks local dev/demo usage that hasn't set
    either).

    Constant-time comparison (`secrets.compare_digest`) for both username and
    password -- a naive `==` on credentials leaks timing information an
    attacker could use to brute-force characters one at a time.
    """
    configured_username = settings.review_ui_username
    configured_password = settings.review_ui_password
    if not configured_username or not configured_password:
        return

    valid = (
        credentials is not None
        and secrets.compare_digest(credentials.username, configured_username)
        and secrets.compare_digest(credentials.password, configured_password)
    )
    if not valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )

# Statuses that put a case in front of a rep (module docstring point 2).
QUEUE_STATUSES: tuple[CaseState, ...] = (CaseState.PENDING_REVIEW, CaseState.ESCALATED)
_REVIEWABLE_STATUS_VALUES = {s.value for s in QUEUE_STATUSES}

# Demo scenarios the "Process all" button deliberately leaves in INTAKE for a
# human to trigger by hand, mirroring the same exclusion `scripts/seed.py`
# makes in `DEMO_CASE_IDS`, for the same reason.
#
# `CASE-9003-REPEAT` is a second case from a merchant that already appears in
# the queue (`CASE-1001`, user_id 334430). It exists to show a rep's
# correction carrying forward into a *later* case's draft -- the brief's
# "Not just on this case. On the next one too." Processing it in the same
# batch as everything else would run it *before* the rep has pushed back on
# anything, so there'd be no correction to carry and the beat would silently
# show nothing. Held back so it can be processed after the pushback, which is
# the only ordering where the behaviour is visible at all.
BULK_PROCESS_HOLDBACK: frozenset[str] = frozenset({"CASE-9003-REPEAT"})

# Everything that's left the active queue -- a case a rep has already acted
# on (approve/deny -> SENT is the common case; APPROVED/DENIED/NEEDS_INFO
# included defensively in case a case is ever found mid-transition rather
# than atomically all the way to SENT, and CLOSED for a fully wrapped-up
# NEEDS_INFO -> SENT -> CLOSED loop). Surfaced on `GET /cases/history` so a
# sent/denied case doesn't just silently vanish from the UI once it leaves
# `QUEUE_STATUSES` -- a rep should be able to look back at what already went
# out, not just what's still pending.
HISTORY_STATUSES: tuple[CaseState, ...] = (
    CaseState.APPROVED,
    CaseState.DENIED,
    CaseState.NEEDS_INFO,
    CaseState.SENT,
    CaseState.CLOSED,
)

# `PENDING_REVIEW` cannot transition straight to `SENT` (module docstring
# point 7) -- this is the decision-labeled intermediate state each
# `Recommendation.decision` value maps to, all three of which *do* legally
# lead to `SENT` per `store.LEGAL_TRANSITIONS`.
_DECISION_TO_INTERMEDIATE_STATE: dict[str, CaseState] = {
    "approve": CaseState.APPROVED,
    "deny": CaseState.DENIED,
    "request_info": CaseState.NEEDS_INFO,
}


@dataclass
class DemoRunState:
    """Progress of an in-flight "fetch & process all" run, held in memory on
    `app.state`.

    In-memory and per-process on purpose: this exists only to drive a
    progress banner during a walkthrough. Persisting it would mean schema,
    migrations and cleanup for something that is meaningless the moment the
    process restarts, and a multi-worker deployment would need shared state
    anyway -- at which point a real job queue is the right answer, not a
    dataclass. Assumes a single worker, which the default deployment uses.
    """

    running: bool = False
    total: int = 0
    done: int = 0
    current: str | None = None
    errors: list[str] = field(default_factory=list)

    def reset(self, total: int) -> None:
        self.running = True
        self.total = total
        self.done = 0
        self.current = None
        self.errors = []

    def finish(self) -> None:
        self.running = False
        self.current = None


def _reset_outbox() -> None:
    """Delete the outbox JSONL file(s), leaving the directory in place
    (recreated lazily on the next real send). Mirrors `scripts/seed.py`'s
    `reset_outbox()` -- kept as a separate small helper here rather than
    imported from `scripts/`, which isn't an importable package from the
    installed app.
    """
    from claimpilot.clients.fixtures import _OUTBOX_DIR

    if not _OUTBOX_DIR.exists():
        return
    for path in _OUTBOX_DIR.glob("*.jsonl"):
        path.unlink()


def _resolve_client(request: Request) -> ShipBobClient:
    return request.app.state.client or get_client()


def _db_path(request: Request) -> Path | str | None:
    return request.app.state.db_path


def _email_subject_for(case_id: str) -> str:
    """The one place the outbound email subject line is computed -- used by
    both `approve()` (what actually gets sent) and `case_detail()` (the
    case page's "Email draft" preview, added after a UI-polish audit found
    the preview showed no recipient/subject at all, just a bare body
    textarea). Factored out specifically so the preview can never drift
    from what `approve()` actually sends -- a hardcoded copy of this
    f-string in the template would silently go stale the moment this
    changed here.
    """
    return f"Update on your ShipBob claim {case_id}"


def _case_row_or_404(case_id: str, *, db_path: Path | str | None) -> object:
    row = store.get_case(case_id, db_path=db_path)
    if row is None:
        raise HTTPException(status_code=404, detail=f"case {case_id!r} not found")
    return row


def _require_reviewable(row) -> None:
    if row["status"] not in _REVIEWABLE_STATUS_VALUES:
        raise HTTPException(
            status_code=409,
            detail=f"case {row['case_id']!r} is not awaiting review (status={row['status']!r})",
        )


async def _sku_product_names(client: ShipBobClient, case: Case) -> dict[str, str]:
    """Best-effort `sku -> product name` lookup for `submit_reimbursement`'s
    required `product_name` argument (module docstring point 5). Falls back
    to an empty mapping -- callers then fall back to the bare SKU -- if the
    order is unavailable for any reason; a missing product *name* is
    cosmetic, not a reason to fail the whole approve flow.
    """
    if not case.order_id:
        return {}
    try:
        order = await client.get_order(case.order_id)
    except NotFoundError:
        return {}
    return {line.sku: line.name for line in order.line_items}


# Audit events that can put a case into ESCALATED (module docstring point 4
# note, and `pipeline.py`'s escalation exits + this module's own
# `outbound_guard_blocked` event) -- used by `_escalation_summary()` below to
# turn the bare word "escalated" into a reason a rep can act on without
# reading the raw audit JSON. Kept as a plain function (not a dict lookup)
# since each event's payload shape is different.
def _escalation_summary(audit_events: list[dict]) -> str | None:
    """Best-effort, human-readable reason a case is sitting in `ESCALATED`,
    surfaced on the case detail page so the status doesn't render
    as a bare word indistinguishable from `PENDING_REVIEW` at a glance.

    Scans the case's own audit log (already loaded for the page) newest-
    first for the most recent event known to cause an escalation, and turns
    its payload into one short sentence. Returns `None` if no such event is
    found -- e.g. a case escalated directly via `store.transition` in a
    test, bypassing the pipeline/guard entirely -- so callers must fall back
    to a generic label rather than assume a reason always exists.
    """
    for ev in reversed(audit_events):
        event = ev["event"]
        payload = ev["payload"] or {}
        if event == "gate:eligibility_insured":
            return "Routed to the insured-claims process -- outside this pipeline's automated decision scope."
        if event == "gate:validation_escalated":
            return f"Low-confidence damage validation: {payload.get('reason') or 'see audit log for detail'}."
        if event == "gate:validation_affected_count_mismatch":
            return f"Affected-count mismatch: {payload.get('reason') or 'see audit log for detail'}."
        if event == "gate:claim_scope_mismatch":
            return (
                "Claim-scope mismatch: "
                f"{payload.get('reason') or 'see audit log for detail'}."
            )
        if event == "gate:invoice_audit_discrepancy":
            return (
                "Retail-invoice discrepancy: "
                f"{payload.get('reason') or 'see the invoice reconciliation panel below'}."
            )
        if event == "gate:calc_exception":
            detail = payload.get("error") or payload.get("error_type") or "see audit log for detail"
            return f"Reimbursement calculation failed: {detail}."
        if event == "outbound_guard_blocked":
            violations = payload.get("violations") or []
            invariants = ", ".join(v.get("invariant", "?") for v in violations) or "an outbound invariant"
            return f"Outbound send blocked by the guard ({invariants}) -- edit the draft and re-approve, or investigate."
    return None


def _pushback_calc_result(recommendation: Recommendation) -> CalcResult | None:
    """Fallback synthetic `CalcResult`, built from a persisted
    `Recommendation`, used by `pushback()` only when `store.load_gate_results`
    has no real `calc` for this case (module docstring point 9). `None` when
    there were no line items to begin with (deny/request_info
    recommendations), so `draft()`'s prompt still says "Calc gate: not run
    for this case" rather than fabricating one.
    """
    if not recommendation.line_items:
        return None
    return CalcResult(amount=recommendation.amount, line_items=list(recommendation.line_items), capped=False)


def create_app(
    *,
    client: ShipBobClient | None = None,
    transport: Transport | None = None,
    db_path: Path | str | None = None,
) -> FastAPI:
    """Build a `FastAPI` app for the review UI.

    `client`/`transport`/`db_path` are stored on `app.state` and default to
    the real singletons/on-disk database when omitted -- tests construct
    their own app via `create_app(client=..., transport=..., db_path=...)`
    so no real network/LLM/on-disk-DB access ever happens under test, same
    test-injection convention as `pipeline.process_case`.
    """
    app = FastAPI(title="claimpilot review")
    app.state.client = client
    app.state.transport = transport
    app.state.db_path = db_path
    app.state.demo_run = DemoRunState()

    @app.get("/health")
    async def health() -> dict:
        """Deliberately registered directly on `app`, NOT on the `protected`
        router below -- independent of `require_auth` and of the on-disk
        DB/real client/LLM transport, so a liveness probe (Docker's
        `HEALTHCHECK`, a load balancer, etc.) can always confirm the process
        itself is up even when Basic Auth is configured, without needing
        credentials baked into every health-check caller. Returns no
        case/business data -- there is nothing here for an unauthenticated
        caller to learn beyond "the process is alive".
        """
        return {"status": "ok"}

    # Every other route in this app is registered on `protected`, not `app`,
    # so `require_auth` (config.py's opt-in HTTP Basic Auth -- a no-op when
    # unconfigured) applies uniformly to the whole review UI without having
    # to remember to add it to each route individually, while `/health`
    # above stays reachable unauthenticated. `app.include_router(protected)`
    # at the end of this function is what actually mounts these routes.
    protected = APIRouter(dependencies=[Depends(require_auth)])

    @protected.get("/cases", response_class=HTMLResponse)
    async def queue(request: Request) -> HTMLResponse:
        db_path = _db_path(request)
        client = _resolve_client(request)

        rows = []
        for status in QUEUE_STATUSES:
            rows.extend(store.list_cases_by_status(status, db_path=db_path))
        rows.sort(key=lambda r: r["updated_at"])

        queue_items = []
        for row in rows:
            case_id = row["case_id"]
            recommendation = (
                Recommendation.model_validate_json(row["recommendation_json"])
                if row["recommendation_json"]
                else None
            )
            try:
                case = await client.get_case(case_id)
                account_name = case.account_name or case_id
            except NotFoundError:
                # Fixture/demo data gap -- still show the row rather than
                # hiding a case a rep needs to act on.
                account_name = case_id
            queue_items.append(
                {
                    "case_id": case_id,
                    "account_name": account_name,
                    "status": row["status"],
                    "risk_tier": recommendation.risk_tier if recommendation else "UNKNOWN",
                    "decision": recommendation.decision if recommendation else None,
                }
            )

        # Freshly-fetched-but-unprocessed cases (`INTAKE`), shown as their
        # own section rather than mixed into the review queue: they have no
        # recommendation, no risk tier and nothing for a rep to decide yet,
        # so listing them alongside real work would pad the queue count with
        # rows that aren't actually waiting on a human.
        intake_items = []
        for row in store.list_cases_by_status(CaseState.INTAKE, db_path=db_path):
            case_id = row["case_id"]
            try:
                account_name = (await client.get_case(case_id)).account_name or case_id
            except NotFoundError:
                account_name = case_id
            intake_items.append(
                {
                    "case_id": case_id,
                    "account_name": account_name,
                    "held_back": case_id in BULK_PROCESS_HOLDBACK,
                }
            )
        intake_items.sort(key=lambda item: item["case_id"])
        bulk_pending = [item for item in intake_items if not item["held_back"]]

        policy_notes = memory.list_policy_notes(db_path=db_path)
        queue_stats = store.stats(db_path=db_path)

        return templates.TemplateResponse(
            request,
            "queue.html",
            {
                "queue_items": queue_items,
                "intake_items": intake_items,
                "bulk_pending_count": len(bulk_pending),
                "policy_notes": policy_notes,
                "stats": queue_stats,
                "demo_controls": settings.demo_controls_enabled,
                "demo_run": request.app.state.demo_run,
                # Surfaced proactively rather than only on failure: someone
                # who has just cloned this should learn a key is needed
                # before they click a button and wait, not after.
                "api_key_missing": not configured_api_key()[0],
                "api_key_env_var": configured_api_key()[1],
                "api_key_provider": configured_api_key()[2],
                "synthetic_total": len(synthetic_case_ids()),
            },
        )

    @protected.get("/cases/history", response_class=HTMLResponse)
    async def history(request: Request) -> HTMLResponse:
        """Cases that have already left the active queue (sent/denied/closed --
        see `HISTORY_STATUSES`). `GET /cases` deliberately only shows what
        still needs a rep's attention; without this route, a case a rep just
        approved/denied would just disappear from the UI the moment it left
        `PENDING_REVIEW`/`ESCALATED`, with no way to look back at what already
        went out short of querying the database directly.
        """
        db_path = _db_path(request)
        client = _resolve_client(request)

        rows = []
        for status in HISTORY_STATUSES:
            rows.extend(store.list_cases_by_status(status, db_path=db_path))
        # Most-recently-acted-on first -- unlike the queue's oldest-first FIFO
        # triage ordering, history is for "what just happened", not "what's
        # been waiting longest".
        rows.sort(key=lambda r: r["updated_at"], reverse=True)

        history_items = []
        for row in rows:
            case_id = row["case_id"]
            recommendation = (
                Recommendation.model_validate_json(row["recommendation_json"])
                if row["recommendation_json"]
                else None
            )
            try:
                case = await client.get_case(case_id)
                account_name = case.account_name or case_id
            except NotFoundError:
                account_name = case_id
            history_items.append(
                {
                    "case_id": case_id,
                    "account_name": account_name,
                    "status": row["status"],
                    "decision": recommendation.decision if recommendation else None,
                    "amount": recommendation.amount if recommendation else None,
                    "updated_at": row["updated_at"],
                }
            )

        return templates.TemplateResponse(request, "history.html", {"history_items": history_items})

    @protected.post("/memory/notes/{note_id}/delete")
    async def delete_policy_note(
        request: Request, note_id: int, next: str | None = Form(None)
    ) -> RedirectResponse:
        """Memory review panel delete action -- a plain HTML
        form POST (no JS/HTMX), consistent with the approve/pushback
        forms. Cap enforcement lives in `memory.record_policy_note`
        (write-time); this route only ever deletes, and
        `memory.delete_note` is already a no-op on a missing/nonexistent
        id (module docstring point 7 in `memory.py`) -- a rep double-
        clicking delete, or deleting a note that's already been evicted by
        the cap, must never 404/500 this route.

        `next` (optional hidden form field): where to redirect back to.
        Defaults to the queue (`/cases`, this route's original behavior --
        the queue page's own delete forms never send this field, so they
        are unaffected). The case detail page's newly-wired-up merchant
        memory panel sends `next=/cases/{case_id}` so deleting a note from
        there returns the rep to the case they were looking at instead of
        bouncing them to the queue. Restricted to same-site relative paths
        (must start with exactly one leading `/`) to avoid turning this
        into an open redirect via a crafted form.
        """
        db_path = _db_path(request)
        memory.delete_note(note_id, db_path=db_path)
        redirect_url = next if next and next.startswith("/") and not next.startswith("//") else "/cases"
        return RedirectResponse(url=redirect_url, status_code=303)

    def _require_api_key() -> None:
        """Refuse to start pipeline work when no LLM key is configured.

        Without this the request goes ahead and every case fails deep inside
        the provider SDK with `Could not resolve authentication method.
        Expected one of api_key, auth_token, or credentials to be set` --
        which is accurate and completely unhelpful to someone who has just
        cloned the repo and not yet filled in `.env`. `scripts/seed.py`
        already fails fast with a clear message for exactly this reason; the
        UI should not be the one path that reports it as five identical
        stack-trace strings.
        """
        is_set, env_var, provider = configured_api_key()
        if not is_set:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"No {provider} API key configured. Processing a case makes real "
                    f"{provider} API calls, so set {env_var} in your .env file and restart "
                    f"the app. (Everything else -- browsing the queue, the test suite, "
                    f"fetching cases from the ShipBob API -- works without one.)"
                ),
            )

    def _require_demo_controls() -> None:
        """404 (not 403) when demo controls are disabled -- a disabled admin
        action shouldn't advertise its own existence to an unauthenticated
        prober. Enforced on the endpoints themselves, not just by hiding the
        buttons, so `DEMO_CONTROLS_ENABLED=false` actually closes the door
        rather than only removing the handle.
        """
        if not settings.demo_controls_enabled:
            raise HTTPException(status_code=404, detail="demo controls are disabled")

    @protected.post("/admin/reset")
    async def reset_demo_data(request: Request) -> RedirectResponse:
        """Wipe every case, audit row, action, LLM-call record and memory
        note, plus the outbox -- back to a genuinely empty system.

        Deliberately also clears `memory`: a demo that resets cases but
        silently keeps merchant policy notes would replay beat 3 (the
        memory carry-forward loop) with the note already present, showing
        nothing. "Reset" has to mean reset.
        """
        _require_demo_controls()
        deleted = store.reset_all(db_path=_db_path(request))
        _reset_outbox()
        logging.getLogger(__name__).info("demo reset -- rows deleted: %s", deleted)
        return RedirectResponse(url="/cases", status_code=303)

    @protected.post("/admin/fetch-cases")
    async def fetch_cases(request: Request) -> RedirectResponse:
        """Pull the live case list from the ShipBob API and create a row for
        each one not already tracked, leaving them in `INTAKE`.

        Deliberately does NOT run the pipeline. Fetching is one cheap API
        call and should feel instant in a demo; processing is several vision
        calls per case and takes the better part of a minute for the full
        set. Splitting them means the queue populates immediately (showing
        the real API integration) and the pipeline can then be run on one
        case at a time, visibly, via `process_case_now` below -- rather than
        one opaque 60-second hang that looks like the app froze.

        Idempotent: re-fetching skips cases that already exist rather than
        erroring on the duplicate primary key, so the button is safe to
        double-click.
        """
        _require_demo_controls()
        created = await _fetch_new_cases(request)
        logging.getLogger(__name__).info("demo fetch -- %d new case(s)", created)
        return RedirectResponse(url="/cases", status_code=303)

    async def _fetch_new_cases(request: Request) -> int:
        """Create an `INTAKE` row for every API case not already tracked,
        returning how many were new.

        `list_cases` returns a summary shape; `user_id` (needed for merchant
        memory attribution) only reliably exists on the detail payload, so
        that's fetched per case rather than storing a null and silently
        losing the merchant link on every fetched case.
        """
        db_path = _db_path(request)
        client = _resolve_client(request)

        cases = await client.list_cases()
        scenarios = set(synthetic_case_ids())
        created = 0
        for case in cases:
            # Belt-and-braces: `SyntheticOverlayClient.list_cases()` already
            # delegates straight through, so the scenarios shouldn't appear
            # here at all. Filtered explicitly anyway so "Fetch" means "what
            # the real API returned" regardless of how the client is
            # composed -- a client that does surface them (a test double, or
            # `FixtureClient(include_synthetic=True)`) must not quietly turn
            # a fetch into a fetch-plus-invented-cases.
            if case.case_id in scenarios:
                continue
            if store.get_case(case.case_id, db_path=db_path) is not None:
                continue
            merchant_id = case.user_id
            if merchant_id is None:
                try:
                    merchant_id = (await client.get_case(case.case_id)).user_id
                except NotFoundError:
                    merchant_id = None
            store.create_case(case.case_id, merchant_id=merchant_id, db_path=db_path)
            created += 1
        return created

    async def _process_all_intake(app: FastAPI, case_ids: list[str]) -> None:
        """Run each fetched case through the pipeline, one at a time, updating
        `app.state.demo_run` as it goes.

        Sequential rather than `asyncio.gather`: the point is that a rep (or
        a room) watches cases land in the queue one by one, and firing seven
        concurrent multi-call vision pipelines at a rate-limited API to save
        thirty seconds would trade the entire demo narrative for latency
        nobody is waiting on.

        One case failing must not abandon the rest -- a single bad attachment
        URL or a transient timeout is recorded against that case and the run
        continues, which is also how the equivalent batch job would need to
        behave in production.
        """
        state: DemoRunState = app.state.demo_run
        log = logging.getLogger(__name__)
        try:
            for case_id in case_ids:
                state.current = case_id
                try:
                    await process_case(
                        case_id,
                        client=app.state.client or get_client(),
                        transport=app.state.transport,
                        db_path=app.state.db_path,
                    )
                except Exception as exc:  # noqa: BLE001 -- one bad case must not stop the batch
                    state.errors.append(f"{case_id}: {type(exc).__name__}: {exc}")
                    log.exception("demo batch: %s failed", case_id)
                finally:
                    state.done += 1
        finally:
            state.finish()
            log.info(
                "demo batch complete -- %d/%d processed, %d error(s)",
                state.done,
                state.total,
                len(state.errors),
            )

    @protected.post("/admin/add-demo-cases")
    async def add_demo_cases(request: Request) -> RedirectResponse:
        """Add the synthetic demo scenarios to the queue, on top of whatever
        the real API returned.

        Separate from "Fetch cases" on purpose. A fetch should show exactly
        what ShipBob's API actually has -- five cases, nothing invented -- so
        the queue you open on is real. These three are local test data
        covering brief requirements the real sample data physically cannot
        exercise (every real shipment is uninsured, all five belong to
        different merchants, and the largest real line item is $59.99), so
        they're opt-in rather than always present.

        Works in live mode as well as fixtures: `get_client()` wraps whatever
        client is configured in a `SyntheticOverlayClient`, so these IDs
        resolve even though the live mock 404s them.

        Idempotent, same as fetch.
        """
        _require_demo_controls()
        db_path = _db_path(request)
        client = _resolve_client(request)

        created = 0
        skipped: list[str] = []
        for case_id in synthetic_case_ids():
            if store.get_case(case_id, db_path=db_path) is not None:
                continue
            try:
                case = await client.get_case(case_id)
            except NotFoundError:
                # The configured client can't resolve this scenario (demo
                # controls disabled at the factory, or a hand-built client in
                # a test). Skip rather than creating a row that would 404 the
                # moment anyone tried to process it.
                skipped.append(case_id)
                continue
            store.create_case(case_id, merchant_id=case.user_id, db_path=db_path)
            created += 1

        logging.getLogger(__name__).info(
            "demo scenarios -- %d added, %d unresolvable %s", created, len(skipped), skipped
        )
        return RedirectResponse(url="/cases", status_code=303)

    @protected.post("/admin/process-all")
    async def process_all(request: Request, background_tasks: BackgroundTasks) -> RedirectResponse:
        """Run every fetched-but-unprocessed case through the pipeline.

        Kept separate from the fetch button on purpose. Fetching is one cheap
        API call and finishes instantly; processing is several vision calls
        per case and takes ~10s each. Two buttons means the queue visibly
        fills from the real API first, and the pipeline is then a distinct,
        narratable step rather than both disappearing into one opaque wait.

        The work itself is handed to a background task so the browser gets
        its redirect immediately; the queue page polls itself while
        `demo_run.running`, so cases land one by one as each completes.

        Refuses to start a second run while one is in flight, so an impatient
        double-click can't process every case twice concurrently.
        """
        _require_demo_controls()
        _require_api_key()
        state: DemoRunState = request.app.state.demo_run
        if state.running:
            return RedirectResponse(url="/cases", status_code=303)

        db_path = _db_path(request)
        case_ids = sorted(
            row["case_id"]
            for row in store.list_cases_by_status(CaseState.INTAKE, db_path=db_path)
            if row["case_id"] not in BULK_PROCESS_HOLDBACK
        )
        if not case_ids:
            return RedirectResponse(url="/cases", status_code=303)

        state.reset(len(case_ids))
        background_tasks.add_task(_process_all_intake, request.app, case_ids)
        return RedirectResponse(url="/cases", status_code=303)

    @protected.post("/cases/{case_id}/process")
    async def process_case_now(request: Request, case_id: str) -> RedirectResponse:
        """Run one fetched case through the full pipeline, on demand.

        Synchronous on purpose: the rep clicked it and the resulting page
        *is* the completion signal. A background task would need polling
        infrastructure to say anything useful, which is a lot of machinery
        for a demo affordance -- and a request that takes ~10s and then
        shows the finished recommendation is easier to narrate than a
        spinner that resolves out of band.

        `process_case` is idempotent enough to re-run (it drives its own
        transitions from whatever state the row is in), but re-processing a
        case a rep has already acted on would overwrite that decision, so
        this refuses anything past the review stage.
        """
        _require_demo_controls()
        _require_api_key()
        db_path = _db_path(request)
        client = _resolve_client(request)

        row = _case_row_or_404(case_id, db_path=db_path)
        if row["status"] != CaseState.INTAKE.value:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"case {case_id!r} is already {row['status']} -- only a freshly fetched "
                    f"(intake) case can be processed from here"
                ),
            )

        await process_case(
            case_id, client=client, transport=request.app.state.transport, db_path=db_path
        )
        return RedirectResponse(url=f"/cases/{case_id}", status_code=303)

    @protected.get("/cases/{case_id}", response_class=HTMLResponse)
    async def case_detail(request: Request, case_id: str) -> HTMLResponse:
        db_path = _db_path(request)
        client = _resolve_client(request)

        row = _case_row_or_404(case_id, db_path=db_path)
        recommendation = store.get_recommendation(case_id, db_path=db_path)

        try:
            case = await client.get_case(case_id)
        except NotFoundError:
            raise HTTPException(status_code=404, detail=f"case {case_id!r} not found upstream") from None

        try:
            attachments = await client.list_attachments(case_id)
        except NotFoundError:
            attachments = []

        audit_events = [
            {
                "actor": r["actor"],
                "event": r["event"],
                "ts": r["ts"],
                "payload": json.loads(r["payload_json"]) if r["payload_json"] else None,
            }
            for r in store.get_audit_log(case_id, db_path=db_path)
        ]

        escalation_reason = (
            _escalation_summary(audit_events) if row["status"] == CaseState.ESCALATED.value else None
        )

        # Retail-invoice reconciliation panel. Rendered whenever the case got
        # far enough for the audit to run at all -- including when it came
        # back unverified -- so "the amount was never independently checked"
        # is visible to the rep rather than looking identical to "checked and
        # clean" (see `gates/invoice_audit.py` module docstring point 1).
        invoice_audit = store.load_gate_results(case_id, db_path=db_path).invoice_audit

        # Merchant memory panel (closes the gap noted in docs/ARCHITECTURE.md
        # section 4 / this template's former static disclaimer): the same
        # `memory.merchant_context()` the drafter's prompt actually sees, plus
        # a policy-notes table pre-filtered to this case's own merchant (+
        # global) instead of the queue page's full unfiltered list. `None`/
        # empty when the case has no `user_id` -- mirrors
        # `pipeline.py`/`web/app.py`'s existing `NO_MERCHANT_ID_MEMORY_CONTEXT`
        # fallback rather than inventing a second message for the same case.
        memory_context = None
        case_policy_notes: list[memory.PolicyNote] = []
        if case.user_id is not None:
            memory_context = memory.merchant_context(
                case.user_id, exclude_case_id=case_id, db_path=db_path
            )
            case_policy_notes = [
                note
                for note in memory.list_policy_notes(db_path=db_path)
                if note.scope == "global" or note.merchant_id == case.user_id
            ]

        # Lightbox data for the Evidence section's enlarged/prev-next viewer
        # (case.html's inline script). Built server-side and JSON-embedded
        # rather than relying on a Jinja `tojson` filter (not registered on
        # this plain `Jinja2Templates` environment, unlike Flask's) --
        # `</script` is defensively escaped so an attachment filename
        # containing that literal substring can never break out of the
        # inline <script> tag it's embedded in.
        attachments_json = json.dumps(
            [
                {"url": f"/cases/{case_id}/attachments/{att.attachment_id}", "name": att.file_name}
                for att in attachments
            ]
        ).replace("</script", "<\\/script")

        context = {
            "case": case,
            "case_id": case_id,
            "status": row["status"],
            "recommendation": recommendation,
            "attachments": attachments,
            "attachments_json": attachments_json,
            "audit_events": audit_events,
            "escalation_reason": escalation_reason,
            "memory_context": memory_context,
            "case_policy_notes": case_policy_notes,
            "no_merchant_id_text": memory.NO_MERCHANT_ID_MEMORY_CONTEXT,
            "invoice_audit": invoice_audit,
            "expected_currency": settings.expected_currency,
            # Email draft preview header (UI-polish audit finding): the same
            # recipient/subject `approve()` actually sends with, shown above
            # the draft textarea so a rep can see who/what before hitting
            # send -- not just the bare body.
            "email_to": case.contact_email,
            "email_subject": _email_subject_for(case_id),
        }
        return templates.TemplateResponse(request, "case.html", context)

    @protected.get("/cases/{case_id}/attachments/{attachment_id}")
    async def attachment_bytes(request: Request, case_id: str, attachment_id: str) -> Response:
        client = _resolve_client(request)
        try:
            attachments = await client.list_attachments(case_id)
        except NotFoundError:
            raise HTTPException(status_code=404, detail=f"case {case_id!r} not found") from None

        attachment = next((a for a in attachments if a.attachment_id == attachment_id), None)
        if attachment is None:
            raise HTTPException(status_code=404, detail=f"attachment {attachment_id!r} not found")

        content = await client.get_attachment_bytes(attachment)
        return Response(content=content, media_type=attachment.content_type or "application/octet-stream")

    @protected.post("/cases/{case_id}/approve")
    async def approve(
        request: Request,
        case_id: str,
        action: str = Form("approve"),
        edited_body: str = Form(""),
    ) -> RedirectResponse:
        db_path = _db_path(request)
        client = _resolve_client(request)

        row = _case_row_or_404(case_id, db_path=db_path)
        _require_reviewable(row)

        recommendation = store.get_recommendation(case_id, db_path=db_path)
        if recommendation is None:
            raise HTTPException(status_code=400, detail=f"case {case_id!r} has no recommendation to approve")

        try:
            case = await client.get_case(case_id)
        except NotFoundError:
            raise HTTPException(status_code=404, detail=f"case {case_id!r} not found upstream") from None

        if not case.contact_email:
            # Nowhere safe to send -- basic hygiene ahead of the guard's own
            # RECIPIENT_MISMATCH check below (which would otherwise report
            # "email.to != contact_email" for a `None` contact_email as a
            # generic guard violation rather than this clearer 422).
            raise HTTPException(status_code=422, detail=f"case {case_id!r} has no contact_email on file")

        original_body = recommendation.email_draft
        use_edited = action == "save_and_approve"
        # An explicit "save edits" submit with an emptied-out textarea falls
        # back to the original draft (and logs no correction) rather than
        # sending a blank email -- safer default, but worth calling out:
        # this is currently silent, undocumented-to-the-rep behavior.
        body_to_send = edited_body if use_edited and edited_body else original_body
        edited = use_edited and body_to_send != original_body

        subject = _email_subject_for(case_id)

        # The outbound guard runs here, BEFORE record_action --
        # not merely before send_email. If it sat between record_action and
        # send_email instead, a blocked send would still leave the "email"
        # action permanently recorded, and the case could never be sent even
        # after the rep fixes the draft -- record_action must stay the last
        # gate before the actual send.
        #
        # `gate_results`/`calc_for_guard` are what the guard cross-checks
        # against each other and against a fresh invoice re-derivation (see
        # `guard.py` module docstring point 1): `calc_for_guard` is built
        # from the persisted `Recommendation` -- the exact amount/line-items
        # the email/`submit_reimbursement` loop below actually use -- not
        # from `gate_results.calc` directly, so a corrupted `recommendation_
        # json` row doesn't silently "check itself" and pass.
        current_status = CaseState(row["status"])
        gate_results = store.load_gate_results(case_id, db_path=db_path)
        calc_for_guard = CalcResult(
            amount=recommendation.amount,
            line_items=list(recommendation.line_items),
            # `capped` is cosmetic (no guard invariant reads it) -- best
            # effort from the stored gate result, `False` if unavailable.
            capped=gate_results.calc.capped if gate_results.calc is not None else False,
        )

        invoice = None
        if recommendation.decision == "approve":
            # Only an approve decision has an approved amount to reconcile
            # against an invoice; deny/request_info recommendations carry no
            # calc, so there's nothing for the guard's amount-re-derivation/
            # SKU checks to compare against on those paths (both skipped
            # gracefully when `invoice is None` there). For an approve
            # decision, by contrast, a failed fetch is deliberately passed
            # through as `invoice=None` rather than swallowed -- the guard's
            # amount check is fail-closed on a missing invoice for
            # `decision == "approve"` (an `AMOUNT_MISMATCH` violation, not a
            # silently-disabled check), so this `except` must not paper over
            # the failure by pretending the check ran and passed.
            try:
                invoice = await client.generate_invoice(shipment_id=case.shipment_id, user_id=case.user_id)
            except NotFoundError:
                invoice = None

        # Mirrors the exact intermediate-hop logic further below (module
        # docstring point 7): a PENDING_REVIEW case's *next* legal hop is
        # the decision-mapped intermediate state, not SENT directly.
        intended_state = (
            CaseState.SENT
            if current_status != CaseState.PENDING_REVIEW
            else _DECISION_TO_INTERMEDIATE_STATE[recommendation.decision]
        )

        violations = check_outbound(
            case,
            gate_results,
            calc_for_guard,
            EmailToSend(to=case.contact_email, subject=subject, body=body_to_send),
            decision=recommendation.decision,
            invoice=invoice,
            current_status=current_status,
            intended_state=intended_state,
        )
        if violations:
            violation_payload = [{"invariant": v.invariant, "detail": v.detail} for v in violations]
            # ESCALATED has no legal self-loop (`store.LEGAL_TRANSITIONS`),
            # so an already-escalated case that fails the guard gets an
            # audit-only `log_event`, not a `transition()` call -- same
            # "doesn't represent a real state change" reasoning `store.py`
            # documents for pushback. A PENDING_REVIEW case, by contrast,
            # legally escalates.
            if current_status == CaseState.PENDING_REVIEW:
                store.transition(
                    case_id,
                    CaseState.ESCALATED,
                    actor="system",
                    event="outbound_guard_blocked",
                    payload={"violations": violation_payload},
                    db_path=db_path,
                )
            else:
                store.log_event(
                    case_id,
                    actor="system",
                    event="outbound_guard_blocked",
                    payload={"violations": violation_payload},
                    db_path=db_path,
                )
            raise HTTPException(
                status_code=422,
                detail={
                    "case_id": case_id,
                    "message": "Outbound guard blocked this send -- case escalated for human review.",
                    "violations": violation_payload,
                },
            )

        try:
            store.record_action(
                case_id,
                "email",
                {"to": case.contact_email, "subject": subject, "body": body_to_send, "edited": edited},
                db_path=db_path,
            )
        except DuplicateActionError:
            raise HTTPException(
                status_code=409, detail=f"case {case_id!r} has already had its email sent"
            ) from None

        if edited:
            # Written only once `record_action` has actually claimed the
            # send -- a duplicate/blocked attempt must never leave a
            # correction row behind for an email that was never sent.
            # `audit_log` (general, always-present accountability trail)
            # and `memory` (purpose-built source for the feedback
            # distiller) both get a row -- see `store.log_event`'s
            # docstring for why both are kept.
            store.log_event(
                case_id,
                actor="rep",
                event="correction_recorded",
                payload={"original_email_draft": original_body, "edited_email_draft": body_to_send},
                db_path=db_path,
            )
            if case.user_id is not None:
                # Guarded per module docstring point 11: `record_correction`
                # raises on a missing `user_id`, and this sits between
                # `record_action` and `send_email` -- a missing merchant
                # identifier must degrade gracefully (skip the memory row),
                # never 500 and strand the case mid-send.
                memory.record_correction(
                    case,
                    original_draft=original_body,
                    final_draft=body_to_send,
                    db_path=db_path,
                )

        # Recipient is ALWAYS case.contact_email fetched fresh from the
        # client -- never edited_body/user-supplied form data -- injection
        # hygiene independently re-verified above by the guard's
        # RECIPIENT_MISMATCH check (`_check_recipient` in `guard.py`).
        await client.send_email(case_id, to=case.contact_email, subject=subject, body=body_to_send)

        if recommendation.decision == "approve":
            try:
                store.record_action(
                    case_id,
                    "reimbursement",
                    {
                        "line_items": [
                            {
                                "sku": li.sku,
                                "quantity": li.quantity,
                                "unit_price": str(li.unit_price),
                                "subtotal": str(li.subtotal),
                            }
                            for li in recommendation.line_items
                        ],
                        "total_amount": str(recommendation.amount),
                    },
                    db_path=db_path,
                )
            except DuplicateActionError:
                pass  # already submitted in a previous (idempotent) attempt
            else:
                product_names = await _sku_product_names(client, case)
                for line_item in recommendation.line_items:
                    product_name = product_names.get(line_item.sku, line_item.sku)
                    await client.submit_reimbursement(
                        case_id,
                        order_id=case.order_id or "",
                        user_id=case.user_id or "",
                        shipment_id=case.shipment_id or "",
                        product_name=product_name,
                        amount=line_item.subtotal,
                    )

        if current_status == CaseState.PENDING_REVIEW:
            # See module docstring point 7 for why this hop exists.
            # `intended_state` is the exact same value the guard's
            # `ILLEGAL_STATE` check above just validated -- reused here
            # (rather than recomputed) so the transition actually performed
            # can never drift from what was checked.
            store.transition(
                case_id,
                intended_state,
                actor="rep",
                event=f"transition:pending_review->{intended_state.value}",
                payload={"decision": recommendation.decision},
                db_path=db_path,
            )

        store.transition(
            case_id,
            CaseState.SENT,
            actor="rep",
            event="approved",
            payload={"to": case.contact_email, "edited": edited},
            db_path=db_path,
        )

        if edited:
            # Deliberately the LAST thing this handler does -- see module
            # docstring point 12 for why this sits here (end of handler,
            # after everything outbound has already completed) rather than
            # right next to the `record_correction` call above. Best-effort,
            # non-blocking: a distiller failure must never turn an
            # already-successful send into a failed request.
            try:
                await evolve.distill_feedback(
                    case,
                    original_draft=original_body,
                    final_draft=body_to_send,
                    db_path=db_path,
                    transport=request.app.state.transport,
                )
            except Exception:
                logging.getLogger(__name__).warning(
                    "distill_feedback failed for case %s during approve; continuing without "
                    "a distilled note",
                    case_id,
                    exc_info=True,
                )

        return RedirectResponse(url=f"/cases/{case_id}", status_code=303)

    @protected.post("/cases/{case_id}/pushback")
    async def pushback(
        request: Request,
        case_id: str,
        feedback: str = Form(...),
    ) -> RedirectResponse:
        db_path = _db_path(request)
        transport = request.app.state.transport
        client = _resolve_client(request)

        row = _case_row_or_404(case_id, db_path=db_path)
        _require_reviewable(row)

        recommendation = store.get_recommendation(case_id, db_path=db_path)
        if recommendation is None:
            raise HTTPException(
                status_code=400, detail=f"case {case_id!r} has no recommendation to push back on"
            )

        feedback = feedback.strip()
        if not feedback:
            raise HTTPException(status_code=422, detail="feedback text is required")

        # Written regardless of what happens below -- this is the
        # rep-feedback audit trail the feedback distiller will mine,
        # independent of whether the redraft below succeeds.
        store.log_event(
            case_id,
            actor="rep",
            event="pushback",
            payload={
                "feedback": feedback,
                "previous_decision": recommendation.decision,
                "previous_amount": str(recommendation.amount),
            },
            db_path=db_path,
        )

        try:
            case = await client.get_case(case_id)
        except NotFoundError:
            raise HTTPException(status_code=404, detail=f"case {case_id!r} not found upstream") from None

        # Composition (module docstring point 9): durable merchant/policy
        # memory first, then the rep's immediate feedback on THIS draft as
        # its own clearly labeled section -- not folded into the merchant
        # memory text, since "fix this now" and "always do X for this
        # merchant" are different kinds of information the drafter needs to
        # tell apart. Same `merchant_context()` call `pipeline.py` makes;
        # the immediate-feedback section is pushback-specific.
        if case.user_id is not None:
            merchant_memory_text = memory.merchant_context(case.user_id, db_path=db_path).to_prompt_text()
        else:
            merchant_memory_text = memory.NO_MERCHANT_ID_MEMORY_CONTEXT
        combined_memory_context = (
            f"{merchant_memory_text}\n\n"
            "--- Immediate rep feedback on the draft just reviewed (address "
            "this now; not necessarily a durable fact about the merchant) ---\n"
            f"{feedback}"
        )

        # See module docstring point 9 for the full discussion: this pulls
        # the real persisted gate results rather than reconstructing a
        # lossy subset from the `Recommendation` alone.
        gate_results = store.load_gate_results(case_id, db_path=db_path)
        validation_decision = (
            combine_validation(gate_results.validation) if gate_results.validation is not None else None
        )
        calc_result = gate_results.calc if gate_results.calc is not None else _pushback_calc_result(recommendation)
        draft_inputs = DraftInputs(
            case=case,
            decision=recommendation.decision,
            amount=recommendation.amount,
            confidence=recommendation.confidence,
            risk_assessment=RiskAssessment(tier=RiskTier(recommendation.risk_tier), flags=[]),
            eligibility_result=gate_results.eligibility,
            evidence_gaps=gate_results.evidence_gaps,
            validation_decision=validation_decision,
            calc_result=calc_result,
            # Same reason the rest of the gate context is reconstructed here:
            # a redraft that lost the invoice-reconciliation finding would
            # regress to inventing an evidence request for a case whose
            # evidence was never the problem.
            invoice_audit=gate_results.invoice_audit,
            memory_context=combined_memory_context,
        )

        new_recommendation = await draft(draft_inputs, transport=transport, db_path=db_path)
        store.save_recommendation(case_id, new_recommendation, db_path=db_path)

        if case.user_id is not None:
            # Guarded the same way as the approve endpoint's correction
            # write (module docstring point 11) -- nothing has been sent
            # yet on the pushback path, so a missing `user_id` here is
            # lower-stakes than in approve, but still shouldn't 500 a
            # request that otherwise succeeded.
            memory.record_correction(
                case,
                original_draft=recommendation.email_draft,
                final_draft=new_recommendation.email_draft,
                feedback=feedback,
                db_path=db_path,
            )

        # Right alongside `record_correction` above (module docstring point
        # 12) -- nothing outbound has been claimed on the pushback path, so
        # there's no crash-window concern here the way there is in approve.
        # Still best-effort/non-blocking: a distiller failure must not turn
        # an otherwise-successful redraft into a failed request.
        try:
            await evolve.distill_feedback(
                case,
                original_draft=recommendation.email_draft,
                final_draft=new_recommendation.email_draft,
                feedback=feedback,
                db_path=db_path,
                transport=transport,
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "distill_feedback failed for case %s during pushback; continuing without a "
                "distilled note",
                case_id,
                exc_info=True,
            )

        # No transition() call -- see store.py's LEGAL_TRANSITIONS comment
        # and module docstring point 8: pushback doesn't advance the case,
        # it stays in the same review queue with a fresher draft.
        return RedirectResponse(url=f"/cases/{case_id}", status_code=303)

    app.include_router(protected)
    return app


# Real-usage entry point (e.g. `uvicorn claimpilot.web.app:app`) -- defaults
# to the real client/transport singletons and the on-disk database.
app = create_app()
