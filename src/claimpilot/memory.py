"""Memory store.

Gives `claimpilot` a real, queryable record of
merchant-specific history (notes, durable policy guidance, and rep
corrections to drafts) plus global policy guidance that applies to every
merchant, feeding two things that would otherwise be empty placeholders:

- `claimpilot.risk.MerchantMemory()` -- `pipeline.py` builds it from this
  module's `merchant_context()`.
- `claimpilot.draft.DraftInputs.memory_context` -- `pipeline.py` and
  `claimpilot.web.app` populate it with `MemoryContext.to_prompt_text()`.

This module owns the `memory` table's *business* API (mirroring how
`claimpilot.store` owns `cases`/`audit_log`/`actions`'s business API on top
of `claimpilot.db`'s raw schema layer). Every public function here is
self-contained: opens its own connection via `claimpilot.db.get_connection`,
calls `ensure_schema()`, does its work, closes the connection -- same
convention as every function in `claimpilot.store`. `db_path` defaults to
`settings.db_path`; tests pass a `tmp_path` file.

Design decisions (judgment calls made explicitly, not left implicit):

1. **`merchant_id` is `Case.user_id`, never `Case.account_name`.**
   `user_id` is the stable ShipBob account identifier already used
   elsewhere as the merchant key (passed to `generate_invoice`/
   `submit_reimbursement`); `account_name` is a human-readable display
   name ("Best Paw Nutrition") with no uniqueness/stability guarantee.
   `claimpilot.db.ensure_cases_table` documents the same choice for
   `cases.merchant_id`.
2. **`MemoryContext.policy_notes` is merchant-scoped ONLY -- global policies
   are a separate field, `global_policy_notes`, and are never folded into
   `policy_notes`.** This matters for two different consumers of
   `MemoryContext`:
     - `claimpilot.pipeline.process_case` maps `policy_notes` (and only
       `policy_notes`) onto `MerchantMemory.flags` for risk tiering (see
       that module for the full mapping rationale). A global policy note
       (e.g. "always mention warranty terms for mug SKUs") applies to every
       merchant equally -- it is not evidence that *this* merchant is
       riskier than any other, and folding it into `policy_notes` would
       make every single case processed after the first global policy note
       exists read as elevated risk, which defeats the entire point of
       risk *tiering*.
     - `to_prompt_text()` (for the drafter) renders both merchant-scoped
       and global policy notes, just under separate headings, so the LLM
       still sees the full guidance context. So: `policy_notes` feeds risk
       flags AND prompt text; `global_policy_notes` feeds prompt text only.
3. **`recent_notes` (kind IN `"note"`, `"correction"`) never feed risk
   flags at all**, merchant-scoped or not. These are raw, per-case,
   automatically-written records (`record_correction` runs on every rep
   edit/pushback) -- noisy and un-curated by design. `flags` in
   `claimpilot.risk` is meant to carry *curated* signals a past rep (or
   the feedback distiller) has deliberately flagged as durable and
   worth a reviewer's attention; that's exactly what `kind="policy"` rows
   are for; raw corrections are not. Using raw corrections as flags would
   make claim tier climb to ELEVATED for any merchant with so much as one
   prior rep edit, which every active merchant will eventually have.
4. **`record_correction`'s `content` column is one JSON object**
   (`{"original_draft", "final_draft", "feedback"}`, `feedback` nullable),
   not three separate columns -- keeps the `memory` table's shape uniform
   across all three `kind`s (`content` is always a single TEXT column) while
   still giving the feedback distiller structured data to parse back
   out. All three values are plain strings (never `Decimal`/float), so
   there is no precision-loss concern here the way there is for
   `store.save_recommendation`'s `Decimal` fields.
5. **Read-time cap on `recent_notes`, distinct from the write-time
   cap.** `merchant_context()` returns at most `settings.max_recent_notes` (5) most
   recent note/correction rows, ordered by `id DESC` (insertion order is a
   reliable recency proxy here since rows are only ever appended, never
   reordered, and `id` is monotonic within a single SQLite file). This is
   purely to keep the drafter's prompt bounded; a separate mechanism
   enforces a hard per-scope cap of 10 stored rows at *write* time (pruning
   old rows), which is a different mechanism for a different purpose and
   is explicitly out of scope here.
6. **`claims_last_90_days` is a bounded window (`now - 90d <= created_at <=
   now`), not just a lower bound, and supports excluding one case_id.**
   Two reasons an upper bound matters, not just the lower one the task
   describes: (a) `claimpilot.store`'s `_now()` always uses the real wall
   clock for `cases.created_at`/`updated_at`, independent of any `now`
   caller code injects for testability (see `pipeline.process_case`'s own
   `now` parameter, used only for eligibility date math) -- without an
   upper bound, a fixed/historical test `now` would still count every row
   actually written *during* the test run, since their real created_at is
   chronologically after the test's `now`. (b) In real usage, `pipeline.
   process_case` creates the case's own `cases` row (via `store.
   create_case`) moments before it computes risk tiering from
   `merchant_context()` -- without excluding that case's own `case_id`
   (`exclude_case_id`), a merchant's very first-ever claim would already
   report `claims_last_90_days=1`, not `0`, because the claim being scored
   would count itself as history. Note this window is a literal timestamp
   comparison (carries `now`'s time-of-day), NOT a calendar-day window
   like `claimpilot.gates.eligibility`'s claim-window check -- a
   deliberate difference (this is "trailing 90 days of wall-clock time",
   not "the last 90 calendar dates"), called out here since it's easy to
   assume the two work the same way.
7. **`delete_note` is a plain, non-raising `DELETE ... WHERE id = ?`.** A
   delete of an already-gone/never-existed id is treated as a no-op
   (matches ordinary SQL `DELETE` semantics) rather than raising -- the
   memory review panel is the only planned caller, and "the note is gone"
   is true either way.
8. **The write-time cap (`settings.policy_note_cap_per_partition` = 10) is
   partitioned per `(scope, merchant_id)`, not a single global cap.**
   Concretely, there are as many independent partitions as there are
   distinct `(scope, merchant_id)` pairs ever written: one partition for
   `scope="global"` (`merchant_id IS NULL`), and one *separate* partition
   per distinct merchant under `scope="merchant"`. Each partition is
   capped at 10 rows independently. Two failure modes a single shared cap
   of 10 would create, which this partitioning avoids:
     - A single chatty merchant writing enough policy notes to hit a
       shared cap would start evicting *every other merchant's* notes,
       even merchants who have only ever had one or two notes recorded --
       one noisy account effectively erases everyone else's memory.
     - Merchant-scoped notes and the `scope="global"` partition would
       compete for the same 10 slots, so enough merchant activity could
       evict a `scope="global"` note (guidance meant to apply to *every*
       merchant) even though it has nothing to do with that merchant.
   Eviction within a partition, once it exceeds 10 rows, deletes the
   oldest rows (ordered by `created_at` ascending, ties broken by lowest
   `id` -- a plain deterministic tiebreaker for ordering, independent of
   whatever timestamp resolution `_now_iso()` happens to have) until
   exactly 10 remain. This happens
   inside `record_policy_note` itself (write time), immediately after the
   insert, in the same connection/transaction -- never in the UI layer
   (`web/app.py`)/template, per the task's explicit instruction that
   enforcement lives in this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from claimpilot.config import settings
from claimpilot.db import ensure_schema, get_connection
from claimpilot.models import Case

Scope = Literal["global", "merchant"]
Kind = Literal["note", "policy", "correction"]

# Read-time cap on how many recent note/correction rows `merchant_context()`
# surfaces to the drafter (module docstring point 5) -- distinct from Task
# 5.4's write-time cap of 10 rows per scope. Now `settings.max_recent_notes`
# (`config.py`), read at call time in `merchant_context()` below -- moved
# off this module as a bare constant so a deployment can tune it without a
# code change, same as every other business-policy value in this codebase.

# Trailing window `merchant_context()` uses for `claims_last_90_days`. Now
# `settings.claim_frequency_window_days` (`config.py`), read at call time
# below. NOTE: `MemoryContext.claim_frequency_90d` and `risk.MerchantMemory.
# claims_last_90_days`'s *field names* still literally say "90" -- overriding
# the setting away from 90 doesn't rename those identifiers, only changes
# the actual math and the rendered "last N days" prompt text (both correctly
# follow the setting's live value; see `config.py`'s own comment on this
# field for the full caveat).

# The write-time cap (max `kind="policy"` rows retained per
# `(scope, merchant_id)` partition -- see module docstring point 8 for why
# the cap is partitioned this way rather than applied globally) is
# `settings.policy_note_cap_per_partition` (`config.py`), read at call time
# in `record_policy_note` below -- moved off this module as a bare constant
# so a deployment can tune it without a code change.

# Shared fallback text for `DraftInputs.memory_context` when a case has no
# `user_id` to look merchant memory up by (shouldn't happen for real ShipBob
# data, but `Case.user_id` is optional) -- used by both `pipeline.py` and
# `web/app.py` so the two call sites never drift. Deliberately distinct from
# `MemoryContext().to_prompt_text()`'s "Claim frequency: 0" framing, which
# would misleadingly read as "confirmed zero prior claims" rather than
# "unknown merchant, can't compute history."
NO_MERCHANT_ID_MEMORY_CONTEXT = (
    "No merchant identifier available for this case -- merchant claim "
    "history/policy notes could not be looked up."
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_now(now: datetime | None) -> datetime:
    """Default to the real wall clock; treat a naive `now` as UTC rather
    than raising, matching this codebase's general tolerance for naive
    datetimes at trust boundaries (see `gates/eligibility.py`'s `_parse_date`
    for the same normalize-rather-than-reject convention).
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now


@dataclass(frozen=True)
class MemoryContext:
    """Bundled merchant memory, as returned by `merchant_context()`.

    See module docstring points 2-3 for exactly which fields feed
    `claimpilot.risk.MerchantMemory.flags` (only `policy_notes`) versus
    which are prompt-only context for the drafter (all three, plus
    `global_policy_notes`).
    """

    recent_notes: list[str] = field(default_factory=list)
    claim_frequency_90d: int = 0
    # Merchant-scoped kind="policy" rows only (module docstring point 2).
    policy_notes: list[str] = field(default_factory=list)
    # kind="policy" rows with scope="global" -- informational context for
    # the drafter only, deliberately never mixed into `policy_notes` (never
    # used as a risk-tiering flag; see module docstring point 2).
    global_policy_notes: list[str] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Render this context as readable text for `draft.py`'s
        `DraftInputs.memory_context` field.
        """
        lines = [
            f"Claim frequency (last {settings.claim_frequency_window_days} days): "
            f"{self.claim_frequency_90d}"
        ]

        lines.append("Merchant-specific policy notes:")
        if self.policy_notes:
            lines.extend(f"  - {note}" for note in self.policy_notes)
        else:
            lines.append("  (none)")

        lines.append("Global policy notes (apply to all merchants):")
        if self.global_policy_notes:
            lines.extend(f"  - {note}" for note in self.global_policy_notes)
        else:
            lines.append("  (none)")

        lines.append("Recent merchant notes/corrections:")
        if self.recent_notes:
            lines.extend(f"  - {note}" for note in self.recent_notes)
        else:
            lines.append("  (none)")

        return "\n".join(lines)


def _render_recent_row(kind: str, content: str) -> str:
    """Render one `kind IN ("note", "correction")` row's `content` as a
    single readable line for `MemoryContext.recent_notes`.

    `kind="note"` content is already plain text -- returned verbatim.
    `kind="correction"` content is the JSON object `record_correction`
    writes (module docstring point 4); rendered as a short human-readable
    summary rather than dumping raw JSON into the prompt. Malformed JSON
    (should not happen, since this module is the only writer, but defensive
    against a hand-edited row) falls back to the raw string rather than
    raising -- a garbled memory row must never crash drafting.
    """
    if kind != "correction":
        return content
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content
    feedback = data.get("feedback")
    if feedback:
        return f"Rep correction with feedback: {feedback}"
    return "Rep edited the drafted email before sending (no separate feedback text given)."


def record_correction(
    case: Case,
    original_draft: str,
    final_draft: str,
    feedback: str | None = None,
    *,
    db_path: Path | str | None = None,
) -> None:
    """Record one merchant-scoped `kind="correction"` memory row for `case`.

    Called from `claimpilot.web.app`'s approve endpoint (rep edits the
    drafted email before sending) and pushback endpoint (rep pushes back
    with feedback and gets a redraft) -- both alongside, not instead of,
    their existing `claimpilot.store.log_event` audit-trail writes (see
    `store.log_event`'s docstring for why both are kept).

    `content` is a single JSON object `{"original_draft", "final_draft",
    "feedback"}` (module docstring point 4) -- the feedback
    distiller is expected to `json.loads()` this back out.

    Raises:
        ValueError: `case.user_id` is `None` -- there is no merchant to
            attribute a merchant-scoped row to. Callers (web/app.py) should
            guard this at the call site rather than relying on this
            exception, exactly the way `web/app.py`'s `_sku_product_names`
            already treats a missing lookup as "a cosmetic downgrade, not a
            reason to fail the whole approve flow" -- letting this raise
            uncaught mid-approve (after `record_action` has already claimed
            the send, before `send_email` actually runs) would strand a
            case in a half-sent state. This function itself still raises
            rather than silently no-op-ing, since a merchant-scoped write
            with no merchant is a genuine caller bug, not an expected case.
    """
    if case.user_id is None:
        raise ValueError(f"case {case.case_id!r} has no user_id -- cannot record a merchant-scoped correction")

    content = json.dumps({"original_draft": original_draft, "final_draft": final_draft, "feedback": feedback})

    conn = get_connection(db_path)
    try:
        ensure_schema(conn)
        conn.execute(
            "INSERT INTO memory (scope, merchant_id, kind, content, source_case_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("merchant", case.user_id, "correction", content, case.case_id, _now_iso()),
        )
        conn.commit()
    finally:
        conn.close()


def merchant_context(
    merchant_id: str,
    *,
    exclude_case_id: str | None = None,
    now: datetime | None = None,
    db_path: Path | str | None = None,
) -> MemoryContext:
    """Return `merchant_id`'s recent notes/corrections, policy notes (both
    merchant-scoped and global), and trailing-90-day claim frequency.

    `now` overrides "today" for the 90-day window (defaults to the real
    wall clock) -- accepted for the same testability reason as
    `gates.eligibility.check_eligibility`'s `now` parameter. `exclude_case_id`
    excludes one `case_id` from the claim-frequency count; `pipeline.
    process_case` passes the case currently being scored so it never counts
    itself as prior history (module docstring point 6).
    """
    now = _normalize_now(now)
    window_start = (now - timedelta(days=settings.claim_frequency_window_days)).isoformat()
    now_iso = now.isoformat()

    conn = get_connection(db_path)
    try:
        ensure_schema(conn)

        recent_rows = conn.execute(
            "SELECT kind, content FROM memory "
            "WHERE scope = 'merchant' AND merchant_id = ? AND kind IN ('note', 'correction') "
            "ORDER BY id DESC LIMIT ?",
            (merchant_id, settings.max_recent_notes),
        ).fetchall()
        recent_notes = [_render_recent_row(row["kind"], row["content"]) for row in recent_rows]

        policy_rows = conn.execute(
            "SELECT content FROM memory WHERE scope = 'merchant' AND merchant_id = ? AND kind = 'policy' "
            "ORDER BY id DESC",
            (merchant_id,),
        ).fetchall()
        policy_notes = [row["content"] for row in policy_rows]

        global_rows = conn.execute(
            "SELECT content FROM memory WHERE scope = 'global' AND kind = 'policy' ORDER BY id DESC",
        ).fetchall()
        global_policy_notes = [row["content"] for row in global_rows]

        freq_sql = (
            "SELECT COUNT(DISTINCT case_id) AS n FROM cases "
            "WHERE merchant_id = ? AND created_at >= ? AND created_at <= ?"
        )
        freq_params: list[str] = [merchant_id, window_start, now_iso]
        if exclude_case_id is not None:
            freq_sql += " AND case_id != ?"
            freq_params.append(exclude_case_id)
        freq_row = conn.execute(freq_sql, freq_params).fetchone()
        claim_frequency_90d = freq_row["n"] if freq_row is not None else 0
    finally:
        conn.close()

    return MemoryContext(
        recent_notes=recent_notes,
        claim_frequency_90d=claim_frequency_90d,
        policy_notes=policy_notes,
        global_policy_notes=global_policy_notes,
    )


def global_policies(*, db_path: Path | str | None = None) -> list[str]:
    """Return every `scope="global"` policy note's content, most recent first."""
    conn = get_connection(db_path)
    try:
        ensure_schema(conn)
        rows = conn.execute(
            "SELECT content FROM memory WHERE scope = 'global' AND kind = 'policy' ORDER BY id DESC"
        ).fetchall()
        return [row["content"] for row in rows]
    finally:
        conn.close()


def record_policy_note(
    content: str,
    *,
    scope: Scope,
    merchant_id: str | None = None,
    source_case_id: str | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Write one `kind="policy"` memory row, then enforce the
    per-partition cap (`settings.policy_note_cap_per_partition` = 10, partitioned by
    `(scope, merchant_id)` -- module docstring point 8) by evicting the
    oldest rows in that same partition until at most 10 remain.

    Used by the feedback distiller (`evolve.py`) and the
    memory review panel's callers alike -- eviction runs on every call so
    no caller has to remember to enforce the cap itself.

    Raises:
        ValueError: `scope="merchant"` with no `merchant_id`, or
            `scope="global"` with a `merchant_id` given -- a global policy
            cannot be scoped to one merchant and vice versa.
    """
    if scope == "merchant" and merchant_id is None:
        raise ValueError('merchant_id is required when scope="merchant"')
    if scope == "global" and merchant_id is not None:
        raise ValueError('merchant_id must be None when scope="global"')

    conn = get_connection(db_path)
    try:
        ensure_schema(conn)
        conn.execute(
            "INSERT INTO memory (scope, merchant_id, kind, content, source_case_id, created_at) "
            "VALUES (?, ?, 'policy', ?, ?, ?)",
            (scope, merchant_id, content, source_case_id, _now_iso()),
        )
        # `merchant_id IS ?` (not `= ?`) so the `scope="global"` partition
        # (`merchant_id IS NULL`) matches correctly -- `NULL = NULL` is
        # `NULL`/falsy in SQL, not true, so a plain `= ?` would never match
        # any row in the global partition.
        excess_ids = conn.execute(
            "SELECT id FROM memory WHERE scope = ? AND merchant_id IS ? AND kind = 'policy' "
            "ORDER BY created_at ASC, id ASC",
            (scope, merchant_id),
        ).fetchall()
        overflow = len(excess_ids) - settings.policy_note_cap_per_partition
        if overflow > 0:
            ids_to_delete = [row["id"] for row in excess_ids[:overflow]]
            conn.executemany("DELETE FROM memory WHERE id = ?", [(i,) for i in ids_to_delete])
        conn.commit()
    finally:
        conn.close()


@dataclass(frozen=True)
class PolicyNote:
    """One `kind="policy"` row, with full provenance -- the memory
    review panel needs the `id` (to delete by) and `scope`/`merchant_id`/
    `source_case_id` (to display where a note came from), not just its text.
    """

    id: int
    scope: str
    merchant_id: str | None
    content: str
    source_case_id: str | None
    created_at: str


def list_policy_notes(*, db_path: Path | str | None = None) -> list[PolicyNote]:
    """Return every `kind="policy"` row (both scopes), most recent first.

    Added for the memory
    review panel, which needs to list all policy notes with provenance.
    """
    conn = get_connection(db_path)
    try:
        ensure_schema(conn)
        rows = conn.execute(
            "SELECT id, scope, merchant_id, content, source_case_id, created_at FROM memory "
            "WHERE kind = 'policy' ORDER BY id DESC"
        ).fetchall()
    finally:
        conn.close()
    return [
        PolicyNote(
            id=row["id"],
            scope=row["scope"],
            merchant_id=row["merchant_id"],
            content=row["content"],
            source_case_id=row["source_case_id"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


def delete_note(note_id: int, *, db_path: Path | str | None = None) -> None:
    """Delete one `memory` row by id (any `kind`).

    Added for the memory
    review panel. A no-op (not an error) if `note_id` doesn't exist --
    module docstring point 7.
    """
    conn = get_connection(db_path)
    try:
        ensure_schema(conn)
        conn.execute("DELETE FROM memory WHERE id = ?", (note_id,))
        conn.commit()
    finally:
        conn.close()
