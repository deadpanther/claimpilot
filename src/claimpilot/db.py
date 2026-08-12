"""SQLite connection + schema layer.

Owns every table this app persists to: `llm_calls` (every LLM call attempt,
for the case timeline + per-claim cost metric), `cases`/`audit_log`/
`actions` (case lifecycle + audit trail), and `memory` (merchant/global
policy notes and corrections). This module owns *schema*
(table definitions + `get_connection()`) only; the higher-level business API
-- creating cases, validating/recording state transitions, recording
idempotent actions -- lives in `claimpilot.store`, and the memory-specific
business API lives in `claimpilot.memory`, both built on top of the tables
defined here. Keep it that way: `db.py` stays a thin connection+schema
layer, `store.py`/`memory.py` are where policy lives.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from claimpilot.config import settings


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open a connection to the claimpilot SQLite database.

    `db_path` defaults to `settings.db_path` (the real on-disk database,
    read at call time -- not snapshotted into a module-level name -- so a
    `monkeypatch.setattr(settings, "db_path", ...)` override or a real env
    var change is honored immediately); tests pass an explicit path (e.g. a
    `tmp_path` fixture) so they never touch the real file or leak state
    between tests. Row access by column name (`row["case_id"]`) is enabled
    via `row_factory`.
    """
    conn = sqlite3.connect(db_path if db_path is not None else settings.db_path)
    conn.row_factory = sqlite3.Row
    # Found in a full-codebase audit: this app opens a fresh connection per
    # operation (never a long-lived shared one), and SQLite's *default*
    # journal mode (rollback journal, not WAL) takes a lock on the whole
    # file for the duration of a write -- meaning one rep's approve/pushback
    # can briefly block another rep's queue page load, not just another
    # write. WAL mode lets readers proceed concurrently with a writer.
    # Idempotent to set on every connect (a no-op once the file is already
    # in WAL mode) and safe for this app's deployment shape (a local file or
    # a Docker volume, never a network filesystem, where WAL is unreliable).
    # `busy_timeout` is a second, independent layer: if two writers still
    # genuinely collide, SQLite retries for up to 5s before raising
    # `OperationalError` ("database is locked") instead of failing instantly.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def ensure_llm_calls_table(conn: sqlite3.Connection) -> None:
    """Create the `llm_calls` table if it doesn't already exist.

    Columns:
      id            -- autoincrement primary key.
      case_id       -- which case this call was made on behalf of.
      prompt_name   -- logical prompt identifier (e.g. "classify_attachment"),
                       matching a `.md` file under `claimpilot/prompts/`.
      prompt_hash   -- sha256 hex digest of the prompt file's raw bytes, so a
                       row records exactly which prompt *content* (not just
                       which filename) produced the call.
      model         -- model ID string used for the call.
      latency_ms    -- wall-clock latency for this attempt, in milliseconds.
      input_tokens  -- input token count from the transport's usage data.
      output_tokens -- output token count from the transport's usage data.
      cost_usd      -- computed cost in USD, stored as TEXT (not REAL/NUMERIC)
                       to preserve exact `Decimal` precision -- SQLite has no
                       native decimal type, and a REAL column would round-trip
                       through floats and drift.
      raw_response  -- JSON-serialized raw response content blocks, for
                       later debugging/audit (exact shape is transport-
                       defined; see `claimpilot.llm.TransportResult.raw_content`).
      created_at    -- ISO 8601 UTC timestamp of when the row was written.

    Idempotent and cheap enough to call on every `structured_call()` --
    avoids needing separate app-startup migration wiring for this task.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL,
            prompt_name TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            model TEXT NOT NULL,
            latency_ms INTEGER NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            cost_usd TEXT NOT NULL,
            raw_response TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def ensure_cases_table(conn: sqlite3.Connection) -> None:
    """Create the `cases` table if it doesn't already exist.

    Columns:
      case_id             -- primary key, matches the ShipBob case ID.
      status              -- current `claimpilot.models.CaseState` value
                             (stored as its plain string, e.g. "pending_review").
      recommendation_json -- JSON-serialized `claimpilot.models.Recommendation`
                             (via `.model_dump_json()`, which renders `Decimal`
                             fields as JSON strings so precision survives the
                             round trip), or NULL before a recommendation has
                             been produced.
      gate_results_json   -- JSON-serialized bundle of the intermediate gate
                             objects (`EligibilityResult`/`evidence_gaps`/
                             `ValidationResult`/`CalcResult`) that
                             `pipeline.process_case` actually computed for
                             this case, written by `claimpilot.store.
                             save_gate_results` alongside `recommendation_json`
                             at every pipeline exit point. NULL before any
                             gate results have been saved. Added for Task
                             4.4's outbound guard, which must re-verify an
                             approve decision against these stored gate
                             results -- not the drafted `Recommendation`,
                             which never carried this detail (see
                             `claimpilot.store.GateResults`/
                             `save_gate_results`/`load_gate_results`).
      merchant_id         -- ShipBob account identifier this case belongs to
                             (`claimpilot.models.Case.user_id` -- the stable
                             ID also passed to `generate_invoice`/
                             `submit_reimbursement`, as opposed to
                             `Case.account_name`, a human-readable display
                             name not guaranteed stable/unique). NULL when
                             the upstream case has no `user_id` (should not
                             happen for real ShipBob data, but the field is
                             optional on `Case`). Present so
                             `claimpilot.memory.merchant_context` can compute
                             "claims in the last 90 days" directly from this
                             table instead of needing a separate merchant
                             index.
      created_at          -- ISO 8601 UTC timestamp of case creation.
      updated_at          -- ISO 8601 UTC timestamp of the last status or
                             recommendation write.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            recommendation_json TEXT,
            gate_results_json TEXT,
            merchant_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    # Migration for any `cases` table created before `gate_results_json`/
    # `merchant_id` existed (e.g. an older on-disk `claimpilot.db`):
    # `CREATE TABLE IF NOT EXISTS` above is a no-op against an
    # already-existing table, so each column needs an explicit, idempotent
    # `ALTER TABLE` guarded by a `PRAGMA table_info` existence check.
    existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(cases)")}
    if "gate_results_json" not in existing_columns:
        conn.execute("ALTER TABLE cases ADD COLUMN gate_results_json TEXT")
        conn.commit()
    if "merchant_id" not in existing_columns:
        conn.execute("ALTER TABLE cases ADD COLUMN merchant_id TEXT")
        conn.commit()


def ensure_audit_log_table(conn: sqlite3.Connection) -> None:
    """Create the `audit_log` table if it doesn't already exist.

    This is the accountability trail for the review panel: every state
    transition (and any other notable system/rep action) writes one row
    here, atomically alongside the `cases.status` update that caused it
    (see `claimpilot.store.transition`).

    Columns:
      id           -- autoincrement primary key.
      case_id      -- which case this event belongs to.
      actor        -- "system" (pipeline-driven) or "rep" (human review
                      action), matching `claimpilot.store.Actor`.
      event        -- short event label (e.g. "transition:eligibility->evidence").
      payload_json -- JSON-serialized event detail, or NULL.
      ts           -- ISO 8601 UTC timestamp.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL,
            actor TEXT NOT NULL,
            event TEXT NOT NULL,
            payload_json TEXT,
            ts TEXT NOT NULL
        )
        """
    )
    conn.commit()


def ensure_actions_table(conn: sqlite3.Connection) -> None:
    """Create the `actions` table if it doesn't already exist.

    This is the idempotency backstop called out in the plan: `UNIQUE(case_id,
    action)` means a case can record at most one "email" action and at most
    one "reimbursement" action, ever. A double-clicked approve button or a
    retried request hits the unique constraint (surfaced by
    `claimpilot.store.record_action` as `DuplicateActionError`) instead of
    double-sending or double-paying.

    Columns:
      id           -- autoincrement primary key.
      case_id      -- which case this action was taken for.
      action       -- "email" or "reimbursement".
      payload_json -- JSON-serialized action detail (e.g. email body, payout
                      amount).
      created_at   -- ISO 8601 UTC timestamp.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL,
            action TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(case_id, action)
        )
        """
    )
    conn.commit()


def ensure_memory_table(conn: sqlite3.Connection) -> None:
    """Create the `memory` table if it doesn't already exist.

    This is the real, final shape of the table -- nothing ever wrote to an
    earlier placeholder shape, so there was no data to migrate.

    Columns:
      id             -- autoincrement primary key.
      scope          -- `"global"` (applies to every merchant) or
                        `"merchant"` (applies to one merchant only).
      merchant_id    -- the merchant this row applies to (see `cases.
                        merchant_id`'s docstring for why `Case.user_id` is
                        the canonical identifier used here), or NULL for
                        `scope="global"` rows. `claimpilot.memory` enforces
                        merchant_id is present iff scope="merchant".
      kind           -- `"note"` (freeform observation), `"policy"`
                        (durable, curated guidance -- written deliberately,
                        e.g. by the feedback distiller or a rep via
                        the review panel), or `"correction"` (a
                        record of a rep editing/pushing back on a draft,
                        written automatically by `claimpilot.web.app` on
                        every such action).
      content        -- freeform text. For `kind="correction"` this is a
                        JSON-encoded object (see `claimpilot.memory.
                        record_correction`'s docstring for the exact keys);
                        for `kind="note"`/`"policy"` it's plain human-
                        readable text, not JSON.
      source_case_id -- the case this row was written about/from, or NULL
                        (e.g. a hand-authored global policy note with no
                        single originating case).
      created_at     -- ISO 8601 UTC timestamp.
    """
    # A real on-disk `claimpilot.db` from before this task (if one ever ran
    # far enough to create the table) would still have the old key/
    # value_json shape -- `CREATE TABLE IF NOT EXISTS` below is a no-op
    # against an already-existing table, so an old-shaped `memory` table
    # needs to be dropped and recreated explicitly. Safe unconditionally:
    # nothing ever wrote to the old shape, so there is nothing to lose.
    existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(memory)")}
    if existing_columns and "scope" not in existing_columns:
        conn.execute("DROP TABLE memory")
        conn.commit()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope TEXT NOT NULL,
            merchant_id TEXT,
            kind TEXT NOT NULL,
            content TEXT NOT NULL,
            source_case_id TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create all claimpilot tables (idempotent) on the given connection.

    Covers `llm_calls`, `cases`/`audit_log`/`actions`,
    and `memory`.
    `claimpilot.store`/`claimpilot.memory` call this at the top of every
    entry point so callers never have to remember a separate migration step
    -- consistent with how `llm.py`'s `_log_call` already calls
    `ensure_llm_calls_table` inline.
    """
    ensure_llm_calls_table(conn)
    ensure_cases_table(conn)
    ensure_audit_log_table(conn)
    ensure_actions_table(conn)
    ensure_memory_table(conn)
