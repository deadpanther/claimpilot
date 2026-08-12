"""Demo seed script.

Resets local state (the on-disk SQLite database + the outbox JSONL file)
and then runs the real pipeline (`claimpilot.pipeline.process_case`)
against `DEMO_CASE_IDS` (or, with `--case`, a caller-chosen subset -- see
Usage below), so the review queue is pre-populated and ready for a live
walkthrough.

**This makes real Claude API calls.** There is no fixture/fake transport
path here on purpose -- `get_transport()`'s default (real Anthropic
transport) is used exactly as the pipeline would use it in production, since
the entire point of a demo is showing the real system work end to end, not
a scripted replay. That means this script:

- Requires `settings.anthropic_api_key` to be set (via `.env` or the
  environment) -- `check_api_key()` fails fast with a clear, actionable
  message and a non-zero exit code if it's missing, *before* touching the
  network or the database, rather than letting the first real LLM call blow
  up with a raw traceback deep in `claimpilot.llm`.
- Costs real money and real latency per case (7 cases x several LLM calls
  each -- classification per attachment, validation, drafting).
- Cannot be run end-to-end in an environment with no configured API key
  (this repo's CI/dev sandbox, for instance) -- that's expected. The parts
  of this script that don't need a real key (`reset_db`, `reset_outbox`,
  `check_api_key`, argument parsing) are covered by `tests/test_seed.py`;
  `seed_all_cases`'s actual LLM-calling behavior is not (and should not be)
  faked just to claim coverage there.

Usage:
  `python scripts/seed.py` -- full reset (DB + outbox) then seed of
  `DEMO_CASE_IDS`. Run this once before the demo, ahead of time.
  `python scripts/seed.py --case CASE-ID` (repeatable) -- process only the
  given case ID(s) via the real pipeline, WITHOUT resetting the database or
  outbox first. This is how the memory carry-forward walkthrough seeds
  `CASE-9003-REPEAT` live, after the CASE-1001 pushback -- a full reset at
  that point would destroy the pushback-derived policy note the whole beat
  depends on.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Demo case IDs (see module docstring above): the 5 real
# fixture cases from `docs/api/postman_collection.json`, plus 2 of the 3
# synthetic cases from `fixtures/synthetic.json` (insured routing and the
# cap trigger).
#
# **`CASE-9003-REPEAT` is deliberately NOT included here.** It's the
# same-merchant repeat case for the memory-carry-forward demo beat
# (scripted end to end in `tests/test_memory_demo.py`): its
# whole point is that its draft reflects a policy note distilled from a
# CASE-1001 pushback that happens live, during the demo. If it were
# pre-seeded here (before any pushback exists), its stored draft would be
# generated with an empty merchant memory context, and the "money shot"
# would render empty when the presenter opens it live. It must be processed
# *after* the CASE-1001 pushback -- done via
# `--case CASE-9003-REPEAT` (see `seed_all_cases`/`main`'s `--case` flag).
# Also, reprocessing an already-seeded case is not just wasted API spend --
# `store.LEGAL_TRANSITIONS` has no `PENDING_REVIEW -> ELIGIBILITY` edge, so a
# case already sitting in the review queue would raise
# `store.IllegalTransitionError` on a second `process_case` call, not
# silently overwrite its draft.
DEMO_CASE_IDS: list[str] = [
    "CASE-1001",
    "CASE-1002",
    "CASE-1003",
    "CASE-1004",
    "CASE-1005",
    "CASE-9001-INSURED",
    "CASE-9002-CAP",
]


def reset_db() -> None:
    """Delete the on-disk SQLite database file, then recreate it with a
    fresh schema.

    Simplest way to guarantee a clean slate ("resets DB" per the task
    description): delete-and-recreate rather than truncating tables in
    place, so a schema change between demo runs can never leave stale
    columns/rows behind. `ensure_schema()` is idempotent and safe to call
    on a brand-new empty file (it's the same function every other entry
    point in this codebase calls at startup).
    """
    from claimpilot.config import settings
    from claimpilot.db import ensure_schema, get_connection

    if settings.db_path.exists():
        settings.db_path.unlink()
    conn = get_connection()
    try:
        ensure_schema(conn)
        conn.commit()
    finally:
        conn.close()


def reset_outbox() -> None:
    """Delete the outbox directory's JSONL file(s), leaving the directory
    itself in place (recreated lazily by `FixtureClient._append_outbox` on
    the first real send during the demo).
    """
    from claimpilot.clients.fixtures import _OUTBOX_DIR

    if not _OUTBOX_DIR.exists():
        return
    for path in _OUTBOX_DIR.glob("*.jsonl"):
        path.unlink()


def check_api_key() -> bool:
    """Return whether the configured `settings.llm_provider`'s API key is set.

    Provider-aware (mirrors `evals/test_golden.py`'s `_llm_provider_has_api_key`
    guard, added alongside OpenAI support) -- checks `anthropic_api_key` or
    `openai_api_key` depending on `settings.llm_provider`, not just Anthropic's.
    Prints a clear, actionable message (not a raw traceback) when it isn't
    set -- the caller (`main()`) is responsible for exiting non-zero based on
    this return value. Kept as a plain predicate (rather than raising/
    exiting itself) so tests can call it directly without needing to catch
    `SystemExit`.
    """
    from claimpilot.config import configured_api_key

    is_set, env_var, provider = configured_api_key()
    if is_set:
        return True

    print(
        f"No {env_var} configured -- set it in .env before seeding "
        f"demo data. Seeding requires the real {provider} API (see "
        "scripts/seed.py's module docstring): there is no fixture/fake "
        "transport path for this script, so nothing can proceed without a "
        "real key.",
        file=sys.stderr,
    )
    return False


async def seed_all_cases(case_ids: list[str] | None = None) -> None:
    """Run the given demo case IDs (default: `DEMO_CASE_IDS`) through the
    real pipeline, sequentially.

    Uses the default (real) transport and (fixture) ShipBobClient --
    `process_case`'s own defaults -- so this exercises exactly the code
    path a live demo run would. One case's failure (LLM error, unexpected
    gate outcome, etc.) is caught and reported without aborting the rest of
    the run, per the task description ("one bad case shouldn't block
    seeding the rest").

    `case_ids` lets `main()`'s `--case` flag seed a single case on demand --
    in particular `CASE-9003-REPEAT` (deliberately excluded from
    `DEMO_CASE_IDS`, see that constant's comment), which the memory
    beat 3 processes live, after the CASE-1001 pushback it depends on.
    """
    from claimpilot.pipeline import process_case

    for case_id in case_ids if case_ids is not None else DEMO_CASE_IDS:
        print(f"Processing {case_id}...", flush=True)
        try:
            recommendation = await process_case(case_id)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
            print(f"  FAILED: {case_id}: {exc!r}", file=sys.stderr, flush=True)
            continue
        print(f"  -> decision={recommendation.decision!r} amount={recommendation.amount}", flush=True)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        default=None,
        help=(
            "Process only this case ID (repeatable), and SKIP the DB/outbox "
            "reset -- for seeding one case live mid-demo (e.g. "
            "CASE-9003-REPEAT after the CASE-1001 pushback beat) without "
            "destroying existing state. Omit for the normal full "
            "reset-then-seed-DEMO_CASE_IDS run."
        ),
    )
    args = parser.parse_args(argv)

    # Checked BEFORE either reset: a presenter re-running this script in a
    # shell that doesn't have the key loaded must not have their already-
    # seeded review queue wiped out only to then refuse to repopulate it --
    # that would leave them strictly worse off than before running it,
    # possibly minutes before the demo. Fail fast, touch nothing.
    if not check_api_key():
        return 1

    if args.cases is None:
        print("Resetting database...", flush=True)
        reset_db()
        print("Resetting outbox...", flush=True)
        reset_outbox()
    else:
        print(f"Skipping reset -- seeding only: {', '.join(args.cases)}", flush=True)

    asyncio.run(seed_all_cases(args.cases))
    print("Done. Review queue is ready -- start the review UI.", flush=True)
    return 0


if __name__ == "__main__":
    # Ensure `import claimpilot...` resolves when this script is invoked
    # directly as `python scripts/seed.py` from the repo root, without
    # requiring the package to already be installed/on PYTHONPATH.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    sys.exit(main())
