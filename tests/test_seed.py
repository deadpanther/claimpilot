"""Tests for `scripts/seed.py`.

Covers only what doesn't require a real `ANTHROPIC_API_KEY` or network
access, per the task's explicit instruction: `reset_db`, `reset_outbox`,
`check_api_key`'s early-exit behavior, and `main()`'s wiring (reset steps
always run, `seed_all_cases` is never invoked when the key check fails).
`seed_all_cases`'s actual pipeline/LLM-calling behavior is NOT exercised
here -- there is no fixture/fake-transport path in `seed.py` by design (see
its module docstring), so faking that just to claim coverage would test
nothing real.

`scripts/` isn't a Python package (no `__init__.py`, not installed), so the
module is loaded directly from its file path via `importlib` rather than a
normal `import scripts.seed` -- this is the simplest way to get the same
module object every test in this file needs to share (so monkeypatching
attributes on it actually take effect where `main()` reads them).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SEED_PATH = _REPO_ROOT / "scripts" / "seed.py"


def _load_seed_module():
    spec = importlib.util.spec_from_file_location("claimpilot_seed_script", _SEED_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def seed():
    return _load_seed_module()


# --- reset_db ----------------------------------------------------------------


def test_reset_db_creates_a_fresh_file_with_schema(seed, tmp_path, monkeypatch):
    from claimpilot import db
    from claimpilot.config import settings

    fake_db_path = tmp_path / "fresh.db"
    monkeypatch.setattr(settings, "db_path", fake_db_path)

    seed.reset_db()

    assert fake_db_path.exists()
    # Fresh connection (not the one reset_db used internally) can see the
    # tables ensure_schema() creates -- confirms the schema really landed,
    # not just that some file exists.
    conn = db.get_connection(fake_db_path)
    try:
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    finally:
        conn.close()
    assert {"cases", "audit_log", "actions", "memory", "llm_calls"} <= tables


def test_reset_db_removes_stale_data_from_a_prior_run(seed, tmp_path, monkeypatch):
    from claimpilot import db
    from claimpilot.config import settings

    fake_db_path = tmp_path / "stale.db"
    monkeypatch.setattr(settings, "db_path", fake_db_path)

    # Seed some stale state, as if a prior demo run had already populated it.
    conn = db.get_connection(fake_db_path)
    try:
        db.ensure_schema(conn)
        conn.execute(
            "INSERT INTO cases (case_id, status, merchant_id, created_at, updated_at) "
            "VALUES ('CASE-STALE', 'pending_review', 'm1', 'x', 'x')"
        )
        conn.commit()
    finally:
        conn.close()

    seed.reset_db()

    conn = db.get_connection(fake_db_path)
    try:
        rows = conn.execute("SELECT * FROM cases").fetchall()
    finally:
        conn.close()
    assert rows == []


# --- reset_outbox --------------------------------------------------------------


def test_reset_outbox_clears_prior_content(seed, tmp_path, monkeypatch):
    from claimpilot.clients import fixtures

    fake_outbox_dir = tmp_path / "outbox"
    fake_outbox_dir.mkdir()
    stale_file = fake_outbox_dir / "outbox.jsonl"
    stale_file.write_text(json.dumps({"stale": True}) + "\n")
    monkeypatch.setattr(fixtures, "_OUTBOX_DIR", fake_outbox_dir)

    seed.reset_outbox()

    assert not stale_file.exists()


def test_reset_outbox_is_a_noop_when_outbox_dir_does_not_exist_yet(seed, tmp_path, monkeypatch):
    from claimpilot.clients import fixtures

    missing_dir = tmp_path / "does-not-exist-yet"
    monkeypatch.setattr(fixtures, "_OUTBOX_DIR", missing_dir)

    seed.reset_outbox()  # must not raise

    assert not missing_dir.exists()


# --- check_api_key -------------------------------------------------------------


def test_check_api_key_false_and_prints_clear_message_when_unset(seed, monkeypatch, capsys):
    from claimpilot.config import settings

    # Pin llm_provider explicitly rather than inheriting whatever a real
    # .env configures (e.g. LLM_PROVIDER=openai) -- check_api_key() is
    # provider-aware, so this test must not depend on environment state.
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "")

    result = seed.check_api_key()

    assert result is False
    captured = capsys.readouterr()
    assert "ANTHROPIC_API_KEY" in captured.err
    assert "seeding" in captured.err.lower() or "seed" in captured.err.lower()


def test_check_api_key_true_when_set(seed, monkeypatch):
    from claimpilot.config import settings

    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-fake-for-test")

    assert seed.check_api_key() is True


def test_check_api_key_false_for_openai_provider_when_unset(seed, monkeypatch, capsys):
    from claimpilot.config import settings

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "")

    result = seed.check_api_key()

    assert result is False
    captured = capsys.readouterr()
    assert "OPENAI_API_KEY" in captured.err


def test_check_api_key_true_for_openai_provider_when_set(seed, monkeypatch):
    from claimpilot.config import settings

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-fake-for-test")

    assert seed.check_api_key() is True


# --- main() wiring: reset always runs, seeding never runs without a key ------


def test_main_exits_nonzero_without_resetting_or_calling_seed_all_cases(
    seed, tmp_path, monkeypatch
):
    """Missing key -> `main()` must exit non-zero WITHOUT touching the DB,
    the outbox, or calling `seed_all_cases` at all. The check runs before
    either reset precisely so a presenter re-running this script with a
    forgotten key doesn't get their already-seeded review queue wiped out
    only to have the run then refuse to repopulate it (see `main()`'s
    inline comment) -- so this test asserts the DB/outbox were left
    untouched, not (as an earlier version of this test did) that they were
    reset.
    """
    from claimpilot.clients import fixtures
    from claimpilot.config import settings

    fake_db_path = tmp_path / "main.db"  # deliberately never created
    fake_outbox_dir = tmp_path / "outbox"
    fake_outbox_dir.mkdir()
    stale_outbox = fake_outbox_dir / "outbox.jsonl"
    stale_outbox.write_text("stale\n")

    monkeypatch.setattr(settings, "db_path", fake_db_path)
    monkeypatch.setattr(fixtures, "_OUTBOX_DIR", fake_outbox_dir)
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "")

    def _fail_if_called(case_ids=None):
        raise AssertionError("seed_all_cases must not be called without a configured API key")

    monkeypatch.setattr(seed, "seed_all_cases", _fail_if_called)

    exit_code = seed.main([])

    assert exit_code == 1
    assert not fake_db_path.exists()  # reset_db did NOT run
    assert stale_outbox.exists()  # reset_outbox did NOT run
    assert stale_outbox.read_text() == "stale\n"


# --- main() argument parsing / reset-skip contract ---------------------------


def test_main_with_case_flag_skips_reset_and_forwards_case_ids(seed, tmp_path, monkeypatch):
    """`--case` is load-bearing for the memory carry-forward walkthrough: it must NOT reset
    the DB/outbox (that would destroy the pushback-derived policy note the
    beat depends on), and it must forward the requested case ID(s) straight
    through to `seed_all_cases`. A future "simplification" that made `--case`
    always reset would silently break that live demo beat without any test
    failing here otherwise.
    """
    from claimpilot.clients import fixtures
    from claimpilot.config import settings

    fake_db_path = tmp_path / "main.db"  # must stay absent
    fake_outbox_dir = tmp_path / "outbox"
    fake_outbox_dir.mkdir()
    stale_outbox = fake_outbox_dir / "outbox.jsonl"
    stale_outbox.write_text("stale\n")

    monkeypatch.setattr(settings, "db_path", fake_db_path)
    monkeypatch.setattr(fixtures, "_OUTBOX_DIR", fake_outbox_dir)
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-fake-for-test")

    received: dict = {}

    async def _fake_seed_all_cases(case_ids=None):
        received["case_ids"] = case_ids

    monkeypatch.setattr(seed, "seed_all_cases", _fake_seed_all_cases)

    exit_code = seed.main(["--case", "CASE-9003-REPEAT"])

    assert exit_code == 0
    assert received["case_ids"] == ["CASE-9003-REPEAT"]
    assert not fake_db_path.exists()  # reset_db did NOT run
    assert stale_outbox.read_text() == "stale\n"  # reset_outbox did NOT run


def test_main_without_case_flag_resets_and_seeds_the_full_default_set(seed, tmp_path, monkeypatch):
    from claimpilot.clients import fixtures
    from claimpilot.config import settings

    fake_db_path = tmp_path / "main.db"
    fake_outbox_dir = tmp_path / "outbox"
    fake_outbox_dir.mkdir()
    stale_outbox = fake_outbox_dir / "outbox.jsonl"
    stale_outbox.write_text("stale\n")

    monkeypatch.setattr(settings, "db_path", fake_db_path)
    monkeypatch.setattr(fixtures, "_OUTBOX_DIR", fake_outbox_dir)
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-fake-for-test")

    received: dict = {}

    async def _fake_seed_all_cases(case_ids=None):
        received["case_ids"] = case_ids

    monkeypatch.setattr(seed, "seed_all_cases", _fake_seed_all_cases)

    exit_code = seed.main([])

    assert exit_code == 0
    assert received["case_ids"] is None  # seed_all_cases falls back to DEMO_CASE_IDS
    assert fake_db_path.exists()  # reset_db ran
    assert not stale_outbox.exists()  # reset_outbox ran


# --- __main__ entrypoint: real subprocess invocation --------------------------


def test_running_as_a_script_exits_nonzero_without_a_key_and_no_raw_traceback():
    """Invokes `python scripts/seed.py` in a real subprocess with
    `ANTHROPIC_API_KEY` cleared from the environment.

    Guarded up front: if this checkout's `settings.anthropic_api_key` OR
    `settings.openai_api_key` is genuinely non-empty (a real `.env` with a
    real key for either provider is present), `env_ignore_empty=True` means
    clearing/blanking the env var in the subprocess's `env=` dict still
    falls through to that `.env` source -- the child process reads the same
    absolute, repo-root `.env` file directly regardless of what's passed in
    `env=`, and would then proceed past `check_api_key()` into a real,
    billed pipeline run (this happened once during manual testing before
    this guard was broadened to cover both providers -- do not narrow it
    back to `anthropic_api_key` alone). Skip rather than risk that; this
    test only asserts the failure path, which only triggers when there is
    genuinely no key configured anywhere for either provider.

    This exercises the real `if __name__ == "__main__"` entrypoint end to
    end, including argument-free invocation exactly as a human running the
    demo would type it. Because `check_api_key()` now runs before either
    reset (see `main()`), this run must NOT touch `claimpilot.db`/`outbox/`
    at all.
    """
    from claimpilot.config import settings

    if settings.anthropic_api_key or settings.openai_api_key:
        pytest.skip(
            "real API key configured for Anthropic and/or OpenAI; subprocess "
            "seed run would hit the live API"
        )

    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
    env["ANTHROPIC_API_KEY"] = ""
    env["OPENAI_API_KEY"] = ""

    db_path = _REPO_ROOT / "claimpilot.db"
    db_existed_before = db_path.exists()
    db_mtime_before = db_path.stat().st_mtime if db_existed_before else None

    result = subprocess.run(
        [sys.executable, str(_SEED_PATH)],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1
    assert "ANTHROPIC_API_KEY" in result.stderr
    # No raw Python traceback on the missing-key path.
    assert "Traceback (most recent call last)" not in result.stderr
    # The real repo DB must be untouched -- check-first ordering in main().
    if db_existed_before:
        assert db_path.stat().st_mtime == db_mtime_before
    else:
        assert not db_path.exists()
