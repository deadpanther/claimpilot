"""Tests for the post-demo-v1 swappable-provider feature's OpenAI side:
`OpenAITransport` and `claimpilot.openai_schema.to_openai_strict_schema`.

Split out of `tests/test_llm.py` (which keeps the provider-agnostic
`structured_call`/`FakeTransport` tests and the `AnthropicTransport`-specific
tests) purely to keep both files under the house file-size guideline --
`OpenAITransport`'s test surface (request-building/response-parsing, its own
fake OpenAI SDK response classes, strict-schema adaptation) is large enough
to earn its own file, mirroring this project's existing one-test-file-
per-feature-module convention (`test_evidence.py`, `test_validation.py`,
etc.). Shared fixtures (`FakeTransport`, `Person`, `PROMPT_NAME`,
`_messages`) are imported from `tests.test_llm`, the same pattern
`test_evidence.py`/`test_validation.py` already use.

The underlying `openai.AsyncOpenAI` client is constructed for real in these
tests (that alone makes no network call -- an empty/dummy API key is enough,
no real credentials needed), but its `chat.completions.create` is always
monkeypatched to an async stub. No real HTTP/SDK network I/O happens
anywhere in this file.
"""

from __future__ import annotations

import base64
import copy
from decimal import Decimal
from pathlib import Path

from claimpilot.config import settings
from claimpilot.db import get_connection
from claimpilot.gates.evidence import AttachmentClassification
from claimpilot.llm import OpenAITransport, get_transport, structured_call
from claimpilot.openai_schema import to_openai_strict_schema
from tests.test_llm import PROMPT_NAME, Person, _messages

# --- OpenAITransport request-building / response-parsing --------------------


class _FakeOpenAIFunctionCall:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeOpenAIToolCall:
    def __init__(self, *, name: str, arguments: str, type_: str = "function") -> None:
        self.type = type_
        self.function = _FakeOpenAIFunctionCall(name, arguments)


class _FakeOpenAIMessage:
    def __init__(self, *, tool_calls: list[_FakeOpenAIToolCall] | None = None, content: str | None = None) -> None:
        self.tool_calls = tool_calls
        self.content = content

    def model_dump(self, mode: str = "json") -> dict:
        return {
            "content": self.content,
            "tool_calls": [
                {"type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in (self.tool_calls or [])
            ],
        }


class _FakeOpenAIChoice:
    def __init__(self, *, message: _FakeOpenAIMessage, finish_reason: str) -> None:
        self.message = message
        self.finish_reason = finish_reason


class _FakeOpenAIUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeOpenAICompletion:
    def __init__(self, *, choices: list[_FakeOpenAIChoice], usage: _FakeOpenAIUsage | None) -> None:
        self.choices = choices
        self.usage = usage


def _openai_transport(monkeypatch) -> OpenAITransport:
    """`openai.AsyncOpenAI` raises `OpenAIError` at construction time given
    an empty API key (unlike `anthropic.AsyncAnthropic`, which allows it) --
    `settings.openai_api_key` defaults to `""`, so every OpenAITransport test
    needs a non-empty (if fake) key monkeypatched in first.
    """
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-not-real")
    return OpenAITransport()


async def test_openai_transport_create_sends_forced_strict_tool_choice_and_parses_response(monkeypatch):
    transport = _openai_transport(monkeypatch)
    captured_kwargs: dict = {}

    async def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        return _FakeOpenAICompletion(
            choices=[
                _FakeOpenAIChoice(
                    message=_FakeOpenAIMessage(
                        tool_calls=[
                            _FakeOpenAIToolCall(name="extract_person", arguments='{"name": "Ann", "age": 30}')
                        ]
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=_FakeOpenAIUsage(prompt_tokens=12, completion_tokens=6),
        )

    transport._client.chat.completions.create = fake_create

    result = await transport.create(
        model="gpt-4o",
        system="a system prompt",
        messages=[{"role": "user", "content": "hi"}],
        tool_name="extract_person",
        tool_description="Return a Person.",
        tool_schema=Person.model_json_schema(),
        timeout=30.0,
    )

    # Request-building: system prompt becomes the leading system message
    # (OpenAI has no separate `system=` parameter, unlike Anthropic), forced
    # strict function tool_choice, timeout/max_completion_tokens passed
    # through, and the tool's `parameters` is the strict-adapted schema (no
    # `X | None` optional fields on `Person`, so strict-adaptation is a
    # no-op here beyond `additionalProperties`/`required` -- the adapter
    # itself is exercised more thoroughly below against a real
    # optional-field project schema).
    assert captured_kwargs["model"] == "gpt-4o"
    assert captured_kwargs["messages"][0] == {"role": "system", "content": "a system prompt"}
    assert captured_kwargs["messages"][1] == {"role": "user", "content": "hi"}
    assert captured_kwargs["tool_choice"] == {"type": "function", "function": {"name": "extract_person"}}
    assert captured_kwargs["timeout"] == 30.0
    assert captured_kwargs["max_completion_tokens"] > 0
    assert captured_kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "extract_person",
                "description": "Return a Person.",
                "parameters": to_openai_strict_schema(Person.model_json_schema()),
                "strict": True,
            },
        }
    ]

    # Response-parsing: tool_input/usage/stop_reason/raw_content all correct.
    assert result.tool_input == {"name": "Ann", "age": 30}
    assert result.input_tokens == 12
    assert result.output_tokens == 6
    assert result.stop_reason == "tool_calls"
    assert result.raw_content == [
        {
            "content": None,
            "tool_calls": [{"type": "function", "function": {"name": "extract_person", "arguments": '{"name": "Ann", "age": 30}'}}],
        }
    ]


async def test_openai_transport_create_returns_none_tool_input_when_tool_not_called(monkeypatch):
    transport = _openai_transport(monkeypatch)

    async def fake_create(**kwargs):
        # Model replied with plain text instead of calling the forced tool.
        return _FakeOpenAICompletion(
            choices=[
                _FakeOpenAIChoice(message=_FakeOpenAIMessage(tool_calls=None, content="hi there"), finish_reason="stop")
            ],
            usage=_FakeOpenAIUsage(prompt_tokens=5, completion_tokens=3),
        )

    transport._client.chat.completions.create = fake_create

    result = await transport.create(
        model="gpt-4o",
        system="a system prompt",
        messages=[{"role": "user", "content": "hi"}],
        tool_name="extract_person",
        tool_description="Return a Person.",
        tool_schema=Person.model_json_schema(),
        timeout=30.0,
    )

    assert result.tool_input is None
    assert result.stop_reason == "stop"


async def test_openai_transport_create_treats_malformed_json_arguments_as_no_tool_input(monkeypatch):
    """OpenAI hands back `function.arguments` as a JSON *string* (unlike
    Anthropic, which hands back an already-parsed `input` dict) -- if that
    string somehow isn't valid JSON, `OpenAITransport` must not let
    `json.JSONDecodeError` escape `create()` as an uncaught/unretried
    transport exception. It's a bad-*output* problem, so it gets treated
    exactly like Anthropic's "no matching tool_use block" case
    (`tool_input=None`), which flows through `structured_call`'s normal
    one-retry-then-raise path.
    """
    transport = _openai_transport(monkeypatch)

    async def fake_create(**kwargs):
        return _FakeOpenAICompletion(
            choices=[
                _FakeOpenAIChoice(
                    message=_FakeOpenAIMessage(
                        tool_calls=[_FakeOpenAIToolCall(name="extract_person", arguments="{not valid json")]
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=_FakeOpenAIUsage(prompt_tokens=5, completion_tokens=3),
        )

    transport._client.chat.completions.create = fake_create

    result = await transport.create(
        model="gpt-4o",
        system="a system prompt",
        messages=[{"role": "user", "content": "hi"}],
        tool_name="extract_person",
        tool_description="Return a Person.",
        tool_schema=Person.model_json_schema(),
        timeout=30.0,
    )

    assert result.tool_input is None
    assert result.stop_reason == "tool_calls"


async def test_openai_transport_create_embeds_images_in_openai_format(monkeypatch):
    transport = _openai_transport(monkeypatch)
    captured_kwargs: dict = {}

    async def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        return _FakeOpenAICompletion(
            choices=[
                _FakeOpenAIChoice(
                    message=_FakeOpenAIMessage(
                        tool_calls=[
                            _FakeOpenAIToolCall(name="extract_person", arguments='{"name": "Ann", "age": 30}')
                        ]
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=_FakeOpenAIUsage(prompt_tokens=1, completion_tokens=1),
        )

    transport._client.chat.completions.create = fake_create
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"fake-png-data"

    await transport.create(
        model="gpt-4o",
        system="a system prompt",
        messages=[{"role": "user", "content": "look at this photo"}],
        tool_name="extract_person",
        tool_description="Return a Person.",
        tool_schema=Person.model_json_schema(),
        timeout=30.0,
        images=[png_bytes],
    )

    # Last message in the request (the user turn, after the leading system
    # message) carries OpenAI's `image_url`/data-URL block shape, not
    # Anthropic's `image`/`source` shape.
    last_content = captured_kwargs["messages"][-1]["content"]
    image_blocks = [block for block in last_content if block["type"] == "image_url"]
    text_blocks = [block for block in last_content if block["type"] == "text"]

    assert len(image_blocks) == 1
    url = image_blocks[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    encoded = url.removeprefix("data:image/png;base64,")
    assert base64.standard_b64decode(encoded) == png_bytes
    assert text_blocks == [{"type": "text", "text": "look at this photo"}]


# --- get_transport() factory: openai branch ----------------------------------


def test_get_transport_returns_openai_transport_when_configured(monkeypatch):
    """`get_transport()`'s provider branch actually works: flipping
    `settings.llm_provider` to `"openai"` changes which concrete `Transport`
    the factory returns, with no other code (`structured_call`,
    `gates/evidence.py`, etc.) needing to know or care. `cache_clear()`
    before and after, matching `test_llm.py`'s anthropic-singleton test
    convention, so this test doesn't leak a cached instance into whichever
    test runs next.
    """
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-not-real")
    get_transport.cache_clear()

    first = get_transport()
    second = get_transport()

    assert first is second
    assert isinstance(first, OpenAITransport)
    get_transport.cache_clear()


# --- structured_call() end-to-end through a real Transport -------------------
# Every `structured_call` test in `test_llm.py` uses `FakeTransport`, whose
# `create(self, **kwargs)` absorbs any kwargs -- so a kwarg-name mismatch
# between what `structured_call` sends and what a real `Transport`
# implementation's `create()` accepts would never surface there. And every
# real-transport test above calls `.create(...)` directly with hand-written
# kwargs, never through `structured_call` itself. This test closes that gap
# for `OpenAITransport` specifically (the new provider): it drives
# `structured_call` with `settings.llm_provider="openai"` and a *real*
# `OpenAITransport` instance (SDK method monkeypatched, no network), proving
# the whole path -- provider-aware default model, untrusted-data rule
# reaching OpenAI's request, images passed through then embedded in OpenAI's
# format, and the parsed result validating into `schema` -- actually wires
# together end to end, not just in its two halves separately.


async def test_structured_call_end_to_end_through_real_openai_transport(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    transport = _openai_transport(monkeypatch)
    captured_kwargs: dict = {}

    async def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        return _FakeOpenAICompletion(
            choices=[
                _FakeOpenAIChoice(
                    message=_FakeOpenAIMessage(
                        tool_calls=[
                            _FakeOpenAIToolCall(name="extract_person", arguments='{"name": "Ann", "age": 30}')
                        ]
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=_FakeOpenAIUsage(prompt_tokens=10, completion_tokens=5),
        )

    transport._client.chat.completions.create = fake_create
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"fake-png-data"
    db_path = tmp_path / "t.db"

    result = await structured_call(
        case_id="CASE-E2E",
        prompt_name=PROMPT_NAME,
        messages=_messages("look at this photo"),
        schema=Person,
        images=[png_bytes],
        transport=transport,
        db_path=db_path,
    )

    assert result == Person(name="Ann", age=30)

    # Provider-aware default model resolution (no explicit `model=` given).
    assert captured_kwargs["model"] == settings.openai_model == "gpt-4o"

    # The standing untrusted-data rule reached OpenAI's request, in the
    # leading system message (OpenAI has no separate `system=` parameter).
    assert captured_kwargs["messages"][0]["role"] == "system"
    assert "<untrusted_data>" in captured_kwargs["messages"][0]["content"]

    # Images were passed through `structured_call` then embedded by
    # `OpenAITransport` itself, in OpenAI's own format.
    last_content = captured_kwargs["messages"][-1]["content"]
    assert any(block["type"] == "image_url" for block in last_content)

    # The `llm_calls` row landed with the `gpt-4o` pricing key applied (not
    # silently priced at $0 via a model-name mismatch with `LLM_PRICING`).
    conn = get_connection(db_path)
    row = conn.execute("SELECT model, cost_usd FROM llm_calls").fetchone()
    conn.close()
    assert row["model"] == "gpt-4o"
    assert Decimal(row["cost_usd"]) > Decimal("0")


# --- to_openai_strict_schema() -----------------------------------------------


def test_to_openai_strict_schema_flat_schema_sets_required_and_additional_properties():
    schema = Person.model_json_schema()
    strict = to_openai_strict_schema(schema)

    assert strict["additionalProperties"] is False
    assert strict["required"] == ["name", "age"]


def test_to_openai_strict_schema_handles_real_project_schema_with_optional_field():
    """`AttachmentClassification` (`gates/evidence.py`) is a real schema this
    project sends as a forced tool/function call, with an actually-optional
    field (`quality_issue: str | None = None`) -- pydantic emits that as
    `anyOf: [{"type": "string"}, {"type": "null"}]` with `default: null`,
    NOT listed in `required`. OpenAI's strict mode requires every property
    listed in `required` (optionality expressed via the `anyOf`/null instead)
    and rejects unrecognized behavior implied by `default`. This is the
    concrete case `to_openai_strict_schema` exists for.
    """
    schema = AttachmentClassification.model_json_schema()
    strict = to_openai_strict_schema(schema)

    assert strict["additionalProperties"] is False
    assert set(strict["required"]) == {"category", "confidence", "usable", "quality_issue"}

    quality_issue = strict["properties"]["quality_issue"]
    assert "default" not in quality_issue
    # The nullable union itself is untouched -- strict mode's fix is only
    # about `required`/`additionalProperties`/`default`, not the type union.
    assert quality_issue["anyOf"] == [{"type": "string"}, {"type": "null"}]

    # `category` is a `$ref`'d enum ($defs/EvidenceItem) -- confirm the walk
    # doesn't choke on `$ref` nodes (they have no "type"/"properties" key of
    # their own, so they pass through unchanged; the *referenced* `$defs`
    # entry is what gets adapted).
    assert strict["properties"]["category"] == {"$ref": "#/$defs/EvidenceItem"}


def test_to_openai_strict_schema_does_not_mutate_input():
    """Immutability (house style): the adapter must return a new structure,
    never mutate the schema dict `schema.model_json_schema()` handed it (that
    dict could be reused/cached elsewhere).
    """
    schema = AttachmentClassification.model_json_schema()
    original = copy.deepcopy(schema)

    to_openai_strict_schema(schema)

    assert schema == original
