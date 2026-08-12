"""OpenAI strict function-calling schema adaptation.

Split out of `claimpilot.llm` (where `OpenAITransport` is the only caller)
purely to keep `llm.py` under the house file-size guideline -- this is a
pure, self-contained function with no dependency on the rest of that
module's runtime state (no `settings`, no SDK client, no logging), so it
costs nothing to give it its own file. See `claimpilot.llm`'s module
docstring point 7 for how this fits into the multi-provider design.
"""

from __future__ import annotations


def to_openai_strict_schema(schema: dict) -> dict:
    """Adapt a pydantic-generated JSON Schema for OpenAI's strict
    function-calling mode (`strict: True` on a function tool).

    Strict mode validates the model's output against the schema at
    generation time, which is what lets `OpenAITransport` guarantee
    well-formed output the way Anthropic's forced `tool_choice` does -- but
    it imposes structural requirements `schema.model_json_schema()` doesn't
    satisfy by default:

    - every object must set `"additionalProperties": false` (pydantic never
      sets this).
    - every property must be listed in `"required"`, even ones that are
      conceptually optional -- optionality is expressed by including
      `"null"` in the field's `anyOf`/type instead of omitting it from
      `required`. Pydantic already emits `anyOf: [..., {"type": "null"}]`
      for `X | None` fields (see `AttachmentClassification.quality_issue`),
      so this only needs to fix up `required`, not the value types.
    - a `"default"` on a property isn't honored by strict mode (the model
      must always emit an explicit value, `null` included for optional
      fields), so `default` keys are stripped to avoid the schema
      documenting a behavior ("this field can be left out") strict mode
      doesn't actually allow.

    Recurses through `properties`, `$defs`/`definitions`, `items`, and
    `anyOf`/`oneOf`/`allOf`, covering the schema shapes this project
    actually generates (flat pydantic models, `$ref`'d enums and nested
    `BaseModel`s, `list[...]` fields, `X | None` optional fields -- see
    `AttachmentClassification`, `ValidationResult`/`Judgment`,
    `DistillOutput`/`DistilledNote`). This is a best-effort adaptation
    tuned against those shapes, not a general JSON-Schema-to-strict-schema
    compiler -- numeric/string/array constraints (`minimum`, `maxItems`,
    etc.) are passed through unchanged and NOT verified against OpenAI's
    supported-keyword list. Revisit against OpenAI's structured-outputs
    docs (https://platform.openai.com/docs/guides/structured-outputs) if a
    real schema is ever rejected by the API -- this has not been exercised
    against the real API in this codebase, same caveat `evidence.py` and
    `validation.py` already document for their `$ref`-heavy schemas against
    Anthropic's API.
    """
    if not isinstance(schema, dict):
        return schema

    node = {key: value for key, value in schema.items() if key != "default"}

    if node.get("type") == "object" and "properties" in node:
        properties = {name: to_openai_strict_schema(value) for name, value in node["properties"].items()}
        node["properties"] = properties
        node["required"] = list(properties.keys())
        node["additionalProperties"] = False

    if "items" in node:
        node["items"] = to_openai_strict_schema(node["items"])

    for defs_key in ("$defs", "definitions"):
        if defs_key in node:
            node[defs_key] = {name: to_openai_strict_schema(value) for name, value in node[defs_key].items()}

    for combinator_key in ("anyOf", "oneOf", "allOf"):
        if combinator_key in node:
            node[combinator_key] = [to_openai_strict_schema(value) for value in node[combinator_key]]

    return node
