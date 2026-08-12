"""Evidence classifier.

Unlike `eligibility.py`, the other module in this `gates/`
package, this module is not entirely pure -- `classify_attachment`
does real I/O (an LLM call via `claimpilot.llm.structured_call`). It lives
here anyway because it's still a *gate*-shaped decision (classify one
attachment, then decide what's missing) and the plan places it in
`gates/evidence.py` explicitly. `evidence_gaps`, the second half of this
module, is kept genuinely pure (no I/O, no LLM) so it stays trivially
testable and reusable by the orchestrator without needing a case_id,
network, or mocked transport.

Design decisions (the task plan's schema/signature were intentionally
abbreviated -- these are the calls made to extend it):

1. **`classify_attachment` branches on `Attachment.content_type`.** The
   model field is `str | None` (real API data may omit it). Per
   `clients/attachment_guard.py`'s current constraint, every real
   attachment fetched today has an `image/*` content-type (even documents
   like an invoice screenshot are still image files) -- so `None`/empty
   content-type defaults to the **image** path, not the text path, since
   that matches every attachment this pipeline can actually produce right
   now. The text path (`text/plain`, `application/pdf`, ...) is implemented
   for the plan's literal "text for PDFs/messages" wording and to
   future-proof against `attachment_guard`'s allowlist loosening later, but
   has no real fixture data to exercise today -- this module's tests construct
   a synthetic case for it. Content-type matching is normalized
   (lowercased, `;`-parameters stripped) so `IMAGE/PNG` or
   `image/jpeg; charset=binary` still match correctly.
2. **Untrusted-data wrapping.** `structured_call`'s standing rule (see
   `llm.py`) only guarantees the model is told what `<untrusted_data>` tags
   mean -- callers must actually wrap untrusted text themselves. Attachment
   `file_name` and (on the text path) decoded attachment content are both
   external, merchant/customer-supplied data, so both are wrapped here.
3. **Text decoding never raises.** The text path decodes `content` with
   `errors="replace"` rather than strict UTF-8, so a mislabeled or
   corrupted attachment degrades to replacement characters instead of
   crashing the pipeline on dirty upstream data (same rationale as
   `eligibility.py`'s malformed-date handling).
4. **`Gap` has three distinct reasons, not two.** The plan's parenthetical
   ("confidence < settings.evidence_min_conf or usable=False -> treat as missing")
   collapses to one *outcome* (act as if missing) but there are three
   distinct *inputs* a drafter needs to tell apart, since "please
   send a photo of X" and "please resend a clearer photo of X" are different
   emails:
     - `"MISSING"` -- the category was never classified at all for this
       case. No `detail` (there is nothing to describe).
     - `"UNUSABLE"` -- classified, but `usable=False`. Checked before
       low-confidence (see point 8) since an unreliable-but-confident
       classification is a more specific complaint than a vague one.
     - `"LOW_CONFIDENCE"` -- classified, `usable=True`, but
       `confidence < settings.evidence_min_conf`.
   `detail` is ALWAYS customer-safe, drop-into-an-email text -- either the
   model's own `quality_issue` verbatim (the expected case for `UNUSABLE`),
   or, when the model didn't supply one (including the *normal* case for
   `LOW_CONFIDENCE`, since the prompt tells the model to set
   `quality_issue: null` whenever `usable` is `true`), a generic but still
   customer-safe fallback phrase -- never an internal diagnostic string like
   a raw confidence number, which would otherwise leak into a real email
   draft. The numeric confidence remains available for audit via the
   `AttachmentClassification` object itself and the `llm_calls` log; `detail`
   is not the place for it. Mirrors `eligibility.py`'s "machine code separate
   from human text" precedent (`EligibilityResult.reason` there /
   `Gap.reason` here).
5. **Reconciliation across duplicate categories.** If multiple attachments
   classify into the same `EvidenceItem`, the category is satisfied (no gap)
   as soon as *any* instance is usable and at/above `settings.evidence_min_conf` --
   a second, worse attachment for the same category doesn't undo a good
   one. If *none* of the instances for a category are good, the first
   matching instance in `classified` order (the order the caller supplied,
   e.g. attachment-fetch order) is used to derive the reported reason/detail
   -- an arbitrary but deterministic tie-break, documented here and covered
   by a test, so golden-set output doesn't flap between runs.
6. **Confidence boundary is inclusive**, matching `eligibility.py`'s
   documented precedent: `confidence == settings.evidence_min_conf` exactly is NOT a
   gap (only strictly `<` is).
7. **`EvidenceItem` iteration order.** `evidence_gaps` walks
   `EvidenceItem`'s enum declaration order (not `classified`'s order), so
   the returned gap list has a stable, deterministic order regardless of
   what order attachments happened to be classified in.
8. **No `attachment_id` on `Gap` or `AttachmentClassification`.** The LLM
   schema deliberately does not ask the model to echo back an
   `attachment_id` -- that would be the model inventing/copying an
   identifier it has no reason to get right, rather than data it actually
   derived from the content. A caller that needs "which specific attachment
   was blurry" (vision damage validation on a specific photo may
   want this) must keep its own `attachment_id -> AttachmentClassification`
   mapping alongside the plain list passed to `evidence_gaps` -- that
   mapping is this module's caller's responsibility, not something
   `evidence_gaps` tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from claimpilot.config import settings
from claimpilot.llm import Transport, structured_call
from claimpilot.models import Attachment, EvidenceItem

PROMPT_NAME = "classify_attachment"

GapReason = Literal["MISSING", "UNUSABLE", "LOW_CONFIDENCE"]


class AttachmentClassification(BaseModel):
    """Per-attachment classification output, forced via
    `structured_call(schema=AttachmentClassification, ...)`.

    Note: `model_json_schema()` for this class emits `$defs`/`$ref` for the
    `EvidenceItem` enum field (the first schema of that shape used as a
    `structured_call` tool `input_schema` in this project -- every existing
    `AnthropicTransport` test in `test_llm.py` uses the flat `Person`
    schema). This is valid JSON Schema and Anthropic's tool-use API accepts
    `$ref`, but it hasn't been exercised against the real API yet in this
    codebase; if it were ever rejected, it would surface as a transport
    exception (not a validation retry), since `structured_call` never
    inspects `tool_schema`'s shape itself.
    """

    category: EvidenceItem
    # Bounded so an out-of-range value (e.g. a model returning `70` instead
    # of `0.7`) fails `schema.model_validate` and triggers `structured_call`'s
    # one retry with a corrective message, rather than silently poisoning the
    # `settings.evidence_min_conf` threshold check downstream.
    confidence: float = Field(ge=0.0, le=1.0)
    usable: bool
    quality_issue: str | None = None


@dataclass(frozen=True)
class Gap:
    """One required `EvidenceItem` category that `evidence_gaps` considers
    unsatisfied for a case. `reason` is the machine code a drafter branches
    on; `detail` is the specific, customer-facing text it should use
    verbatim-ish (e.g. "please resend a clearer photo of the outer
    packaging" is built from `detail` like "the photo is too blurry to make
    out the damage"). `detail` is `None` only for `reason="MISSING"`, where
    there is nothing to describe.
    """

    item: EvidenceItem
    reason: GapReason
    detail: str | None = None


def _is_image_content_type(content_type: str | None) -> bool:
    """True for `image/*` content types, and also for missing/empty
    content-type (see module docstring point 1: every real attachment this
    pipeline can fetch today is an image, so an absent content-type is not
    treated as "definitely not an image").
    """
    normalized = (content_type or "").split(";")[0].strip().lower()
    return normalized == "" or normalized.startswith("image/")


async def classify_attachment(
    case_id: str,
    attachment: Attachment,
    content: bytes,
    *,
    transport: Transport | None = None,
    db_path: Path | str | None = None,
) -> AttachmentClassification:
    """Classify one attachment into an `EvidenceItem` category via the LLM.

    Branches on `attachment.content_type`: `image/*` (or missing/empty) goes
    through the vision path (`content` passed as `structured_call(images=...)`);
    anything else (`text/plain`, `application/pdf`, ...) goes through the
    text path (`content` decoded and inlined into `messages`, wrapped in
    `<untrusted_data>` tags since attachment content is external/untrusted
    per the untrusted-data wrapping convention described in `llm.py`).

    `transport`/`db_path` are pass-through overrides to `structured_call`,
    mirroring its own test-injection story (see `tests/test_llm.py`) --
    tests here inject a fake transport rather than mocking this function's
    internals, so the real prompt file and untrusted-data wrapping are
    actually exercised.
    """
    file_name_tag = f"<untrusted_data>{attachment.file_name}</untrusted_data>"

    if _is_image_content_type(attachment.content_type):
        messages = [
            {
                "role": "user",
                "content": (
                    "Classify the attached image per the system instructions. "
                    f"File name: {file_name_tag}"
                ),
            }
        ]
        return await structured_call(
            case_id=case_id,
            prompt_name=PROMPT_NAME,
            messages=messages,
            schema=AttachmentClassification,
            images=[content],
            transport=transport,
            db_path=db_path,
        )

    text = content.decode("utf-8", errors="replace")
    content_type = attachment.content_type or "unknown"
    messages = [
        {
            "role": "user",
            "content": (
                "Classify this text-based attachment per the system "
                f"instructions. File name: {file_name_tag}\n"
                f"Content type: {content_type}\n"
                f"Attachment content:\n<untrusted_data>\n{text}\n</untrusted_data>"
            ),
        }
    ]
    return await structured_call(
        case_id=case_id,
        prompt_name=PROMPT_NAME,
        messages=messages,
        schema=AttachmentClassification,
        transport=transport,
        db_path=db_path,
    )


def _gap_reason_and_detail(classification: AttachmentClassification) -> tuple[GapReason, str]:
    """Derive the `(reason, detail)` for a single classified-but-unsatisfying
    attachment. Only called for instances that are NOT (usable and
    confidence >= settings.evidence_min_conf) -- see `evidence_gaps`.

    `detail` is always customer-safe, drop-into-an-email text (per module
    docstring point 4) -- never an internal diagnostic like a raw confidence
    number. The fallback strings below are deliberately generic rather than
    numeric/technical, since the fallback path is the *normal* case for
    `LOW_CONFIDENCE` (the prompt tells the model to set `quality_issue: null`
    whenever `usable` is `true`), not a rare defensive branch.
    """
    if not classification.usable:
        detail = classification.quality_issue or (
            "the attachment provided couldn't be relied on to show this"
        )
        return "UNUSABLE", detail

    # usable, but below the confidence threshold -- checked after `usable`
    # (module docstring point 4: an unusable-but-confident classification is
    # a more specific complaint than a merely-uncertain one, so it takes
    # precedence when both conditions happen to be true).
    detail = classification.quality_issue or (
        "we couldn't confirm this attachment clearly shows the required evidence"
    )
    return "LOW_CONFIDENCE", detail


def evidence_gaps(classified: list[AttachmentClassification]) -> list[Gap]:
    """Pure function: which of the 4 required `EvidenceItem` categories are
    missing or unusable for a case, given the classifications already
    produced for its attachments.

    No I/O, no LLM calls -- see module docstring points 4-7 for the
    reconciliation/ordering rules this implements.
    """
    gaps: list[Gap] = []
    for item in EvidenceItem:  # enum declaration order, for deterministic output
        matches = [c for c in classified if c.category == item]

        if not matches:
            gaps.append(Gap(item=item, reason="MISSING", detail=None))
            continue

        # Read at call time (not a module-level snapshot) so a
        # `monkeypatch.setattr(settings, "evidence_min_conf", ...)` override
        # in a test -- or a real env var change -- is honored immediately.
        has_good_instance = any(
            c.usable and c.confidence >= settings.evidence_min_conf for c in matches
        )
        if has_good_instance:
            continue

        # None of the instances for this category are good enough -- report
        # based on the first one in caller-supplied order (documented
        # tie-break, module docstring point 5).
        reason, detail = _gap_reason_and_detail(matches[0])
        gaps.append(Gap(item=item, reason=reason, detail=detail))

    return gaps
