"""Can a small local model say WHAT an unrecognised field contains?

A feasibility spike, not a baseline. Nothing here is published, scored in
``results/``, or imported by the package. It exists to produce one number.

THE QUESTION
------------
Three published tools score 0.000, 0.023 and 0.143 on *typed* recall: they can
often tell you something is PHI, but not what kind. Typing is what drives
format- and length-preserving surrogate replacement, so it is the capability
that decides whether a de-identified message survives in a clinically usable
form. This asks whether a small open-weight model running locally can supply it.

WHY A FALLBACK DETECTOR, AND NOT JUST A NARRATIVE ONE
-----------------------------------------------------
``SchemaConditionedDetector.candidates()`` contains::

    if rule is None or rule.action is not Action.DEFER_TO_MODEL:
        continue

When the policy has no rule for a location -- an unseen Z-segment, say -- the
span is *skipped*. It is not deferred to the narrative detector; it is dropped.
That single line is the mechanical cause of the measured 0.00 recall on ZIN,
where a crude regex over the same flattened text scores 0.94.

:class:`HybridFallbackDetector` changes exactly one thing: when ``rule is
None``, build a candidate anyway and send it to the model. Everything else --
the rules for recognised locations, the deferral for narrative ones -- is
inherited unchanged, so any difference in the score is attributable to the
fallback.

THE INDEPENDENT VARIABLE
------------------------
``NarrativeCandidate.context_line()`` is documented upstream as "the
independent variable of the Phase I experiment: the same span offered to a
model with and without this description attached". :class:`OllamaTypingDetector`
takes ``use_context`` so both sides of that comparison run through identical
code, differing only in whether the structural description is in the prompt.

SCOPE
-----
No fine-tuning -- that is what a Phase I would pay for. No engine integration.
No tests. One script, one number. Building a working v1 here would delete the
open research question the proposal rests on.

Usage::

    python experiments/ollama_typing.py --help
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deidbench import hl7  # noqa: E402
from deidbench.eval.detectors import (  # noqa: E402
    Detection,
    NarrativeCandidate,
    SchemaConditionedDetector,
    _resolve_overlaps,
    _rule_for,
)
from deidbench.surrogate.types import Action, PhiType  # noqa: E402

DEFAULT_HOST = "http://mini.local:11434"
DEFAULT_MODEL = "qwen3:14b"
TIMEOUT_SECONDS = 180

#: The model may only answer with these. Giving it the closed set is what stops
#: it inventing a category, or -- as the first smoke test did -- echoing the
#: context description back as if it were a type.
ALLOWED_TYPES = tuple(t.value for t in PhiType)

PROMPT = """\
You label protected health information (PHI) inside a single field taken from \
an HL7 v2 healthcare message.

{context}Field content:
<<<{text}>>>

List every span of PHI in the field content above.

Rules:
- Return ONLY a JSON array. No prose, no markdown fences, no explanation.
- One object per identifier. Never group several identifiers into one span.
- "text" MUST be copied character-for-character from the field content, and \
must be the shortest span that covers the identifier.
- "type" MUST be exactly one of: {types}
- If the field contains no PHI, return [].

Example of the required shape:
[{{"text":"Barnabus Drummond","type":"family_name"}},\
{{"text":"716-555-0142","type":"phone"}}]
"""


@dataclass
class Usage:
    """Enough to report throughput honestly, and nothing more."""

    calls: int = 0
    failures: int = 0
    unparseable: int = 0
    total_ns: int = 0
    eval_tokens: int = 0

    @property
    def seconds(self) -> float:
        return self.total_ns / 1e9

    def summary(self) -> str:
        per = self.seconds / self.calls if self.calls else 0.0
        return (
            f"{self.calls} calls, {self.failures} failed, "
            f"{self.unparseable} unparseable, "
            f"{self.seconds:.1f}s total, {per:.2f}s/call, "
            f"{self.eval_tokens} output tokens"
        )


def _strip_fences(raw: str) -> str:
    """Models wrap JSON in ```json fences despite being told not to."""
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    start, end = s.find("["), s.rfind("]")
    return s[start : end + 1] if start != -1 and end > start else s


def _locate(needle: str, haystack: str, claimed: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Find `needle` in `haystack`, skipping offsets already claimed.

    The model returns text, not offsets. A field may legitimately repeat a
    value, so each match is consumed once rather than every occurrence being
    attributed to the first hit.
    """
    if not needle:
        return None
    pos = 0
    while True:
        i = haystack.find(needle, pos)
        if i == -1:
            return None
        span = (i, i + len(needle))
        if not any(span[0] < c[1] and c[0] < span[1] for c in claimed):
            return span
        pos = i + 1


class OllamaTypingDetector:
    """A local open-weight model behind the narrative/fallback seam.

    ``use_context`` is the experiment. With it False the model sees only the
    field's raw text; with it True the same model sees the same text plus
    ``candidate.context_line()``. Nothing else differs.
    """

    #: How the structural description is worded. The Oct 28 sweep compared only
    #: ``none`` against ``terse`` and found no measurable difference -- which is
    #: a statement about *that wording*, not about schema context as an idea.
    #: ``terse`` renders a fallback span as
    #: ``segment=ZIN, field=3, description=no policy rule for this location``:
    #: key-value noise whose only content announces that nothing is known.
    CONTEXT_STYLES = ("none", "terse", "plain", "domain")

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        use_context: bool = True,
        context_style: str = "terse",
        temperature: float = 0.0,
        seed: int = 20261013,
        hardware: str | None = None,
    ):
        style = context_style if use_context else "none"
        if style not in self.CONTEXT_STYLES:
            raise ValueError(f"context_style must be one of {self.CONTEXT_STYLES}")
        self.model = model
        self.host = host.rstrip("/")
        #: Free text describing the Ollama host, e.g. "Mac mini M2 Pro, 48 GB".
        #: Falls back to $DEID_BENCH_HARDWARE so a run can set it once in the
        #: environment rather than threading it through every constructor.
        self.hardware = hardware or os.environ.get("DEID_BENCH_HARDWARE") or "unrecorded"
        self.context_style = style
        self.use_context = style != "none"
        self.temperature = temperature
        self.seed = seed
        self.usage = Usage()
        self.name = f"ollama:{model}+{style}"

    # -- the independent variable ---------------------------------------

    def _context(self, candidate: NarrativeCandidate) -> str:
        """Build the structural description in the configured wording.

        Deliberately built here rather than by extending
        ``NarrativeCandidate.context_line()``: the package is the published
        instrument and this is an experiment on prompt design, not a change to
        what the benchmark measures.
        """
        style = self.context_style
        if style == "none":
            return ""

        if style == "terse":
            return f"Structural context of this field: {candidate.context_line()}\n\n"

        seg, fld = candidate.segment, candidate.field
        where = f"field {fld} of the {seg} segment"
        if candidate.component is not None:
            where = f"component {candidate.component} of field {fld} of the {seg} segment"

        if style == "plain":
            s = f"This value was taken from {where} of an HL7 v2 message."
            if candidate.data_type:
                s += f" Its HL7 data type is {candidate.data_type}."
            if candidate.note and "no policy rule" not in candidate.note:
                s += f" That location holds: {candidate.note.strip()}."
            return s + "\n\n"

        # style == "domain": say what the structure does and does not tell you,
        # and in the fallback case say so explicitly rather than emitting a
        # description whose content is "there is no description".
        s = f"This value was taken from {where} of an HL7 v2 message."
        if candidate.data_type:
            s += f" The HL7 specification gives this location the data type {candidate.data_type}"
            if candidate.note and "no policy rule" not in candidate.note:
                s += f", holding {candidate.note.strip()}"
            s += "."
        elif seg.upper().startswith("Z"):
            s += (
                f" {seg} is a site-defined Z-segment and is not part of the HL7"
                " standard, so the specification says nothing about what this"
                " field contains. Judge the value on its own terms."
            )
        else:
            s += (
                " The specification has no entry for this location, so its"
                " contents cannot be inferred from the structure. Judge the"
                " value on its own terms."
            )
        return s + "\n\n"

    @property
    def provenance(self) -> dict:
        """What a reader needs to reproduce this arm.

        Hardware is included because the throughput figures are otherwise
        unreproducible: a host name says nothing about how fast the box is,
        and s/call is the one number here that is a property of the machine
        rather than of the model. It describes the host running Ollama, which
        is not necessarily the machine running this script.
        """
        return {
            "detector": "ollama-typing-spike",
            "model": self.model,
            "host": self.host,
            "hardware": self.hardware,
            "use_context": self.use_context,
            "context_style": self.context_style,
            "temperature": self.temperature,
            "seed": self.seed,
        }

    # -- the model call ------------------------------------------------

    def _generate(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                # qwen3 reasons by default; the reasoning is not wanted here and
                # costs seconds per call.
                "think": False,
                "options": {
                    "temperature": self.temperature,
                    "seed": self.seed,
                    "num_predict": 512,
                },
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            payload = json.load(resp)
        self.usage.calls += 1
        self.usage.total_ns += int(payload.get("total_duration", 0))
        self.usage.eval_tokens += int(payload.get("eval_count", 0))
        return payload.get("response", "")

    # -- the NarrativeDetector protocol --------------------------------

    def detect_in(self, candidate: NarrativeCandidate) -> list[tuple[int, int, str]]:
        text = candidate.text
        if not text.strip():
            return []

        context = self._context(candidate)

        prompt = PROMPT.format(
            context=context, text=text, types=", ".join(ALLOWED_TYPES)
        )

        try:
            raw = self._generate(prompt)
        except (urllib.error.URLError, TimeoutError, OSError):
            self.usage.failures += 1
            return []

        try:
            items = json.loads(_strip_fences(raw))
        except (ValueError, TypeError):
            self.usage.unparseable += 1
            return []
        if not isinstance(items, list):
            self.usage.unparseable += 1
            return []

        found: list[tuple[int, int, str]] = []
        claimed: list[tuple[int, int]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            span_text = str(item.get("text") or "")
            phi_type = str(item.get("type") or "").strip().lower()
            if phi_type not in ALLOWED_TYPES:
                # An invented category is a miss, not a crash. Counting it as
                # untyped would flatter the typed score this spike exists to
                # measure.
                continue
            located = _locate(span_text, text, claimed)
            if located is None:
                continue
            claimed.append(located)
            found.append((located[0], located[1], phi_type))

        # Stable order so repeated runs produce identical output.
        found.sort(key=lambda d: (d[0], d[1], d[2]))
        return found


class HybridFallbackDetector(SchemaConditionedDetector):
    """Schema conditioning, plus a fallback for structure it does not recognise.

    The one behavioural change from the parent: a span whose policy lookup
    returns ``None`` is routed to the detector instead of being dropped. That is
    the whole of Aim 2's architectural proposal, and the only difference the
    measurement can be attributing anything to.
    """

    def __init__(self, *args, **kwargs):
        name = kwargs.pop("name", None)
        super().__init__(*args, **kwargs)
        self.name = name or f"hybrid+{self.narrative_detector.name}"
        self.fallback_locations = 0

    def fallback_candidates(self, text: str) -> list[NarrativeCandidate]:
        """Only the locations the policy has NO rule for.

        Deliberately not every candidate. ``SchemaConditionedDetector.detect``
        already calls the detector on ``DEFER_TO_MODEL`` spans, so returning
        those here too would invoke the model twice on the same text -- double
        the cost, and two chances to produce the same span under different
        types.
        """
        message = hl7.parse(text)
        _, spans = hl7.serialize_with_spans(message)
        result: list[NarrativeCandidate] = []

        for span in spans:
            if not span.text.strip():
                continue
            if _rule_for(span, self._index) is not None:
                continue
            self.fallback_locations += 1
            result.append(
                NarrativeCandidate(
                    text=span.text, start=span.start, end=span.end,
                    path=span.path, policy_path=span.policy_path,
                    segment=span.segment, field=span.field,
                    component=span.component,
                    # No rule means no schema knowledge: no data_type, no note.
                    # The model gets only the message's own shape.
                    data_type="", note="no policy rule for this location",
                )
            )
        return result

    def detect(self, text: str) -> list[Detection]:
        detections = list(super().detect(text))

        for cand in self.fallback_candidates(text):
            for local_start, local_end, phi_type in (
                self.narrative_detector.detect_in(cand)
            ):
                detections.append(
                    Detection(
                        start=cand.start + local_start,
                        end=cand.start + local_end,
                        text=cand.text[local_start:local_end],
                        phi_type=phi_type,
                        source=f"{self.narrative_detector.name}/fallback",
                    )
                )
        return _resolve_overlaps(detections)


WHOLE_MESSAGE_PROMPT = """\
You label protected health information (PHI) inside a complete HL7 v2 \
healthcare message.

Message:
<<<{text}>>>

List every span of PHI anywhere in the message above.

Rules:
- Return ONLY a JSON array. No prose, no markdown fences, no explanation.
- One object per identifier. Never group several identifiers into one span.
- "text" MUST be copied character-for-character from the message, and must be \
the shortest span that covers the identifier.
- "type" MUST be exactly one of: {types}
- Do NOT label segment names, field delimiters, codes, or structural tokens.
- If the message contains no PHI, return [].

Example of the required shape:
[{{"text":"Barnabus Drummond","type":"family_name"}},\
{{"text":"716-555-0142","type":"phone"}}]
"""


class WholeMessageDetector(OllamaTypingDetector):
    """No schema at all: hand the model the entire message in one call.

    The obvious question a reviewer asks about the hybrid design -- if telling
    the model where a field sits makes it worse, why scope it to fields at all?
    This is that arm, and it was missing.

    Two things it trades. It gives up the rules' exact behaviour on structured
    fields, which are 1,002 of the corpus's 1,275 spans and where rules score
    1.000 precision. And it makes offset attribution ambiguous: the model
    returns text, and a value appearing five times in a message has to be
    matched to the right occurrence rather than the only one.

    What it buys is throughput -- one call per message rather than one per
    field, roughly 36x fewer on this corpus.
    """

    #: A whole message holds far more PHI than one field, so the field-sized
    #: output budget truncates the JSON array and every span after the cut is
    #: silently lost.
    NUM_PREDICT = 4096

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("context_style", "none")
        super().__init__(*args, **kwargs)
        self.name = f"whole-message:{self.model}"

    def _generate(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": self.temperature,
                    "seed": self.seed,
                    "num_predict": self.NUM_PREDICT,
                },
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            payload = json.load(resp)
        self.usage.calls += 1
        self.usage.total_ns += int(payload.get("total_duration", 0))
        self.usage.eval_tokens += int(payload.get("eval_count", 0))
        return payload.get("response", "")

    def detect(self, text: str) -> list[Detection]:
        prompt = WHOLE_MESSAGE_PROMPT.format(
            text=text, types=", ".join(ALLOWED_TYPES)
        )
        try:
            raw = self._generate(prompt)
        except (urllib.error.URLError, TimeoutError, OSError):
            self.usage.failures += 1
            return []

        try:
            items = json.loads(_strip_fences(raw))
        except (ValueError, TypeError):
            self.usage.unparseable += 1
            return []
        if not isinstance(items, list):
            self.usage.unparseable += 1
            return []

        detections: list[Detection] = []
        claimed: list[tuple[int, int]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            span_text = str(item.get("text") or "")
            phi_type = str(item.get("type") or "").strip().lower()
            if phi_type not in ALLOWED_TYPES:
                continue
            located = _locate(span_text, text, claimed)
            if located is None:
                continue
            claimed.append(located)
            detections.append(
                Detection(start=located[0], end=located[1],
                          text=text[located[0]:located[1]],
                          phi_type=phi_type, source=self.name)
            )
        return _resolve_overlaps(detections)
