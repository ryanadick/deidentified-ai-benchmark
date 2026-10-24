"""The detector arms being compared.

Three arms, matching the comparison the proposal commits to:

1. ``RulesOnlyDetector`` -- the policy, nothing else. Should be near-perfect on
   structured fields and score zero on narrative, by construction rather than
   by weakness: the policy marks narrative ``defer_to_model`` and this arm has
   no model to defer to.

2. ``FlattenedTextDetector`` -- the message handed to a detector as a text blob,
   which is what every free-text de-identification tool does. Its failure mode
   is the interesting part: it finds names in prose, where linguistic context
   exists, and misses them in ``Drummond^Barnabus``, where the only thing
   marking a name as a name is the field it sits in.

3. ``SchemaConditionedDetector`` -- the policy for structured locations plus a
   narrative detector applied *only* to the locations rules cannot resolve,
   with the structural context of each location available to it. This is the
   architecture the Phase I hypothesis is about.

**On the text-only baseline.** Microsoft Presidio, Philter and NLM Scrubber are
the comparators that belong in a paper, and this is not one of them. It is a
transparent stand-in whose rules are written out below so the comparison can be
read and criticized rather than taken on faith. Its name heuristic requires
either a title cue or two space-separated capitalized words -- a deliberate,
stated choice that models how NER behaves on flattened HL7, and the single
assumption most worth challenging when the real baselines are wired in.

``NarrativeDetector`` is the seam the local open-weight model plugs into. The
regex implementation here lets the harness produce complete three-arm numbers
today; swapping in the model changes one constructor argument.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .. import hl7
from ..hl7.spans import LeafSpan, candidate_policy_paths
from ..policy.registry import Policy, Rule, load_default_policy
from ..surrogate.types import Action, PhiType


@dataclass(frozen=True)
class Detection:
    """One predicted PHI span."""

    start: int
    end: int
    text: str
    phi_type: str
    #: Which arm produced it, for error analysis.
    source: str = ""

    def covers(self, start: int, end: int) -> bool:
        return self.start <= start and end <= self.end

    def overlaps(self, start: int, end: int) -> bool:
        return start < self.end and end > self.start


@runtime_checkable
class Detector(Protocol):
    name: str

    def detect(self, text: str) -> list[Detection]: ...


# -- arm 1: rules only --------------------------------------------------


class RulesOnlyDetector:
    """Policy-driven detection. Finds exactly what the rules address.

    Walks leaf spans rather than resolving each rule's path, which mirrors how
    the de-identifier actually works and means a leaf governed by a field-level
    rule is handled without special-casing.
    """

    name = "rules-only"

    def __init__(self, policy: Policy | None = None):
        self.policy = policy or load_default_policy()
        self._index = _rule_index(self.policy)

    def detect(self, text: str) -> list[Detection]:
        message = hl7.parse(text)
        _, spans = hl7.serialize_with_spans(message)
        detections: list[Detection] = []

        for span in spans:
            if not span.text.strip():
                continue
            rule = _rule_for(span, self._index)
            if rule is None or not rule.is_write:
                continue
            detections.append(
                Detection(
                    start=span.start,
                    end=span.end,
                    text=span.text,
                    phi_type=str(rule.phi_type),
                    source=self.name,
                )
            )
        return detections


# -- arm 2: flattened text ---------------------------------------------


#: Patterns a text-only detector can apply without any schema. Order matters
#: only for reporting; overlaps are resolved longest-first by the caller.
TEXT_PATTERNS: tuple[tuple[str, str], ...] = (
    (PhiType.EMAIL, r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    (PhiType.SSN, r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    (PhiType.PHONE, r"\(\d{3}\)\d{3}-\d{4}"),
    (PhiType.PHONE, r"(?<!\d)\d{3}-\d{3}-\d{4}(?!\d)"),
    (PhiType.BIRTH_DATE, r"(?<!\d)\d{1,2}/\d{1,2}/\d{4}(?!\d)"),
    (PhiType.EVENT_DATE, r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)"),
    (PhiType.EVENT_DATE, r"(?<!\d)\d{8}(?!\d)"),
    (PhiType.STREET_ADDRESS, r"\b\d{1,5}\s+[A-Z][a-z]+\s+(?:Street|Avenue|Road|Lane|Court|Rd|Ave|St|Ln|Ct)\b"),
    (PhiType.MEDICAL_RECORD_NUMBER, r"\b[A-Z]{1,3}\d{5,9}\b"),
    # Ten consecutive digits: a bare phone number, but also any long numeric
    # identifier. The ambiguity is the point -- it is what a detector without
    # a schema actually faces.
    (PhiType.PHONE, r"(?<!\d)\d{10}(?!\d)"),
    (PhiType.POSTAL_CODE, r"(?<!\d)\d{5}(?!\d)"),
)

#: Names, by linguistic cue only. Two space-separated capitalized words, or a
#: title followed by a capitalized word. Nothing here can fire on
#: ``Drummond^Barnabus``, because a caret is not a linguistic cue -- which is
#: precisely the gap structure-aware detection is meant to close.
TEXT_NAME_PATTERNS: tuple[str, ...] = (
    r"\b(?:Mr|Mrs|Ms|Mx|Dr|Prof)\.?\s+[A-Z][a-z]{2,}\b",
    r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b",
    r"\b[A-Z][a-z]{2,}'s\b",
)

#: Capitalized words that are HL7 furniture or clinical vocabulary rather than
#: names. Without this the baseline is a straw man, firing on every segment
#: name and observation label in the message.
TEXT_NAME_STOPWORDS = frozenset(
    {
        "Progress",
        "Note",
        "Contact",
        "Birth",
        "Reference",
        "Penicillin",
        "Patient",
        "Chronic",
        "Discussed",
        "Called",
        "Spoke",
        "Reviewed",
        "Left",
        "Second",
        "Prior",
        "Home",
        "Best",
        "Transport",
        "Driver",
        "Daughter",
        "Follow",
        "Referred",
        "Reaction",
        "Insurance",
        "Address",
        "Chart",
        "Account",
        "Street",
        "Avenue",
        "Road",
        "Lane",
        "Court",
        "Memorial",
        "Hospital",
        "Regional",
        "Medical",
        "Center",
        "Community",
        "Clinic",
        "County",
        "Apt",
        "Suite",
        "Unit",
        "Per",
        "Onset",
        "Health",
        "Care",
        "Proxy",
        "Holds",
        "Managed",
        "Since",
        "Attending",
        "Requests",
        "Correspondence",
        "Verified",
        "Located",
        "Under",
        "Booked",
        "From",
        "Scheduled",
        "Confirmed",
        "Declined",
        "Translated",
        "Interpreter",
        "Arranged",
        "Presented",
        "Accompanied",
        "Family",
        "Member",
        "Discharge",
        "Planning",
        "Underway",
        "Number",
        "File",
        "Alternate",
        "Today",
        "Visit",
        "Protocol",
        "Condition",
        "Documented",
        "Childhood",
        "Eldest",
        "Physician",
        "Referring",
        "Following",
        "Tuesday",
        "Attempt",
        "Message",
        "Record",
        "License",
        "Present",
        "Will",
        "The",
        "Reached",
        "Lives",
        "Seen",
        "Rather",
        "Than",
        "Home",
        "All",
        "And",
        "With",
        "Was",
        "She",
        "His",
        "Her",
    }
)


class FlattenedTextDetector:
    """The text-only baseline: the message as a blob, no schema at all."""

    name = "text-only"

    def detect(self, text: str) -> list[Detection]:
        detections: list[Detection] = []

        for phi_type, pattern in TEXT_PATTERNS:
            for match in re.finditer(pattern, text):
                detections.append(
                    Detection(
                        start=match.start(),
                        end=match.end(),
                        text=match.group(0),
                        phi_type=str(phi_type),
                        source=self.name,
                    )
                )

        for pattern in TEXT_NAME_PATTERNS:
            for match in re.finditer(pattern, text):
                surface = match.group(0)
                if any(word in TEXT_NAME_STOPWORDS for word in _words(surface)):
                    continue
                detections.append(
                    Detection(
                        start=match.start(),
                        end=match.end(),
                        text=surface,
                        phi_type=str(PhiType.FAMILY_NAME),
                        source=self.name,
                    )
                )

        return _resolve_overlaps(detections)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text)


# -- arm 3: schema-conditioned -----------------------------------------


@dataclass
class NarrativeCandidate:
    """A location rules could not resolve, with its structural context.

    The context is the independent variable of the Phase I experiment: the same
    span offered to a model with and without this description attached.
    """

    text: str
    start: int
    end: int
    path: str
    policy_path: str
    segment: str
    field: int | None
    component: int | None
    data_type: str
    note: str

    def context_line(self) -> str:
        """The structural description handed to a model as a prompt prefix."""
        parts = [
            f"segment={self.segment}",
            f"field={self.field}",
        ]
        if self.component is not None:
            parts.append(f"component={self.component}")
        if self.data_type:
            parts.append(f"data_type={self.data_type}")
        if self.note:
            parts.append(f"description={self.note.strip()}")
        return ", ".join(parts)


@runtime_checkable
class NarrativeDetector(Protocol):
    """The seam the local open-weight model plugs into.

    Offsets returned must be *relative to the candidate text*; the caller
    translates them into message offsets, so an implementation never has to
    know where in the message it is.
    """

    name: str

    def detect_in(self, candidate: NarrativeCandidate) -> list[tuple[int, int, str]]: ...


class RegexNarrativeDetector:
    """Placeholder narrative detector, so the third arm runs today.

    It applies the same patterns as the text-only arm, but only inside
    locations the policy flagged as narrative. That restriction alone should
    improve precision substantially, since the patterns never see the coded
    fields that generate most of the text-only arm's false positives.

    It is **not** a model and makes no claim to be one. Its purpose is to prove
    the harness end to end and establish the floor the model has to beat.
    """

    name = "regex-narrative"

    def detect_in(self, candidate: NarrativeCandidate) -> list[tuple[int, int, str]]:
        found: list[tuple[int, int, str]] = []
        text = candidate.text

        for phi_type, pattern in TEXT_PATTERNS:
            for match in re.finditer(pattern, text):
                found.append((match.start(), match.end(), str(phi_type)))

        for pattern in TEXT_NAME_PATTERNS:
            for match in re.finditer(pattern, text):
                if any(word in TEXT_NAME_STOPWORDS for word in _words(match.group(0))):
                    continue
                found.append((match.start(), match.end(), str(PhiType.FAMILY_NAME)))

        return found


class SchemaConditionedDetector:
    """Rules for structured locations, a narrative detector for the rest.

    The division of labour is the design claim: rules handle what the schema
    determines, and the model is confined to the content rules cannot resolve,
    which is what keeps it fast enough to run on one on-premises server.
    """

    def __init__(
        self,
        narrative_detector: NarrativeDetector | None = None,
        policy: Policy | None = None,
        name: str | None = None,
    ):
        self.policy = policy or load_default_policy()
        self._index = _rule_index(self.policy)
        self.narrative_detector = narrative_detector or RegexNarrativeDetector()
        self.name = name or f"schema+{self.narrative_detector.name}"

    def candidates(self, text: str) -> list[NarrativeCandidate]:
        """Locations the policy defers, with structural context attached."""
        message = hl7.parse(text)
        _, spans = hl7.serialize_with_spans(message)
        result: list[NarrativeCandidate] = []

        for span in spans:
            if not span.text.strip():
                continue
            rule = _rule_for(span, self._index)
            if rule is None or rule.action is not Action.DEFER_TO_MODEL:
                continue
            result.append(
                NarrativeCandidate(
                    text=span.text,
                    start=span.start,
                    end=span.end,
                    path=span.path,
                    policy_path=span.policy_path,
                    segment=span.segment,
                    field=span.field,
                    component=span.component,
                    data_type=rule.data_type,
                    note=rule.note,
                )
            )
        return result

    def detect(self, text: str) -> list[Detection]:
        message = hl7.parse(text)
        _, spans = hl7.serialize_with_spans(message)
        detections: list[Detection] = []

        for span in spans:
            if not span.text.strip():
                continue
            rule = _rule_for(span, self._index)
            if rule is None:
                continue

            if rule.is_write:
                detections.append(
                    Detection(
                        start=span.start,
                        end=span.end,
                        text=span.text,
                        phi_type=str(rule.phi_type),
                        source="rules",
                    )
                )
            elif rule.action is Action.DEFER_TO_MODEL:
                candidate = NarrativeCandidate(
                    text=span.text,
                    start=span.start,
                    end=span.end,
                    path=span.path,
                    policy_path=span.policy_path,
                    segment=span.segment,
                    field=span.field,
                    component=span.component,
                    data_type=rule.data_type,
                    note=rule.note,
                )
                for local_start, local_end, phi_type in self.narrative_detector.detect_in(
                    candidate
                ):
                    detections.append(
                        Detection(
                            start=span.start + local_start,
                            end=span.start + local_end,
                            text=span.text[local_start:local_end],
                            phi_type=phi_type,
                            source=self.narrative_detector.name,
                        )
                    )

        return _resolve_overlaps(detections)


# -- shared -------------------------------------------------------------


def _rule_index(policy: Policy) -> dict[str, Rule]:
    return {str(rule.path): rule for rule in policy.rules}


def _rule_for(span: LeafSpan, index: dict[str, Rule]) -> Rule | None:
    """The most specific rule governing a leaf, or None."""
    for candidate in candidate_policy_paths(span.policy_path):
        rule = index.get(candidate)
        if rule is not None:
            return rule
    return None


def _resolve_overlaps(detections: list[Detection]) -> list[Detection]:
    """Keep the longest of any overlapping predictions.

    A detector that reports both ``Barnabus Drummond`` and ``Drummond`` has
    found one thing, not two, and counting it twice would distort precision.
    """
    ordered = sorted(
        detections, key=lambda d: (d.end - d.start, -d.start), reverse=True
    )
    kept: list[Detection] = []
    for detection in ordered:
        if any(k.overlaps(detection.start, detection.end) for k in kept):
            continue
        kept.append(detection)
    return sorted(kept, key=lambda d: d.start)
