"""Ground-truth annotations and the location classes that make results readable.

A single recall number is a weak claim. "Recall 0.94" could mean a detector
handles everything competently, or that it is perfect on structured fields and
useless on narrative -- and those two systems have completely different
consequences for a privacy officer. Splitting by location class is what turns
the evaluation into an argument:

    structured    0.99    PHI in the field the standard puts it in
    narrative     0.61    PHI in prose: OBX-5, NTE-3, DG1-4
    nonstandard   0.43    PHI in a Z-segment no schema describes
    misused       0.38    PHI in a field not meant to hold it

That shape is the hypothesis the proposal rests on, stated as a measurement.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum


class LocationClass(StrEnum):
    """Where a piece of PHI sits, which predicts who can find it."""

    #: In the field the standard designates for it. Rules handle these.
    STRUCTURED = "structured"
    #: Inside free text. No path-based rule can resolve it.
    NARRATIVE = "narrative"
    #: In a non-standard segment. No schema says what the field means.
    NONSTANDARD = "nonstandard"
    #: In a standard field that is not meant to hold this kind of value --
    #: an SSN in the alternate-ID field, a phone number in a comment. Common
    #: in production and invisible to a policy written from the specification.
    MISUSED = "misused"


@dataclass(frozen=True)
class Annotation:
    """One piece of PHI, located both as characters and as an HL7 path."""

    start: int
    end: int
    text: str
    phi_type: str
    location_class: str
    #: Explicit path of the leaf containing this span, e.g. ``OBX-5[1].1.1``.
    path: str
    #: Same path without repetition indices, as policies are written.
    policy_path: str
    segment: str

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class AnnotatedMessage:
    """A generated message with complete ground truth."""

    message_id: str
    text: str
    annotations: list[Annotation] = field(default_factory=list)
    #: Generation parameters, recorded so a corpus is reproducible and a
    #: reviewer can see which variants were exercised.
    variant: dict = field(default_factory=dict)

    def by_location_class(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for annotation in self.annotations:
            counts[annotation.location_class] = (
                counts.get(annotation.location_class, 0) + 1
            )
        return counts

    def by_phi_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for annotation in self.annotations:
            counts[annotation.phi_type] = counts.get(annotation.phi_type, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "text": self.text,
            "variant": self.variant,
            "annotations": [asdict(a) for a in self.annotations],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AnnotatedMessage":
        return cls(
            message_id=data["message_id"],
            text=data["text"],
            variant=data.get("variant", {}),
            annotations=[Annotation(**a) for a in data["annotations"]],
        )


@dataclass
class Corpus:
    """A set of annotated messages, serializable as one JSON Lines file.

    JSON Lines rather than one blob: a corpus is appended to, streamed and
    diffed, and a 10,000-message single JSON document is none of those.
    """

    messages: list[AnnotatedMessage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.messages)

    @property
    def annotation_count(self) -> int:
        return sum(len(m.annotations) for m in self.messages)

    def totals_by_location_class(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for message in self.messages:
            for key, value in message.by_location_class().items():
                counts[key] = counts.get(key, 0) + value
        return counts

    def totals_by_phi_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for message in self.messages:
            for key, value in message.by_phi_type().items():
                counts[key] = counts.get(key, 0) + value
        return counts

    def to_jsonl(self) -> str:
        lines = [json.dumps({"_metadata": self.metadata}, sort_keys=True)]
        lines.extend(
            json.dumps(message.to_dict(), sort_keys=True) for message in self.messages
        )
        return "\n".join(lines) + "\n"

    @classmethod
    def from_jsonl(cls, text: str) -> "Corpus":
        corpus = cls()
        for line in text.splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            if "_metadata" in data:
                corpus.metadata = data["_metadata"]
                continue
            corpus.messages.append(AnnotatedMessage.from_dict(data))
        return corpus

    def summary(self) -> dict:
        return {
            "messages": len(self.messages),
            "annotations": self.annotation_count,
            "by_location_class": self.totals_by_location_class(),
            "by_phi_type": self.totals_by_phi_type(),
            "metadata": self.metadata,
        }
