"""Span scoring.

Two definitions do the work, and both are chosen for what de-identification
actually requires rather than for convenience:

**Recall counts a gold span as found only if a prediction fully covers it.**
Partial coverage is not a hit. A detector that finds ``Drummond`` inside
``Barnabus Drummond`` has left a given name in the output, and a scoring rule
that awarded that as a true positive would report safety the system does not
have. Partial overlaps are counted separately, because they are a different and
more recoverable failure than missing a span entirely.

**Precision counts a prediction as correct if it overlaps any gold span.** This
is deliberately the lenient direction. Over-predicting is a utility cost, not a
privacy failure, and a detector that flags two extra characters around a name
should not be punished as though it invented the name.

The asymmetry is intentional: strict where a mistake leaks PHI, lenient where a
mistake only costs data.

Typed scores additionally require the predicted PHI category to match. Both are
reported, because they answer different questions -- untyped asks "would this
have been removed?", typed asks "would it have been replaced with the right
kind of surrogate?"
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..corpus.annotation import Annotation
from .detectors import Detection


@dataclass
class Score:
    """Counts and the rates derived from them."""

    gold: int = 0
    predicted: int = 0
    #: Gold spans fully covered by some prediction.
    covered: int = 0
    #: Gold spans overlapped but not fully covered. Still a leak.
    partial: int = 0
    #: Predictions overlapping some gold span.
    useful_predictions: int = 0

    @property
    def missed(self) -> int:
        return self.gold - self.covered

    @property
    def recall(self) -> float:
        return self.covered / self.gold if self.gold else 0.0

    @property
    def precision(self) -> float:
        return self.useful_predictions / self.predicted if self.predicted else 0.0

    @property
    def f1(self) -> float:
        if not (self.recall + self.precision):
            return 0.0
        return 2 * self.recall * self.precision / (self.recall + self.precision)

    @property
    def false_positives(self) -> int:
        return self.predicted - self.useful_predictions

    def add(self, other: "Score") -> None:
        self.gold += other.gold
        self.predicted += other.predicted
        self.covered += other.covered
        self.partial += other.partial
        self.useful_predictions += other.useful_predictions

    def to_dict(self) -> dict:
        return {
            "gold": self.gold,
            "predicted": self.predicted,
            "covered": self.covered,
            "partial": self.partial,
            "missed": self.missed,
            "false_positives": self.false_positives,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class MessageScores:
    """One message's scores, untyped and typed, with breakdowns."""

    untyped: Score = field(default_factory=Score)
    typed: Score = field(default_factory=Score)
    by_location_class: dict[str, Score] = field(default_factory=dict)
    by_phi_type: dict[str, Score] = field(default_factory=dict)
    #: Per-segment scores. These separate a Z-segment the policy already
    #: describes (ZPD) from one it has never seen (ZIN) -- averaging those two
    #: into one "nonstandard" number would flatter any rules-based arm.
    by_segment: dict[str, Score] = field(default_factory=dict)
    #: Gold spans no prediction covered, kept for error analysis.
    misses: list[Annotation] = field(default_factory=list)
    #: Predictions overlapping no gold span.
    spurious: list[Detection] = field(default_factory=list)


def score_message(
    annotations: list[Annotation], detections: list[Detection]
) -> MessageScores:
    """Score one message's predictions against its ground truth."""
    scores = MessageScores()

    scores.untyped.gold = len(annotations)
    scores.untyped.predicted = len(detections)
    scores.typed.gold = len(annotations)
    scores.typed.predicted = len(detections)

    for annotation in annotations:
        location = scores.by_location_class.setdefault(
            annotation.location_class, Score()
        )
        phi = scores.by_phi_type.setdefault(annotation.phi_type, Score())
        segment = scores.by_segment.setdefault(annotation.segment, Score())
        location.gold += 1
        phi.gold += 1
        segment.gold += 1

        covering = [
            d for d in detections if d.covers(annotation.start, annotation.end)
        ]
        overlapping = [
            d for d in detections if d.overlaps(annotation.start, annotation.end)
        ]

        if covering:
            scores.untyped.covered += 1
            location.covered += 1
            phi.covered += 1
            segment.covered += 1
            if any(d.phi_type == annotation.phi_type for d in covering):
                scores.typed.covered += 1
        else:
            scores.misses.append(annotation)
            if overlapping:
                scores.untyped.partial += 1
                location.partial += 1
                phi.partial += 1

    for detection in detections:
        hits = [
            a
            for a in annotations
            if detection.overlaps(a.start, a.end)
        ]
        if hits:
            scores.untyped.useful_predictions += 1
            if any(a.phi_type == detection.phi_type for a in hits):
                scores.typed.useful_predictions += 1
            # Attribute the prediction to the class and type it hit, so
            # per-class precision is meaningful and not just per-class recall.
            for annotation in hits[:1]:
                scores.by_location_class.setdefault(
                    annotation.location_class, Score()
                ).useful_predictions += 1
                scores.by_phi_type.setdefault(
                    annotation.phi_type, Score()
                ).useful_predictions += 1
                scores.by_segment.setdefault(
                    annotation.segment, Score()
                ).useful_predictions += 1
                scores.by_location_class[annotation.location_class].predicted += 1
                scores.by_phi_type[annotation.phi_type].predicted += 1
                scores.by_segment[annotation.segment].predicted += 1
        else:
            scores.spurious.append(detection)

    return scores


@dataclass
class DetectorReport:
    """Aggregate results for one detector across a corpus."""

    detector: str
    messages: int = 0
    untyped: Score = field(default_factory=Score)
    typed: Score = field(default_factory=Score)
    by_location_class: dict[str, Score] = field(default_factory=dict)
    by_phi_type: dict[str, Score] = field(default_factory=dict)
    by_segment: dict[str, Score] = field(default_factory=dict)
    #: Predictions matching no gold span, by PHI type. These are where a
    #: detector's precision goes, and naming them is more useful than a rate.
    spurious_by_phi_type: dict[str, int] = field(default_factory=dict)
    #: Gold spans never covered, by PHI type and location class.
    misses_by_location_class: dict[str, int] = field(default_factory=dict)
    duration_ms: float = 0.0

    def add(self, scores: MessageScores) -> None:
        self.messages += 1
        self.untyped.add(scores.untyped)
        self.typed.add(scores.typed)

        for key, value in scores.by_location_class.items():
            self.by_location_class.setdefault(key, Score()).add(value)
        for key, value in scores.by_phi_type.items():
            self.by_phi_type.setdefault(key, Score()).add(value)
        for key, value in scores.by_segment.items():
            self.by_segment.setdefault(key, Score()).add(value)

        for detection in scores.spurious:
            self.spurious_by_phi_type[detection.phi_type] = (
                self.spurious_by_phi_type.get(detection.phi_type, 0) + 1
            )
        for annotation in scores.misses:
            self.misses_by_location_class[annotation.location_class] = (
                self.misses_by_location_class.get(annotation.location_class, 0) + 1
            )

    def to_dict(self) -> dict:
        return {
            "detector": self.detector,
            "messages": self.messages,
            "duration_ms": round(self.duration_ms, 1),
            "untyped": self.untyped.to_dict(),
            "typed": self.typed.to_dict(),
            "by_location_class": {
                key: value.to_dict()
                for key, value in sorted(self.by_location_class.items())
            },
            "by_phi_type": {
                key: value.to_dict()
                for key, value in sorted(self.by_phi_type.items())
            },
            "by_segment": {
                key: value.to_dict() for key, value in sorted(self.by_segment.items())
            },
            "spurious_by_phi_type": dict(sorted(self.spurious_by_phi_type.items())),
            "misses_by_location_class": dict(
                sorted(self.misses_by_location_class.items())
            ),
        }
