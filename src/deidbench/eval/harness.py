"""Run detectors over a corpus and report.

The report is built to be read by a skeptic. It leads with the location-class
breakdown rather than a single headline number, names the PHI types each arm
misses, and prints the per-arm false-positive counts next to the recall --
because an arm that reaches high recall by flagging half the message has not
solved the problem, and a table that hides that is not evidence.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from ..corpus.annotation import Corpus
from .detectors import Detector
from .metrics import DetectorReport, score_message

LOCATION_CLASS_ORDER = ("structured", "narrative", "nonstandard", "misused")


@dataclass
class EvaluationReport:
    """Results for every arm, plus the corpus they were measured on."""

    corpus_summary: dict = field(default_factory=dict)
    detectors: list[DetectorReport] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "corpus": self.corpus_summary,
            "detectors": [report.to_dict() for report in self.detectors],
            "scoring": {
                "recall": "a gold span counts as found only if a prediction "
                "fully covers it; partial coverage still leaks PHI",
                "precision": "a prediction counts as correct if it overlaps any "
                "gold span; over-prediction costs utility, not privacy",
                "caveat": "synthetic planted-PHI corpus: a floor and a "
                "methodology demonstration, not production measurement",
            },
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def evaluate(corpus: Corpus, detectors: list[Detector]) -> EvaluationReport:
    """Score every detector over every message."""
    report = EvaluationReport(corpus_summary=corpus.summary())

    for detector in detectors:
        detector_report = DetectorReport(detector=detector.name)
        started = time.perf_counter()

        for message in corpus.messages:
            detections = detector.detect(message.text)
            detector_report.add(score_message(message.annotations, detections))

        detector_report.duration_ms = (time.perf_counter() - started) * 1000
        report.detectors.append(detector_report)

    return report


def render(report: EvaluationReport) -> str:
    """A plain-text report. The location-class table is the point of it."""
    lines: list[str] = []
    corpus = report.corpus_summary

    lines.append("CORPUS")
    lines.append(
        f"  {corpus['messages']} messages, {corpus['annotations']} annotated PHI spans"
    )
    classes = corpus.get("by_location_class", {})
    lines.append(
        "  "
        + "  ".join(
            f"{name}={classes.get(name, 0)}" for name in LOCATION_CLASS_ORDER
        )
    )
    lines.append("")

    lines.append("OVERALL (untyped: would this have been removed?)")
    lines.append(
        f"  {'detector':<26} {'recall':>7} {'prec':>7} {'F1':>7} "
        f"{'missed':>7} {'partial':>8} {'FP':>7} {'ms':>8}"
    )
    for detector in report.detectors:
        score = detector.untyped
        lines.append(
            f"  {detector.detector:<26} {score.recall:>7.3f} {score.precision:>7.3f} "
            f"{score.f1:>7.3f} {score.missed:>7} {score.partial:>8} "
            f"{score.false_positives:>7} {detector.duration_ms:>8.0f}"
        )
    lines.append("")

    lines.append("TYPED (would it have been replaced with the right kind of surrogate?)")
    lines.append(f"  {'detector':<26} {'recall':>7} {'prec':>7} {'F1':>7}")
    for detector in report.detectors:
        score = detector.typed
        lines.append(
            f"  {detector.detector:<26} {score.recall:>7.3f} "
            f"{score.precision:>7.3f} {score.f1:>7.3f}"
        )
    lines.append("")

    lines.append("RECALL BY LOCATION CLASS")
    header = f"  {'detector':<26}" + "".join(
        f"{name:>14}" for name in LOCATION_CLASS_ORDER
    )
    lines.append(header)
    for detector in report.detectors:
        row = f"  {detector.detector:<26}"
        for name in LOCATION_CLASS_ORDER:
            score = detector.by_location_class.get(name)
            if score is None or not score.gold:
                row += f"{'-':>14}"
            else:
                row += f"{score.recall:>9.3f}{'':>5}"
        lines.append(row)
    lines.append("")

    lines.append("RECALL BY SEGMENT")
    segments = sorted(
        {name for d in report.detectors for name in d.by_segment},
        key=lambda name: (not name.startswith("Z"), name),
    )
    lines.append(f"  {'detector':<26}" + "".join(f"{name:>8}" for name in segments))
    for detector in report.detectors:
        row = f"  {detector.detector:<26}"
        for name in segments:
            score = detector.by_segment.get(name)
            row += f"{'-':>8}" if score is None or not score.gold else f"{score.recall:>8.2f}"
        lines.append(row)
    lines.append(
        "  ZPD is a Z-segment the shipped policy describes; ZIN is one it has\n"
        "  never seen. The gap between them is how much of the non-standard\n"
        "  problem is solved only by having read the feed first."
    )
    lines.append("")

    lines.append("WORST PHI TYPES BY RECALL")
    for detector in report.detectors:
        ranked = sorted(
            ((name, s) for name, s in detector.by_phi_type.items() if s.gold >= 5),
            key=lambda pair: pair[1].recall,
        )[:6]
        lines.append(f"  {detector.detector}")
        for name, score in ranked:
            lines.append(
                f"    {name:<24} recall {score.recall:.3f}  "
                f"({score.covered}/{score.gold})"
            )
    lines.append("")

    lines.append("TOP FALSE-POSITIVE CATEGORIES")
    for detector in report.detectors:
        ranked = sorted(
            detector.spurious_by_phi_type.items(), key=lambda p: -p[1]
        )[:5]
        if not ranked:
            lines.append(f"  {detector.detector}: none")
            continue
        summary = ", ".join(f"{name}={count}" for name, count in ranked)
        lines.append(f"  {detector.detector}: {summary}")
    lines.append("")

    lines.append(
        "Scoring: recall requires full coverage of a gold span (partial coverage\n"
        "still leaks PHI); precision credits any overlap. Synthetic planted-PHI\n"
        "corpus -- a floor and a methodology demonstration, not production\n"
        "measurement."
    )
    return "\n".join(lines)
