"""Detector arms, span scoring and the evaluation harness."""

from .detectors import (
    Detection,
    Detector,
    FlattenedTextDetector,
    NarrativeCandidate,
    NarrativeDetector,
    RegexNarrativeDetector,
    RulesOnlyDetector,
    SchemaConditionedDetector,
)
from .harness import EvaluationReport, evaluate, render
from .metrics import DetectorReport, MessageScores, Score, score_message

__all__ = [
    "Detection",
    "Detector",
    "DetectorReport",
    "EvaluationReport",
    "FlattenedTextDetector",
    "MessageScores",
    "NarrativeCandidate",
    "NarrativeDetector",
    "RegexNarrativeDetector",
    "RulesOnlyDetector",
    "SchemaConditionedDetector",
    "Score",
    "evaluate",
    "render",
    "score_message",
]
