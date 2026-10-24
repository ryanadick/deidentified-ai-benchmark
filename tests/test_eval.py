"""Scoring semantics and the behaviour each detector arm is supposed to have.

The scoring tests matter more than they look. Recall and precision here are
defined asymmetrically on purpose -- strict where a mistake leaks PHI, lenient
where it only costs utility -- and if those definitions drift, every reported
number changes meaning without any test failing.
"""

from __future__ import annotations

import pytest

from deidbench.corpus import generate_corpus, GenerationOptions
from deidbench.corpus.annotation import Annotation
from deidbench.eval.detectors import (
    Detection,
    FlattenedTextDetector,
    RegexNarrativeDetector,
    RulesOnlyDetector,
    SchemaConditionedDetector,
    _resolve_overlaps,
)
from deidbench.eval.harness import evaluate, render
from deidbench.eval.metrics import score_message

CORPUS = generate_corpus(GenerationOptions(messages=60))


def gold(start: int, end: int, text: str, phi_type: str = "family_name") -> Annotation:
    return Annotation(
        start=start,
        end=end,
        text=text,
        phi_type=phi_type,
        location_class="structured",
        path="PID-5[1].1.1",
        policy_path="PID-5.1.1",
        segment="PID",
    )


def predicted(start: int, end: int, text: str, phi_type: str = "family_name") -> Detection:
    return Detection(start=start, end=end, text=text, phi_type=phi_type)


# -- scoring semantics --------------------------------------------------


def test_full_coverage_counts_as_found() -> None:
    scores = score_message([gold(10, 18, "Vanbrugh")], [predicted(10, 18, "Vanbrugh")])
    assert scores.untyped.covered == 1
    assert scores.untyped.recall == 1.0


def test_a_wider_prediction_still_counts_as_found() -> None:
    """Flagging extra characters costs utility, not privacy."""
    scores = score_message([gold(10, 18, "Vanbrugh")], [predicted(8, 20, "  Vanbrugh  ")])
    assert scores.untyped.covered == 1
    assert scores.untyped.precision == 1.0


def test_partial_coverage_is_not_a_true_positive() -> None:
    """A detector that finds "brugh" inside "Vanbrugh" has left PHI in the output.

    Scoring this as a hit would report safety the system does not have.
    """
    scores = score_message([gold(10, 18, "Vanbrugh")], [predicted(13, 18, "brugh")])
    assert scores.untyped.covered == 0
    assert scores.untyped.partial == 1
    assert scores.untyped.recall == 0.0
    assert scores.misses


def test_a_prediction_matching_nothing_is_a_false_positive() -> None:
    scores = score_message([gold(10, 18, "Vanbrugh")], [predicted(40, 48, "Progress")])
    assert scores.untyped.false_positives == 1
    assert scores.untyped.precision == 0.0
    assert scores.spurious


def test_typed_scoring_requires_the_category_to_match() -> None:
    """Untyped asks if it would be removed; typed asks if it would be replaced
    with the right kind of surrogate."""
    scores = score_message(
        [gold(10, 18, "Vanbrugh", "family_name")],
        [predicted(10, 18, "Vanbrugh", "city")],
    )
    assert scores.untyped.covered == 1
    assert scores.typed.covered == 0


def test_empty_ground_truth_and_predictions_do_not_divide_by_zero() -> None:
    scores = score_message([], [])
    assert scores.untyped.recall == 0.0
    assert scores.untyped.precision == 0.0
    assert scores.untyped.f1 == 0.0


def test_overlapping_predictions_collapse_to_the_longest() -> None:
    """Reporting both "Bartram Vanbrugh" and "Vanbrugh" is one find, not two."""
    kept = _resolve_overlaps(
        [predicted(0, 16, "Bartram Vanbrugh"), predicted(8, 16, "Vanbrugh")]
    )
    assert len(kept) == 1
    assert kept[0].text == "Bartram Vanbrugh"


# -- arm behaviour ------------------------------------------------------


def test_rules_are_perfect_on_structured_and_blind_to_narrative() -> None:
    """Both halves are by construction, which is the point of the comparison.

    Note the structured number is flattered by the corpus: the policy and the
    generator were written from the same reading of HL7, so the rules are being
    tested on the fields they were authored against. Production feeds put PHI
    in fields no specification predicts -- which is what the misused and ZIN
    categories exist to approximate.
    """
    report = evaluate(CORPUS, [RulesOnlyDetector()])
    arm = report.detectors[0]
    assert arm.by_location_class["structured"].recall == 1.0
    assert arm.by_location_class["narrative"].recall == 0.0


def test_rules_never_predict_inside_narrative() -> None:
    detector = RulesOnlyDetector()
    narrative_messages = [m for m in CORPUS.messages if m.variant["narrative"]]
    assert narrative_messages

    for message in narrative_messages[:10]:
        narrative_spans = [
            a for a in message.annotations if a.location_class == "narrative"
        ]
        for detection in detector.detect(message.text):
            for annotation in narrative_spans:
                assert not detection.covers(annotation.start, annotation.end)


def test_text_only_finds_prose_names_and_misses_delimited_ones() -> None:
    """The central claim about flattening: a caret is not a linguistic cue.

    ``Vanbrugh^Bartram`` carries no signal that either token is a name; the
    same two words separated by a space do.
    """
    detector = FlattenedTextDetector()

    # The date is found either way -- digits need no linguistic context. The
    # claim is specifically about names, so that is what gets asserted.
    delimited = detector.detect("PID|1||||Vanbrugh^Bartram||19550312|F\r")
    assert not any("Vanbrugh" in d.text or "Bartram" in d.text for d in delimited)

    prose = detector.detect("Spoke with Bartram Vanbrugh about discharge.")
    assert any("Vanbrugh" in d.text for d in prose)


def test_text_only_recall_is_higher_on_narrative_than_structured() -> None:
    """The inversion relative to rules is the result worth reporting."""
    report = evaluate(CORPUS, [FlattenedTextDetector()])
    arm = report.detectors[0]
    assert (
        arm.by_location_class["narrative"].recall
        > arm.by_location_class["structured"].recall
    )


def test_schema_conditioned_beats_both_arms_on_overall_recall() -> None:
    report = evaluate(
        CORPUS,
        [RulesOnlyDetector(), FlattenedTextDetector(), SchemaConditionedDetector()],
    )
    rules, text, schema = report.detectors
    assert schema.untyped.recall > rules.untyped.recall
    assert schema.untyped.recall > text.untyped.recall
    # And it does so without giving away precision, because the narrative
    # detector never sees the coded fields that generate most of the
    # text-only arm's false positives.
    assert schema.untyped.precision > text.untyped.precision


def test_schema_conditioned_keeps_the_rules_structured_recall() -> None:
    report = evaluate(CORPUS, [RulesOnlyDetector(), SchemaConditionedDetector()])
    rules, schema = report.detectors
    assert schema.by_location_class["structured"].recall == (
        rules.by_location_class["structured"].recall
    )


def test_an_unknown_z_segment_defeats_the_rules_arm() -> None:
    """ZPD is in the policy, ZIN is not. The gap is what reading a feed buys."""
    report = evaluate(CORPUS, [RulesOnlyDetector()])
    arm = report.detectors[0]
    assert arm.by_segment["ZIN"].recall == 0.0
    assert arm.by_segment["ZPD"].recall > 0.5


# -- the structural context that is the Phase I variable ---------------


def test_deferred_candidates_carry_structural_context() -> None:
    """This context is the independent variable of the detection experiment."""
    detector = SchemaConditionedDetector()
    message = next(m for m in CORPUS.messages if m.variant["narrative"])
    candidates = detector.candidates(message.text)

    assert candidates
    for candidate in candidates:
        assert candidate.segment
        assert candidate.field is not None
        assert message.text[candidate.start : candidate.end] == candidate.text
        line = candidate.context_line()
        assert f"segment={candidate.segment}" in line
        assert "field=" in line


def test_narrative_detector_offsets_are_translated_to_message_offsets() -> None:
    """A narrative detector reports offsets inside its own candidate text.

    If the caller failed to translate them, predictions would land at the start
    of the message and quietly score as false positives.
    """
    detector = SchemaConditionedDetector(RegexNarrativeDetector())
    message = next(
        m
        for m in CORPUS.messages
        if any(a.location_class == "narrative" for a in m.annotations)
    )
    for detection in detector.detect(message.text):
        assert message.text[detection.start : detection.end] == detection.text


# -- reporting ----------------------------------------------------------


def test_the_rendered_report_names_what_a_reader_needs() -> None:
    report = evaluate(CORPUS, [RulesOnlyDetector(), SchemaConditionedDetector()])
    text = render(report)
    for expected in (
        "RECALL BY LOCATION CLASS",
        "RECALL BY SEGMENT",
        "WORST PHI TYPES BY RECALL",
        "TOP FALSE-POSITIVE CATEGORIES",
        "structured",
        "narrative",
        "ZIN",
    ):
        assert expected in text
    # The caveat has to survive into the output, not live only in a docstring.
    assert "not production" in text


def test_the_json_report_is_serializable() -> None:
    report = evaluate(CORPUS, [RulesOnlyDetector()])
    assert "by_location_class" in report.to_json()
