"""The Presidio baseline arms.

Split in two deliberately. The mapping and bookkeeping tests run everywhere,
because they encode the decisions that make the comparison fair and those must
not drift silently in an environment without the optional dependency. The tests
that actually invoke Presidio are skipped when it is not installed.
"""

from __future__ import annotations

import pytest

from deidbench.corpus import GenerationOptions, generate_corpus
from deidbench.eval.detectors import NarrativeCandidate, SchemaConditionedDetector
from deidbench.eval.presidio_baseline import (
    PRESIDIO_ENTITY_MAP,
    map_entity_type,
    presidio_available,
)

requires_presidio = pytest.mark.skipif(
    not presidio_available(), reason="Presidio is an optional dependency"
)


def candidate(text: str) -> NarrativeCandidate:
    return NarrativeCandidate(
        text=text,
        start=0,
        end=len(text),
        path="OBX-5[1].1.1",
        policy_path="OBX-5.1.1",
        segment="OBX",
        field=5,
        component=1,
        data_type="TX",
        note="observation value",
    )


# -- the mapping, which is where the comparison is made fair or unfair ----


def test_known_entities_map_into_the_corpus_vocabulary() -> None:
    assert map_entity_type("PERSON") == "family_name"
    assert map_entity_type("US_SSN") == "ssn"
    assert map_entity_type("EMAIL_ADDRESS") == "email"


def test_unmapped_entities_are_kept_under_a_prefix_not_dropped() -> None:
    """Dropping them would flatter Presidio by hiding predictions it made.

    They have to survive into the report so they still cost precision and show
    up by name in the false-positive breakdown.
    """
    assert map_entity_type("CREDIT_CARD") == "unmapped:CREDIT_CARD"
    assert map_entity_type("URL") == "unmapped:URL"


def test_unmapped_prefix_cannot_collide_with_a_real_phi_type() -> None:
    """An unmapped type must never score as a typed match by accident."""
    for value in PRESIDIO_ENTITY_MAP.values():
        assert not value.startswith("unmapped:")


# -- the arms themselves --------------------------------------------------


@requires_presidio
def test_presidio_finds_a_name_in_prose() -> None:
    from deidbench.eval.presidio_baseline import PresidioTextDetector

    detector = PresidioTextDetector()
    text = "Patient Barnabus Drummond was seen on 2024-03-14."
    detections = detector.detect(text)

    assert any(
        "Drummond" in d.text and d.phi_type == "family_name" for d in detections
    )


@requires_presidio
def test_detected_offsets_address_the_text_they_claim() -> None:
    """An offset bug here would corrupt every number the arm produces."""
    from deidbench.eval.presidio_baseline import PresidioTextDetector

    message = generate_corpus(GenerationOptions(messages=3)).messages[0]
    for detection in PresidioTextDetector().detect(message.text):
        assert message.text[detection.start : detection.end] == detection.text


@requires_presidio
def test_narrative_offsets_are_relative_to_the_candidate() -> None:
    """The seam's contract: local offsets, translated by the caller.

    Returning message-absolute offsets from here would silently shift every
    narrative detection, so this is worth asserting rather than assuming.
    """
    from deidbench.eval.presidio_baseline import PresidioNarrativeDetector

    text = "Spoke with Barnabus Drummond about discharge planning."
    found = PresidioNarrativeDetector().detect_in(candidate(text))

    assert found
    for start, end, _phi_type in found:
        assert 0 <= start < end <= len(text)
    assert any("Drummond" in text[start:end] for start, end, _ in found)


@requires_presidio
def test_presidio_plugs_into_the_schema_conditioned_arm() -> None:
    from deidbench.eval.presidio_baseline import PresidioNarrativeDetector

    detector = SchemaConditionedDetector(
        narrative_detector=PresidioNarrativeDetector()
    )
    assert detector.name == "schema+presidio-narrative"

    message = generate_corpus(GenerationOptions(messages=3)).messages[0]
    detections = detector.detect(message.text)
    assert detections
    for detection in detections:
        assert message.text[detection.start : detection.end] == detection.text


@requires_presidio
def test_provenance_records_what_makes_the_score_reproducible() -> None:
    """"Presidio scored X" is not a result without the versions behind it."""
    from deidbench.eval.presidio_baseline import (
        PresidioNarrativeDetector,
        PresidioTextDetector,
    )

    provenance = PresidioTextDetector().provenance
    assert provenance["presidio-analyzer"] != "not installed"
    assert provenance["spacy_model"] == "en_core_web_lg"
    assert "score_threshold" in provenance

    # The wrapper arm must surface the wrapped tool's provenance, or the
    # schema-conditioned Presidio row would report no versions at all.
    wrapped = SchemaConditionedDetector(
        narrative_detector=PresidioNarrativeDetector()
    )
    assert wrapped.provenance["presidio-analyzer"] == provenance["presidio-analyzer"]


def test_regex_arm_provenance_is_empty() -> None:
    """Only third-party arms declare provenance; ours have nothing to declare."""
    assert SchemaConditionedDetector().provenance == {}


# -- arm selection --------------------------------------------------------


PRESIDIO_SPEC = ("presidio", "PresidioTextDetector", "PresidioNarrativeDetector")


def test_no_presidio_runs_the_three_arm_subset() -> None:
    from deidbench.cli import _baseline_detectors

    assert _baseline_detectors(PRESIDIO_SPEC, "skip", None) == []


def test_missing_presidio_is_a_note_in_auto_and_an_error_when_required(
    monkeypatch, capsys
) -> None:
    """A run that quietly dropped the baseline would look like the full table."""
    import deidbench.eval.presidio_baseline as baseline
    from deidbench.cli import _baseline_detectors

    monkeypatch.setattr(baseline, "presidio_available", lambda: False)

    assert _baseline_detectors(PRESIDIO_SPEC, "auto", None) == []
    assert "skipped" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        _baseline_detectors(PRESIDIO_SPEC, "require", None)


@requires_presidio
def test_auto_adds_both_presidio_arms_when_available() -> None:
    from deidbench.cli import _baseline_detectors

    arms = _baseline_detectors(PRESIDIO_SPEC, "auto", None)
    assert [arm.name for arm in arms] == [
        "presidio-text",
        "schema+presidio-narrative",
    ]


def test_every_baseline_arm_follows_the_module_contract() -> None:
    """The table in cli.py is only safe if every row honours the convention."""
    from importlib import import_module

    from deidbench.cli import BASELINE_ARMS

    for name, text_cls, narrative_cls in BASELINE_ARMS:
        module = import_module(f"deidbench.eval.{name}_baseline")
        assert callable(getattr(module, f"{name}_available"))
        assert isinstance(module._IMPORT_HINT, str)
        assert hasattr(module, text_cls)
        assert hasattr(module, narrative_cls)
