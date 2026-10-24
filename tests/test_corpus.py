"""Corpus integrity.

A corpus whose labels are subtly wrong is worse than no corpus: the error
surfaces as a detector's apparent failure, and nobody can tell the difference.
So the properties asserted here are the ones that would silently invalidate
every number the harness produces.
"""

from __future__ import annotations

import random
from datetime import date

import pytest

from deidbench import hl7
from deidbench.corpus import generate_corpus, GenerationOptions
from deidbench.corpus.annotation import Corpus, LocationClass
from deidbench.corpus.generate import MISUSE_KINDS, _assert_unambiguous, _Planted
from deidbench.corpus.people import (
    CORPUS_CITIES,
    CORPUS_FACILITIES,
    CORPUS_GIVEN_NAMES,
    CORPUS_SURNAMES,
    make_person,
)
from deidbench.surrogate.types import PhiType

SMALL = GenerationOptions(messages=40)


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return generate_corpus(SMALL)


# -- the property everything else depends on ---------------------------


def test_every_annotation_points_at_its_own_text(corpus: Corpus) -> None:
    """If offsets are wrong, every recall and precision number is wrong."""
    for message in corpus.messages:
        for annotation in message.annotations:
            actual = message.text[annotation.start : annotation.end]
            assert actual == annotation.text, (
                f"{message.message_id} {annotation.path}: "
                f"{actual!r} != {annotation.text!r}"
            )


def test_every_message_parses_and_round_trips(corpus: Corpus) -> None:
    for message in corpus.messages:
        assert hl7.serialize(hl7.parse(message.text)) == message.text


def test_annotations_do_not_overlap(corpus: Corpus) -> None:
    """A character belongs to one annotation, or the metrics are ill-defined."""
    for message in corpus.messages:
        spans = sorted((a.start, a.end) for a in message.annotations)
        for (_, first_end), (second_start, _) in zip(spans, spans[1:]):
            assert first_end <= second_start, message.message_id


def test_annotation_paths_resolve_to_the_annotated_text(corpus: Corpus) -> None:
    """The HL7 path and the character offsets must agree with each other."""
    for message in corpus.messages:
        _, spans = hl7.serialize_with_spans(hl7.parse(message.text))
        by_path = {span.path: span for span in spans}
        for annotation in message.annotations:
            span = by_path.get(annotation.path)
            assert span is not None, annotation.path
            assert span.covers(annotation.start, annotation.end)


# -- disjointness from the surrogate pools -----------------------------
#
# One test is absent here by design. The engine asserts that corpus names are
# disjoint from the surrogate lexicons, so a leak-check hit can never be a
# coincidence between the two pools. That check needs the lexicons, which are
# part of the replacement machinery and not published with the benchmark. The
# property it guards still holds; it is simply verified upstream.


def test_facility_words_are_disjoint_from_surnames() -> None:
    """Otherwise one string is ground truth for two different PHI types."""
    assert not set(CORPUS_FACILITIES) & set(CORPUS_SURNAMES)


def test_distinct_people_get_distinct_names() -> None:
    rng = random.Random(99)
    for index in range(200):
        person = make_person(rng, index, date(2026, 10, 13))
        names = [
            person.given,
            person.middle,
            person.nok_given,
            person.doctor_given,
            *[given for given, _, _ in person.narrative_people],
        ]
        surnames = [
            person.family,
            person.doctor_family,
            *[family for _, family, _ in person.narrative_people],
        ]
        assert len(names) == len(set(names))
        assert len(surnames) == len(set(surnames))
        # The next of kin may deliberately share the patient's surname.
        assert person.nok_family == person.family or person.nok_family not in surnames


def test_name_affixes_do_not_collide_with_hl7_code_values() -> None:
    """Uppercase "MR" as a name prefix is byte-identical to the PID-3.5 type code."""
    rng = random.Random(5)
    for index in range(100):
        person = make_person(rng, index, date(2026, 10, 13))
        assert person.prefix in {"Mr", "Ms", "Mrs", "Dr", ""}
        assert person.suffix in {"Jr", "Sr", "III", ""}


# -- the ambiguity guard ------------------------------------------------


def test_ambiguous_ground_truth_is_refused() -> None:
    """One string with two PHI types has no correct label, so generation stops."""
    with pytest.raises(ValueError, match="ambiguous ground truth"):
        _assert_unambiguous(
            {
                ("Vanbrugh", "family_name"): _Planted("Vanbrugh", PhiType.FAMILY_NAME),
                ("Vanbrugh", "clinician_name"): _Planted(
                    "Vanbrugh", PhiType.CLINICIAN_NAME
                ),
            }
        )


def test_unambiguous_ground_truth_passes() -> None:
    _assert_unambiguous(
        {
            ("Vanbrugh", "family_name"): _Planted("Vanbrugh", PhiType.FAMILY_NAME),
            ("Loxley", "clinician_name"): _Planted("Loxley", PhiType.CLINICIAN_NAME),
        }
    )


# -- coverage of the cases the corpus exists to exercise ---------------


def test_all_four_location_classes_are_represented(corpus: Corpus) -> None:
    present = set(corpus.totals_by_location_class())
    assert present == {str(c) for c in LocationClass}


def test_narrative_phi_is_present_and_substantial(corpus: Corpus) -> None:
    totals = corpus.totals_by_location_class()
    assert totals["narrative"] > 100


def test_both_known_and_unknown_z_segments_appear(corpus: Corpus) -> None:
    """ZPD is in the shipped policy; ZIN is not. The pair is the honest test."""
    segments = {a.segment for m in corpus.messages for a in m.annotations}
    assert "ZPD" in segments
    assert "ZIN" in segments


def test_every_misuse_kind_is_exercised() -> None:
    corpus = generate_corpus(GenerationOptions(messages=400))
    kinds = {
        m.variant.get("misuse_kind")
        for m in corpus.messages
        if m.variant.get("misuse_kind")
    }
    assert kinds == set(MISUSE_KINDS)


def test_format_variety_is_exercised() -> None:
    corpus = generate_corpus(GenerationOptions(messages=200))
    variants = [m.variant for m in corpus.messages]
    assert {v["delimiters"] for v in variants} == {"standard", "odd"}
    assert {v["terminator"] for v in variants} == {"cr", "crlf"}
    assert len({v["version"] for v in variants}) >= 3
    assert len({v["message_type"] for v in variants}) >= 3
    assert any(v["loose_date"] for v in variants)


def test_odd_delimiter_messages_are_annotated_correctly() -> None:
    """Non-standard delimiters must not shift any offsets."""
    corpus = generate_corpus(GenerationOptions(messages=200))
    odd = [m for m in corpus.messages if m.variant["delimiters"] == "odd"]
    assert odd
    for message in odd:
        assert message.annotations
        for annotation in message.annotations:
            assert message.text[annotation.start : annotation.end] == annotation.text


# -- reproducibility ----------------------------------------------------


def test_the_same_seed_produces_an_identical_corpus() -> None:
    """A published corpus has to regenerate byte-identically to be checkable."""
    first = generate_corpus(GenerationOptions(messages=25, seed=7))
    second = generate_corpus(GenerationOptions(messages=25, seed=7))
    assert first.to_jsonl() == second.to_jsonl()


def test_different_seeds_produce_different_corpora() -> None:
    first = generate_corpus(GenerationOptions(messages=25, seed=7))
    second = generate_corpus(GenerationOptions(messages=25, seed=8))
    assert first.to_jsonl() != second.to_jsonl()


def test_jsonl_round_trips(corpus: Corpus) -> None:
    restored = Corpus.from_jsonl(corpus.to_jsonl())
    assert len(restored) == len(corpus)
    assert restored.metadata == corpus.metadata
    assert restored.annotation_count == corpus.annotation_count
    assert restored.messages[0].annotations == corpus.messages[0].annotations


def test_metadata_records_the_synthetic_caveat(corpus: Corpus) -> None:
    """The caveat travels with the data, not just the write-up."""
    assert "Synthetic" in corpus.metadata["note"]
    assert corpus.metadata["seed"] == SMALL.seed
