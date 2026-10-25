"""The Philter baseline arms, and the evidence that the port is faithful.

The second group of tests is the unusual one. This arm runs ``philter-lite``
rather than the ``philter-ucsf`` distribution UCSF published, because the
published filter set cannot compile on any Python this package supports. That
substitution is only defensible if the fork's patterns really are the published
patterns, so the claim is pinned here as a test rather than asserted in a
docstring.
"""

from __future__ import annotations

import re

import pytest

from deidbench.corpus import GenerationOptions, generate_corpus
from deidbench.eval.detectors import NarrativeCandidate, SchemaConditionedDetector
from deidbench.eval.philter_baseline import (
    PHILTER_PHI_TYPE_MAP,
    map_phi_type,
    philter_available,
)

requires_philter = pytest.mark.skipif(
    not philter_available(), reason="Philter is an optional dependency"
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


# -- the mapping ----------------------------------------------------------


def test_known_phi_types_map_into_the_corpus_vocabulary() -> None:
    assert map_phi_type("Name") == "family_name"
    assert map_phi_type("Email") == "email"
    assert map_phi_type("Patient_Social_Security_Number") == "ssn"


def test_other_is_never_silently_given_a_category() -> None:
    """Philter's OTHER means "unsafe, type unknown".

    Mapping it into a corpus category would manufacture typed accuracy Philter
    never demonstrated, and it is the dominant label on HL7 input, so the
    mistake would not be a small one.
    """
    assert map_phi_type("OTHER") == "unmapped:OTHER"
    assert "OTHER" not in PHILTER_PHI_TYPE_MAP


def test_unmapped_prefix_cannot_collide_with_a_real_phi_type() -> None:
    for value in PHILTER_PHI_TYPE_MAP.values():
        assert not value.startswith("unmapped:")


# -- the port is faithful -------------------------------------------------

#: Patterns transcribed verbatim from philter-ucsf's published
#: ``philter_delta`` filter files, paired with philter-lite's form of the same
#: filter. The only difference permitted is relocating an inline ``(?i)`` to
#: the front, which is semantics-preserving: the flag always applied to the
#: whole pattern regardless of position, which is exactly why Python 3.11 made
#: the misleading position an error.
PUBLISHED_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (
        "gross_safe",
        r"\b(?i)gross\s(description|per)\b",
        r"(?i)\bgross\s(description|per)\b",
    ),
    (
        "onset_safe",
        r"\b(?i)age\sof\sonset\b",
        r"(?i)\bage\sof\sonset\b",
    ),
    (
        "day_safe",
        r"\b(?i)(a|per|the|same|next|each|every|by)\sday\b",
        r"(?i)\b(a|per|the|same|next|each|every|by)\sday\b",
    ),
    (
        "hearing_safe",
        r"\b(?i)hearing\s(loss|aid(s)?)\b",
        r"(?i)\bhearing\s(loss|aid(s)?)\b",
    ),
    (
        "found_safe",
        r"\b(?i)((were|was|is|results|data)\sfound|found\s(in|for))\b",
        r"(?i)\b((were|was|is|results|data)\sfound|found\s(in|for))\b",
    ),
    # No inline flag, so the fork should not have touched it at all.
    (
        "sao2_safe",
        r"\bSaO2(\%)?:\s\d{2,3}\b",
        r"\bSaO2(\%)?:\s\d{2,3}\b",
    ),
    # The fingerprint. Upstream wrote "b(" where "\b(" was meant, so this
    # pattern matches a literal "b". The fork preserves the bug verbatim --
    # which is the evidence that it transcribed the published filter set rather
    # than rewriting it. If this ever starts passing with a corrected "\b", the
    # fork has diverged and this arm needs re-examining.
    (
        "birth_safe",
        r"b(?i)((at|since)\sbirth|birth\s(control|weight|mother|time|day))\b",
        r"(?i)b((at|since)\sbirth|birth\s(control|weight|mother|time|day))\b",
    ),
)


def _strip_inline_flag(pattern: str) -> str:
    return pattern.replace("(?i)", "", 1)


@pytest.mark.parametrize(
    "title,published,ported", PUBLISHED_PATTERNS, ids=lambda v: v if isinstance(v, str) and "\\" not in v else ""
)
def test_ported_pattern_differs_only_by_flag_position(
    title: str, published: str, ported: str
) -> None:
    assert _strip_inline_flag(published) == _strip_inline_flag(ported), title
    if "(?i)" in ported:
        assert ported.startswith("(?i)"), title


def test_the_published_patterns_really_are_broken_on_this_python() -> None:
    """The reason the fork is necessary, asserted rather than claimed.

    If a future Python accepts these again, the honest move is to go back to
    the distribution UCSF published, and this failing test is the trigger.
    """
    broken = [
        title
        for title, published, _ported in PUBLISHED_PATTERNS
        if _raises_re_error(published)
    ]
    assert "gross_safe" in broken
    assert "sao2_safe" not in broken


def _raises_re_error(pattern: str) -> bool:
    try:
        re.compile(pattern)
    except re.error:
        return True
    return False


@requires_philter
def test_the_ported_patterns_all_compile() -> None:
    from deidbench.eval.philter_baseline import DEFAULT_CONFIG, _load_filters

    assert _load_filters(DEFAULT_CONFIG)


# -- the arms themselves --------------------------------------------------


@requires_philter
def test_philter_finds_phi_in_a_flattened_message() -> None:
    from deidbench.eval.philter_baseline import PhilterTextDetector

    message = generate_corpus(GenerationOptions(messages=3)).messages[0]
    detections = PhilterTextDetector().detect(message.text)
    assert detections


@requires_philter
def test_detected_offsets_address_the_text_they_claim() -> None:
    from deidbench.eval.philter_baseline import PhilterTextDetector

    message = generate_corpus(GenerationOptions(messages=3)).messages[0]
    for detection in PhilterTextDetector().detect(message.text):
        assert message.text[detection.start : detection.end] == detection.text


@requires_philter
def test_narrative_offsets_are_relative_to_the_candidate() -> None:
    from deidbench.eval.philter_baseline import PhilterNarrativeDetector

    text = "Spoke with Barnabus Drummond about discharge planning."
    found = PhilterNarrativeDetector().detect_in(candidate(text))

    assert found
    for start, end, _phi_type in found:
        assert 0 <= start < end <= len(text)


@requires_philter
def test_philter_plugs_into_the_schema_conditioned_arm() -> None:
    from deidbench.eval.philter_baseline import PhilterNarrativeDetector

    detector = SchemaConditionedDetector(
        narrative_detector=PhilterNarrativeDetector()
    )
    assert detector.name == "schema+philter-narrative"

    message = generate_corpus(GenerationOptions(messages=3)).messages[0]
    for detection in detector.detect(message.text):
        assert message.text[detection.start : detection.end] == detection.text


@requires_philter
def test_provenance_names_the_port_explicitly() -> None:
    """The row must not be readable as "the tool UCSF ships"."""
    from deidbench.eval.philter_baseline import PhilterTextDetector

    provenance = PhilterTextDetector().provenance
    assert "philter-lite" in provenance["distribution"]
    assert "philter-ucsf" in provenance["distribution"]
    assert provenance["config"] == "philter_delta"
    assert provenance["philter-lite"] != "not installed"
