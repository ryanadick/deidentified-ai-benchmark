"""The NLM Scrubber arm, and the alignment its numbers depend on.

Most of this file tests :func:`recover_spans` with handwritten inputs and no
binary, because the alignment is the only part of this arm that can be wrong
*and still look right*. Scrubber reports redacted text, not offsets, so every
span is inferred; a subtly wrong aligner yields a plausible recall figure built
on mis-located spans. That failure mode nearly shipped here -- a character-level
diff produced a believable 0.284 while losing every tag name -- so the aligner
is pinned hard.
"""

from __future__ import annotations

import pytest

from deidbench.eval.scrubber_baseline import (
    SCRUBBER_TAG_MAP,
    apply_spans,
    map_tag,
    recover_spans,
    scrubber_available,
    strip_footer,
    verify_recovery,
)

requires_scrubber = pytest.mark.skipif(
    not scrubber_available(),
    reason="NLM Scrubber is an optional binary dependency",
)

FOOTER = (
    "\n##### DOCUMENT ####################\n"
    "##### ConfigFile:config.cfg\n"
    "#NLM-Scrubber v.19.0403L\n"
)


# -- the tag mapping ------------------------------------------------------


def test_known_tags_map_into_the_corpus_vocabulary() -> None:
    assert map_tag("PERSONALNAME") == "family_name"
    assert map_tag("ADDRESS") == "street_address"
    assert map_tag("DATE") == "event_date"
    assert map_tag("ALPHANUMERICID") == "identifier"


def test_age_tag_is_unmapped_rather_than_forced() -> None:
    """HIPAA only treats ages over 89 as PHI and this corpus plants none."""
    assert map_tag("AGE90+") == "unmapped:AGE90+"
    assert "AGE90+" not in SCRUBBER_TAG_MAP


# -- the footer -----------------------------------------------------------


def test_footer_is_stripped_before_alignment() -> None:
    """Left on, Scrubber's banner makes every message fail the round-trip."""
    assert strip_footer("Call me [PERSONALNAME].\r\n" + FOOTER) == (
        "Call me [PERSONALNAME].\r\n"
    )


def test_stripping_is_a_no_op_without_a_footer() -> None:
    assert strip_footer("no banner here") == "no banner here"


# -- the aligner ----------------------------------------------------------


def test_a_single_redaction_is_located_in_the_original() -> None:
    original = "Call me Ishmael."
    spans = recover_spans(original, "Call me [PERSONALNAME].")

    assert len(spans) == 1
    assert (spans[0].start, spans[0].end) == (8, 15)
    assert original[spans[0].start : spans[0].end] == "Ishmael"
    assert spans[0].phi_type == "family_name"
    assert not spans[0].merged


def test_tags_are_not_shredded_by_character_similarity() -> None:
    """The regression that nearly shipped.

    A character-level diff aligns the letters inside ``[ALPHANUMERICID]``
    against similar letters in the original, producing many fragments with no
    recoverable tag. Tag-anchored recovery must return one span, correctly
    typed.
    """
    original = "MSH|^~\\&|REGADT|ACMEHOSP|202608010800|ADT^A04|MSG00001|P|2.7\r"
    spans = recover_spans(original, "[ALPHANUMERICID]\r")

    assert len(spans) == 1
    assert spans[0].phi_type == "identifier"
    assert spans[0].start == 0


def test_a_surviving_delimiter_splits_adjacent_redactions() -> None:
    """Why HL7's carets save this arm rather than breaking it.

    Adjacent identifiers looked like the dominant risk, since half this
    corpus's gold spans touch another. Scrubber leaves the delimiter standing,
    so the two spans are separable and neither is merged.
    """
    original = "PID|5|Everman^Clara\r"
    spans = recover_spans(original, "PID|5|[PERSONALNAME]^[PERSONALNAME]\r")

    assert len(spans) == 2
    assert [original[s.start : s.end] for s in spans] == ["Everman", "Clara"]
    assert not any(s.merged for s in spans)


def test_consecutive_tags_with_no_anchor_between_them_are_merged() -> None:
    """The genuinely unrecoverable case, marked rather than guessed at.

    With nothing between the tags there is no evidence of where one value
    ended and the next began, so the region is reported as one merged span and
    typed under a prefix that can never count as a typed match.
    """
    original = "to Everman Clara now"
    spans = recover_spans(original, "to [PERSONALNAME][PERSONALNAME] now")

    assert len(spans) == 1
    assert spans[0].merged
    assert spans[0].tags == ("PERSONALNAME", "PERSONALNAME")
    assert spans[0].phi_type.startswith("merged:")


def test_round_trip_is_what_makes_the_spans_trustworthy() -> None:
    original = "Patient Drummond, DOB 19550312, at 12 Maple Street.\r"
    scrubbed = "Patient [PERSONALNAME], DOB [DATE], at [ADDRESS].\r"

    spans = recover_spans(original, scrubbed)
    assert apply_spans(original, spans) == scrubbed
    assert verify_recovery(original, scrubbed + FOOTER)


def test_an_unanchorable_output_is_refused_not_guessed() -> None:
    """If a literal run is absent from the original, the alignment is void.

    Returning None here is what lets the batch exclude the message instead of
    scoring wrong offsets.
    """
    assert recover_spans("Call me Ishmael.", "Totally different [PERSONALNAME].") is None
    assert not verify_recovery("Call me Ishmael.", "Totally different text.")


def test_an_output_with_no_redactions_yields_no_spans() -> None:
    assert recover_spans("nothing to remove", "nothing to remove") == []


def test_cr_terminators_must_survive_the_round_trip(tmp_path) -> None:
    """Regression: reading Scrubber's output in text mode silently broke this.

    HL7 terminates segments with CR. ``Path.read_text()`` applies
    universal-newline translation, so a file written with CR comes back with
    LF, no anchor containing a line ending matches the original, and every
    message is discarded as unalignable. The arm reported recall 0.000 across
    500 messages and looked like a tool that found nothing rather than a
    reading bug. Read bytes and decode.
    """
    original = "PID|5|Everman\rNTE|1|L|Called Everman\r"
    scrubbed = "PID|5|[PERSONALNAME]\rNTE|1|L|Called [PERSONALNAME]\r"
    path = tmp_path / "msg.nphi.txt"
    path.write_bytes(scrubbed.encode("utf-8"))

    as_bytes = path.read_bytes().decode("utf-8")
    assert "\r" in as_bytes
    assert recover_spans(original, as_bytes) is not None
    assert verify_recovery(original, as_bytes)

    # The failure mode, pinned so it cannot silently return.
    translated = path.read_text(encoding="utf-8")
    assert "\r" not in translated
    assert recover_spans(original, translated) is None


# -- the live binary ------------------------------------------------------


@requires_scrubber
def test_scrubber_runs_and_offsets_address_their_own_text() -> None:
    from deidbench.corpus import GenerationOptions, generate_corpus
    from deidbench.eval.scrubber_baseline import ScrubberTextDetector

    messages = generate_corpus(GenerationOptions(messages=3)).messages
    detector = ScrubberTextDetector()
    detector.prime([m.text for m in messages])

    for message in messages:
        for detection in detector.detect(message.text):
            assert message.text[detection.start : detection.end] == detection.text


@requires_scrubber
def test_provenance_records_the_build_and_that_spans_are_inferred() -> None:
    from deidbench.eval.scrubber_baseline import ScrubberTextDetector

    provenance = ScrubberTextDetector().provenance
    assert provenance["version"] == "19.0403L"
    assert "alignment" in provenance["spans"]
    assert "archive_sha256" in provenance
