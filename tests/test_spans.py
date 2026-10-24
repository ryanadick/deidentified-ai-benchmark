"""Span tracking must agree with serialization exactly.

If the two ever disagree, annotations and predictions silently point at the
wrong text and every evaluation number is quietly wrong.
"""

from __future__ import annotations

import pathlib

import pytest

from deidbench import hl7
from deidbench.hl7.spans import candidate_policy_paths, serialize_with_spans

SAMPLES = sorted((pathlib.Path(__file__).parent.parent / "samples").glob("*.hl7"))


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda p: p.name)
def test_span_walk_reproduces_the_message(sample) -> None:
    text = sample.read_text(encoding="utf-8", newline="")
    rebuilt, _ = serialize_with_spans(hl7.parse(text))
    assert rebuilt == text


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda p: p.name)
def test_every_span_points_at_its_own_text(sample) -> None:
    text = sample.read_text(encoding="utf-8", newline="")
    rebuilt, spans = serialize_with_spans(hl7.parse(text))
    for span in spans:
        assert rebuilt[span.start : span.end] == span.text, span.path


def test_paths_are_addressed_as_expected() -> None:
    text = (
        "MSH|^~\\&|A|B|C|D|20261013||ADT^A04|1|P|2.5.1\r"
        "PID|1||MR1^^^MCM^MR~SSN1^^^SSA^SS||Everman^Clara||19550312|F\r"
    )
    _, spans = serialize_with_spans(hl7.parse(text))
    by_path = {s.path: s.text for s in spans}

    assert by_path["PID-3[1].1.1"] == "MR1"
    assert by_path["PID-3[1].4.1"] == "MCM"
    assert by_path["PID-3[2].1.1"] == "SSN1"
    assert by_path["PID-5[1].1.1"] == "Everman"
    assert by_path["PID-5[1].2.1"] == "Clara"
    assert by_path["PID-7[1].1.1"] == "19550312"


def test_repetitions_are_distinguished() -> None:
    _, spans = serialize_with_spans(
        hl7.parse("MSH|^~\\&|A\rPID|||||Smith^John~Smyth^Jon\r")
    )
    families = {s.path: s.text for s in spans if s.policy_path == "PID-5.1.1"}
    assert families == {"PID-5[1].1.1": "Smith", "PID-5[2].1.1": "Smyth"}


def test_subcomponents_are_addressed() -> None:
    # PID-11 is the patient address, so the segment needs eleven field
    # separators before the value -- counted, not eyeballed.
    pid = "|".join(["PID"] + [""] * 10 + ["12 Oak Ln&Suite 3^^Lockport"])
    _, spans = serialize_with_spans(hl7.parse(f"MSH|^~\\&|A\r{pid}\r"))
    by_path = {s.path: s.text for s in spans}
    assert by_path["PID-11[1].1.1"] == "12 Oak Ln"
    assert by_path["PID-11[1].1.2"] == "Suite 3"
    assert by_path["PID-11[1].3.1"] == "Lockport"


def test_msh_delimiters_get_their_own_spans() -> None:
    _, spans = serialize_with_spans(hl7.parse("MSH|^~\\&|A\r"))
    by_path = {s.path: s.text for s in spans}
    assert by_path["MSH-1[1]"] == "|"
    assert by_path["MSH-2[1]"] == "^~\\&"


def test_repeated_segments_are_numbered() -> None:
    _, spans = serialize_with_spans(
        hl7.parse("MSH|^~\\&|A\rNK1|1|Doe^Jane\rNK1|2|Roe^John\r")
    )
    families = {s.path: s.text for s in spans if s.policy_path == "NK1-2.1.1"}
    assert families == {"NK1-2[1].1.1": "Doe", "NK1[2]-2[1].1.1": "Roe"}


def test_candidate_policy_paths_narrow_from_specific_to_general() -> None:
    assert list(candidate_policy_paths("PID-5.1.1")) == [
        "PID-5.1.1",
        "PID-5.1",
        "PID-5",
    ]


# -- leaf addressing ----------------------------------------------------


def test_leaf_paths_descends_into_subcomponents() -> None:
    """A component holding subcomponents expands to one path per subcomponent.

    Writing at the component level instead reads the subcomponent separator as
    part of the text and escapes it on the way back out, collapsing two
    subcomponents into one.
    """
    fields = {1: "1", 11: "12 Main St&Apt 2^^Lockport^NY^14094"}
    body = "|".join(fields.get(i, "") for i in range(1, 12))
    message = hl7.parse("MSH|^~\\&|A\rPID|" + body + "\r")
    segment = message.segment("PID")

    leaves = hl7.leaf_paths(segment, hl7.Path.parse("PID-11.1"))
    assert [str(p) for p in leaves] == ["PID-11[1].1.1", "PID-11[1].1.2"]

    # A component with nothing deeper is its own leaf, so feeds that send no
    # subcomponents behave exactly as they did before.
    flat = hl7.leaf_paths(segment, hl7.Path.parse("PID-11.3"))
    assert [str(p) for p in flat] == ["PID-11.3"]


def test_writing_a_leaf_preserves_the_subcomponent_separator() -> None:
    """The regression in one assertion: read a leaf, write it back, unchanged."""
    fields = {1: "1", 11: "12 Main St&Apt 2^^Lockport^NY^14094"}
    body = "|".join(fields.get(i, "") for i in range(1, 12))
    text = "MSH|^~\\&|A\rPID|" + body + "\r"
    message = hl7.parse(text)
    segment = message.segment("PID")

    for path in hl7.leaf_paths(segment, hl7.Path.parse("PID-11.1")):
        node = hl7.ensure(message, path)
        node.set_raw(hl7.escape(hl7.unescape(node.serialize(), segment.delimiters),
                                segment.delimiters))

    assert message.serialize() == text
