"""The gate: parse then serialize must reproduce the input byte for byte.

Every other guarantee in the engine rests on this one. If the parser silently
normalizes a trailing delimiter or a stray newline, then a message that passes
through de-identification differs from the original in ways nobody asked for,
and a diff against production output becomes unreadable.
"""

from __future__ import annotations

import pathlib

import pytest

from deidbench import hl7

SAMPLES = sorted((pathlib.Path(__file__).parent.parent / "samples").glob("*.hl7"))


def read_sample(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", newline="")


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda p: p.name)
def test_sample_round_trips_exactly(sample: pathlib.Path) -> None:
    text = read_sample(sample)
    assert hl7.serialize(hl7.parse(text)) == text


@pytest.mark.parametrize(
    "text",
    [
        "",
        "MSH",
        "MSH|",
        "MSH|^~\\&",
        "MSH|^~\\&|\r",
        "PID|||\r",
        "\r\r\r",
        "\n",
        "GARBAGE",
        "PID|a~b~~c|\r",
        "PID|a^^b&&c|\r",
        "MSH|^~\\&|A|B|C|D|1||ADT^A01|1|P|2.5.1\r\nPID|1\r\n",
        "MSH|^~\\&|A|B|C|D|1||ADT^A01|1|P|2.5.1\nPID|1\r",
        "MSH|^~\\&|A\rPID|||||Smith^John\r",
        # Truncation character present in MSH-2.
        "MSH|^~\\&#|A|B|C|D|1||ADT^A01|1|P|2.7\rPID|1\r",
        # Unterminated escape sequence.
        "MSH|^~\\&|A\rPID|||||Sm\\ith\r",
    ],
)
def test_edge_cases_round_trip_exactly(text: str) -> None:
    assert hl7.serialize(hl7.parse(text)) == text


def test_delimiters_are_read_from_msh() -> None:
    message = hl7.parse(read_sample(pathlib.Path(__file__).parent.parent / "samples" / "odd_delimiters.hl7"))
    d = message.delimiters
    assert (d.field, d.component, d.repetition, d.escape, d.subcomponent) == (
        "#",
        "!",
        "*",
        "%",
        "@",
    )


def test_missing_encoding_characters_fall_back_to_defaults() -> None:
    d = hl7.parse("MSH|\rPID|1\r").delimiters
    assert (d.component, d.repetition, d.escape, d.subcomponent) == ("^", "~", "\\", "&")


def test_tree_depth_is_uniform_below_the_segment() -> None:
    """A single-valued field still has one repetition, component and subcomponent."""
    message = hl7.parse("MSH|^~\\&|A\rPID|1\r")
    field = message.segment("PID").field(1)
    repetition = field.children[0]
    component = repetition.children[0]
    subcomponent = component.children[0]
    assert len(field.children) == 1
    assert len(repetition.children) == 1
    assert len(component.children) == 1
    assert subcomponent.is_leaf
    assert subcomponent.raw == "1"


def test_msh_1_and_2_stay_opaque() -> None:
    """Splitting the delimiter definitions on the delimiters they define is circular."""
    message = hl7.parse("MSH|^~\\&|A\r")
    msh = message.segment("MSH")
    assert msh.field(1).serialize() == "|"
    assert msh.field(2).serialize() == "^~\\&"
    assert msh.field(2).is_leaf
