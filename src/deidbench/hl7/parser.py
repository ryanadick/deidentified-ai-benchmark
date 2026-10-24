"""HL7 v2 parser and serializer.

Design commitment: ``serialize(parse(text)) == text`` for every input, valid or
not. Real-world v2 is full of blank lines, missing trailing fields, stray
``\\n`` terminators, segments with no delimiters at all and non-standard
Z-segments. A parser that normalizes any of that away would quietly rewrite
parts of the message the engine was never asked to touch, and would hide the
very irregularities this project exists to handle.

The tree is uniform below the segment: every field holds repetitions, every
repetition holds components, every component holds subcomponents. Single-valued
fields simply have one of each. Uniform depth means path resolution never has to
guess whether ``PID-5``'s children are repetitions or components.

MSH-1 and MSH-2 are the one exception HL7 itself creates -- they *contain* the
delimiters -- so they are kept as opaque leaves.
"""

from __future__ import annotations

from .model import Delimiters, Message, Node, Segment

DEFAULT_DELIMITERS = Delimiters()

_TERMINATOR_CHARS = "\r\n"
_MLLP_START = "\x0b"
_MLLP_END = "\x1c"


def parse(text: str) -> Message:
    """Parse an HL7 v2 message. Never raises on malformed input."""
    preamble, body = _strip_preamble(text)
    delimiters = _sniff_delimiters(body)

    segments = [
        _parse_segment(line, terminator, delimiters)
        for line, terminator in _split_lines(body)
    ]
    return Message(segments, delimiters, preamble)


def serialize(message: Message) -> str:
    """Render a message back to wire format."""
    return message.serialize()


# -- delimiters ---------------------------------------------------------


def _strip_preamble(text: str) -> tuple[str, str]:
    """Split off MLLP framing or leading blank lines before the first segment."""
    index = 0
    while index < len(text) and text[index] in (_MLLP_START, *_TERMINATOR_CHARS, " ", "\t"):
        index += 1
    return text[:index], text[index:]


def _sniff_delimiters(body: str) -> Delimiters:
    """Read the delimiters from MSH-1 and MSH-2, falling back to the defaults.

    A message whose MSH-2 is truncated or absent still has to parse, so each
    delimiter falls back independently rather than rejecting the message.
    """
    if not body.startswith("MSH") or len(body) < 4:
        return Delimiters()

    field_sep = body[3]
    rest = body[4:]
    end = _find_any(rest, _TERMINATOR_CHARS)
    line_rest = rest[:end] if end is not None else rest
    encoding = line_rest.split(field_sep)[0] if line_rest else ""

    def char_at(position: int, default: str) -> str:
        return encoding[position] if len(encoding) > position else default

    return Delimiters(
        field=field_sep,
        component=char_at(0, "^"),
        repetition=char_at(1, "~"),
        escape=char_at(2, "\\"),
        subcomponent=char_at(3, "&"),
        truncation=encoding[4] if len(encoding) > 4 else "",
    )


def _find_any(text: str, chars: str) -> int | None:
    for index, ch in enumerate(text):
        if ch in chars:
            return index
    return None


# -- segments -----------------------------------------------------------


def _split_lines(body: str) -> list[tuple[str, str]]:
    """Split into (line, terminator) pairs, preserving the exact terminators.

    Concatenating every line with its terminator reproduces ``body`` exactly.
    Blank lines are kept as empty segments so that messages padded with extra
    carriage returns survive a round trip.
    """
    pairs: list[tuple[str, str]] = []
    index = 0
    length = len(body)

    while index < length:
        cursor = index
        while cursor < length and body[cursor] not in _TERMINATOR_CHARS:
            cursor += 1

        line = body[index:cursor]
        terminator = ""
        if cursor < length:
            first = body[cursor]
            second = body[cursor + 1] if cursor + 1 < length else ""
            if (first, second) in (("\r", "\n"), ("\n", "\r")):
                terminator = first + second
                cursor += 2
            else:
                terminator = first
                cursor += 1

        pairs.append((line, terminator))
        index = cursor

    return pairs


def _parse_segment(line: str, terminator: str, d: Delimiters) -> Segment:
    stripped = line.rstrip(_MLLP_END)
    trailer = line[len(stripped) :]

    if stripped.startswith("MSH") and len(stripped) >= 4:
        return _parse_msh(stripped, trailer + terminator, d)

    parts = stripped.split(d.field)
    name = parts[0]
    fields = [_parse_field(part, d) for part in parts[1:]]
    return Segment(name, fields, trailer + terminator, d)


def _parse_msh(line: str, terminator: str, d: Delimiters) -> Segment:
    """Parse MSH, where MSH-1 is the field separator and MSH-2 the encoding chars.

    Both are stored as opaque leaves: splitting them on the very characters they
    define would be circular, and nothing in the engine should rewrite them.
    """
    field_sep = line[3]
    tail = line[4:].split(field_sep)

    fields: list[Node] = [Node(field_sep)]
    if tail:
        fields.append(Node(tail[0]))
        fields.extend(_parse_field(part, d) for part in tail[1:])

    return Segment("MSH", fields, terminator, d)


# -- fields and below ---------------------------------------------------


def _parse_field(raw: str, d: Delimiters) -> Node:
    repetitions = [
        _parse_repetition(part, d) for part in raw.split(d.repetition)
    ]
    return Node(raw, d.repetition, repetitions)


def _parse_repetition(raw: str, d: Delimiters) -> Node:
    components = [_parse_component(part, d) for part in raw.split(d.component)]
    return Node(raw, d.component, components)


def _parse_component(raw: str, d: Delimiters) -> Node:
    subcomponents = [Node(part) for part in raw.split(d.subcomponent)]
    return Node(raw, d.subcomponent, subcomponents)


def reparse_field(segment: Segment, number: int, raw: str) -> Node:
    """Replace a whole field with freshly parsed text, keeping depth uniform.

    Used when a rule rewrites an entire field rather than one component -- for
    example collapsing a repeating phone field to a single surrogate.
    """
    node = _parse_field(raw, segment.delimiters)
    while len(segment.fields) < number:
        segment.fields.append(_parse_field("", segment.delimiters))
    segment.fields[number - 1] = node
    return node
