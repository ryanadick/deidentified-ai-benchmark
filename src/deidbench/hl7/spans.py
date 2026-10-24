"""Character spans for every leaf in a message.

Evaluation needs to speak two languages at once. Ground-truth annotations and
detector predictions are character ranges in the serialized message, because a
text-only detector sees nothing else. Policies, structural context and the
location classes that make the results interpretable are HL7 paths. This module
is the bridge: it serializes a message while recording where each leaf landed,
so a character offset can be resolved to ``PID-5[1].1.1`` and back.

It mirrors ``Segment.serialize`` and ``Node.serialize`` exactly. If the two ever
disagree the spans would silently point at the wrong text, so
``test_spans.py`` asserts that the reconstructed string equals ``serialize()``
for every sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .model import Message, Node, Segment


@dataclass(frozen=True)
class LeafSpan:
    """One leaf value and where it sits in the serialized message."""

    #: Fully explicit path, e.g. ``PID-5[1].1.1``.
    path: str
    #: Path without repetition indices, e.g. ``PID-5.1.1``. This is the form
    #: policy rules are written in.
    policy_path: str
    segment: str
    segment_occurrence: int
    field: int | None
    repetition: int
    component: int | None
    subcomponent: int | None
    start: int
    end: int
    text: str

    @property
    def length(self) -> int:
        return self.end - self.start

    def covers(self, start: int, end: int) -> bool:
        return self.start <= start and end <= self.end

    def overlaps(self, start: int, end: int) -> bool:
        return start < self.end and end > self.start


def serialize_with_spans(message: Message) -> tuple[str, list[LeafSpan]]:
    """Serialize a message and return it with a span for every leaf."""
    pieces: list[str] = [message.preamble]
    spans: list[LeafSpan] = []
    position = len(message.preamble)
    occurrences: dict[str, int] = {}

    for segment in message.segments:
        occurrences[segment.name] = occurrences.get(segment.name, 0) + 1
        position = _walk_segment(
            segment, occurrences[segment.name], position, pieces, spans
        )

    return "".join(pieces), spans


def leaf_index(message: Message) -> dict[str, LeafSpan]:
    """Spans keyed by explicit path, for direct lookup."""
    _, spans = serialize_with_spans(message)
    return {span.path: span for span in spans}


def find_enclosing(spans: list[LeafSpan], start: int, end: int) -> LeafSpan | None:
    """The leaf containing a character range, or None if it straddles leaves.

    A range that straddles a delimiter is not inside any single value. That
    happens when a detector predicts across a field boundary, and it has to be
    distinguishable from a clean hit rather than snapped to the nearest leaf.
    """
    for span in spans:
        if span.covers(start, end):
            return span
    return None


def candidate_policy_paths(policy_path: str) -> Iterator[str]:
    """Progressively shorter forms of a path, most specific first.

    ``PID-5.1.1`` yields ``PID-5.1.1``, ``PID-5.1``, ``PID-5``. A policy may
    address any of those levels -- ``PID-19`` governs a whole field while
    ``PID-5.1`` governs one component -- so a lookup has to try each.
    """
    parts = policy_path.split(".")
    for stop in range(len(parts), 0, -1):
        yield ".".join(parts[:stop])


# -- internals ----------------------------------------------------------


def _walk_segment(
    segment: Segment,
    occurrence: int,
    position: int,
    pieces: list[str],
    spans: list[LeafSpan],
) -> int:
    delimiters = segment.delimiters

    if segment.is_msh and segment.fields:
        # MSH-1 *is* the field separator, so the head is "MSH" plus that
        # character with nothing joining them. Fields from MSH-2 onward are
        # then joined normally, which is why the first of them gets no
        # preceding separator.
        pieces.append(segment.name)
        position += len(segment.name)
        position = _walk_node(
            segment.fields[0], segment, occurrence, 1, position, pieces, spans
        )
        for number, node in enumerate(segment.fields[1:], start=2):
            if number > 2:
                pieces.append(delimiters.field)
                position += len(delimiters.field)
            position = _walk_node(
                node, segment, occurrence, number, position, pieces, spans
            )
    else:
        pieces.append(segment.name)
        position += len(segment.name)
        for number, node in enumerate(segment.fields, start=1):
            pieces.append(delimiters.field)
            position += len(delimiters.field)
            position = _walk_node(
                node, segment, occurrence, number, position, pieces, spans
            )

    pieces.append(segment.terminator)
    return position + len(segment.terminator)


def _walk_node(
    node: Node,
    segment: Segment,
    occurrence: int,
    field: int,
    position: int,
    pieces: list[str],
    spans: list[LeafSpan],
    repetition: int = 1,
    component: int | None = None,
    subcomponent: int | None = None,
) -> int:
    if node.children is None:
        pieces.append(node.raw)
        spans.append(
            _make_span(
                segment,
                occurrence,
                field,
                repetition,
                component,
                subcomponent,
                position,
                node.raw,
            )
        )
        return position + len(node.raw)

    separator = node.sep or ""
    for index, child in enumerate(node.children, start=1):
        if index > 1:
            pieces.append(separator)
            position += len(separator)

        # Depth determines which coordinate this index names: a field's
        # children are repetitions, a repetition's are components, a
        # component's are subcomponents.
        if component is None and subcomponent is None and node.sep == segment.delimiters.repetition:
            next_repetition, next_component, next_subcomponent = index, None, None
        elif component is None:
            next_repetition, next_component, next_subcomponent = repetition, index, None
        else:
            next_repetition, next_component, next_subcomponent = (
                repetition,
                component,
                index,
            )

        position = _walk_node(
            child,
            segment,
            occurrence,
            field,
            position,
            pieces,
            spans,
            next_repetition,
            next_component,
            next_subcomponent,
        )
    return position


def _make_span(
    segment: Segment,
    occurrence: int,
    field: int,
    repetition: int,
    component: int | None,
    subcomponent: int | None,
    start: int,
    text: str,
) -> LeafSpan:
    explicit = f"{segment.name}"
    policy = f"{segment.name}"
    if occurrence != 1:
        explicit += f"[{occurrence}]"
    explicit += f"-{field}[{repetition}]"
    policy += f"-{field}"
    if component is not None:
        explicit += f".{component}"
        policy += f".{component}"
        if subcomponent is not None:
            explicit += f".{subcomponent}"
            policy += f".{subcomponent}"

    return LeafSpan(
        path=explicit,
        policy_path=policy,
        segment=segment.name,
        segment_occurrence=occurrence,
        field=field,
        repetition=repetition,
        component=component,
        subcomponent=subcomponent,
        start=start,
        end=start + len(text),
        text=text,
    )
