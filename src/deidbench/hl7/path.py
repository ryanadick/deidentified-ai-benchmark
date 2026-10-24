"""HL7 v2 path addressing.

Paths are the vocabulary shared by the policy registry, the audit log and the
structural context handed to a model. The syntax follows what integration
engineers already write by hand:

    PID-5              whole field, every repetition
    PID-5[1]           first repetition only
    PID-5.1            first repetition, family name component
    PID-5[2].1.1       second repetition, family name, first subcomponent
    PID[2]-3           second PID segment in the message
    ZPD-4              non-standard segments address identically

Component and subcomponent indices are 1-based, matching the standard rather
than Python.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .escape import unescape
from .model import Message, Node, Segment

_PATH_RE = re.compile(
    r"""
    \A
    (?P<segment>[A-Za-z0-9]{3})
    (?:\[(?P<occurrence>\d+)\])?
    (?:
        -(?P<field>\d+)
        (?:\[(?P<repetition>\d+)\])?
        (?:
            \.(?P<component>\d+)
            (?:\.(?P<subcomponent>\d+))?
        )?
    )?
    \Z
    """,
    re.VERBOSE,
)


class PathError(ValueError):
    """Raised when a path string cannot be parsed."""


@dataclass(frozen=True)
class Path:
    """A parsed location in an HL7 v2 message."""

    segment: str
    occurrence: int = 1
    field: int | None = None
    repetition: int = 1
    component: int | None = None
    subcomponent: int | None = None
    #: Whether ``[n]`` was written explicitly. ``PID-5`` addresses the whole
    #: field including every repetition; ``PID-5[1]`` addresses only the first.
    repetition_explicit: bool = False

    @classmethod
    def parse(cls, text: str) -> "Path":
        match = _PATH_RE.match(text.strip())
        if not match:
            raise PathError(f"not a valid HL7 path: {text!r}")

        groups = match.groupdict()
        repetition = groups["repetition"]
        return cls(
            segment=groups["segment"].upper(),
            occurrence=int(groups["occurrence"] or 1),
            field=int(groups["field"]) if groups["field"] else None,
            repetition=int(repetition or 1),
            component=int(groups["component"]) if groups["component"] else None,
            subcomponent=(
                int(groups["subcomponent"]) if groups["subcomponent"] else None
            ),
            repetition_explicit=repetition is not None,
        )

    def with_repetition(self, repetition: int) -> "Path":
        """The same path pinned to one repetition. Used when fanning out rules."""
        return Path(
            segment=self.segment,
            occurrence=self.occurrence,
            field=self.field,
            repetition=repetition,
            component=self.component,
            subcomponent=self.subcomponent,
            repetition_explicit=True,
        )

    def with_component(self, component: int) -> "Path":
        """The same path narrowed to one component."""
        return Path(
            segment=self.segment,
            occurrence=self.occurrence,
            field=self.field,
            repetition=self.repetition,
            component=component,
            subcomponent=None,
            repetition_explicit=True,
        )

    def with_subcomponent(self, subcomponent: int) -> "Path":
        """The same path narrowed to one subcomponent."""
        return Path(
            segment=self.segment,
            occurrence=self.occurrence,
            field=self.field,
            repetition=self.repetition,
            component=self.component,
            subcomponent=subcomponent,
            repetition_explicit=True,
        )

    def with_occurrence(self, occurrence: int) -> "Path":
        """The same path pinned to one segment occurrence."""
        return Path(
            segment=self.segment,
            occurrence=occurrence,
            field=self.field,
            repetition=self.repetition,
            component=self.component,
            subcomponent=self.subcomponent,
            repetition_explicit=self.repetition_explicit,
        )

    def __str__(self) -> str:
        text = self.segment
        if self.occurrence != 1:
            text += f"[{self.occurrence}]"
        if self.field is None:
            return text
        text += f"-{self.field}"
        if self.repetition_explicit:
            text += f"[{self.repetition}]"
        if self.component is not None:
            text += f".{self.component}"
            if self.subcomponent is not None:
                text += f".{self.subcomponent}"
        return text


def resolve(message: Message, path: Path) -> Node | None:
    """Return the node a path addresses, or None when it is absent."""
    segment = message.segment(path.segment, path.occurrence)
    if segment is None or path.field is None:
        return None
    return resolve_in_segment(segment, path)


def resolve_in_segment(segment: Segment, path: Path) -> Node | None:
    """Resolve the field-and-below part of a path against one segment."""
    if path.field is None:
        return None

    field = segment.field(path.field)
    if field is None:
        return None

    # Whole field, every repetition.
    if path.component is None and not path.repetition_explicit:
        return field

    repetition = _descend(field, path.repetition)
    if repetition is None or path.component is None:
        return repetition

    component = _descend(repetition, path.component)
    if component is None or path.subcomponent is None:
        return component

    return _descend(component, path.subcomponent)


def ensure(message: Message, path: Path) -> Node | None:
    """Resolve a path, creating missing fields and children along the way.

    Returns None when the *segment* is missing: the engine writes surrogates
    into locations that exist, it does not invent segments a sending system
    never produced.
    """
    segment = message.segment(path.segment, path.occurrence)
    if segment is None or path.field is None:
        return None
    return ensure_in_segment(segment, path)


def ensure_in_segment(segment: Segment, path: Path) -> Node | None:
    if path.field is None:
        return None

    d = segment.delimiters
    # MSH-1 and MSH-2 hold the delimiters themselves and are never rewritten.
    if segment.is_msh and path.field in (1, 2):
        return segment.field(path.field)

    field = segment.ensure_field(path.field)
    if path.component is None and not path.repetition_explicit:
        return field

    repetition = field.ensure_child(path.repetition, d.repetition)
    if path.component is None:
        return repetition

    component = repetition.ensure_child(path.component, d.component)
    if path.subcomponent is None:
        return component

    return component.ensure_child(path.subcomponent, d.subcomponent)


def _descend(node: Node, index: int) -> Node | None:
    """Step one level down a uniformly parsed tree.

    A leaf asked for its first child answers with itself, which is how HL7
    treats ``PID-5.1`` and ``PID-5.1.1`` as the same location when no
    subcomponent delimiter was sent.
    """
    if node.is_leaf:
        return node if index == 1 else None
    return node.child(index)


def leaf_paths(segment: Segment, path: Path) -> list[Path]:
    r"""Expand a path to the leaves beneath it, deepest level only.

    A value is only safe to *write* at a leaf. Addressing ``PID-11.1`` when the
    sender put two subcomponents there reads the subcomponent separator as part
    of the text, and writing that text back escapes the separator -- turning
    ``12 Main St&Apt 2`` into ``12 Main St\T\Apt 2`` and collapsing two
    subcomponents into one. The de-identified message then has no ``PID-11.1.2``
    for a downstream system to read, which is a fidelity loss the leak check and
    the parser's own round-trip gate both miss, because the gate runs on the
    input before anything is written.

    Where nothing deeper was sent, the path is its own leaf, so a feed without
    subcomponents behaves exactly as before.
    """
    node = resolve_in_segment(segment, path)
    if node is None:
        return []
    if node.is_leaf or node.children is None:
        return [path]
    # A subcomponent is already the deepest addressable level.
    if path.subcomponent is not None:
        return [path]
    # One child means no delimiter was sent at this level, and HL7 treats
    # ``PID-5.1`` and ``PID-5.1.1`` as the same location. Staying put keeps the
    # path reported in the manifest the one the sender actually used.
    if len(node.children) == 1:
        return [path]

    if path.component is not None:
        return [path.with_subcomponent(index) for index in range(1, len(node.children) + 1)]

    # A whole repetition: descend through its components to their leaves.
    leaves: list[Path] = []
    for index in range(1, len(node.children) + 1):
        leaves.extend(leaf_paths(segment, path.with_component(index)))
    return leaves


def repetition_nodes(field: Node) -> list[Node]:
    """Every repetition of a field, as a list of at least one node."""
    if field.is_leaf or field.children is None:
        return [field]
    return list(field.children)


def read(message: Message, path: Path | str) -> str:
    """Read a path's text with escape sequences decoded. Empty when absent."""
    if isinstance(path, str):
        path = Path.parse(path)
    node = resolve(message, path)
    if node is None:
        return ""
    return unescape(node.serialize(), message.delimiters)
