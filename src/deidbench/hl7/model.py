"""HL7 v2 structural model.

The tree is deliberately dumb: every node keeps the exact text it was built from
and rebuilds itself by joining its children with the delimiter it was split on.
Because ``str.split(sep)`` and ``sep.join(...)`` are exact inverses, a message
that is parsed and immediately serialized is byte-identical -- including empty
trailing fields, odd repetition counts and malformed segments. That property is
what lets the engine touch one component of one field and leave production
quirks everywhere else untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field


@dataclass
class Delimiters:
    """The five HL7 v2 delimiters, taken from MSH-1 and MSH-2."""

    field: str = "|"
    component: str = "^"
    repetition: str = "~"
    escape: str = "\\"
    subcomponent: str = "&"
    truncation: str = ""

    @property
    def encoding_characters(self) -> str:
        """The MSH-2 value these delimiters came from."""
        return (
            self.component
            + self.repetition
            + self.escape
            + self.subcomponent
            + self.truncation
        )


class Node:
    """A node in the message tree.

    ``children`` is ``None`` for leaves (subcomponents). Non-leaf nodes join
    their children with ``sep`` to serialize.
    """

    __slots__ = ("raw", "sep", "children")

    def __init__(self, raw: str, sep: str | None = None, children: list["Node"] | None = None):
        self.raw = raw
        self.sep = sep
        self.children = children

    def serialize(self) -> str:
        if self.children is None:
            return self.raw
        assert self.sep is not None
        return self.sep.join(child.serialize() for child in self.children)

    # -- leaf access ----------------------------------------------------

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    def set_raw(self, value: str) -> None:
        """Replace this node's text, collapsing it to a leaf."""
        self.raw = value
        self.children = None
        self.sep = None

    def child(self, index: int) -> "Node | None":
        """1-based child access. Returns None when the child is absent."""
        if self.children is None or index < 1 or index > len(self.children):
            return None
        return self.children[index - 1]

    def ensure_child(self, index: int, sep: str) -> "Node":
        """1-based child access that grows the node with empty children.

        A leaf is promoted to a parent whose first child holds the former text,
        which is how HL7 treats ``PID-5`` and ``PID-5.1`` as the same location.
        """
        if self.children is None:
            self.children = [Node(self.raw)]
            self.sep = sep
            self.raw = ""
        while len(self.children) < index:
            self.children.append(Node(""))
        return self.children[index - 1]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Node({self.serialize()!r})"


class Segment:
    """One HL7 v2 segment plus the line terminator that followed it.

    Field numbering is HL7's, not Python's: ``segment.field(5)`` is PID-5. MSH
    is the exception HL7 built in -- MSH-1 *is* the field separator and MSH-2 is
    the encoding characters -- so for MSH a synthetic field holding the
    separator occupies position 1 and the on-wire text is reassembled to match.
    """

    __slots__ = ("name", "fields", "terminator", "delimiters")

    def __init__(
        self,
        name: str,
        fields: list[Node],
        terminator: str,
        delimiters: Delimiters,
    ):
        self.name = name
        self.fields = fields
        self.terminator = terminator
        self.delimiters = delimiters

    @property
    def is_msh(self) -> bool:
        return self.name == "MSH"

    def field(self, number: int) -> Node | None:
        """1-based field access. Returns None when the field is absent."""
        if number < 1 or number > len(self.fields):
            return None
        return self.fields[number - 1]

    def ensure_field(self, number: int) -> Node:
        """1-based field access that grows the segment with empty fields."""
        while len(self.fields) < number:
            self.fields.append(Node(""))
        return self.fields[number - 1]

    def serialize(self) -> str:
        d = self.delimiters
        if self.is_msh and self.fields:
            # fields[0] is the separator itself; it is not *joined* by anything.
            head = self.name + self.fields[0].serialize()
            rest = d.field.join(f.serialize() for f in self.fields[1:])
            body = head + rest
        else:
            body = d.field.join(
                [self.name] + [f.serialize() for f in self.fields]
            )
        return body + self.terminator

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Segment({self.name}, {len(self.fields)} fields)"


class Message:
    """A parsed HL7 v2 message."""

    __slots__ = ("segments", "delimiters", "preamble")

    def __init__(
        self,
        segments: list[Segment],
        delimiters: Delimiters,
        preamble: str = "",
    ):
        self.segments = segments
        self.delimiters = delimiters
        #: Any bytes before the first segment (MLLP framing, stray newlines).
        self.preamble = preamble

    def serialize(self) -> str:
        return self.preamble + "".join(s.serialize() for s in self.segments)

    def segments_named(self, name: str) -> list[Segment]:
        return [s for s in self.segments if s.name == name]

    def segment(self, name: str, occurrence: int = 1) -> Segment | None:
        """1-based occurrence of a named segment, e.g. the 2nd NK1."""
        matches = self.segments_named(name)
        if occurrence < 1 or occurrence > len(matches):
            return None
        return matches[occurrence - 1]

    @property
    def control_id(self) -> str:
        """MSH-10, the message control ID. Empty string when absent."""
        msh = self.segment("MSH")
        if msh is None:
            return ""
        node = msh.field(10)
        return node.serialize() if node else ""

    @property
    def message_type(self) -> str:
        """MSH-9 as sent, e.g. ``ADT^A04^ADT_A01``."""
        msh = self.segment("MSH")
        if msh is None:
            return ""
        node = msh.field(9)
        return node.serialize() if node else ""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        names = ",".join(s.name for s in self.segments)
        return f"Message({names})"
