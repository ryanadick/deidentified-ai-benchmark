"""HL7 v2 parsing with byte-exact round-trip fidelity."""

from .escape import escape, unescape
from .model import Delimiters, Message, Node, Segment
from .parser import parse, reparse_field, serialize
from .path import (
    Path,
    PathError,
    ensure,
    leaf_paths,
    read,
    repetition_nodes,
    resolve,
)
from .spans import (
    LeafSpan,
    candidate_policy_paths,
    find_enclosing,
    leaf_index,
    serialize_with_spans,
)

__all__ = [
    "Delimiters",
    "LeafSpan",
    "Message",
    "Node",
    "Path",
    "PathError",
    "Segment",
    "candidate_policy_paths",
    "ensure",
    "find_enclosing",
    "leaf_index",
    "escape",
    "parse",
    "read",
    "leaf_paths",
    "repetition_nodes",
    "reparse_field",
    "resolve",
    "serialize",
    "serialize_with_spans",
    "unescape",
]
