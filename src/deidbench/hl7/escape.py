"""HL7 v2 escape sequence handling.

Values are stored in the tree exactly as they arrived on the wire, so a field
that is never touched round-trips byte for byte. Unescaping happens only when
something needs to *read* a value -- a rule matching on text, or the structural
context handed to a model. Escaping happens only when a surrogate is written
back.
"""

from __future__ import annotations

import re

from .model import Delimiters


def _sequence_map(d: Delimiters) -> dict[str, str]:
    return {
        "F": d.field,
        "S": d.component,
        "T": d.subcomponent,
        "R": d.repetition,
        "E": d.escape,
    }


def unescape(text: str, d: Delimiters) -> str:
    """Decode HL7 escape sequences into their literal characters.

    Sequences this engine does not interpret -- highlighting (``\\H\\``,
    ``\\N\\``), charset switches (``\\Cxxyy\\``, ``\\Mxxyyzz\\``) and locally
    defined ``\\Zxx\\`` -- are left in place rather than dropped, so no
    information is silently lost.
    """
    if not text or d.escape not in text:
        return text

    esc = d.escape
    simple = _sequence_map(d)
    out: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch != esc:
            out.append(ch)
            i += 1
            continue

        end = text.find(esc, i + 1)
        if end == -1:
            # Unterminated escape: a real-world quirk. Keep it verbatim.
            out.append(text[i:])
            break

        body = text[i + 1 : end]
        if body in simple:
            out.append(simple[body])
        elif body[:1] == "X" and _is_hex(body[1:]):
            out.append(_decode_hex(body[1:]))
        else:
            out.append(text[i : end + 1])
        i = end + 1

    return "".join(out)


def escape(text: str, d: Delimiters) -> str:
    """Encode delimiter characters in ``text`` as HL7 escape sequences.

    The escape character is handled first so that newly written delimiter
    sequences are not themselves re-escaped.
    """
    if not text:
        return text

    esc = d.escape
    result = text.replace(esc, esc + "E" + esc)
    for code, char in (
        ("F", d.field),
        ("S", d.component),
        ("T", d.subcomponent),
        ("R", d.repetition),
    ):
        if char:
            result = result.replace(char, esc + code + esc)
    return result


_HEX_RE = re.compile(r"\A(?:[0-9A-Fa-f]{2})+\Z")


def _is_hex(body: str) -> bool:
    return bool(body) and bool(_HEX_RE.match(body))


def _decode_hex(body: str) -> str:
    try:
        return bytes.fromhex(body).decode("latin-1")
    except ValueError:  # pragma: no cover - guarded by _is_hex
        return body
