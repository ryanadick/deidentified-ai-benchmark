"""Philter as a published baseline, via the ``philter-lite`` port.

Philter is UCSF's clinical de-identifier, published in *npj Digital Medicine*
(Norgeot et al., 2020) and run there over 130M notes. It is the second
published comparator in this benchmark, after Microsoft Presidio in
:mod:`.presidio_baseline`, and it is interesting precisely because it fails
differently.

**Philter is a redactor, not a classifier.** Presidio looks for things it
recognizes as PHI. Philter inverts that: it keeps only what its filter set can
confirm is *safe* and redacts the rest. On a 978-byte HL7 message it returns
57 spans where Presidio returns 27, and it finds the caret-delimited name
components Presidio cannot see -- not because it understands HL7, but because
it does not need to understand anything to redact. That is a real privacy
advantage and a real utility cost, and the ``chars x`` column is where the cost
shows up.

**Which Philter this is, and why.** The arm runs ``philter-lite``, a BSD fork
of ``philter-ucsf`` by Sirona Medical, not the original distribution. This is a
compatibility necessity, not a preference: 324 of the 617 regex filter files in
the published ``philter_delta`` set place an inline ``(?i)`` flag mid-pattern,
which Python has rejected as an error since 3.11. The published tool cannot run
on a supported Python at all.

The fork's patterns are the published patterns with ``(?i)`` moved to the
front. That relocation is **semantics-preserving by construction**: an inline
global flag always applied to the entire pattern, whichever position it
appeared in, which is exactly why Python made the misleading position an error.
``test_philter_baseline.py`` pins a sample of patterns against the published
filter files verbatim, including an upstream typo the fork faithfully preserves
-- a transcription fingerprint, since a rewrite would have fixed it.

Report this row as "Philter (philter-lite port)". It is not a measurement of
code UCSF ships.

**On the PHI type mapping.** Philter's type vocabulary is seven values wide,
and on HL7 input it labels the overwhelming majority of what it finds ``OTHER``
-- it knows the span is unsafe without claiming to know what it is. There is no
honest mapping from ``OTHER`` into this corpus's categories, so it is reported
under the same ``unmapped:`` prefix the Presidio arm uses: visible in the
false-positive breakdown, and never a typed match. The consequence is that
**Philter's typed score is close to meaningless** and its untyped score is the
only comparable number. Treating ``OTHER`` as a wildcard that matched any gold
category would manufacture typed accuracy Philter never demonstrated.
"""

from __future__ import annotations

import functools

from ..surrogate.types import PhiType
from .detectors import (
    Detection,
    NarrativeCandidate,
    _resolve_overlaps,
    stable_span_order,
)

#: The published filter configuration, shipped inside the package. Philter's
#: behaviour *is* its filter set, so the config name belongs in the provenance
#: next to the version.
DEFAULT_CONFIG = "philter_delta"

#: Philter's PHI type vocabulary -> this corpus's. Lossy in the same direction
#: as the Presidio mapping, and far more so: ``OTHER`` dominates real output.
PHILTER_PHI_TYPE_MAP: dict[str, str] = {
    "Name": str(PhiType.FAMILY_NAME),
    "Email": str(PhiType.EMAIL),
    "Patient_Social_Security_Number": str(PhiType.SSN),
    "Provider_Address_or_Location": str(PhiType.STREET_ADDRESS),
    "DATE": str(PhiType.EVENT_DATE),
    "Age": str(PhiType.IDENTIFIER),
    # "OTHER" is deliberately absent. Philter uses it for "this is unsafe and
    # I am not claiming to know what it is", which has no counterpart here.
}

_IMPORT_HINT = (
    "Philter is not installed. Install the optional baseline extra and the "
    "NLTK tagger it needs:\n"
    '    pip install -e ".[philter]"\n'
    "    python -m nltk.downloader averaged_perceptron_tagger "
    "averaged_perceptron_tagger_eng punkt punkt_tab"
)


class PhilterUnavailable(RuntimeError):
    """Philter, or the NLTK data it needs, is not importable."""


def philter_available() -> bool:
    """Whether the Philter arms can run in this environment."""
    try:
        import philter_lite  # noqa: F401
    except ImportError:
        return False
    return True


def map_phi_type(phi_type: str) -> str:
    """The corpus PHI category for a Philter PHI type.

    ``OTHER`` and anything else unmapped keeps its identity under a prefix
    rather than being forced into a category Philter did not claim.
    """
    mapped = PHILTER_PHI_TYPE_MAP.get(phi_type)
    return mapped if mapped is not None else f"unmapped:{phi_type}"


@functools.lru_cache(maxsize=4)
def _load_filters(config: str):
    """Philter's filter set, loaded and compiled once.

    Cached because ``philter_delta`` compiles 313 filters over several hundred
    pattern files; doing that per message would dominate the runtime.
    """
    try:
        import philter_lite
    except ImportError as exc:  # pragma: no cover - exercised by env, not tests
        raise PhilterUnavailable(_IMPORT_HINT) from exc

    from importlib.resources import files

    config_path = files("philter_lite") / "configs" / f"{config}.toml"
    if not config_path.is_file():
        raise PhilterUnavailable(
            f"Philter config {config!r} is not shipped with philter_lite"
        )
    return philter_lite.load_filters(str(config_path))


def philter_provenance(config: str = DEFAULT_CONFIG) -> dict:
    """Versions and configuration that define this baseline.

    Records the port explicitly. "Philter scored X" without saying which
    Philter, on which filter set, is not a reproducible claim.
    """
    provenance: dict = {
        "tool": "philter",
        "distribution": "philter-lite (fork of philter-ucsf)",
        "config": config,
        "configuration": "published philter_delta filter set, untuned",
    }
    try:
        from importlib.metadata import PackageNotFoundError, version

        for package in ("philter-lite", "nltk"):
            try:
                provenance[package] = version(package)
            except PackageNotFoundError:
                provenance[package] = "not installed"
    except ImportError:  # pragma: no cover
        pass
    return provenance


def _detect_spans(text: str, config: str) -> list[tuple[int, int, str]]:
    """Philter's PHI spans for one string, as ``(start, end, phi_type)``.

    ``detect_phi`` returns an include map, an exclude map and a tracker. The
    tracker's ``phi`` list is the typed view of the exclude map, which is what
    this benchmark scores.
    """
    import philter_lite

    _include, _exclude, tracker = philter_lite.detect_phi(
        text, _load_filters(config)
    )
    return stable_span_order(
        [
            (entry.start, entry.stop, map_phi_type(entry.phi_type))
            for entry in tracker.phi
        ]
    )


class PhilterTextDetector:
    """Philter over the flattened message.

    Directly comparable to ``PresidioTextDetector`` and to the hand-written
    ``FlattenedTextDetector``: same input, same scoring, a different published
    tool.
    """

    def __init__(self, config: str = DEFAULT_CONFIG, name: str = "philter-text"):
        self.name = name
        self.config = config

    @property
    def provenance(self) -> dict:
        return philter_provenance(self.config)

    def detect(self, text: str) -> list[Detection]:
        detections = [
            Detection(
                start=start,
                end=end,
                text=text[start:end],
                phi_type=phi_type,
                source=self.name,
            )
            for start, end, phi_type in _detect_spans(text, self.config)
        ]
        return _resolve_overlaps(detections)


class PhilterNarrativeDetector:
    """Philter behind the schema-conditioning seam.

    Satisfies the ``NarrativeDetector`` protocol, so it drops into
    ``SchemaConditionedDetector`` as one constructor argument.

    As with Presidio, ``candidate.context_line()`` is **not** passed in.
    Philter has no interface for structural context, and prepending text to the
    value would shift every offset and make this a tool we modified rather than
    the tool other people run.

    Philter's redact-what-is-not-safe design makes this arm worth watching for
    the opposite reason to Presidio's. Confining it to deferred locations
    removes the coded fields it was redacting wholesale, which should help
    precision -- but Philter was never losing much recall to those fields in
    the first place, so there is less for the architecture to recover.
    """

    def __init__(
        self, config: str = DEFAULT_CONFIG, name: str = "philter-narrative"
    ):
        self.name = name
        self.config = config

    @property
    def provenance(self) -> dict:
        return philter_provenance(self.config)

    def detect_in(self, candidate: NarrativeCandidate) -> list[tuple[int, int, str]]:
        return _detect_spans(candidate.text, self.config)
