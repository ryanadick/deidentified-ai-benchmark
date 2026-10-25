"""Microsoft Presidio as a real, published baseline.

The text-only arm in :mod:`.detectors` is a stand-in we wrote ourselves. That
makes it readable and criticizable, but it also makes it *ours*: a comparison
against an opponent of our own construction, which cannot settle the question
of whether a published tool does better. This module wires in
`Microsoft Presidio <https://microsoft.github.io/presidio/>`_ so the comparison
is measured rather than self-made.

Two arms are provided, and the pair is the point:

``PresidioTextDetector``
    Presidio over the flattened message, which is how a free-text
    de-identification tool meets an HL7 v2 feed today. Directly comparable to
    ``FlattenedTextDetector``: same input, same scoring, different detector.

``PresidioNarrativeDetector``
    Presidio behind the schema-conditioning seam, applied only to the locations
    the policy defers. Without this arm, "schema-conditioned beats text-only"
    would confound two separate variables -- the architecture and the quality
    of the detector inside it. With it, Presidio appears on both sides of the
    architectural comparison and the remaining gap is attributable to the
    architecture alone.

**Configuration is deliberately out of the box.** Default recognizers, default
confidence scores, no score threshold, no custom deny-lists, no HL7-specific
tuning. A baseline we tuned ourselves would have the same credibility problem
as a baseline we wrote ourselves. ``score_threshold`` is exposed because it is
Presidio's own knob and readers will ask, but the published figures use the
untuned default.

**On the entity mapping.** Presidio's vocabulary is coarser than this corpus's
ground truth, and the mapping below is lossy in one direction only. ``PERSON``
cannot distinguish a family name from a given name from a clinician's name, and
``DATE_TIME`` cannot distinguish a birth date from an event date. Each is
mapped to a single representative category, which costs Presidio *typed*
recall for a distinction it was never built to draw. Read Presidio's untyped
score as the comparable number and its typed score as a floor. The flattened
regex arm is mapped the same way -- every name it finds is reported as
``family_name`` -- so the two text-only arms are penalized identically.

Entity types with no counterpart in the corpus vocabulary (``URL``,
``CREDIT_CARD``, ``NRP`` and friends) are **not** discarded. They are reported
under an ``unmapped:`` prefix so they still count against precision and show up
by name in the report's false-positive breakdown. Dropping them would quietly
flatter Presidio by hiding predictions it actually made.

**Ties are broken deterministically, which Presidio does not do for us.**
Presidio can return the same span under two entity types with equal confidence
-- a bare number that is equally a bank account and a driver's licence. Which
one survives overlap resolution depended on the order Presidio emitted them,
and that order varies between processes with Python's hash randomization. The
untyped score was unaffected (the span is identical either way) but the typed
score drifted by ~13 spans per 500 messages between runs. ``stable_span_order``
imposes a total order so every process resolves ties the same way. See its
docstring for why alphabetical-by-type is a fine answer and drift is not.

**One post-processing step is applied, and it is ours, not Presidio's.**
Overlapping predictions are reduced to the longest, via the same
``_resolve_overlaps`` every other arm uses. Presidio's recognizers do overlap
each other, and counting one finding twice would distort precision. Applying
the same reduction to every arm keeps them comparable, and the direction of the
effect favours Presidio: it lowers its prediction count without costing it any
coverage. It does mean a wide span can suppress a precise one nested inside it,
which shows up as a partial rather than a hit -- visible in the ``partial``
column, where the flattened Presidio arm is the only one with a non-zero count.
"""

from __future__ import annotations

import functools
import importlib.util
from dataclasses import dataclass

from ..surrogate.types import PhiType
from .detectors import (
    Detection,
    NarrativeCandidate,
    _resolve_overlaps,
    stable_span_order,
)

#: The spaCy model behind the published figures. Presidio's own default, and
#: pinned here because the NER model is part of the result: a different model
#: is a different baseline, not the same baseline run again.
DEFAULT_SPACY_MODEL = "en_core_web_lg"

#: Presidio entity type -> this corpus's PHI vocabulary. Lossy where Presidio
#: is coarser; see the module docstring. Anything absent here is reported as
#: ``unmapped:<ENTITY_TYPE>`` rather than dropped.
PRESIDIO_ENTITY_MAP: dict[str, str] = {
    # Presidio's one name category has to stand for all six of ours. Mapped to
    # family_name for parity with the regex arm's identical collapse.
    "PERSON": str(PhiType.FAMILY_NAME),
    "EMAIL_ADDRESS": str(PhiType.EMAIL),
    "PHONE_NUMBER": str(PhiType.PHONE),
    "US_SSN": str(PhiType.SSN),
    # One date category for birth_date, event_date and death_date.
    "DATE_TIME": str(PhiType.EVENT_DATE),
    # One location category for street_address, city, state, county, country
    # and postal_code.
    "LOCATION": str(PhiType.CITY),
    "ORGANIZATION": str(PhiType.ORGANIZATION),
    "US_DRIVER_LICENSE": str(PhiType.LICENSE_NUMBER),
    "MEDICAL_LICENSE": str(PhiType.LICENSE_NUMBER),
    "US_PASSPORT": str(PhiType.IDENTIFIER),
    "US_ITIN": str(PhiType.IDENTIFIER),
    "US_BANK_NUMBER": str(PhiType.ACCOUNT_NUMBER),
    "AGE": str(PhiType.IDENTIFIER),
}

#: Shown when the arm is skipped. The model is installed by URL rather than
#: with ``spacy download`` because that command shells out to pip, which a
#: uv-created venv does not have, and resolves a model version that need not
#: match the pinned spaCy.
SPACY_MODEL_WHEEL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl"
)

_IMPORT_HINT = (
    "Presidio, or its language model, is not available. Install the optional "
    "baseline extra and the model:\n"
    '    pip install -e ".[presidio]"\n'
    f"    pip install {SPACY_MODEL_WHEEL}"
)


class PresidioUnavailable(RuntimeError):
    """Presidio, or its language model, is not importable."""


def presidio_available(spacy_model: str = DEFAULT_SPACY_MODEL) -> bool:
    """Whether the Presidio arms can run in this environment.

    Checks the language model as well as the package. Presidio imports fine
    without a model and then fails when the engine is built, which turns a
    missing optional dependency into a crash mid-run and makes the test suite
    error instead of skipping. The model is checked by package presence rather
    than by loading it, because loading costs seconds and this is called during
    test collection.
    """
    try:
        import presidio_analyzer  # noqa: F401
    except ImportError:
        return False
    return importlib.util.find_spec(spacy_model) is not None


def map_entity_type(entity_type: str) -> str:
    """The corpus PHI category for a Presidio entity type.

    Unmapped types keep their identity under a prefix rather than being
    silently dropped or forced into an unrelated category.
    """
    mapped = PRESIDIO_ENTITY_MAP.get(entity_type)
    return mapped if mapped is not None else f"unmapped:{entity_type}"


@functools.lru_cache(maxsize=4)
def _build_analyzer(spacy_model: str):
    """An ``AnalyzerEngine`` with default recognizers and a pinned NER model.

    Cached because construction loads the spaCy model, which costs seconds and
    hundreds of megabytes; the engine is stateless across ``analyze`` calls, so
    one instance serves every arm and every message.
    """
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
    except ImportError as exc:  # pragma: no cover - exercised by env, not tests
        raise PresidioUnavailable(_IMPORT_HINT) from exc

    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": spacy_model}],
        }
    )
    try:
        nlp_engine = provider.create_engine()
    except OSError as exc:  # spaCy model absent
        raise PresidioUnavailable(
            f"The spaCy model {spacy_model!r} is not installed.\n"
            f"    python -m spacy download {spacy_model}"
        ) from exc

    return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])


def presidio_provenance(spacy_model: str = DEFAULT_SPACY_MODEL) -> dict:
    """Versions that define the baseline, for the report.

    A baseline's version is part of its result. "Presidio scored X" is not
    reproducible without it.
    """
    provenance: dict = {"spacy_model": spacy_model, "configuration": "default recognizers, no score threshold"}
    try:
        from importlib.metadata import PackageNotFoundError, version

        for package in ("presidio-analyzer", "spacy"):
            try:
                provenance[package] = version(package)
            except PackageNotFoundError:
                provenance[package] = "not installed"
    except ImportError:  # pragma: no cover
        pass

    try:
        import spacy

        meta = spacy.load(spacy_model).meta
        provenance["spacy_model_version"] = meta.get("version", "unknown")
    except Exception:  # pragma: no cover - provenance is best-effort
        provenance.setdefault("spacy_model_version", "unknown")

    return provenance


@dataclass
class _PresidioRunner:
    """Shared analyze-and-map logic for both Presidio arms."""

    spacy_model: str = DEFAULT_SPACY_MODEL
    #: Presidio's own confidence filter. ``0.0`` keeps everything the default
    #: recognizers return, which is the untuned configuration.
    score_threshold: float = 0.0

    def analyze(self, text: str) -> list[tuple[int, int, str]]:
        analyzer = _build_analyzer(self.spacy_model)
        results = analyzer.analyze(
            text=text,
            language="en",
            score_threshold=self.score_threshold,
        )
        return stable_span_order(
            [
                (result.start, result.end, map_entity_type(result.entity_type))
                for result in results
            ]
        )


class PresidioTextDetector:
    """Presidio over the flattened message: the published text-only baseline.

    The arm ``FlattenedTextDetector`` was standing in for. Same input, same
    scoring; the difference in score is the difference between a detector we
    wrote and one a team published and other people cite.
    """

    def __init__(
        self,
        spacy_model: str = DEFAULT_SPACY_MODEL,
        score_threshold: float = 0.0,
        name: str = "presidio-text",
    ):
        self.name = name
        self.spacy_model = spacy_model
        self.score_threshold = score_threshold
        self._runner = _PresidioRunner(spacy_model, score_threshold)

    @property
    def provenance(self) -> dict:
        return presidio_provenance(self.spacy_model) | {
            "score_threshold": self.score_threshold
        }

    def detect(self, text: str) -> list[Detection]:
        detections = [
            Detection(
                start=start,
                end=end,
                text=text[start:end],
                phi_type=phi_type,
                source=self.name,
            )
            for start, end, phi_type in self._runner.analyze(text)
        ]
        return _resolve_overlaps(detections)


class PresidioNarrativeDetector:
    """Presidio behind the schema-conditioning seam.

    Satisfies the ``NarrativeDetector`` protocol, so it drops into
    ``SchemaConditionedDetector`` as one constructor argument -- the same seam
    the local open-weight model is meant to occupy.

    It receives only the locations the policy defers, and receives them one at
    a time. That is a real handicap as well as a real advantage: Presidio sees
    none of the coded fields that generate most of its flattened-mode false
    positives, but it also sees each narrative value with no surrounding
    sentence to draw on. Both effects are part of what the arm measures.

    ``candidate.context_line()`` is deliberately **not** passed to Presidio.
    Presidio has no interface for structural context, and inventing one by
    prepending text to the value would change the offsets and make this a tool
    we modified rather than the tool other people run.
    """

    def __init__(
        self,
        spacy_model: str = DEFAULT_SPACY_MODEL,
        score_threshold: float = 0.0,
        name: str = "presidio-narrative",
    ):
        self.name = name
        self.spacy_model = spacy_model
        self.score_threshold = score_threshold
        self._runner = _PresidioRunner(spacy_model, score_threshold)

    @property
    def provenance(self) -> dict:
        return presidio_provenance(self.spacy_model) | {
            "score_threshold": self.score_threshold
        }

    def detect_in(self, candidate: NarrativeCandidate) -> list[tuple[int, int, str]]:
        return self._runner.analyze(candidate.text)
