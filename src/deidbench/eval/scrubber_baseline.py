"""NLM Scrubber as a published baseline, with spans recovered by alignment.

NLM Scrubber is the National Library of Medicine's clinical de-identifier
(Kayaalp et al., Lister Hill Center). It is the third published comparator,
after Presidio in :mod:`.presidio_baseline` and Philter in
:mod:`.philter_baseline`, and it is the most awkward of the three to measure
for a reason worth stating plainly.

**Scrubber does not report spans.** Presidio returns offsets; Philter returns a
coordinate map; Scrubber returns *redacted text* with inline bracketed tags --
``Call me [PERSONALNAME].`` -- which changes the length of the document and
shifts every offset after it. Every span this module reports is therefore
**recovered by aligning Scrubber's output against the original input**, not
read off the tool. That is a methodological difference from the other two arms
and it is disclosed in the report, because alignment can fail where the other
two cannot.

The approach follows Ferrell et al. (2020), who hit the same problem and solved
it with an ``nlm2brat.py`` converter anchoring bracketed redactions to
surrounding context, reporting 99.77-100% successful anchoring by category.
That script does not appear to be publicly released, so :func:`recover_spans`
is our own implementation of the same idea.

**It must be tag-anchored, not a character diff.** A ``difflib`` comparison of
the two strings looks like the obvious approach and is quietly wrong: it
matches the letters of ``[ALPHANUMERICID]`` against similar letters in the
original, shattering one redaction into dozens of fragments and losing the tag
names altogether. It still produces a plausible-looking recall number -- which
is how this nearly shipped. The working version splits the output into literal
runs and tags, anchors the literals forward through the original, and treats
what lies between anchors as redacted. ``verify_recovery`` then requires that
substituting the recovered regions back reproduces Scrubber's output exactly.
All 500 corpus messages and all 984 deferred narrative locations pass; a
message that fails is excluded rather than scored.

The second way this arm went quietly wrong is worth recording too, because the
guard is what caught it. Reading Scrubber's output with ``Path.read_text()``
applies universal-newline translation, turning the CR terminators HL7 uses into
LF. No anchor containing a line ending then matches, every message is refused,
and the arm reports recall 0.000 -- indistinguishable from a tool that simply
found nothing. Always ``read_bytes().decode()`` here.

**Adjacent identifiers do not merge, which was the opposite of the guess.**
HL7 packs identifiers behind single-character delimiters -- 50.1% of gold spans
in this corpus sit within one character of another -- so merged redactions
looked like the dominant risk. They do not occur: Scrubber leaves the delimiter
standing, so ``[PERSONALNAME]^[PERSONALNAME]`` has a surviving anchor between
the tags and both spans are located precisely. ``RecoveredSpan.merged`` and
``credit_merged`` handle the case for correctness, but it measured zero times.

**The over-breadth is in single wide tags instead.** One tag can replace a long
run spanning several fields. Under this benchmark's rule -- a gold span is found
when a prediction fully covers it -- every identifier inside such a region
counts as found, so some of this arm's recall rests on regions far wider than
the PHI they cover. Aggregate over-flagging is nonetheless low at **0.6x**
characters: wide regions in a few places, not indiscriminate redaction. The
README reports the strict figure alongside the headline one.

**What it finds, and what that means.** Scrubber behaves like Presidio, not
like Philter, and more sharply than either: **0.271 recall on structured fields
against 0.743 on narrative**, the widest split of the three published tools. It
catches names in ``OBX``/``NTE`` prose and leaves ``Everman^Clara^Jean^JR``
untouched, scores 0.02 on ``PV1`` and 0.05 on ``NK1``, and finds nothing at all
in Z-segments. Its published NAME recall on i2b2 prose notes is 0.9393. One
prose-trained tool dropping that far when only the *shape* of the data changes
is the clearest single piece of evidence for what this benchmark measures.

**Operational notes.** The binary is a 2019 x86-64 Linux ELF, not a Python
package, so there is no pip extra. It is not vendored: the download page states
no redistribution terms, so users fetch it themselves and this module pins the
version and SHA-256 in its provenance. On non-x86-64 hosts it runs under
container emulation. Loading the program costs most of a minute *per
invocation*, so detection is **batched**: one Scrubber run per corpus, cached
by content hash. See :class:`ScrubberBatch`.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..surrogate.types import PhiType
from .detectors import (
    Detection,
    NarrativeCandidate,
    _resolve_overlaps,
    stable_span_order,
)

#: The build these figures were produced with. A different build is a
#: different baseline, not the same baseline run again.
DEFAULT_VERSION = "19.0403L"

#: SHA-256 of ``scrubber.19.0403L.zip`` as downloaded from the NLM site.
#: Recorded because the binary is not redistributed with this package, so the
#: only way a reader can confirm they ran the same tool is to check the digest.
KNOWN_ARCHIVE_SHA256 = (
    "4dbe08a458433c3dac71007ce17ae81b7d0f392e1370daedb3c65da47f79e48d"
)

#: Where to find the binary, and how to run it on a host that is not x86-64.
ENV_BINARY = "DEIDBENCH_SCRUBBER"
ENV_IMAGE = "DEIDBENCH_SCRUBBER_IMAGE"
DEFAULT_IMAGE = "debian:bullseye"

#: Scrubber's five tags -> this corpus's PHI vocabulary. Coarser even than
#: Philter's, and lossy in the same single direction.
SCRUBBER_TAG_MAP: dict[str, str] = {
    "PERSONALNAME": str(PhiType.FAMILY_NAME),
    "ADDRESS": str(PhiType.STREET_ADDRESS),
    "DATE": str(PhiType.EVENT_DATE),
    "ALPHANUMERICID": str(PhiType.IDENTIFIER),
    # AGE90+ has no counterpart: HIPAA only treats ages over 89 as an
    # identifier, this corpus plants no ages at all, and the 30 age-like
    # strings it does contain are all "54-year-old". Left unmapped rather than
    # forced into a category.
}

_IMPORT_HINT = (
    "NLM Scrubber is not available. It is a binary distribution, not a Python\n"
    "package, and it is not redistributed with this benchmark.\n"
    "  1. Download the Linux x86_64 build (v.19.0403L) from\n"
    "     https://lhncbc.nlm.nih.gov/scrubber/download.html\n"
    f"     Expected archive SHA-256: {KNOWN_ARCHIVE_SHA256}\n"
    "  2. Unzip it and point the benchmark at the binary:\n"
    f"     export {ENV_BINARY}=/path/to/scrubber.19.0403.lnx\n"
    "  3. On a non-x86-64 host, Docker is used to run it under emulation.\n"
    f"     Override the image with {ENV_IMAGE} if needed."
)


class ScrubberUnavailable(RuntimeError):
    """The Scrubber binary, or the runtime needed to execute it, is missing."""


def _binary_path() -> Path | None:
    raw = os.environ.get(ENV_BINARY)
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_file() else None


def _needs_emulation() -> bool:
    """Whether the host can execute an x86-64 Linux ELF directly."""
    return not (
        platform.system() == "Linux" and platform.machine() in {"x86_64", "amd64"}
    )


def scrubber_available() -> bool:
    """Whether the Scrubber arms can run in this environment."""
    if _binary_path() is None:
        return False
    if _needs_emulation() and shutil.which("docker") is None:
        return False
    return True


def map_tag(tag: str) -> str:
    """The corpus PHI category for a Scrubber tag.

    Unmapped tags keep their identity under a prefix rather than being dropped
    or forced into an unrelated category.
    """
    mapped = SCRUBBER_TAG_MAP.get(tag)
    return mapped if mapped is not None else f"unmapped:{tag}"


# -- span recovery ------------------------------------------------------


@dataclass(frozen=True)
class RecoveredSpan:
    """One redacted region, located in the *original* text.

    ``tags`` holds every Scrubber tag found in the replacement text. More than
    one means Scrubber redacted across a boundary between two values and this
    region cannot be split -- the case ``merged`` marks.
    """

    start: int
    end: int
    tags: tuple[str, ...]
    merged: bool

    @property
    def phi_type(self) -> str:
        """A single category for scoring.

        A merged region has no one category, so it is reported under the
        ``merged:`` prefix: visible in the report, and never a typed match.
        Picking one of its tags would assert a precision the output does not
        contain.
        """
        if self.merged:
            return "merged:" + "+".join(sorted(self.tags))
        return map_tag(self.tags[0]) if self.tags else "unmapped:UNKNOWN"


#: A Scrubber tag in its output. ``AGE90+`` is why ``+`` is in the class.
TAG_PATTERN = re.compile(r"\[([A-Z][A-Z0-9+]*)\]")

#: Scrubber appends a provenance banner to every output file, recording the
#: config path, output path, timestamp and version. It is not part of the
#: de-identified document and must come off before alignment -- left on, it
#: makes every message fail the round-trip check.
FOOTER_MARKER = "\n##### DOCUMENT "


def strip_footer(scrubbed: str) -> str:
    """Scrubber's output without its trailing banner."""
    index = scrubbed.find(FOOTER_MARKER)
    return scrubbed if index < 0 else scrubbed[:index]


def recover_spans(original: str, scrubbed: str) -> list[RecoveredSpan] | None:
    """Locate Scrubber's redactions in the original text, or ``None``.

    Scrubber's output cannot be scored directly: the bracketed tags are a
    different length from the text they replace, so every offset after the
    first redaction is wrong.

    The recovery is **tag-anchored**, not a character diff. A character-level
    diff is wrong here in a way that is easy to miss: it happily matches the
    letters of ``[ALPHANUMERICID]`` against similar letters in the original,
    fragmenting one redaction into dozens of regions and losing the tag names
    entirely. Instead the scrubbed output is split into alternating literal
    runs and tags; the literal runs are text Scrubber left alone, so they are
    matched forward through the original, and whatever lies between two
    consecutive anchors is the region the intervening tag replaced.

    This also answers the merged-region question exactly rather than by
    heuristic. If two tags are separated by a literal -- ``[PERSONALNAME]^``
    ``[PERSONALNAME]`` -- the surviving delimiter splits them and both spans
    are located precisely. Only when the literal between two tags is *empty*
    is the boundary genuinely unrecoverable, and that is the case ``merged``
    marks.

    Returns ``None`` when an anchor cannot be found, which means the alignment
    is untrustworthy for this message. Callers must not score a ``None``.
    """
    parts: list[tuple[str, str]] = []
    position = 0
    for match in TAG_PATTERN.finditer(scrubbed):
        parts.append(("literal", scrubbed[position : match.start()]))
        parts.append(("tag", match.group(1)))
        position = match.end()
    parts.append(("literal", scrubbed[position:]))

    spans: list[RecoveredSpan] = []
    cursor = 0
    pending: list[str] = []
    pending_start: int | None = None

    for kind, value in parts:
        if kind == "tag":
            if pending_start is None:
                pending_start = cursor
            pending.append(value)
            continue
        if not value:
            # Consecutive tags with nothing between them: the boundary is not
            # in the output, so keep accumulating into one merged region.
            continue
        found = original.find(value, cursor)
        if found < 0:
            return None
        if pending:
            spans.append(
                RecoveredSpan(
                    start=pending_start if pending_start is not None else cursor,
                    end=found,
                    tags=tuple(pending),
                    merged=len(pending) > 1,
                )
            )
            pending, pending_start = [], None
        cursor = found + len(value)

    if pending:
        spans.append(
            RecoveredSpan(
                start=pending_start if pending_start is not None else cursor,
                end=len(original),
                tags=tuple(pending),
                merged=len(pending) > 1,
            )
        )
    return [span for span in spans if span.end > span.start]


def apply_spans(original: str, spans: list[RecoveredSpan]) -> str:
    """Rebuild Scrubber's output from the original and the recovered spans."""
    pieces: list[str] = []
    cursor = 0
    for span in spans:
        pieces.append(original[cursor : span.start])
        pieces.extend(f"[{tag}]" for tag in span.tags)
        cursor = span.end
    pieces.append(original[cursor:])
    return "".join(pieces)


def verify_recovery(original: str, scrubbed: str) -> bool:
    """Whether the recovered spans reproduce Scrubber's output exactly.

    The guard on the alignment, and the reason this arm can be trusted at all.
    Mis-located spans would corrupt every number the arm produces while
    looking entirely plausible, so a message that fails this is excluded rather
    than scored.
    """
    scrubbed = strip_footer(scrubbed)
    spans = recover_spans(original, scrubbed)
    if spans is None:
        return False
    return apply_spans(original, spans) == scrubbed


# -- running the binary -------------------------------------------------


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class ScrubberBatch:
    """One Scrubber invocation over many texts, cached by content hash.

    Scrubber spends most of a minute loading its dictionaries *per
    invocation*, and the narrative arm asks about thousands of short strings.
    Calling it per string would take hours, so every text is written into one
    directory, Scrubber runs once, and results are looked up by digest.

    This is why the arm needs priming before the harness runs: see
    :meth:`prime`.
    """

    image: str = DEFAULT_IMAGE
    #: digest -> recovered spans.
    _cache: dict[str, list[RecoveredSpan]] = field(default_factory=dict)
    #: digests whose alignment failed to round-trip, which must not be scored.
    _unverified: set[str] = field(default_factory=set)

    def prime(self, texts: list[str]) -> None:
        """Scrub every text not already cached, in a single invocation."""
        pending = {
            _digest(text): text
            for text in texts
            if _digest(text) not in self._cache
        }
        if not pending:
            return

        binary = _binary_path()
        if binary is None:
            raise ScrubberUnavailable(_IMPORT_HINT)

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            indir, outdir = root / "in", root / "out"
            indir.mkdir()
            # Scrubber renames a pre-existing output directory, so leave it.
            names = {}
            for index, (digest, text) in enumerate(sorted(pending.items())):
                name = f"msg{index:06d}"
                names[name] = (digest, text)
                (indir / f"{name}.txt").write_text(
                    text, encoding="utf-8", newline=""
                )

            local_binary = root / binary.name
            shutil.copy2(binary, local_binary)
            config = root / "config.cfg"
            config.write_text(
                "ROOT1 = {root}\n"
                "ClinicalReports_dir = ROOT1/in\n"
                "ClinicalReports_files = ^msg.*\\.txt\n"
                "nPHI_outdir = ROOT1/out\n".format(
                    root="/work" if _needs_emulation() else root
                ),
                encoding="utf-8",
            )

            self._invoke(root, local_binary, config)

            for name, (digest, text) in names.items():
                produced = outdir / f"{name}.nphi.txt"
                if not produced.is_file():
                    # Scrubber failed on this input (it is known to fail on
                    # non-ASCII). Record it rather than scoring it as "found
                    # nothing", which would be a silent recall penalty.
                    self._unverified.add(digest)
                    self._cache[digest] = []
                    continue
                # read_bytes().decode(), never read_text(): text mode applies
                # universal-newline translation, turning the CR terminators
                # HL7 uses into LF. The original still has CR, so every
                # anchor containing a line ending fails to match and the
                # whole message is discarded as unalignable.
                scrubbed = strip_footer(
                    produced.read_bytes().decode("utf-8", errors="replace")
                )
                spans = recover_spans(text, scrubbed)
                if spans is None or apply_spans(text, spans) != scrubbed:
                    # Alignment is untrustworthy for this message. Record it
                    # so the report can say so, and score nothing rather than
                    # scoring wrong offsets.
                    self._unverified.add(digest)
                    spans = spans or []
                self._cache[digest] = spans

    def _invoke(self, root: Path, binary: Path, config: Path) -> None:
        if _needs_emulation():
            command = [
                "docker", "run", "--rm",
                "--platform", "linux/amd64",
                "-v", f"{root}:/work",
                "-w", "/work",
                self.image,
                f"./{binary.name}", config.name,
            ]
        else:
            command = [str(binary), str(config)]

        result = subprocess.run(
            command, cwd=root, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise ScrubberUnavailable(
                f"NLM Scrubber exited {result.returncode}.\n"
                f"stdout: {result.stdout[-800:]}\n"
                f"stderr: {result.stderr[-800:]}"
            )

    def spans_for(self, text: str) -> list[RecoveredSpan]:
        digest = _digest(text)
        if digest not in self._cache:
            self.prime([text])
        return self._cache[digest]

    def is_verified(self, text: str) -> bool:
        return _digest(text) not in self._unverified

    @property
    def unverified_count(self) -> int:
        return len(self._unverified)


def scrubber_provenance(batch: ScrubberBatch | None = None) -> dict:
    """Versions and configuration that define this baseline."""
    binary = _binary_path()
    provenance: dict = {
        "tool": "nlm-scrubber",
        "version": DEFAULT_VERSION,
        "archive_sha256": KNOWN_ARCHIVE_SHA256,
        "configuration": "default parameters, no preserved-terms whitelist",
        "spans": "recovered by alignment; not reported by the tool",
        "emulated": _needs_emulation(),
        # The harness times detect() calls, but this arm has already done its
        # work by then. Flagged so the report can say the ms column does not
        # mean for this arm what it means for the others.
        "batched": True,
    }
    if binary is not None:
        provenance["binary_sha256"] = hashlib.sha256(
            binary.read_bytes()
        ).hexdigest()
    if _needs_emulation():
        provenance["image"] = (
            batch.image if batch is not None else os.environ.get(ENV_IMAGE, DEFAULT_IMAGE)
        )
    if batch is not None and batch.unverified_count:
        provenance["unverified_alignments"] = batch.unverified_count
    return provenance


# -- the arms -----------------------------------------------------------


def _to_spans(
    recovered: list[RecoveredSpan], credit_merged: bool
) -> list[tuple[int, int, str]]:
    return stable_span_order(
        [
            (span.start, span.end, span.phi_type)
            for span in recovered
            if credit_merged or not span.merged
        ]
    )


class ScrubberTextDetector:
    """NLM Scrubber over the flattened message.

    ``credit_merged`` selects which end of this arm's band to report. ``True``
    credits regions where Scrubber redacted across a boundary between two PHI
    values, which is the generous reading; ``False`` drops them, which is the
    conservative one. Neither is the truth -- the tool's output does not
    contain enough information to say -- so the README reports both.
    """

    def __init__(
        self,
        batch: ScrubberBatch | None = None,
        credit_merged: bool = True,
        name: str | None = None,
    ):
        self.batch = batch if batch is not None else ScrubberBatch(
            image=os.environ.get(ENV_IMAGE, DEFAULT_IMAGE)
        )
        self.credit_merged = credit_merged
        self.name = name or (
            "scrubber-text" if credit_merged else "scrubber-text-strict"
        )

    @property
    def provenance(self) -> dict:
        return scrubber_provenance(self.batch) | {
            "credit_merged_regions": self.credit_merged
        }

    def prime(self, texts: list[str]) -> None:
        self.batch.prime(texts)

    def detect(self, text: str) -> list[Detection]:
        recovered = self.batch.spans_for(text)
        detections = [
            Detection(
                start=start,
                end=end,
                text=text[start:end],
                phi_type=phi_type,
                source=self.name,
            )
            for start, end, phi_type in _to_spans(recovered, self.credit_merged)
        ]
        return _resolve_overlaps(detections)


class ScrubberNarrativeDetector:
    """NLM Scrubber behind the schema-conditioning seam.

    Satisfies the ``NarrativeDetector`` protocol. Like the other two published
    tools it receives no structural context, because Scrubber has no interface
    to accept any -- it takes files of text and returns files of text.

    Priming matters more here than anywhere else: the candidates are thousands
    of short strings, and one Scrubber invocation per string would make the arm
    unrunnable. The CLI collects every candidate up front via
    ``SchemaConditionedDetector.candidates`` and primes the batch once.
    """

    def __init__(
        self,
        batch: ScrubberBatch | None = None,
        credit_merged: bool = True,
        name: str = "scrubber-narrative",
    ):
        self.batch = batch if batch is not None else ScrubberBatch(
            image=os.environ.get(ENV_IMAGE, DEFAULT_IMAGE)
        )
        self.credit_merged = credit_merged
        self.name = name

    @property
    def provenance(self) -> dict:
        return scrubber_provenance(self.batch) | {
            "credit_merged_regions": self.credit_merged
        }

    def prime(self, texts: list[str]) -> None:
        self.batch.prime(texts)

    def detect_in(self, candidate: NarrativeCandidate) -> list[tuple[int, int, str]]:
        recovered = self.batch.spans_for(candidate.text)
        return _to_spans(recovered, self.credit_merged)
