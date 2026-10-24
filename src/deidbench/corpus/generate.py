"""Corpus generation with derived ground truth.

Annotations are *located*, not declared. The generator plants a person's values
into a message, serializes it, then searches the serialized bytes for each
planted value and resolves every occurrence to the leaf containing it. That
ordering matters: hand-declared offsets drift the moment a template changes by
one character, and a corpus whose labels are subtly wrong is worse than no
corpus, because the error shows up as a detector's apparent failure.

Randomness here uses a *seeded* ``random.Random``, deliberately unlike the
CSPRNG the surrogate factory uses. A published corpus has to regenerate
byte-identically from its seed so that reported numbers can be checked; a
surrogate has to be unpredictable. Opposite requirements, opposite generators.

The corpus plants PHI into four kinds of location, because that split is the
measurement the whole exercise is for -- see ``annotation.LocationClass``.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import date

from .. import hl7
from ..hl7.spans import candidate_policy_paths, find_enclosing
from ..policy.registry import Policy, load_default_policy
from ..surrogate.types import Action, PhiType
from . import narrative
from .annotation import Annotation, AnnotatedMessage, Corpus, LocationClass
from .people import Person, make_person

#: Delimiter sets the corpus uses. The odd set avoids every character that
#: appears in corpus data, so values never need escaping -- and
#: ``_assert_no_delimiters`` fails loudly if that ever stops being true, rather
#: than emitting a corrupt message with plausible-looking annotations.
STANDARD_DELIMITERS = ("|", "^", "~", "\\", "&")
ODD_DELIMITERS = ("#", "!", "*", "%", "+")

MESSAGE_TYPES = (
    "ADT^A04^ADT_A01",
    "ADT^A08^ADT_A01",
    "ADT^A03^ADT_A03",
    "ADT^A01^ADT_A01",
    "ORU^R01^ORU_R01",
)

VERSIONS = ("2.3", "2.4", "2.5.1", "2.7")

#: Deliberate field misuse, each case drawn from the same real pattern: a
#: sending system ran out of the field it wanted and used the nearest one.
#:
#: All five target fields the policy *does* cover, but with a different PHI
#: type than what gets planted -- that mismatch is what makes them misuse
#: rather than narrative. Putting a stray phone number in OBX-5 would not
#: qualify, because the policy already defers all of OBX-5 to the narrative
#: detector; it would be measuring the narrative gap a second time.
MISUSE_KINDS = (
    "phone_in_alternate_id",
    "ssn_in_county_code",
    "phone_in_geographic_designation",
    "name_in_hospital_service",
    "name_in_diagnosis",
)


@dataclass
class GenerationOptions:
    """Corpus shape. Rates are probabilities per message."""

    messages: int = 50
    seed: int = 20261013
    reference_date: date = date(2026, 10, 13)
    #: Fraction of messages carrying OBX/NTE narrative. Not 1.0, because a
    #: corpus where every message has prose overstates how much narrative a
    #: real ADT feed contains.
    narrative_rate: float = 0.7
    #: Fraction carrying ZPD, a Z-segment the shipped policy describes.
    nonstandard_rate: float = 0.45
    #: Fraction carrying ZIN, a Z-segment the policy does *not* describe. This
    #: is the honest measure of non-standard segments: ZPD recall reflects a
    #: policy written with that segment in hand, which no site gets for free.
    nonstandard_unknown_rate: float = 0.3
    #: Fraction with at least one deliberately misused field.
    misuse_rate: float = 0.35
    #: Fraction using non-standard delimiters.
    odd_delimiter_rate: float = 0.12
    #: Fraction terminated with CRLF rather than CR.
    crlf_rate: float = 0.2
    #: Fraction where a date field holds a human-formatted date.
    loose_date_rate: float = 0.15


@dataclass
class _Planted:
    """A value planted into the message, awaiting location."""

    value: str
    phi_type: PhiType


def generate_corpus(
    options: GenerationOptions | None = None, policy: Policy | None = None
) -> Corpus:
    """Build a reproducible annotated corpus."""
    options = options or GenerationOptions()
    policy = policy or load_default_policy()
    policy_index = _policy_index(policy)
    rng = random.Random(options.seed)

    corpus = Corpus(
        metadata={
            "generator": "deid.corpus.generate",
            "seed": options.seed,
            "messages": options.messages,
            "reference_date": options.reference_date.isoformat(),
            "policy": policy.name,
            "policy_version": policy.version,
            "rates": {
                "narrative": options.narrative_rate,
                "nonstandard": options.nonstandard_rate,
                "misuse": options.misuse_rate,
                "odd_delimiter": options.odd_delimiter_rate,
                "crlf": options.crlf_rate,
                "loose_date": options.loose_date_rate,
                "nonstandard_unknown": options.nonstandard_unknown_rate,
            },
            "note": (
                "Synthetic. No production data. Numbers from a planted-PHI "
                "corpus are a floor and a methodology demonstration, not a "
                "substitute for measurement on production messages."
            ),
        }
    )

    for index in range(options.messages):
        person = make_person(rng, index, options.reference_date)
        corpus.messages.append(
            _generate_message(rng, index, person, options, policy_index)
        )

    return corpus


# -- one message --------------------------------------------------------


def _generate_message(
    rng: random.Random,
    index: int,
    person: Person,
    options: GenerationOptions,
    policy_index: dict[str, PhiType],
) -> AnnotatedMessage:
    delimiters = (
        ODD_DELIMITERS
        if rng.random() < options.odd_delimiter_rate
        else STANDARD_DELIMITERS
    )
    terminator = "\r\n" if rng.random() < options.crlf_rate else "\r"

    variant = {
        "message_type": rng.choice(MESSAGE_TYPES),
        "version": rng.choice(VERSIONS),
        "delimiters": "odd" if delimiters is ODD_DELIMITERS else "standard",
        "terminator": "crlf" if terminator == "\r\n" else "cr",
        "narrative": rng.random() < options.narrative_rate,
        "nonstandard": rng.random() < options.nonstandard_rate,
        "nonstandard_unknown": rng.random() < options.nonstandard_unknown_rate,
        "misuse": rng.random() < options.misuse_rate,
        "loose_date": rng.random() < options.loose_date_rate,
    }

    builder = _MessageBuilder(person, delimiters, terminator, variant, rng)
    text = builder.build()

    message = hl7.parse(text)
    rebuilt, spans = hl7.serialize_with_spans(message)
    assert rebuilt == text, "span walk disagreed with the generated text"

    annotations = _locate_annotations(
        text, spans, builder.planted, policy_index
    )

    return AnnotatedMessage(
        message_id=f"msg-{index:05d}",
        text=text,
        annotations=annotations,
        variant=variant,
    )


class _MessageBuilder:
    """Assembles segments and records every PHI value it plants."""

    def __init__(
        self,
        person: Person,
        delimiters: tuple[str, ...],
        terminator: str,
        variant: dict,
        rng: random.Random,
    ):
        self.person = person
        self.field, self.component, self.repetition, self.escape, self.subcomponent = (
            delimiters
        )
        self.terminator = terminator
        self.variant = variant
        self.rng = rng
        self.planted: list[_Planted] = []
        self.misuse_kind: str | None = (
            rng.choice(MISUSE_KINDS) if variant["misuse"] else None
        )
        variant["misuse_kind"] = self.misuse_kind

    # -- helpers --------------------------------------------------------

    def plant(self, value: str, phi_type: PhiType) -> str:
        """Record a value as ground-truth PHI and return it for placement."""
        if value:
            _assert_no_delimiters(value, self)
            self.planted.append(_Planted(value, phi_type))
        return value

    def segment(self, name: str, fields: dict[int, str]) -> str:
        highest = max(fields) if fields else 0
        return self.field.join(
            [name] + [fields.get(i, "") for i in range(1, highest + 1)]
        )

    def components(self, *values: str) -> str:
        return self.component.join(values)

    # -- assembly -------------------------------------------------------

    def build(self) -> str:
        segments = [self._msh(), self._evn(), self._pid(), self._nk1(), self._pv1()]

        if self.variant["narrative"]:
            segments.extend(self._narrative_segments())
        if self.variant["nonstandard"]:
            segments.append(self._zpd())
        if self.variant["nonstandard_unknown"]:
            segments.append(self._zin())
        if self.misuse_kind == "name_in_diagnosis":
            segments.append(self._misused_diagnosis())

        return "".join(segment + self.terminator for segment in segments if segment)

    def _msh(self) -> str:
        encoding = (
            self.component + self.repetition + self.escape + self.subcomponent
        )
        body = self.field.join(
            [
                f"MSH{self.field}{encoding}",
                "REGADT",
                "ACMEHOSP",
                "IFENG",
                "ACMEHOSP",
                self.plant(self.person.admit_hl7, PhiType.EVENT_DATE),
                "",
                self.variant["message_type"],
                f"MSG{self.rng.randrange(100000, 999999)}",
                "P",
                self.variant["version"],
            ]
        )
        return body

    def _evn(self) -> str:
        return self.segment(
            "EVN",
            {
                1: self.variant["message_type"].split(self.component)[0][:3],
                2: self.plant(self.person.admit_hl7, PhiType.EVENT_DATE),
                5: self.components(
                    "OP1",
                    self.plant(self.person.doctor_family, PhiType.CLINICIAN_NAME),
                    self.plant(self.person.doctor_given, PhiType.CLINICIAN_NAME),
                ),
                6: self.plant(self.person.admit_hl7, PhiType.EVENT_DATE),
            },
        )

    def _pid(self) -> str:
        person = self.person
        identifier_list = self.repetition.join(
            [
                self.components(
                    self.plant(person.mrn, PhiType.MEDICAL_RECORD_NUMBER),
                    "",
                    "",
                    "ACMEHOSP",
                    "MR",
                ),
                self.components(
                    self.plant(
                        person.ssn.replace("-", ""), PhiType.SSN
                    ),
                    "",
                    "",
                    "SSA",
                    "SS",
                ),
            ]
        )

        birth = (
            self.plant(person.birth_date_slashes, PhiType.BIRTH_DATE)
            if self.variant["loose_date"]
            else self.plant(person.birth_date_hl7, PhiType.BIRTH_DATE)
        )

        phones = self.repetition.join(
            [
                self.components(
                    self.plant(person.phone_home, PhiType.PHONE),
                    "PRN",
                    "PH",
                    self.plant(person.email, PhiType.EMAIL),
                ),
                self.components(
                    self.plant(person.phone_mobile, PhiType.PHONE), "ORN", "CP"
                ),
            ]
        )

        return self.segment(
            "PID",
            {
                1: "1",
                3: identifier_list,
                4: self.components(
                    self.plant(person.phone_home, PhiType.PHONE)
                    if self.misuse_kind == "phone_in_alternate_id"
                    else self.plant(
                        person.alternate_id, PhiType.MEDICAL_RECORD_NUMBER
                    )
                ),
                5: self.components(
                    self.plant(person.family, PhiType.FAMILY_NAME),
                    self.plant(person.given, PhiType.GIVEN_NAME),
                    self.plant(person.middle, PhiType.MIDDLE_NAME),
                    self.plant(person.suffix, PhiType.NAME_SUFFIX),
                    # Placed but NOT annotated as PHI. A bare honorific is not
                    # a name: "Mr" comes from a closed set of five values and
                    # identifies nobody, so HIPAA's "Names" identifier does not
                    # reach it. Annotating it charged every arm a miss for
                    # correctly leaving it alone, and buried the real
                    # field-misuse cases under 400 artifacts.
                    person.prefix,
                    "",
                    "L",
                ),
                7: birth,
                8: person.sex,
                10: "2106-3",
                11: self.components(
                    self.plant(person.street, PhiType.STREET_ADDRESS),
                    self.plant(person.unit, PhiType.ADDRESS_OTHER),
                    self.plant(person.city, PhiType.CITY),
                    self.plant(person.state, PhiType.STATE),
                    self.plant(person.postal_code, PhiType.POSTAL_CODE),
                    "USA",
                    "H",
                    # PID-11.8 is "other geographic designation". A phone number
                    # here is a classic overflow misuse.
                    self.plant(person.phone_work, PhiType.PHONE)
                    if self.misuse_kind == "phone_in_geographic_designation"
                    else "",
                    self.plant(person.county, PhiType.COUNTY),
                ),
                13: phones,
                14: self.components(
                    self.plant(person.phone_work, PhiType.PHONE), "WPN", "PH"
                ),
                15: "eng",
                16: "M",
                18: self.components(
                    self.plant(person.account, PhiType.ACCOUNT_NUMBER)
                ),
                19: self.plant(person.ssn, PhiType.SSN),
                20: self.components(
                    self.plant(person.license_number, PhiType.LICENSE_NUMBER),
                    person.state,
                ),
                12: self.plant(person.ssn, PhiType.SSN)
                if self.misuse_kind == "ssn_in_county_code"
                else "",
                22: "N",
                23: self.plant(person.birth_place, PhiType.ORGANIZATION),
                26: "USA",
                30: "N",
            },
        )

    def _nk1(self) -> str:
        person = self.person
        return self.segment(
            "NK1",
            {
                1: "1",
                2: self.components(
                    self.plant(person.nok_family, PhiType.FAMILY_NAME),
                    self.plant(person.nok_given, PhiType.GIVEN_NAME),
                ),
                3: person.nok_relationship,
                4: self.components(
                    person.street,
                    person.unit,
                    person.city,
                    person.state,
                    person.postal_code,
                ),
                5: self.components(
                    self.plant(person.nok_phone, PhiType.PHONE), "PRN", "PH"
                ),
            },
        )

    def _pv1(self) -> str:
        person = self.person
        return self.segment(
            "PV1",
            {
                1: "1",
                2: self.rng.choice(["I", "O", "E"]),
                3: self.components("2000", "2012", "01"),
                7: self.components(
                    "DR1",
                    self.plant(person.doctor_family, PhiType.CLINICIAN_NAME),
                    self.plant(person.doctor_given, PhiType.CLINICIAN_NAME),
                ),
                10: self.plant(
                    f"{person.given} {person.family}", PhiType.NAME_FREEFORM
                )
                if self.misuse_kind == "name_in_hospital_service"
                else "MED",
                19: self.components(
                    self.plant(person.account, PhiType.ACCOUNT_NUMBER)
                ),
                44: self.plant(person.admit_hl7, PhiType.EVENT_DATE),
                45: self.plant(person.discharge_hl7, PhiType.EVENT_DATE),
            },
        )

    def _narrative_segments(self) -> list[str]:
        """OBX and NTE text, where PHI arrives with no schema at all."""
        note = narrative.progress_note(self.rng, self.person)
        if self.rng.randrange(2):
            note += narrative.orphan_mention(self.rng, self.person)

        segments = [
            self.segment(
                "OBX",
                {
                    1: "1",
                    2: "TX",
                    3: self.components("NOTE", "Progress Note"),
                    5: self.plant_narrative(note),
                    11: "F",
                },
            ),
            self.segment(
                "NTE",
                {1: "1", 2: "L", 3: self.plant_narrative(narrative.comment(self.rng, self.person))},
            ),
        ]

        if self.rng.randrange(2):
            segments.append(
                self.segment(
                    "DG1",
                    {
                        1: "1",
                        3: self.components("I10", "E11.9"),
                        4: self.plant_narrative(
                            narrative.diagnosis_text(self.rng, self.person)
                        ),
                    },
                )
            )
        if self.rng.randrange(3) == 0:
            segments.append(
                self.segment(
                    "AL1",
                    {
                        1: "1",
                        3: self.components("", "Penicillin"),
                        5: self.plant_narrative(
                            narrative.allergy_text(self.rng, self.person)
                        ),
                    },
                )
            )
        return segments

    def plant_narrative(self, text: str) -> str:
        """Narrative text is placed verbatim.

        The PHI inside it was already registered when the person's fields were
        planted into structured locations, so the locator finds these
        occurrences too and classifies them as narrative. That is the whole
        trick: narrative ground truth comes free, with no hand-labelling.

        Names appearing *only* in narrative are registered here, since no
        structured field ever held them.
        """
        _assert_no_delimiters(text, self)
        for given, family, _ in self.person.narrative_people:
            if given in text:
                self.planted.append(_Planted(given, PhiType.GIVEN_NAME))
            if family in text:
                self.planted.append(_Planted(family, PhiType.FAMILY_NAME))
        return text

    def _zpd(self) -> str:
        """A site-defined segment. No standard says what any of this means."""
        person = self.person
        return self.segment(
            "ZPD",
            {
                1: "1",
                2: self.plant(
                    f"{person.given} {person.middle[:1]} {person.family}",
                    PhiType.NAME_FREEFORM,
                ),
                3: self.plant(person.phone_home, PhiType.PHONE),
                4: self.plant(person.mrn, PhiType.MEDICAL_RECORD_NUMBER),
                5: self.plant(person.facility, PhiType.ORGANIZATION),
                6: self.plant(person.birth_date_hl7, PhiType.BIRTH_DATE),
            },
        )

    def _zin(self) -> str:
        """A site-defined segment with no policy rule anywhere.

        Nothing in the shipped policy mentions ZIN, so rules score zero here by
        construction. That is the point: it isolates how much of the
        non-standard problem is solved by having already seen the segment.
        """
        person = self.person
        return self.segment(
            "ZIN",
            {
                1: "1",
                2: self.plant(
                    f"{person.given} {person.family}", PhiType.NAME_FREEFORM
                ),
                3: self.plant(person.mrn, PhiType.MEDICAL_RECORD_NUMBER),
                4: self.plant(person.birth_date_hl7, PhiType.BIRTH_DATE),
                5: self.plant(person.phone_mobile, PhiType.PHONE),
            },
        )

    def _misused_diagnosis(self) -> str:
        """A patient name in DG1-20, the confidential-indicator field.

        Appended as its own segment because, unlike the other misuse kinds,
        there is no DG1 in the baseline message to modify.
        """
        person = self.person
        return self.segment(
            "DG1",
            {
                1: "2",
                3: self.components("I10", "Z00.00"),
                20: self.plant(
                    f"{person.given} {person.family}", PhiType.NAME_FREEFORM
                ),
            },
        )


def _assert_no_delimiters(value: str, builder: _MessageBuilder) -> None:
    """Fail loudly rather than emit a corrupt message.

    The corpus avoids escaping by choosing delimiter sets that do not occur in
    its data. That is a real constraint, so it is checked rather than assumed:
    a value containing a delimiter would silently split into two fields, and
    the annotation offsets would point at text that no longer exists.
    """
    for delimiter in (
        builder.field,
        builder.component,
        builder.repetition,
        builder.escape,
        builder.subcomponent,
    ):
        if delimiter and delimiter in value:
            raise ValueError(
                f"corpus value {value!r} contains delimiter {delimiter!r}; "
                "either change the value or add escaping to the generator"
            )


# -- locating ground truth ---------------------------------------------


def _locate_annotations(
    text: str,
    spans: list[hl7.LeafSpan],
    planted: list[_Planted],
    policy_index: dict[str, PhiType],
) -> list[Annotation]:
    """Find every occurrence of every planted value and classify it.

    Longest values are claimed first so that a city named inside a county
    ("Kettlewell" within "Kettlewell County") does not steal the longer span.
    Overlapping claims are dropped rather than nested: a character belongs to
    exactly one annotation, or recall and precision stop being well defined.
    """
    unique: dict[tuple[str, str], _Planted] = {}
    for entry in planted:
        unique[(entry.value, str(entry.phi_type))] = entry

    _assert_unambiguous(unique)
    ordered = sorted(unique.values(), key=lambda e: len(e.value), reverse=True)

    annotations: list[Annotation] = []
    claimed: list[tuple[int, int]] = []

    for entry in ordered:
        for start, end in _occurrences(text, entry.value):
            if any(start < c_end and end > c_start for c_start, c_end in claimed):
                continue
            span = find_enclosing(spans, start, end)
            if span is None:
                # Straddles a delimiter, so it is not a single field value.
                continue
            claimed.append((start, end))
            annotations.append(
                Annotation(
                    start=start,
                    end=end,
                    text=entry.value,
                    phi_type=str(entry.phi_type),
                    location_class=str(
                        _classify(span, entry.phi_type, policy_index)
                    ),
                    path=span.path,
                    policy_path=span.policy_path,
                    segment=span.segment,
                )
            )

    return sorted(annotations, key=lambda a: a.start)


def _assert_unambiguous(unique: dict[tuple[str, str], _Planted]) -> None:
    """Refuse to emit a message where one string is two different PHI types.

    If a patient's middle name is also the attending physician's given name,
    then an occurrence of that string has no single correct label, and whichever
    one the locator picked would score a detector arbitrarily. The generator
    controls the data, so this is preventable rather than something to tolerate
    -- see the ``rng.sample`` draws in ``people.make_person``.
    """
    by_value: dict[str, set[str]] = {}
    for value, phi_type in unique:
        by_value.setdefault(value, set()).add(phi_type)

    ambiguous = {v: t for v, t in by_value.items() if len(t) > 1}
    if ambiguous:
        detail = "; ".join(
            f"{value!r} planted as {sorted(types)}" for value, types in ambiguous.items()
        )
        raise ValueError(f"ambiguous ground truth: {detail}")


def _occurrences(text: str, value: str) -> list[tuple[int, int]]:
    """Every occurrence of a value, with word boundaries where meaningful.

    A bare ``re.finditer`` on ``"NY"`` would match inside ``NY1234567``, which
    is a different identifier entirely. Word boundaries are applied only where
    the value starts and ends with a word character, since a phone number like
    ``(201)555-0142`` has none.
    """
    if not value:
        return []

    pattern = re.escape(value)
    if value[0].isalnum():
        pattern = r"(?<![0-9A-Za-z])" + pattern
    if value[-1].isalnum():
        pattern = pattern + r"(?![0-9A-Za-z])"

    return [(m.start(), m.end()) for m in re.finditer(pattern, text)]


def _classify(
    span: hl7.LeafSpan, phi_type: PhiType, policy_index: dict[str, PhiType]
) -> LocationClass:
    """Decide which location class an occurrence falls into."""
    if span.segment.startswith("Z"):
        return LocationClass.NONSTANDARD

    for candidate in candidate_policy_paths(span.policy_path):
        if candidate in policy_index:
            expected = policy_index[candidate]
            if expected is PhiType.FREE_TEXT:
                return LocationClass.NARRATIVE
            if expected == phi_type or _compatible(expected, phi_type):
                return LocationClass.STRUCTURED
            return LocationClass.MISUSED

    # No policy rule covers this location at all, so nothing was ever going to
    # look there.
    return LocationClass.MISUSED


#: Types the policy and the corpus may legitimately disagree about without it
#: being a misuse. The policy assigns one category per path, but PID-3 is the
#: *Patient Identifier List* -- it carries MRNs, SSNs, driver's licences and
#: plan IDs side by side, distinguished by the type code in component 5. An SSN
#: there is correctly placed, and counting it as a misuse would both inflate
#: that class and understate how well rules do on structured data.
_COMPATIBLE = (
    {
        PhiType.MEDICAL_RECORD_NUMBER,
        PhiType.ACCOUNT_NUMBER,
        PhiType.IDENTIFIER,
        PhiType.LICENSE_NUMBER,
        PhiType.SSN,
    },
    {PhiType.GIVEN_NAME, PhiType.MIDDLE_NAME},
    {PhiType.FAMILY_NAME, PhiType.NAME_FREEFORM, PhiType.CLINICIAN_NAME},
    {PhiType.CITY, PhiType.COUNTY},
    {PhiType.BIRTH_DATE, PhiType.EVENT_DATE, PhiType.DEATH_DATE},
)


def _compatible(expected: PhiType, actual: PhiType) -> bool:
    return any(
        expected in group and actual in group for group in _COMPATIBLE
    )


def _policy_index(policy: Policy) -> dict[str, PhiType]:
    """Policy path to the PHI type the policy expects there.

    Retained fields are excluded: a retained categorical is not a place the
    policy claims PHI lives, so planting PHI there is a misuse, not a match.
    """
    index: dict[str, PhiType] = {}
    for rule in policy.rules:
        if rule.action is Action.RETAIN:
            continue
        index[str(rule.path)] = rule.phi_type
    return index
