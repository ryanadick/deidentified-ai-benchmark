"""Synthetic people for the evaluation corpus.

**These pools are deliberately disjoint from ``surrogate/lexicon.py``.** If a
test patient could be named ``Eldridge`` and a surrogate could also be
``Eldridge``, then a leak check hit might be a coincidence and a detector's true
positive might be an artifact. Evaluation would be measuring its own noise.
``test_corpus.py`` asserts the disjointness, so the two files cannot drift back
together unnoticed.

Nothing here comes from patient data. The names are drawn from fiction-safe
constructions and uncommon-but-plausible forms, chosen so that a detector cannot
succeed by memorizing a common-names list -- which is a real failure mode when a
corpus is built from the most frequent US surnames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

CORPUS_SURNAMES = """
Alderwick Banfield Brackmoor Cavanaugh Cholmondeley Daventry Dunsmore Eadweard
Fairweather Featherstone Garrowby Grimsditch Haverstock Hollingbrook Inchcape
Jellicoe Kirkbride Lanthorne Loxley Marchbanks Mottram Netherby Oglethorpe
Pinkerton Quarrington Rookwood Sandiford Shawcross Templeton Thistlewood
Underhill Vanbrugh Wetherell Winterslow Yoxall Zouche Abernethy Blagdon
Carnforth Drummond Elphinstone Fotherby Greenhalgh Harkaway Ilkeston Jessamine
Kelsterbane Lydiard Marrowbone Nunnington Ollerton Pargeter Rivenhall Scrivener
Tollemache Uppington Vexley Wrothesley Yelverton
""".split()

CORPUS_GIVEN_NAMES = """
Ambrosine Bartram Bethia Caradoc Cressida Dorcas Eulalia Ferdinand Grizelda
Hepzibah Ignatia Jocasta Kenelm Lysander Marmaduke Nerissa Obadiah Peregrine
Quintilla Rosamund Sebastiane Tryphena Ulric Verity Wilhelmine Xanthe Yseult
Zephaniah Alvina Barnabus Clorinda Dunstan Emmeline Fenwicke Gervase Honoria
Isambard Jerusha Kerensa Loveday Mehitabel Nathanael Oriel Philomena Quenilda
Rafferty Sidonie Thaddea Ursinus Vashti Wendeline Yolande Zenobia Carys Dilwyn
""".split()

CORPUS_STREETS = """
Bramblewick Candlewick Chaffinch Coppergate Cuttlebrook Dovecote Fernycroft
Gallowtree Gorsemoor Hollowmead Kilnhouse Lampwright Marlpit Nettlebed
Oxleasow Pennycress Quillgate Ramsbottom Saltwheel Shuttleworth Tanneryard
Thrushcross Vetchfield Wainwright Whitsunday Yarrowgate
""".split()

CORPUS_CITIES = """
Ashwickham Brindlecombe Cranmoreton Dunwallow Elvermead Frostwickle Garnsworth
Hobbleton Inverkirk Jarrowdale Kettlewell Lammermoor Mirthwaite Nethercote
Orrisdale Pottersby Quillinghame Rushmoorly Sedgewinter Tarnbridge Ullswater
Vexcombe Warblington Yealmpton
""".split()

#: Disjoint from CORPUS_SURNAMES on purpose. If a facility could be called
#: "Brackmoor Memorial" while a patient was also named Brackmoor, the same
#: string would be ground truth for two different PHI types.
CORPUS_FACILITIES = """
Ardenshaw Belmontry Calderhall Dunmorrow Eastgarth Fenwycke Glenbarrow
Hartsmere Ivybourne Jedmonton Kingsmoult Larkenham Monksfield Norbridge
""".split()

#: Area codes outside the plausible-area-code list the surrogate factory draws
#: from, so a corpus phone number and a surrogate phone number cannot collide.
CORPUS_AREA_CODES = ["201", "202", "203", "204", "206", "207", "208", "210"]

STATES = ["NY", "PA", "OH", "MI", "VT", "ME", "NH", "MA", "CT", "RI"]

RELATIONSHIPS = [("SPO", "spouse"), ("CHD", "daughter"), ("PAR", "mother"),
                 ("SIB", "brother"), ("GRD", "granddaughter")]


@dataclass
class Person:
    """One synthetic patient and the people around them.

    Every field here is PHI that the corpus plants and the harness expects a
    detector to find. Grouping them on one object keeps the generator from
    accidentally planting a value it forgot to annotate.
    """

    family: str
    given: str
    middle: str
    prefix: str
    suffix: str

    mrn: str
    ssn: str
    account: str
    license_number: str
    alternate_id: str

    birth_date: date
    sex: str

    street: str
    unit: str
    city: str
    state: str
    postal_code: str
    county: str
    birth_place: str

    phone_home: str
    phone_mobile: str
    phone_work: str
    email: str

    nok_family: str
    nok_given: str
    nok_phone: str
    nok_relationship: str
    nok_relationship_word: str

    doctor_family: str
    doctor_given: str
    facility: str

    admit: date
    discharge: date

    #: Extra people named only in narrative, which is where rules cannot reach.
    narrative_people: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def birth_date_hl7(self) -> str:
        return self.birth_date.strftime("%Y%m%d")

    @property
    def birth_date_slashes(self) -> str:
        return self.birth_date.strftime("%m/%d/%Y")

    @property
    def admit_hl7(self) -> str:
        return self.admit.strftime("%Y%m%d") + "0800"

    @property
    def discharge_hl7(self) -> str:
        return self.discharge.strftime("%Y%m%d") + "1545"

    @property
    def admit_slashes(self) -> str:
        return self.admit.strftime("%m/%d/%Y")

    @property
    def full_name(self) -> str:
        return f"{self.given} {self.family}"

    @property
    def address_line(self) -> str:
        return f"{self.street}, {self.city} {self.state} {self.postal_code}"


def make_person(rng, index: int, reference: date) -> Person:
    """Build one synthetic patient.

    ``index`` seeds the identifier shapes so that a corpus contains a realistic
    mix of MRN formats rather than one template repeated -- an identifier
    detector that only ever sees ``MR#####`` learns the wrong lesson.
    """
    # Draw every role's name without replacement. Two different people sharing
    # a given name is realistic but it makes ground truth ambiguous -- the same
    # string would be annotated once as the patient's name and once as the
    # clinician's -- and the generator controls this, so it should avoid it.
    surnames = rng.sample(CORPUS_SURNAMES, 6)
    givens = rng.sample(CORPUS_GIVEN_NAMES, 8)
    places = rng.sample(CORPUS_CITIES, 4)

    family = surnames[0]
    given = givens[0]

    age = rng.choice([0, 0, 2, 7, 14, 23, 34, 41, 52, 58, 63, 67, 74, 81, 88, 91, 96])
    birth_date = reference - timedelta(days=int(age * 365.2425) + rng.randrange(0, 360))

    admit = reference - timedelta(days=rng.randrange(0, 90))
    discharge = admit + timedelta(days=rng.randrange(1, 12))

    mrn_shape = index % 4
    if mrn_shape == 0:
        mrn = f"MR{rng.randrange(100000, 999999)}"
    elif mrn_shape == 1:
        mrn = f"{rng.randrange(10000000, 99999999)}"
    elif mrn_shape == 2:
        mrn = f"{rng.choice('ABCDEFGH')}{rng.randrange(1000, 9999)}-{rng.randrange(10, 99)}"
    else:
        mrn = f"H{rng.randrange(1000000, 9999999)}"

    state = rng.choice(STATES)
    relationship, relationship_word = rng.choice(RELATIONSHIPS)

    # Mixed case, not upper. "MR" as a name prefix is byte-identical to the
    # HL7 identifier type code "MR" in PID-3.5, so a corpus using the
    # uppercase form cannot tell the two apart and neither can ground truth.
    prefix = rng.choice(["Mr", "Ms", "Mrs", "Dr", ""])
    suffix = rng.choice(["Jr", "Sr", "III", ""])

    # Next of kin usually but not always shares the patient's surname. Both
    # cases matter: a shared surname is a linkage risk, a different one is a
    # detection problem.
    nok_family = family if rng.randrange(3) else surnames[1]

    return Person(
        family=family,
        given=given,
        middle=givens[1],
        prefix=prefix,
        suffix=suffix,
        mrn=mrn,
        ssn=f"{rng.randrange(100, 700)}-{rng.randrange(10, 99)}-{rng.randrange(1000, 9999)}",
        account=f"ACC{rng.randrange(10000000, 99999999)}",
        license_number=f"{state}{rng.randrange(1000000, 9999999)}",
        alternate_id=f"ALT{rng.randrange(100000, 999999)}",
        birth_date=birth_date,
        sex=rng.choice(["M", "F"]),
        street=f"{rng.randrange(1, 9999)} {rng.choice(CORPUS_STREETS)} "
        f"{rng.choice(['Street', 'Avenue', 'Road', 'Lane', 'Court'])}",
        unit=rng.choice(["", "Apt 2C", "Suite 410", "Unit 7", "Apt 19B"]),
        city=places[0],
        state=state,
        postal_code=f"{rng.randrange(10000, 99999)}",
        county=f"{places[1]} County",
        birth_place=places[2],
        phone_home=_phone(rng, "paren"),
        phone_mobile=_phone(rng, "dashes"),
        phone_work=_phone(rng, "plain"),
        email=f"{given.lower()}.{family.lower()}@{places[3].lower()}.example",
        nok_family=nok_family,
        nok_given=givens[2],
        nok_phone=_phone(rng, "dashes"),
        nok_relationship=relationship,
        nok_relationship_word=relationship_word,
        doctor_family=surnames[2],
        doctor_given=givens[3],
        facility=f"{rng.choice(CORPUS_FACILITIES)} "
        f"{rng.choice(['Memorial Hospital', 'Regional Medical Center', 'Community Clinic'])}",
        admit=admit,
        discharge=discharge,
        narrative_people=[
            (givens[4 + offset], surnames[3 + offset], rng.choice(RELATIONSHIPS)[1])
            for offset in range(rng.randrange(1, 3))
        ],
    )


def _phone(rng, style: str) -> str:
    """A phone number in one of three formats.

    Format variety is not decoration. A detector tuned to ``(xxx)xxx-xxxx``
    quietly fails on the bare ten-digit form that HL7 feeds actually carry in
    XTN.12, and a corpus with one format would never reveal it.
    """
    area = rng.choice(CORPUS_AREA_CODES)
    exchange = rng.randrange(200, 999)
    line = f"{rng.randrange(0, 10000):04d}"
    if style == "paren":
        return f"({area}){exchange}-{line}"
    if style == "dashes":
        return f"{area}-{exchange}-{line}"
    return f"{area}{exchange}{line}"
