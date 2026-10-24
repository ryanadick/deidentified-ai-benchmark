"""Narrative templates: PHI written the way clinicians actually write it.

This is the hard half of the corpus. Structured PHI is easy to plant and easy
to find; the interesting question is narrative, where identifiers arrive with no
schema, in whatever form someone typed at three in the morning.

The templates deliberately include the forms that break naive detectors:

* a name with no title or sentence cue (``Discussed with Vanbrugh``);
* a relative date instead of an absolute one (``three weeks after admission``);
* a phone number split by prose (``reached at 201-555-0142 ext 4``);
* an address written in postal shorthand (``88 Yarrowgate Rd``);
* an MRN with no label (``chart 48213771``);
* a second person whose name is the only occurrence in the whole message, so a
  detector cannot find it by matching a structured field.

Every planted value is a real field of the ``Person``, which is what lets the
generator annotate narrative spans without hand-labelling them.
"""

from __future__ import annotations

from .people import Person

#: Each template is a format string over Person attributes. They are chosen for
#: variety of phrasing, not volume: a corpus of one template measures how well a
#: detector memorized that template.
PROGRESS_NOTE_TEMPLATES = (
    "Patient {given} {family}, DOB {birth_date_slashes}, reached at {phone_home}. "
    "MRN {mrn}. Lives at {street}, {city} {state} {postal_code}.",

    "{prefix} {family} was seen today. Contact number on file is {phone_mobile}; "
    "alternate contact is {nok_given} {nok_family} ({nok_relationship_word}) at {nok_phone}.",

    "Spoke at length with {given}'s {nok_relationship_word}, {nok_given} {nok_family}. "
    "Follow-up arranged at {facility}. Chart {mrn}.",

    "{given} {family} presented on {admit_slashes} accompanied by a family member. "
    "Discharge planning to {street} is underway. Insurance verified against SSN {ssn}.",

    "Discussed with {doctor_given} {doctor_family}, attending. Patient {family} "
    "requests all correspondence to {email} rather than the home address.",

    "Home visit scheduled. Address confirmed as {street} {unit}, {city}. "
    "Best number {phone_work}. Patient's {nok_relationship_word} {nok_given} will be present.",
)

COMMENT_TEMPLATES = (
    "Called {nok_given} {nok_family} on {admit_slashes} re discharge to {street}, {city} {state} {postal_code}.",

    "Left message at {phone_home}. Second attempt {phone_mobile}. "
    "Patient account {account}.",

    "Transport booked from {facility} to {street}. Driver to ask for {given}.",

    "{given} {family} declined the interpreter. Daughter {nok_given} translated.",

    "Prior record located under account {account} and license {license_number}.",
)

DIAGNOSIS_TEMPLATES = (
    "Follow-up per {doctor_family} protocol, patient {family}",
    "Chronic condition, managed since {birth_date_slashes}",
    "Referred by {doctor_given} {doctor_family}",
)

ALLERGY_TEMPLATES = (
    "Reaction documented by {doctor_family}; contact {phone_home}",
    "Per {nok_given} {nok_family}, onset in childhood",
)

#: Phrases with no resolvable referent. They are included so the corpus
#: contains the paraphrase problem Innovation 2 addresses, but they are NOT
#: annotated as PHI -- "the 54-year-old" identifies nobody on its own, and
#: scoring a detector for missing it would be measuring the wrong thing.
UNRESOLVABLE_PHRASES = (
    "three weeks after admission",
    "the following Tuesday",
    "her eldest daughter",
    "the referring physician",
    "a 54-year-old",
)


def render(template: str, person: Person) -> str:
    """Fill a template from a Person, collapsing whitespace from empty fields.

    Empty optional fields -- a patient with no prefix or unit -- would otherwise
    leave double spaces that shift every downstream character offset and make
    the text read as machine-generated.
    """
    values = {
        name: getattr(person, name)
        for name in dir(person)
        if not name.startswith("_") and isinstance(getattr(person, name), str)
    }
    values.update(
        {
            "birth_date_slashes": person.birth_date_slashes,
            "admit_slashes": person.admit_slashes,
        }
    )
    text = template.format(**values)
    return " ".join(text.split())


def progress_note(rng, person: Person) -> str:
    text = render(rng.choice(PROGRESS_NOTE_TEMPLATES), person)
    if rng.randrange(2):
        text += f" Reviewed {rng.choice(UNRESOLVABLE_PHRASES)}."
    return text


def comment(rng, person: Person) -> str:
    return render(rng.choice(COMMENT_TEMPLATES), person)


def diagnosis_text(rng, person: Person) -> str:
    return render(rng.choice(DIAGNOSIS_TEMPLATES), person)


def allergy_text(rng, person: Person) -> str:
    return render(rng.choice(ALLERGY_TEMPLATES), person)


def orphan_mention(rng, person: Person) -> str:
    """A person named only here, nowhere in any structured field.

    This is the case that separates a structure-aware detector from a lookup:
    there is no field to copy the answer from, so the only way to find the name
    is to recognize it as a name.
    """
    if not person.narrative_people:
        return ""
    given, family, relationship = rng.choice(person.narrative_people)
    return (
        f" Patient's {relationship}, {given} {family}, holds health care proxy."
    )
