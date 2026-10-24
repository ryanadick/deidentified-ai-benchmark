"""The PHI categories the engine recognizes.

These are the shared vocabulary of the policy registry, the surrogate factory,
the vault and the evidence manifest. Keeping them in one enum means a policy
file cannot name a category nothing knows how to generate.
"""

from __future__ import annotations

from enum import StrEnum


class PhiType(StrEnum):
    """A category of protected health information."""

    # Names
    FAMILY_NAME = "family_name"
    GIVEN_NAME = "given_name"
    MIDDLE_NAME = "middle_name"
    NAME_PREFIX = "name_prefix"
    NAME_SUFFIX = "name_suffix"
    NAME_FREEFORM = "name_freeform"

    # Addresses
    STREET_ADDRESS = "street_address"
    ADDRESS_OTHER = "address_other"
    CITY = "city"
    STATE = "state"
    POSTAL_CODE = "postal_code"
    COUNTY = "county"
    COUNTRY = "country"

    # Contact
    PHONE = "phone"
    EMAIL = "email"

    # Identifiers
    MEDICAL_RECORD_NUMBER = "medical_record_number"
    ACCOUNT_NUMBER = "account_number"
    SSN = "ssn"
    LICENSE_NUMBER = "license_number"
    IDENTIFIER = "identifier"

    # Dates
    BIRTH_DATE = "birth_date"
    EVENT_DATE = "event_date"
    DEATH_DATE = "death_date"

    # Organizations and people other than the patient
    CLINICIAN_NAME = "clinician_name"
    ORGANIZATION = "organization"

    # Narrative. Rules deliberately do not resolve these; they are the content
    # the Phase I model work exists to handle, and leaving them flagged rather
    # than silently passed through is what makes the gap measurable.
    FREE_TEXT = "free_text"


class Action(StrEnum):
    """What a policy does with a location."""

    #: Replace with a surrogate and record the mapping in the vault.
    SURROGATE = "surrogate"
    #: Shift by the patient's event offset.
    SHIFT_EVENT_DATE = "shift_event_date"
    #: Shift by the patient's birth offset, bounded by the age band.
    SHIFT_BIRTH_DATE = "shift_birth_date"
    #: Blank the value outright. Irreversible by design.
    REDACT = "redact"
    #: Leave as sent. Used for values that carry no identifying information.
    RETAIN = "retain"
    #: Hand to the narrative detector. Rules alone cannot resolve this.
    DEFER_TO_MODEL = "defer_to_model"


class Sex(StrEnum):
    """The administrative sex a name surrogate should agree with.

    Only the two values that select a name pool are represented. Everything
    else HL7 can send in PID-8 -- other, unknown, ambiguous, not applicable --
    is absence of a constraint rather than a third pool, and maps to ``None``.
    """

    MALE = "M"
    FEMALE = "F"


def parse_sex(raw: str | None) -> Sex | None:
    """Read an HL7 administrative sex code, or ``None`` if it constrains nothing.

    ``None`` is the safe answer for every unrecognized value: it widens the
    name pool rather than narrowing it to a guess, and a surrogate from the
    combined pool is still format-preserving, still unique and still reversible.
    """
    if not raw:
        return None
    code = raw.strip().upper()[:1]
    if code == "M":
        return Sex.MALE
    if code == "F":
        return Sex.FEMALE
    return None


#: Categories whose mappings may be reversed from the vault. Redaction and
#: narrative deferral are excluded because nothing was stored to reverse.
REVERSIBLE_TYPES = frozenset(
    {
        PhiType.FAMILY_NAME,
        PhiType.GIVEN_NAME,
        PhiType.MIDDLE_NAME,
        PhiType.NAME_PREFIX,
        PhiType.NAME_SUFFIX,
        PhiType.NAME_FREEFORM,
        PhiType.STREET_ADDRESS,
        PhiType.ADDRESS_OTHER,
        PhiType.CITY,
        PhiType.STATE,
        PhiType.POSTAL_CODE,
        PhiType.COUNTY,
        PhiType.PHONE,
        PhiType.EMAIL,
        PhiType.MEDICAL_RECORD_NUMBER,
        PhiType.ACCOUNT_NUMBER,
        PhiType.SSN,
        PhiType.LICENSE_NUMBER,
        PhiType.IDENTIFIER,
        PhiType.CLINICIAN_NAME,
        PhiType.ORGANIZATION,
    }
)
