"""The policy registry: which locations hold PHI, and what to do with them.

Policies are data, not code. That is a deliberate choice with three payoffs:

1. A site can add the Z-segment its vendor invented without touching Python.
2. The same table that drives the rules supplies the *structural context* a
   model needs -- segment, field position, component, HL7 data type, PHI
   category. Conditioning a small model on that context is the central Phase I
   hypothesis, and having the registry already hold it means the experiment does
   not need a parallel description of the message schema.
3. A policy file is reviewable by a privacy officer, who can read YAML and
   cannot read a parser.

Locations rules deliberately cannot resolve are marked ``defer_to_model`` rather
than passed through silently. An unhandled narrative field that looks handled is
the failure mode that makes rules-only de-identification unsafe, and naming it
in the policy is what turns it into something measurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path as FilePath

import yaml

from ..hl7.path import Path, PathError
from ..surrogate.types import Action, PhiType, Sex

POLICY_DIR = FilePath(__file__).parent
DEFAULT_POLICIES = (
    "hl7v2-pid.yaml",
    "hl7v2-extras.yaml",
    "hl7v2-financial.yaml",
)


class PolicyError(ValueError):
    """Raised when a policy file is malformed. Never silently ignored."""


@dataclass(frozen=True)
class Rule:
    """One policy entry."""

    path: Path
    phi_type: PhiType
    action: Action
    #: Apply to every repetition of the field, or only the first. Repeating
    #: name and phone fields are the norm in production, not the exception.
    all_repetitions: bool = True
    #: Apply to every occurrence of the segment, or only the first. NK1 and
    #: OBX segments repeat; PID normally does not.
    all_occurrences: bool = True
    #: Whether this value identifies the patient across messages.
    patient_key: bool = False
    #: Where the sex of *this location's person* is sent, as a path in the same
    #: segment -- ``PID-8`` for the patient, ``GT1-9`` for the guarantor,
    #: ``IN1-43`` for the insured, ``NK1-15`` for a contact. A name surrogate
    #: here is drawn to agree with it.
    #:
    #: Per person rather than per message, because a record holds several
    #: people and each has their own sex. Matching a guarantor's name to the
    #: *patient's* sex would invent a fact; reading the guarantor's own sex
    #: field preserves one.
    sex_from: Path | None = None
    #: Where the relationship of this location's person to the patient is sent,
    #: used when the sex field is absent. ``MTH`` means female whether or not
    #: anyone filled in GT1-9, and in practice the relationship is sent far
    #: more reliably than the sex.
    sex_from_relationship: Path | None = None
    #: The sex this location's person has by the field's own definition, for
    #: fields where no code is sent because none is needed: PID-6 is the
    #: mother's maiden name, so the person is female.
    assumed_sex: Sex | None = None
    #: Static namespace for a patient key, e.g. ``ssn``.
    key_namespace: str | None = None
    #: Component within the same repetition holding the assigning authority,
    #: e.g. ``4`` for PID-3.4. MRN 12345 from two hospitals is two patients.
    key_namespace_component: int | None = None
    #: Component within the same repetition holding an identifier type code,
    #: e.g. ``5`` for PID-3.5.
    phi_type_component: int | None = None
    #: Type codes to PHI categories, for fields whose content varies by code.
    #: PID-3 repeats, carrying an MRN in one repetition and an SSN in the next;
    #: typing the whole field as one category gives the same value two
    #: surrogates when it also appears in PID-19, which breaks joins in the
    #: de-identified data and is itself a signal. Empty means the static
    #: ``phi_type`` always applies.
    phi_type_by_code: dict[str, PhiType] = field(default_factory=dict)
    #: HL7 data type, used as model context rather than for dispatch.
    data_type: str = ""
    note: str = ""

    @property
    def is_write(self) -> bool:
        return self.action in (
            Action.SURROGATE,
            Action.SHIFT_EVENT_DATE,
            Action.SHIFT_BIRTH_DATE,
            Action.REDACT,
        )

    def context(self) -> dict:
        """Structural context for this location, for the narrative detector.

        This is the independent variable of the schema-conditioned detection
        experiment: the same span with and without this description attached.
        """
        return {
            "path": str(self.path),
            "segment": self.path.segment,
            "field": self.path.field,
            "component": self.path.component,
            "data_type": self.data_type,
            "phi_type": str(self.phi_type),
            "note": self.note,
        }


@dataclass
class Policy:
    """A named, versioned collection of rules."""

    name: str
    version: int
    rules: list[Rule] = field(default_factory=list)

    def rules_for_segment(self, segment_name: str) -> list[Rule]:
        return [r for r in self.rules if r.path.segment == segment_name.upper()]

    @property
    def segment_names(self) -> list[str]:
        seen: list[str] = []
        for rule in self.rules:
            if rule.path.segment not in seen:
                seen.append(rule.path.segment)
        return seen

    def merged_with(self, other: "Policy") -> "Policy":
        """Combine two policies. Later rules win on an identical path.

        Site overrides are the expected use: load the baseline, then a local
        file that changes one field without restating the other hundred.
        """
        combined: dict[str, Rule] = {str(r.path): r for r in self.rules}
        for rule in other.rules:
            combined[str(rule.path)] = rule
        return Policy(
            name=f"{self.name}+{other.name}",
            version=max(self.version, other.version),
            rules=list(combined.values()),
        )


def load_policy(path: FilePath | str) -> Policy:
    """Load and validate one policy file."""
    file_path = FilePath(path)
    if not file_path.exists():
        raise PolicyError(f"policy file not found: {file_path}")

    try:
        document = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PolicyError(f"{file_path.name}: invalid YAML: {exc}") from exc

    if not isinstance(document, dict):
        raise PolicyError(f"{file_path.name}: expected a mapping at the top level")

    name = document.get("name") or file_path.stem
    version = document.get("version", 1)
    entries = document.get("rules")
    if not isinstance(entries, list):
        raise PolicyError(f"{file_path.name}: 'rules' must be a list")

    rules = [
        _build_rule(entry, file_path.name, index)
        for index, entry in enumerate(entries)
    ]
    _reject_duplicate_paths(rules, file_path.name)
    return Policy(name=name, version=int(version), rules=rules)


def load_default_policy() -> Policy:
    """The shipped baseline: every PID field, plus common dates and narrative."""
    policies = [load_policy(POLICY_DIR / name) for name in DEFAULT_POLICIES]
    result = policies[0]
    for extra in policies[1:]:
        result = result.merged_with(extra)
    return result


def _build_rule(entry: object, source: str, index: int) -> Rule:
    where = f"{source} rule {index + 1}"
    if not isinstance(entry, dict):
        raise PolicyError(f"{where}: expected a mapping")

    raw_path = entry.get("path")
    if not raw_path:
        raise PolicyError(f"{where}: missing 'path'")
    try:
        parsed_path = Path.parse(str(raw_path))
    except PathError as exc:
        raise PolicyError(f"{where}: {exc}") from exc
    if parsed_path.field is None:
        raise PolicyError(f"{where}: path must address at least a field")

    try:
        phi_type = PhiType(str(entry.get("phi_type", "")))
    except ValueError as exc:
        raise PolicyError(
            f"{where}: unknown phi_type {entry.get('phi_type')!r}; "
            f"valid values are {', '.join(sorted(t.value for t in PhiType))}"
        ) from exc

    try:
        action = Action(str(entry.get("action", "")))
    except ValueError as exc:
        raise PolicyError(
            f"{where}: unknown action {entry.get('action')!r}; "
            f"valid values are {', '.join(sorted(a.value for a in Action))}"
        ) from exc

    by_code = _build_phi_type_by_code(entry, where)
    type_component = (
        int(entry["phi_type_component"])
        if entry.get("phi_type_component") is not None
        else None
    )
    if bool(by_code) != (type_component is not None):
        # Either half alone is a silent no-op: a map with nothing to read, or a
        # component read and then ignored.
        raise PolicyError(
            f"{where}: 'phi_type_component' and 'phi_type_by_code' must be "
            f"given together"
        )

    return Rule(
        path=parsed_path,
        phi_type=phi_type,
        action=action,
        all_repetitions=bool(entry.get("all_repetitions", True)),
        all_occurrences=bool(entry.get("all_occurrences", True)),
        patient_key=bool(entry.get("patient_key", False)),
        sex_from=_build_sex_path(entry, "sex_from", parsed_path, where),
        sex_from_relationship=_build_sex_path(
            entry, "sex_from_relationship", parsed_path, where
        ),
        assumed_sex=_build_assumed_sex(entry, where),
        key_namespace=entry.get("key_namespace"),
        key_namespace_component=(
            int(entry["key_namespace_component"])
            if entry.get("key_namespace_component") is not None
            else None
        ),
        phi_type_component=type_component,
        phi_type_by_code=by_code,
        data_type=str(entry.get("data_type", "")),
        note=str(entry.get("note", "")),
    )


def _build_sex_path(
    entry: dict, key: str, rule_path: Path, where: str
) -> Path | None:
    """Parse a sex-source path, which must live in the rule's own segment.

    Same segment, and therefore the same *occurrence*: the second NK1's name
    takes the second NK1's sex. Allowing a path into another segment would make
    the engine guess which occurrence was meant, and guessing wrong assigns one
    person's sex to another.
    """
    raw = entry.get(key)
    if raw is None:
        return None
    try:
        parsed = Path.parse(str(raw))
    except PathError as exc:
        raise PolicyError(f"{where}: {key}: {exc}") from exc
    if parsed.segment != rule_path.segment:
        raise PolicyError(
            f"{where}: {key} must address the same segment as the rule "
            f"({rule_path.segment}), not {parsed.segment}"
        )
    return parsed


def _build_assumed_sex(entry: dict, where: str) -> Sex | None:
    raw = entry.get("assumed_sex")
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in ("m", "male"):
        return Sex.MALE
    if text in ("f", "female"):
        return Sex.FEMALE
    raise PolicyError(
        f"{where}: assumed_sex must be 'male' or 'female', not {raw!r}"
    )


def _build_phi_type_by_code(entry: dict, where: str) -> dict[str, PhiType]:
    raw = entry.get("phi_type_by_code")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise PolicyError(f"{where}: 'phi_type_by_code' must be a mapping")

    resolved: dict[str, PhiType] = {}
    for code, name in raw.items():
        try:
            resolved[str(code).strip().upper()] = PhiType(str(name))
        except ValueError as exc:
            raise PolicyError(
                f"{where}: unknown phi_type {name!r} for code {code!r}; "
                f"valid values are {', '.join(sorted(t.value for t in PhiType))}"
            ) from exc
    return resolved


def _reject_duplicate_paths(rules: list[Rule], source: str) -> None:
    """Two rules for one path is always a mistake, and a dangerous one.

    Whichever won would depend on list order, so a field a reviewer believes is
    surrogated might in fact be retained.
    """
    seen: dict[str, int] = {}
    for index, rule in enumerate(rules):
        key = str(rule.path)
        if key in seen:
            raise PolicyError(
                f"{source}: path {key} appears in rule {seen[key] + 1} "
                f"and rule {index + 1}"
            )
        seen[key] = index
