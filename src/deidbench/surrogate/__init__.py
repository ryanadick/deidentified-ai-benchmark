"""The shared PHI vocabulary.

Only the type enum lives here. In the full engine this package also holds the
surrogate factory and its lexicons; the benchmark needs none of that, because it
measures detection rather than performing replacement. The enum is shared so a
policy file cannot name a PHI category the harness does not understand.
"""

from .types import REVERSIBLE_TYPES, Action, PhiType

__all__ = [
    "REVERSIBLE_TYPES",
    "Action",
    "PhiType",
]
