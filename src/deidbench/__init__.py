"""A reproducible benchmark for PHI detection in HL7 v2 messages.

Three things live here: a generator that plants annotated PHI in synthetic HL7
v2 messages, a declarative policy describing where PHI is allowed to sit, and a
harness that scores detectors against the resulting ground truth.

What this package is *not* is a de-identification tool. It removes nothing and
replaces nothing; it measures how much PHI a detector finds, and reports that
number broken down by where in the message the PHI was hiding. See the README
for the four caveats that bound what the numbers can be used for.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
