# deidbench: a reproducible PHI-detection benchmark for HL7 v2

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23623786.svg)](https://doi.org/10.5281/zenodo.23623786)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Existing de-identification tools and the papers that evaluate them were built for
free-text clinical notes. Clinical data does not move as notes. It moves as HL7 v2
messages, C-CDA documents and FHIR resources, where protected health information
(PHI) sits in structured fields, in non-standard segments, in standard fields the
sending system has misused, and in narrative text embedded inside the message.

No public benchmark measures PHI detection across those locations. This one does,
for HL7 v2. It generates synthetic messages with planted, annotated PHI, labels
every span by **where** it sits, and scores detectors with the breakdown that
matters: a tool can be perfect on `PID-5` and useless on a name typed into a
progress note, and a single overall recall figure hides exactly that.

**This is not a de-identification tool.** It removes nothing and replaces nothing.
It measures detection and reports the result.

## The headline result

500 messages, 25,585 annotated PHI spans, seed `20261013`. Reproduce with the two
commands below; the committed run is in [`results/`](results/).

| detector arm | recall | precision | structured | narrative | known Z-seg | unseen Z-seg | misused |
|---|---|---|---|---|---|---|---|
| rules-only | 0.825 | 1.000 | 1.000 | 0.000 | 0.80 | 0.00 | 0.596 |
| text-only (flattened) | 0.451 | 0.919 | 0.389 | 0.662 | 0.55 | 0.94 | 1.000 |
| schema-conditioned + regex narrative | 0.919 | 0.980 | 1.000 | 0.662 | 0.80 | 0.00 | 0.596 |

The point of the table is the **inversion**. Path-based rules are perfect on the
fields they were written for and score zero on narrative, because no path-based
rule finds a name someone typed into `OBX-5`. A text-only detector that flattens
the message to a string is the mirror image: it finds names in prose and misses
them in `Vanbrugh^Bartram`, because a caret is not a linguistic cue. Neither
approach alone is sufficient, and the gap between `ZPD` (0.80, a Z-segment the
shipped policy describes) and `ZIN` (0.00, one it has never seen) measures how
much of the non-standard problem is solved only by having already read the feed.

## Install and reproduce

Requires Python 3.12+. The only runtime dependency is PyYAML.

```bash
pip install -e ".[dev]"

deidbench corpus --out corpus.jsonl --messages 500 --seed 20261013
deidbench evaluate corpus.jsonl --json results.json

pytest            # 80 tests
```

The same seed regenerates the corpus byte for byte. That is the property the
whole repository exists to provide: a number nobody else can regenerate is an
assertion, not a result.

## What the corpus contains

Every message is synthetic. No production data, no real patients, no PHI, and no
IRB involvement. Names, addresses, cities and facilities come from pools written
for this corpus.

Each annotation is labelled with one of four **location classes**, which is what
makes the per-class breakdown possible:

| class | what it means |
|---|---|
| `structured` | PHI in the field the standard puts it in — `PID-5`, `NK1-2`, `PV1-7` |
| `narrative` | PHI inside free text — `OBX-5` observation values, `NTE-3` comments |
| `nonstandard` | PHI in a Z-segment. `ZPD` is in the shipped policy; `ZIN` deliberately is not |
| `misused` | PHI in a standard field the sending system has used for something else |

The generator also varies what production actually varies: standard and odd
delimiters, CR and CRLF terminators, several HL7 versions and message types, and
loose date formats. Offsets are asserted to survive all of it — a test fails if a
non-standard delimiter shifts any annotation by one byte.

## Scoring, and why it is asymmetric

**Recall** counts a gold span as found only if a prediction *fully covers* it.
Finding `brugh` inside `Vanbrugh` leaves PHI in the output, and scoring that as a
hit would report safety the system does not have. **Precision** counts a
prediction as correct if it *overlaps* any gold span — over-prediction costs
utility, not privacy.

Strict where a mistake leaks, lenient where it only costs. The harness also
reports a **typed** score, which asks the harder question: would this span have
been replaced with the right *kind* of surrogate, not merely removed?

## Five caveats that bound what these numbers can be used for

These travel with the numbers. The last one is printed at the bottom of every
report so it cannot be separated from the figures.

1. **The rules arm is flattered by construction.** The policy and the generator
   were written from the same reading of HL7, so `structured 1.000` measures rules
   against the fields they were authored against. The `misused` and `ZIN`
   categories exist to approximate what production does instead, and rules score
   0.596 and 0.000 there.
2. **The text-only arm is a stand-in, not a baseline.** Presidio, Philter and NLM
   Scrubber are the comparators that belong in a paper. This is a transparent
   regex-and-heuristic detector, written out in `eval/detectors.py` so the
   comparison can be criticized. Its name heuristic — requiring a title cue or two
   space-separated capitalized words — is the single assumption most worth
   challenging. Do not describe the 0.451 as any named tool's score.
3. **The third arm contains no model.** `RegexNarrativeDetector` is a placeholder
   that proves the harness end to end and establishes the floor a local
   open-weight model has to beat. `NarrativeDetector` is the seam it plugs into;
   swapping it changes one constructor argument. The architecture is the claim
   here, not the detector.
4. **Bare honorifics are not annotated as PHI.** `Mr`/`Dr` is placed in `PID-5.5`
   but not treated as ground truth: a title from a closed set of five values is
   not a name, so HIPAA's "Names" identifier does not reach it. Annotating it
   charged every arm a miss for correctly leaving it alone, and buried the 171 real
   field-misuse cases under 400 artifacts.
5. **Planted-PHI corpora measure what you planted.** These numbers are a floor and
   a methodology demonstration. They are not a substitute for measurement on
   production messages, and nothing written up from them should imply otherwise.

## What is deliberately not here

This repository is the measurement half of a larger system. The published part is
everything needed to regenerate the corpus and reproduce the figures above:
the HL7 v2 parser, the corpus generator, the HL7 v2 policy, the three detector
arms and the scoring harness.

Not published, because none of it is needed to check a number: surrogate
generation and its lexicons, the encrypted mapping vault, re-identification,
the FHIR R4 implementation and its policies, the HTTP API and the container.

One test is absent for the same reason. The engine asserts that corpus name pools
are disjoint from the surrogate lexicons, so a leak-check hit can never be a
coincidence between the two pools. That check needs the lexicons; the property it
guards still holds and is verified upstream.

## Scope and roadmap

HL7 v2 only, today. C-CDA is the harder case — identifiers sit in both coded
entries and a narrative block that routinely disagrees with them — and FHIR R4
annotation is not yet in the corpus. Both are planned. The open research question
the benchmark was built to answer is whether a language model small enough to run
on one modest on-premises server, given the structural context of each location,
can reach the recall that rules cannot.

Issues and pull requests are welcome, particularly: real baselines wired in as
additional arms, location classes the four here do not capture, and message
variants from production feeds that break the generator's assumptions.

## Citing this work

> Dick, R. (2026). *deidbench: a reproducible PHI-detection benchmark for HL7 v2*.
> Zenodo. https://doi.org/10.5281/zenodo.23623786

That DOI always resolves to the latest version. If you are reporting a number,
cite the **version** DOI of the release you actually ran — each release has its
own — so the figure and the code that produced it stay attached. Machine-readable
metadata is in [`CITATION.cff`](CITATION.cff).

Please quote the caveats with the numbers.

## License and patents

MIT, © 2026 Active Crossing LLC. The corpus this code generates is synthetic and
carries no patient privacy restrictions.

**No patent is or will be sought on the method described in this repository.** It
is published deliberately as prior art: the method it describes —
parsing interoperability messages into native structure, resolving what
path-based rules can resolve, and deferring only the remainder to a detector
together with its structural context — is free for anyone to implement, and this
publication is intended to prevent it being patented by anyone else.
