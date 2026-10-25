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

500 messages, 25,585 annotated PHI spans, seed `20261013`. Reproduce with the
commands below; the committed runs are in [`results/`](results/).

| detector arm | recall | precision | structured | narrative | known Z-seg | unseen Z-seg | misused |
|---|---|---|---|---|---|---|---|
| rules-only | 0.825 | 1.000 | 1.000 | 0.000 | 0.80 | 0.00 | 0.596 |
| text-only, ours (flattened) | 0.451 | 0.919 | 0.389 | 0.662 | 0.55 | 0.94 | 1.000 |
| **Presidio** (flattened) | 0.506 | 0.841 | 0.460 | 0.717 | 0.55 | 0.71 | 0.585 |
| schema-conditioned + regex narrative | 0.919 | 0.980 | 1.000 | 0.662 | 0.80 | 0.00 | 0.596 |
| **schema-conditioned + Presidio narrative** | 0.932 | 0.977 | 1.000 | 0.755 | 0.80 | 0.00 | 0.596 |

The point of the table is the **inversion**. Path-based rules are perfect on the
fields they were written for and score zero on narrative, because no path-based
rule finds a name someone typed into `OBX-5`. A detector that flattens the
message to a string is the mirror image: it finds names in prose and misses them
in `Vanbrugh^Bartram`, because a caret is not a linguistic cue. Neither approach
alone is sufficient, and the gap between `ZPD` (0.80, a Z-segment the shipped
policy describes) and `ZIN` (0.00, one it has never seen) measures how much of
the non-standard problem is solved only by having already read the feed.

**The inversion is not an artifact of a home-made opponent.** Microsoft Presidio
— a published, widely cited tool — is run here out of the box, and it reproduces
the text-only failure pattern: 0.460 on structured fields against 0.717 on
narrative. Flattening HL7 and handing it to a free-text de-identifier is a
measured 0.506, not an assumption.

Running Presidio also audits the stand-in it replaced, and the stand-in comes out
*mildly pessimistic on recall and flattering on precision*: 0.451 against
Presidio's 0.506, with precision 0.919 against 0.841. So the hand-written arm was
a somewhat weak opponent on the metric that matters most — worth knowing, and the
reason this was measured rather than argued — but it was not a straw man, and no
conclusion in the table rests on it.

The strongest row is the last one, because it puts the **same published tool on
both sides** of the architectural comparison. Presidio flattened scores 0.506;
Presidio behind schema conditioning, seeing only the locations the policy defers,
scores 0.932 — and gains on narrative specifically (0.717 → 0.755), where the
coded fields that generate most of its false positives are no longer in front of
it. Holding the detector fixed and changing only the architecture is worth 0.43
recall, which is the claim this repository exists to test.

One caveat travels with the Presidio rows and is marked in every report: its
**typed** scores (0.163 typed recall) are a floor, not a measurement. Presidio
has one `PERSON` category where the ground truth distinguishes six kinds of name,
and one `DATE_TIME` where the corpus separates birth from event dates, so typed
scoring charges it for a distinction it was never built to draw. Read its untyped
column as the comparable number.

## Install and reproduce

Requires Python 3.12. The three core arms need only PyYAML.

```bash
pip install -e ".[dev]"

deidbench corpus --out corpus.jsonl --messages 500 --seed 20261013
deidbench evaluate corpus.jsonl --json results.json

pytest            # 95 tests; the Presidio ones skip without the extra
```

To reproduce the **Presidio** rows as well, install the optional baseline extra
and its language model, then ask for the arms explicitly:

```bash
pip install -e ".[dev,presidio]"
python -m spacy download en_core_web_lg

deidbench evaluate corpus.jsonl --presidio --json results.json
```

`--presidio` fails loudly if Presidio is missing, which is what a reproduction
wants. Without the flag the Presidio arms are included when importable and
skipped with a note on stderr when not, so a three-arm run can never be mistaken
for the full published table. `--no-presidio` runs the three-arm subset.

Two notes on that extra, both of which cost time to rediscover:

- **Python 3.12, not 3.13.** The pinned spaCy stack has no 3.13 wheels. The
  package declares 3.12 and is tested there.
- **The pins in `[presidio]` are a verified working set, not caution.** spaCy
  3.8 against the current `murmurhash`/`preshed`/`cymem` wheels raises
  `OverflowError` inside the tokenizer when loading *any* model, so the detector
  cannot run at all, and numpy 2 breaks the matching `thinc` build. The extra
  pins the combination that actually loads `en_core_web_lg`.

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

Both rules share one exploit, so both are reported against a character count. A
prediction that swallows an entire segment fully covers every gold span inside it
*and* overlaps them, collecting recall and precision while flagging most of the
message. This is not hypothetical — it is what a general-purpose NER model does
to caret-delimited HL7, which has no sentence structure to work from. The
`chars x` column is the guard: characters flagged per character of real PHI,
where 1.0 is a detector flagging exactly the PHI. Every arm in the table above
sits between 0.6 and 1.0, so none of these figures are bought by over-flagging —
but that is now a measured fact in the report rather than an assumption, and a
future arm that games the rules will show it in that column.

## Six caveats that bound what these numbers can be used for

These travel with the numbers. The last one is printed at the bottom of every
report so it cannot be separated from the figures.

1. **The rules arm is flattered by construction.** The policy and the generator
   were written from the same reading of HL7, so `structured 1.000` measures rules
   against the fields they were authored against. The `misused` and `ZIN`
   categories exist to approximate what production does instead, and rules score
   0.596 and 0.000 there.
2. **`text-only` is our own detector; `presidio-text` is the published one.** The
   `text-only` arm is a regex-and-heuristic detector we wrote, kept because it is
   the arm behind the originally published figures and because its rules can be
   read in `eval/detectors.py` where a statistical model can only be run. Its
   **0.451 is not Presidio's score and must never be reported as one** — Presidio
   has its own row. Philter and NLM Scrubber are still to come, so the set of
   published comparators here is one tool, not the field.
3. **Neither narrative detector is the intended model.** `RegexNarrativeDetector`
   is a placeholder that establishes a floor. Presidio in the narrative seam is a
   real statistical model and the best arm in the table, but it is a general-
   purpose NER model, not the local open-weight clinical model the Phase I
   hypothesis is about — and it gets no structural context, because Presidio has
   no interface to accept any. `schema+presidio-narrative` is therefore the floor
   that model has to beat, not a result about it. `NarrativeDetector` is the seam;
   swapping it changes one constructor argument. The architecture is the claim
   here, not the detector.
4. **Presidio's typed scores are a floor, and its entity mapping is lossy.**
   Presidio's vocabulary is coarser than this corpus's ground truth — one `PERSON`
   for six kinds of name, one `DATE_TIME` for birth and event dates — so typed
   scoring charges it for distinctions it was never built to draw. It is run in
   its **default** configuration: default recognizers, no score threshold, no
   HL7-specific tuning, no deny-lists. A baseline we tuned ourselves would have
   the same credibility problem as one we wrote ourselves, but the untuned
   configuration is a floor for Presidio too, and the mapping in
   `eval/presidio_baseline.py` is the first thing to challenge. Entity types with
   no counterpart in the corpus vocabulary are reported under an `unmapped:`
   prefix rather than dropped, so they still cost precision.
5. **Bare honorifics are not annotated as PHI.** `Mr`/`Dr` is placed in `PID-5.5`
   but not treated as ground truth: a title from a closed set of five values is
   not a name, so HIPAA's "Names" identifier does not reach it. Annotating it
   charged every arm a miss for correctly leaving it alone, and buried the 171 real
   field-misuse cases under 400 artifacts.
6. **Planted-PHI corpora measure what you planted.** These numbers are a floor and
   a methodology demonstration. They are not a substitute for measurement on
   production messages, and nothing written up from them should imply otherwise.

## What is deliberately not here

This repository is the measurement half of a larger system. The published part is
everything needed to regenerate the corpus and reproduce the figures above:
the HL7 v2 parser, the corpus generator, the HL7 v2 policy, the five detector
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
