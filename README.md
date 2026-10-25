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

| detector arm | recall | precision | typed recall | structured | narrative | known Z-seg | unseen Z-seg |
|---|---|---|---|---|---|---|---|
| rules-only | 0.825 | 1.000 | 0.801 | 1.000 | 0.000 | 0.80 | 0.00 |
| text-only, ours (flattened) | 0.451 | 0.919 | 0.308 | 0.389 | 0.662 | 0.55 | 0.94 |
| **Presidio** (flattened) | 0.506 | 0.841 | 0.143 | 0.460 | 0.717 | 0.55 | 0.71 |
| **Philter** (flattened) | 0.738 | 0.879 | 0.000 | 0.731 | 0.785 | 0.75 | 0.69 |
| **NLM Scrubber** (flattened) | 0.321 | 0.895 | 0.023 | 0.271 | 0.743 | 0.00 | 0.00 |
| schema-conditioned + regex narrative | 0.919 | 0.980 | 0.859 | 1.000 | 0.662 | 0.80 | 0.00 |
| schema-conditioned + Presidio narrative | 0.932 | 0.977 | 0.845 | 1.000 | 0.755 | 0.80 | 0.00 |
| schema-conditioned + Philter narrative | 0.936 | 0.977 | 0.801 | 1.000 | 0.779 | 0.80 | 0.00 |
| **schema-conditioned + Scrubber narrative** | **0.936** | **0.984** | 0.825 | 1.000 | 0.784 | 0.80 | 0.00 |

Path-based rules are perfect on the fields they were written for and score zero
on narrative, because no path-based rule finds a name someone typed into `OBX-5`.
The gap between `ZPD` (0.80, a Z-segment the shipped policy describes) and `ZIN`
(0.00, one it has never seen) measures how much of the non-standard problem is
solved only by having already read the feed.

**The three published tools line up on exactly one axis: how much they depend on
linguistic context.** Compare each tool's structured recall against its own
narrative recall:

| tool | structured | narrative | ratio | approach |
|---|---|---|---|---|
| NLM Scrubber | 0.271 | 0.743 | **0.36** | NLP over prose |
| Presidio | 0.460 | 0.717 | **0.64** | NER + pattern recognizers |
| Philter | 0.731 | 0.785 | **0.93** | whitelist — redact what isn't provably safe |

That ordering is the central finding. The more a tool reasons about text the way
a reader does, the worse it does on a structured message — because in `PID-5`
there is no sentence, and `Vanbrugh^Bartram` offers no linguistic cue at all.
Philter nearly closes the gap precisely because it is *not* looking for PHI: it
keeps only what its filters confirm is safe and redacts the rest, so
caret-delimited name components fall to it with no signal required.

So the blanket claim that free-text tools fail on structured HL7 is **too broad,
and this benchmark corrects it.** It holds, strongly, for context-dependent
detection. It does not hold for whitelist-based redaction. That distinction was
invisible with one comparator — the argument for not stopping at Presidio.

**Scrubber is the sharpest illustration and the weakest arm.** Its *published*
NAME recall is 0.9393 on i2b2 prose notes. Here it catches names in `OBX`/`NTE`
prose and leaves `Everman^Clara^Jean^JR` untouched, scoring 0.02 on `PV1`, 0.05
on `NK1`, and **0.00 in Z-segments of either kind**. One tool, the same
identifier type, only the shape of the data changed. That is the thesis of this
repository in a single row.

Philter's cost shows up in the three columns next to its recall. It has the
highest false-positive count of any arm (2,788), and its **typed recall is
0.000** — not a mapping artifact but a structural fact: Philter labels almost
everything it removes `OTHER`, because it knows a span is unsafe without
claiming to know what it is. A tool that cannot say *what* it removed cannot
drive format-preserving surrogate replacement, which is the question the typed
column asks. For redaction Philter is excellent; for reversible
de-identification it is not a candidate, whatever its recall.

Note also what the `chars x` column rules out: at 0.9, Philter is **not** buying
that recall by flagging everything. The obvious explanation for an aggressive
redactor scoring well is over-flagging, and the guard says no.

The architecture wins with a published tool on **both sides** of the comparison,
and now three times independently. Holding the detector fixed and changing only
the architecture is worth **+0.198 recall** against Philter, **+0.426** against
Presidio and **+0.615** against Scrubber. The gain is real and robust across
three tools of quite different design — and its size depends heavily on which
tool you measure it with, so quoting any single figure would be cherry-picking.

The best arm is schema-conditioned + Scrubber at 0.936 recall and **0.984
precision**, the highest precision of any arm with a model in it. That is worth
reading carefully rather than as a win: Scrubber contributes *less* to that arm
than the others do, so more of it is the rules, which is also why its false
positives are the lowest (378). A weak narrative detector behind strong schema
conditioning can look like a strong system.

## Install and reproduce

Requires Python 3.12. The three core arms need only PyYAML.

```bash
pip install -e ".[dev]"

deidbench corpus --out corpus.jsonl --messages 500 --seed 20261013
deidbench evaluate corpus.jsonl --json results.json

pytest            # 128 tests; baseline tests skip without their extras
```

To reproduce the **published-baseline** rows as well, install the optional
extras and their data files, then ask for the arms explicitly:

```bash
pip install -e ".[dev,presidio,philter]"

# spaCy model, pinned to the build that matches the pinned spaCy. Install it by
# URL rather than with `spacy download`: that command shells out to pip (absent
# in a uv-created venv) and resolves a model version that may not match.
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl

python -m nltk.downloader averaged_perceptron_tagger \
    averaged_perceptron_tagger_eng punkt punkt_tab

deidbench evaluate corpus.jsonl --presidio --philter --json results.json
```

**NLM Scrubber** is a binary distribution, not a Python package, so it has no
pip extra and is **not redistributed here** — the download page states no
redistribution terms, so you fetch it yourself:

```bash
# Linux x86_64 build, v.19.0403L
# https://lhncbc.nlm.nih.gov/scrubber/download.html
# archive sha256: 4dbe08a458433c3dac71007ce17ae81b7d0f392e1370daedb3c65da47f79e48d
unzip scrubber.19.0403L.zip
export DEIDBENCH_SCRUBBER=$PWD/scrubber.19.0403L/scrubber.19.0403.lnx

deidbench evaluate corpus.jsonl --presidio --philter --scrubber --json results.json
```

On a host that is not x86-64 Linux — including Apple Silicon — the benchmark runs
it through Docker under `linux/amd64` emulation automatically. Override the image
with `DEIDBENCH_SCRUBBER_IMAGE` if `debian:bullseye` doesn't suit.

`--presidio` and `--philter` fail loudly if the tool is missing, which is what a
reproduction wants. Without the flags each baseline's arms are included when
importable and skipped with a note on stderr when not, so a partial run can
never be mistaken for the full published table. `--no-presidio` and
`--no-philter` drop them.

Notes on those extras, all of which cost time to rediscover:

- **Python 3.12, not 3.13.** The pinned spaCy stack has no 3.13 wheels. The
  package declares 3.12 and is tested there.
- **The pins in `[presidio]` are a verified working set, not caution.** spaCy
  3.8 against the current `murmurhash`/`preshed`/`cymem` wheels raises
  `OverflowError` inside the tokenizer when loading *any* model, so the detector
  cannot run at all, and numpy 2 breaks the matching `thinc` build. The extra
  pins the combination that actually loads `en_core_web_lg`.
- **The Philter arm is `philter-lite`, not `philter-ucsf`.** This is forced, not
  preferred. 324 of the 617 regex filter files in UCSF's published
  `philter_delta` set place an inline `(?i)` flag mid-pattern, which Python has
  rejected as an error since 3.11 — the published distribution cannot run on any
  Python this package supports. `philter-lite` is a BSD fork carrying the same
  filter set with the flag moved to the front, which is semantics-preserving
  (an inline global flag always applied to the whole pattern; the misleading
  position is *why* it became an error). See the caveat below for how that claim
  is verified rather than asserted.

The same seed regenerates the corpus byte for byte. That is the property the
whole repository exists to provide: a number nobody else can regenerate is an
assertion, not a result.

The baseline arms hold that property too, and it took work. Presidio can return
one span under two entity types with equal confidence, and which one survived
depended on the order it emitted them — an order that varies between processes
with Python's hash randomization. Untyped scores were unaffected, but Presidio's
*typed* score drifted by around 13 spans per 500 messages from run to run.
`stable_span_order` in `eval/detectors.py` imposes a total order so ties resolve
identically everywhere. Verified by running the full evaluation under different
`PYTHONHASHSEED` values and diffing: every score in every arm matches.

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

## Eight caveats that bound what these numbers can be used for

These travel with the numbers. The last one is printed at the bottom of every
report so it cannot be separated from the figures.

1. **The rules arm is flattered by construction.** The policy and the generator
   were written from the same reading of HL7, so `structured 1.000` measures rules
   against the fields they were authored against. The `misused` and `ZIN`
   categories exist to approximate what production does instead, and rules score
   0.596 and 0.000 there.
2. **`text-only` is our own detector; the Presidio, Philter and Scrubber arms
   are the published ones.** The `text-only` arm is a regex-and-heuristic detector we
   wrote, kept because it is the arm behind the originally published figures and
   because its rules can be read in `eval/detectors.py` where a statistical model
   can only be run. Its **0.451 is not any named tool's score and must never be
   reported as one** — each tool has its own row. Measured against the published
   pair, our stand-in was a weak opponent: 0.451 against Presidio's 0.506 and
   Philter's 0.738, though stronger than Scrubber's 0.321. Three published tools
   is not the field, but it is enough to show they do not fail alike.
3. **None of the narrative detectors is the intended model.**
   `RegexNarrativeDetector` is a placeholder that establishes a floor. Presidio
   and Philter in the narrative seam are real published tools and the two best
   arms in the table, but neither is the local open-weight clinical model the
   Phase I hypothesis is about, and neither receives any structural context —
   because neither has an interface to accept it, and prepending context to the
   value would shift every offset and make it a tool we modified rather than the
   tool other people run. So the schema-conditioned baseline arms measure *the
   architecture's floor*, with a general-purpose detector in the seam, and that
   floor (0.936) is what the model has to beat. `NarrativeDetector` is the seam;
   swapping it changes one constructor argument. The architecture is the claim
   here, not the detector.
4. **The published baselines' typed scores are floors, and their type mappings
   are lossy.** Both tools have a coarser vocabulary than this corpus's ground
   truth. Presidio has one `PERSON` for six kinds of name and one `DATE_TIME` for
   birth and event dates. Philter is more extreme: it labels almost everything
   `OTHER`, which has no honest counterpart here, so its typed score is 0.000.
   Unmapped categories are reported under an `unmapped:` prefix rather than
   dropped, so they still cost precision — and we deliberately do **not** treat
   Philter's `OTHER` as a wildcard matching any gold type, which would
   manufacture typed accuracy it never demonstrated. Read the untyped column as
   the comparable number for both. Both run in their **default** configuration:
   stock recognizers and the published `philter_delta` filter set, no score
   threshold, no HL7-specific tuning, no deny-lists. A baseline we tuned would
   have the same credibility problem as one we wrote — but untuned is also a
   floor for them, and the mappings in `eval/presidio_baseline.py` and
   `eval/philter_baseline.py` are the first thing to challenge.
5. **The Philter arm is a port, not the distribution UCSF ships.** It runs
   `philter-lite`, a BSD fork, because 324 of the 617 regex filter files in the
   published `philter_delta` set place an inline `(?i)` flag mid-pattern, which
   Python has rejected as an error since 3.11 — `philter-ucsf` cannot run on any
   Python this package supports. Rather than assert the fork is faithful, we
   checked: of 308 comparable regex patterns, **306 are byte-identical to the
   published files or differ only by relocating `(?i)` to the front** (which is
   semantics-preserving, since an inline global flag always applied to the whole
   pattern). `tests/test_philter_baseline.py` pins a sample verbatim, including
   an upstream typo the fork preserves — a transcription fingerprint, since a
   rewrite would have fixed it. The two genuine divergences are both fork
   bug-fixes in *whitelist* filters, so the fork treats marginally more text as
   safe and if anything under-redacts relative to published Philter. Report this
   row as "Philter (philter-lite port)".
6. **The Scrubber arm's spans are inferred, not reported — and its recall is a
   band.** Presidio returns offsets and Philter returns a coordinate map;
   Scrubber returns *redacted text* with inline `[PERSONALNAME]`-style tags,
   which changes the document length and shifts every later offset. Every span
   in that arm is recovered by aligning its output against the original, so this
   is the one arm whose numbers depend on our inference rather than the tool's
   report. The guard is a round-trip requirement: recovered spans must
   reconstruct Scrubber's output byte-for-byte, and a message that fails is
   excluded rather than scored. All 500 messages and all 984 deferred locations
   pass. Separately, one Scrubber tag can replace a run far wider than the PHI
   inside it — 17.7% of regions cover more than one gold span, carrying 62.1% of
   its credited spans. Under this benchmark's full-coverage rule those count, so
   the arm scores **0.321**; crediting only spans that do not depend on an
   over-broad region gives **0.121**. Over-flagging in aggregate stays low (0.6×
   characters), so this is a few wide regions rather than blanket redaction.
   Also note the binary is unmaintained since 2019, has five tags, runs under
   emulation here, and — because it is batched — its `ms` figure counts cache
   lookups, not work, and is not comparable to the other arms.
7. **Bare honorifics are not annotated as PHI.** `Mr`/`Dr` is placed in `PID-5.5`
   but not treated as ground truth: a title from a closed set of five values is
   not a name, so HIPAA's "Names" identifier does not reach it. Annotating it
   charged every arm a miss for correctly leaving it alone, and buried the 171 real
   field-misuse cases under 400 artifacts.
8. **Planted-PHI corpora measure what you planted.** These numbers are a floor and
   a methodology demonstration. They are not a substitute for measurement on
   production messages, and nothing written up from them should imply otherwise.

## What is deliberately not here

This repository is the measurement half of a larger system. The published part is
everything needed to regenerate the corpus and reproduce the figures above:
the HL7 v2 parser, the corpus generator, the HL7 v2 policy, the nine detector
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
