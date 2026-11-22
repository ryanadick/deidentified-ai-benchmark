# Spike results

Raw output from the Phase I feasibility probes. These are **probes, not
baselines**: 25 messages at seed 20261013, one model family, no fine-tuning.
Nothing here is a production measurement, and the corpus is machine-generated
with planted identifiers, so the structured figure is a ceiling and the
narrative figure a floor.

| run | question | headline |
|---|---|---|
| `typing-spike-20261028-1045` | can a local model type unrecognised fields, and does capacity matter? | non-standard recall 0.509 → 0.964; an unseen Z-segment 0.000 → 0.900; typed recall 0.788 → 0.895. **14B beats 32B** on untyped precision (0.986 vs 0.934) and typed precision (0.916 vs 0.877), and is 2.3× faster |
| `context-variants-20261030` | was the "schema context does not help" result an artifact of wording? | **No.** Four wordings at 14B: typed recall 0.890–0.895, no effect. False positives rose 18 → 92 as context grew richer |
| `whole-message-20261109` | why scope the model to fields rather than the whole message? | Whole-message drops typed recall 0.895 → 0.456 and degrades structured fields 1.000 → 0.618 — worse than no model at all, and 2× the wall clock despite 36× fewer calls |

> **Corrected in v0.2.1.** The Z-segment figure in the first row read 0.95 in
> v0.2.0. That value is the `+ctx` arm; the figure for the arm every other
> number in that row describes is 0.900. See [ERRATA.md](../../ERRATA.md).

The second result contradicted the hypothesis the work started from. Schema
conditioning was expected to help; it does not, and the aims were rewritten to
claim **routing** unrecognised structure to a model, with the value of schema
context stated as the open question.

Reproduce with `experiments/run_typing_spike.py`, `run_context_variants.py`,
and the `WholeMessageDetector` in `ollama_typing.py`. All require a local
Ollama host; set `--host`.

**Throughput figures are hardware-bound and these runs did not record theirs.**
Recall and precision are properties of the model and reproduce anywhere; `s/call`
is a property of the box. The runs above were made against a 48 GB
workstation-class host, but that was not captured in their provenance, so the
timings cannot be checked the way every other figure can. Later runs record it —
pass `hardware=` or set `DEID_BENCH_HARDWARE` — and anyone comparing timings
should state their own.
