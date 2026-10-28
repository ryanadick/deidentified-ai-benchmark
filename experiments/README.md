# experiments/

Throwaway probes. Nothing here is published, scored in `results/`, imported by
`deidbench`, or covered by the test suite. Each exists to answer one question
and then stop being interesting.

## `ollama_typing.py` + `run_typing_spike.py`

**Question:** can a small local open-weight model say *what kind* of identifier
an unrecognised field contains?

**Why it matters:** three published tools score 0.000, 0.023 and 0.143 on typed
recall. Typing is what drives format-preserving surrogate replacement, so it is
the capability that decides whether a de-identified message stays clinically
usable. It is the uncontested gap the NIH Phase I Aim 2 rests on.

**What it adds:** `SchemaConditionedDetector.candidates()` contains
`if rule is None ... continue` — a location the policy has no rule for is
*skipped*, not deferred. That one line is the mechanical cause of the measured
0.00 recall on ZIN, the Z-segment the policy has never seen, where a crude
regex over the same flattened text scores 0.94. `HybridFallbackDetector`
changes exactly that: no rule means send it to the model anyway.

**The variable:** `OllamaTypingDetector(use_context=...)`. Both sides run
identical code; only whether `candidate.context_line()` appears in the prompt
differs. That is the Aim 2 hypothesis stated as an experiment.

```bash
python experiments/run_typing_spike.py --messages 25 --model qwen3:14b
python experiments/run_typing_spike.py --messages 25 --sweep   # 8b, 14b, 32b
```

Needs Ollama reachable (default `http://mini.local:11434`) with the models
pulled. Every unresolved field is one model call, so keep `--messages` small.

### First run, 2026-10-27 — n=2 messages, qwen3:8b

Not a result. A working harness, and one early signal worth chasing.

| arm | untyped recall | typed recall | ZIN | precision |
|---|---|---|---|---|
| rules-only | 0.755 | 0.726 | **0.00** | 1.000 |
| hybrid, no context | 1.000 | 0.849 | **1.00** | 0.766 |
| hybrid, with context | 0.915 | 0.811 | 0.75 | 0.734 |

**The fallback premise holds:** ZIN moves 0.00 → 1.00. Skipping unrecognised
structure really was the whole cause.

**The context hypothesis did not:** the no-context arm beat the with-context arm
on every measure. At n=2 that is noise — 106 spans, 4 of them non-standard — but
it is the opposite of what Aim 2 predicts, and it is exactly what this spike
exists to find out before a proposal commits to it.

Also note `chars x` 1.2 on both model arms: they flag ~20% more characters than
there is real PHI, and precision falls from 1.000 to ~0.75. Over-flagging buys
recall. Any headline number must be read with that.

**Next:** `--messages 25 --sweep`, which is the run that can actually say
something. Roughly an hour.
