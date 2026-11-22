# Errata

Corrections to published figures, newest first. Each entry says what was
wrong, what the correct value is, how to verify it from the data in this
repository, and which releases carry the error.

The raw result files have never been edited. Every correction here is to a
figure that was transcribed out of them into prose or a summary table, and
each can be checked against the JSON.

---

## v0.2.1 — unrecognised Z-segment recall reported from the wrong arm

**Affects:** v0.2.0 (DOI [10.5281/zenodo.24224414][v020]). Not present in
v0.1.0, which predates the spike.

**Where:** `experiments/results/README.md`, the `typing-spike-20261028-1045`
summary row.

| | published in v0.2.0 | correct |
|---|---|---|
| unseen Z-segment (`ZIN`), rules + 14B | 0.95 | **0.900** |

**What happened.** The spike ran seven detector arms. The summary row
describes `hybrid+ollama:qwen3:14b+noctx` — that is the arm its other figures
come from, and it is the configuration the project adopted. The Z-segment
value was taken instead from `hybrid+ollama:qwen3:14b+ctx`, the
schema-context arm. One row therefore mixed two arms.

The mix-up is easy to make and worth naming precisely: the context arm scores
*higher* on this class (0.950 against 0.900), which is why the error inflated
rather than deflated the number. It does not change the conclusion that
context does not help — `context-variants-20261030` tests that directly, on
typed recall overall, and that result is unaffected. But anyone comparing arms
on the Z-segment class alone was given the wrong pairing.

**Verify it:**

```python
import json
d = json.load(open("experiments/results/typing-spike-20261028-1045.json"))
for det in d["detectors"]:
    z = det["by_segment"]["ZIN"]
    print(f'{det["detector"]:<32} {z["covered"]}/{z["gold"]} = {z["covered"]/z["gold"]:.3f}')
```

```
rules-only                        0/20 = 0.000
hybrid+ollama:qwen3:8b+noctx     17/20 = 0.850
hybrid+ollama:qwen3:8b+ctx       13/20 = 0.650
hybrid+ollama:qwen3:14b+noctx    18/20 = 0.900   <- the arm the row describes
hybrid+ollama:qwen3:14b+ctx      19/20 = 0.950   <- the value published
hybrid+ollama:qwen3:32b+noctx    17/20 = 0.850
hybrid+ollama:qwen3:32b+ctx      15/20 = 0.750
```

Note the denominator: `ZIN` carries 20 annotated spans in a 25-message sample,
so one span is 0.05. A figure on this class should not be read as precise to
better than that, and the class is reported because it is the *only* segment
type the shipped policy has never seen — not because 20 spans is a sufficient
sample.

**Also corrected in the same row**, for clarity rather than accuracy:
`0.509 → 0.96` is now written `0.509 → 0.964` to match the three-decimal
convention used everywhere else, and the "14B beats 32B on precision
(0.986 vs 0.934)" claim now says *untyped* precision, with the typed figures
(0.916 vs 0.877) given alongside. Both values were correct; neither said which
precision it meant.

[v020]: https://doi.org/10.5281/zenodo.24224414

---

## Reporting an error

Open an issue. Corrections to published figures are released as a patch
version with an entry here rather than amended silently, so that a citation to
an earlier version still resolves to what was actually read at the time.
