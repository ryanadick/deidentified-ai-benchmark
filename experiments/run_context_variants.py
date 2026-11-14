#!/usr/bin/env python3
"""Does the WORDING of the schema context change anything?

The Oct 28 sweep compared context-off against one context wording and found a
difference of 0.001-0.007 in typed recall -- nothing, at every capacity tier.
That result is often read as "schema context does not help". It does not
support that. It supports a narrower claim: *this wording* did not help.

The wording in question renders a fallback span as

    segment=ZIN, field=3, description=no policy rule for this location

which is key-value noise whose only semantic content announces that nothing is
known about the location. A model given that has been told less than nothing.

So this holds the model fixed at the tier that won the sweep (14b) and varies
only the wording:

    none     no structural description at all        (the Oct 28 baseline)
    terse    segment=.., field=.., description=..    (the Oct 28 context arm)
    plain    the same facts as an English sentence
    domain   plus what the structure does NOT tell you, said explicitly

``domain`` is the one with a real hypothesis behind it. On an unrecognised
Z-segment it says the specification is silent and the value must be judged on
its own terms -- which is the true state of affairs, and is the opposite of
what ``terse`` communicates by emitting an empty description field.

    python experiments/run_context_variants.py
    python experiments/run_context_variants.py --messages 25 --model qwen3:14b
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from deidbench.corpus.generate import GenerationOptions, generate_corpus  # noqa: E402
from deidbench.eval.detectors import (  # noqa: E402
    NarrativeCandidate,
    RulesOnlyDetector,
)
from deidbench.eval.harness import evaluate, render  # noqa: E402
from ollama_typing import (  # noqa: E402
    DEFAULT_HOST,
    HybridFallbackDetector,
    OllamaTypingDetector,
)

SEED = 20261013
STYLES = ("none", "terse", "plain", "domain")

#: Two candidates that stand in for the two cases that matter: one the policy
#: resolves, one it does not. Printed so a reader can see exactly what varied
#: without running anything.
SAMPLES = (
    NarrativeCandidate(
        text="Barnabus Drummond", start=0, end=17,
        path="PID-5-1", policy_path="PID-5-1", segment="PID", field=5,
        component=1, data_type="XPN", note="Patient Name",
    ),
    NarrativeCandidate(
        text="716-555-0142", start=0, end=12,
        path="ZIN-3", policy_path="ZIN-3", segment="ZIN", field=3,
        component=None, data_type="", note="no policy rule for this location",
    ),
)


def reachable(host: str) -> str | None:
    try:
        with urllib.request.urlopen(f"{host}/api/version", timeout=5) as r:
            return json.load(r).get("version")
    except Exception:  # noqa: BLE001
        return None


def show_wordings(model: str, host: str) -> None:
    print("=" * 72)
    print("WHAT ACTUALLY VARIES -- the prompt prefix, for two candidates")
    print("=" * 72)
    for cand in SAMPLES:
        kind = "policy HAS a rule" if cand.data_type else "policy has NO rule"
        print(f"\n  {cand.path}  ({kind})   value: {cand.text!r}")
        for style in STYLES:
            d = OllamaTypingDetector(model=model, host=host, context_style=style)
            ctx = d._context(cand).strip()
            print(f"    {style:<7} {ctx if ctx else '(nothing)'}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--messages", type=int, default=25)
    ap.add_argument("--model", default="qwen3:14b",
                    help="held fixed; 14b won the Oct 28 capacity sweep")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--json", help="write the raw report here")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the wordings and exit, no model calls")
    args = ap.parse_args()

    show_wordings(args.model, args.host)
    if args.dry_run:
        return 0

    version = reachable(args.host)
    if version is None:
        print(f"no Ollama at {args.host}", file=sys.stderr)
        return 1
    print(f"ollama {version} at {args.host}, model {args.model}")

    print(f"generating {args.messages} messages at seed {args.seed} ...")
    corpus = generate_corpus(
        GenerationOptions(messages=args.messages, seed=args.seed)
    )

    detectors = [RulesOnlyDetector()]
    built: list[OllamaTypingDetector] = []
    for style in STYLES:
        d = OllamaTypingDetector(
            model=args.model, host=args.host,
            context_style=style, seed=args.seed,
        )
        built.append(d)
        detectors.append(HybridFallbackDetector(narrative_detector=d))

    print(f"arms: {', '.join(d.name for d in detectors)}")
    print("running ...\n")

    report = evaluate(corpus, detectors)
    print(render(report))

    print("\nmodel usage")
    for d in built:
        print(f"  {d.name:<28} {d.usage.summary()}")

    if args.json:
        import dataclasses
        payload = (dataclasses.asdict(report)
                   if dataclasses.is_dataclass(report) else report)
        Path(args.json).write_text(json.dumps(payload, default=str, indent=2))
        print(f"\nwrote {args.json}")

    print("\nRead the NON-STANDARD and ZIN rows, not the aggregate. The aggregate")
    print("is dominated by structured fields the rules already resolve, where the")
    print("wording cannot matter. If any variant moves a number, it moves it where")
    print("the policy is silent -- which is the whole of Aim 2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
