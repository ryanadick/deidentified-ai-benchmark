#!/usr/bin/env python3
"""Run the typing spike and print one table.

Three arms over the same spans, one variable between them:

    rules-only          what ships today  (0.00 on ZIN, 0.596 on misused)
    hybrid + model      context NOT given to the model
    hybrid + model+ctx  context given     <- the Aim 2 hypothesis

The number worth taking to a program officer is the TYPED recall in the
non-standard and misused rows, against the published 0.000 / 0.023 / 0.143.

Keep the sample small. This is a feasibility probe, not a benchmark run: at a
few seconds per model call, 500 messages would take hours and tell you nothing
the first 25 do not.

    python experiments/run_typing_spike.py --messages 25 --model qwen3:14b
    python experiments/run_typing_spike.py --messages 25 --sweep
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
from deidbench.eval.detectors import RulesOnlyDetector  # noqa: E402
from deidbench.eval.harness import evaluate  # noqa: E402
from ollama_typing import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_MODEL,
    HybridFallbackDetector,
    OllamaTypingDetector,
)

SEED = 20261013


def reachable(host: str) -> str | None:
    try:
        with urllib.request.urlopen(f"{host}/api/version", timeout=5) as r:
            return json.load(r).get("version")
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--messages", type=int, default=25,
                    help="corpus size; keep small, every span is a model call")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--sweep", action="store_true",
                    help="run 8b, 14b and 32b instead of one model")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--json", help="write the raw report here")
    args = ap.parse_args()

    version = reachable(args.host)
    if version is None:
        print(f"no Ollama at {args.host}\n"
              f"  ssh into the host and check: ollama list", file=sys.stderr)
        return 1
    print(f"ollama {version} at {args.host}")

    print(f"generating {args.messages} messages at seed {args.seed} ...")
    corpus = generate_corpus(
        GenerationOptions(messages=args.messages, seed=args.seed)
    )

    models = ["qwen3:8b", "qwen3:14b", "qwen3:32b"] if args.sweep else [args.model]

    detectors = [RulesOnlyDetector()]
    built: list[OllamaTypingDetector] = []
    for model in models:
        for use_context in (False, True):
            d = OllamaTypingDetector(
                model=model, host=args.host,
                use_context=use_context, seed=args.seed,
            )
            built.append(d)
            detectors.append(HybridFallbackDetector(narrative_detector=d))

    print(f"arms: {', '.join(d.name for d in detectors)}")
    print("running (expect several seconds per field) ...\n")

    report = evaluate(corpus, detectors)
    from deidbench.eval.harness import render

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

    print("\nThe number for the call is TYPED recall on the non-standard and")
    print("misused rows. Published tools: 0.000 Philter / 0.023 Scrubber /")
    print("0.143 Presidio. Rules-only today: 0.00 non-standard, 0.596 misused.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
