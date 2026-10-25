"""Two commands: generate an annotated corpus, then score detectors against it.

    deidbench corpus --out corpus.jsonl --messages 500
    deidbench evaluate corpus.jsonl --json results.json

Both are deterministic. The same seed produces the same corpus byte for byte,
which is the whole point: a number nobody else can regenerate is an assertion,
not a result.

``evaluate`` runs three arms with no optional dependency, plus two arms for
each published baseline that is installed -- Microsoft Presidio and Philter
today, NLM Scrubber to come. ``--presidio`` and ``--philter`` make those arms
mandatory and fail loudly without them, which is what a reproduction of the
published table wants; ``--no-presidio`` and ``--no-philter`` drop them.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

from .corpus.annotation import Corpus as AnnotatedCorpus
from .corpus.generate import GenerationOptions, generate_corpus
from .eval.detectors import (
    FlattenedTextDetector,
    RulesOnlyDetector,
    SchemaConditionedDetector,
)
from .eval.harness import evaluate, render
from .policy.registry import load_default_policy, load_policy

DEFAULT_SEED = 20261013

#: The published baselines, in report order. Each is an optional dependency
#: behind one uniform contract: a ``deidbench.eval.<name>_baseline`` module
#: exposing ``<name>_available()``, ``_IMPORT_HINT``, and a flattened plus a
#: narrative detector class. Adding NLM Scrubber should be one more row.
BASELINE_ARMS: tuple[tuple[str, str, str], ...] = (
    ("presidio", "PresidioTextDetector", "PresidioNarrativeDetector"),
    ("philter", "PhilterTextDetector", "PhilterNarrativeDetector"),
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 2
    return args.handler(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deidbench",
        description="Reproducible PHI-detection benchmark for HL7 v2 messages",
    )
    subparsers = parser.add_subparsers(dest="command")

    corpus = subparsers.add_parser(
        "corpus", help="generate a synthetic annotated evaluation corpus"
    )
    corpus.add_argument("--out", type=Path, required=True, help="JSON Lines output")
    corpus.add_argument("--messages", type=int, default=200)
    corpus.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"generation seed (default {DEFAULT_SEED}, the seed behind the published figures)",
    )
    corpus.add_argument(
        "--messages-dir",
        type=Path,
        default=None,
        help="also write each message as its own .hl7 file",
    )
    corpus.set_defaults(handler=_cmd_corpus)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="score detector arms against an annotated corpus"
    )
    evaluate_parser.add_argument("corpus", type=Path, help="JSON Lines corpus")
    evaluate_parser.add_argument(
        "--json", type=Path, default=None, help="write the full report as JSON"
    )
    evaluate_parser.add_argument(
        "--policy",
        type=Path,
        action="append",
        default=[],
        help="override the bundled policy; repeatable, merged in order",
    )
    defaults = {"handler": _cmd_evaluate}
    for name, _text_cls, _narrative_cls in BASELINE_ARMS:
        group = evaluate_parser.add_mutually_exclusive_group()
        group.add_argument(
            f"--{name}",
            dest=name,
            action="store_const",
            const="require",
            help=f"require the {name} baseline arms; fail if {name} is missing",
        )
        group.add_argument(
            f"--no-{name}",
            dest=name,
            action="store_const",
            const="skip",
            help=f"skip the {name} arms even if {name} is installed",
        )
        defaults[name] = "auto"
    evaluate_parser.set_defaults(**defaults)

    return parser


def _cmd_corpus(args) -> int:
    corpus = generate_corpus(
        GenerationOptions(messages=args.messages, seed=args.seed)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(corpus.to_jsonl(), encoding="utf-8")

    if args.messages_dir:
        args.messages_dir.mkdir(parents=True, exist_ok=True)
        for message in corpus.messages:
            # newline="" so the terminators the generator chose survive the
            # write. open() rather than Path.write_text(), whose newline
            # argument is Python 3.13+ only.
            path = args.messages_dir / f"{message.message_id}.hl7"
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(message.text)

    print(json.dumps(corpus.summary(), indent=2))
    print(f"wrote {len(corpus)} messages to {args.out}", file=sys.stderr)
    return 0


def _cmd_evaluate(args) -> int:
    corpus = AnnotatedCorpus.from_jsonl(args.corpus.read_text(encoding="utf-8"))
    policy = _resolve_policy(args.policy)
    detectors = [
        RulesOnlyDetector(policy),
        FlattenedTextDetector(),
        SchemaConditionedDetector(policy=policy),
    ]
    for spec in BASELINE_ARMS:
        detectors.extend(_baseline_detectors(spec, getattr(args, spec[0]), policy))

    report = evaluate(corpus, detectors)

    print(render(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(report.to_json(), encoding="utf-8")
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


def _baseline_detectors(spec: tuple[str, str, str], mode: str, policy) -> list:
    """The two arms for one published baseline, if they can run.

    Every baseline is an optional dependency, so a missing install is not an
    error by default -- but it is never silent either. The published table
    includes these arms, and a run that quietly omitted them would look like
    the published run while measuring something smaller.
    """
    name, text_cls, narrative_cls = spec
    if mode == "skip":
        return []

    module = import_module(f"deidbench.eval.{name}_baseline")

    if not getattr(module, f"{name}_available")():
        if mode == "require":
            print(module._IMPORT_HINT, file=sys.stderr)
            raise SystemExit(
                f"--{name} was requested but {name} is not importable"
            )
        print(
            f"note: {name} is not installed, so its two baseline arms were\n"
            "skipped. The published table includes them; this run is a subset.\n"
            "To match the published figures:\n"
            f"{module._IMPORT_HINT}",
            file=sys.stderr,
        )
        return []

    return [
        getattr(module, text_cls)(),
        SchemaConditionedDetector(
            narrative_detector=getattr(module, narrative_cls)(), policy=policy
        ),
    ]


def _resolve_policy(paths: list[Path]):
    if not paths:
        return load_default_policy()
    policy = load_policy(paths[0])
    for extra in paths[1:]:
        policy = policy.merged_with(load_policy(extra))
    return policy


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
