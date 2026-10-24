"""Two commands: generate an annotated corpus, then score detectors against it.

    deidbench corpus --out corpus.jsonl --messages 500
    deidbench evaluate corpus.jsonl --json results.json

Both are deterministic. The same seed produces the same corpus byte for byte,
which is the whole point: a number nobody else can regenerate is an assertion,
not a result.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    evaluate_parser.set_defaults(handler=_cmd_evaluate)

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
            (args.messages_dir / f"{message.message_id}.hl7").write_text(
                message.text, encoding="utf-8", newline=""
            )

    print(json.dumps(corpus.summary(), indent=2))
    print(f"wrote {len(corpus)} messages to {args.out}", file=sys.stderr)
    return 0


def _cmd_evaluate(args) -> int:
    corpus = AnnotatedCorpus.from_jsonl(args.corpus.read_text(encoding="utf-8"))
    policy = _resolve_policy(args.policy)
    report = evaluate(
        corpus,
        [
            RulesOnlyDetector(policy),
            FlattenedTextDetector(),
            SchemaConditionedDetector(policy=policy),
        ],
    )

    print(render(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(report.to_json(), encoding="utf-8")
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


def _resolve_policy(paths: list[Path]):
    if not paths:
        return load_default_policy()
    policy = load_policy(paths[0])
    for extra in paths[1:]:
        policy = policy.merged_with(load_policy(extra))
    return policy


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
