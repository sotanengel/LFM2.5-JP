"""ifeval_ja orchestration CLI: generate / score / all (Issue #104).

Unlike run_llm_jp_eval.py, this harness never touches the WSL llm-jp-eval /
llm-jp-eval-inference checkouts or their `_find_latest_prompts_dir` mtime
lookup -- it reads configs/eval/ifeval_ja.yaml's own dataset_path directly,
so that bug class does not apply here.
"""

from __future__ import annotations

import argparse
import logging

from lfm25_ja.eval.generate_ifeval_ja import run_generation
from lfm25_ja.eval.score_ifeval_ja import print_summary, run_scoring

logger = logging.getLogger(__name__)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="ifeval_ja eval config YAML path")
    parser.add_argument("--models", nargs="+", default=None, help="Subset of model names to run")


def _add_generate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resume-from", default=None, help="Resume starting at this model name")
    parser.add_argument("--limit", type=int, default=None, help="Only generate first N prompts")
    parser.add_argument("--force", action="store_true", help="Regenerate even if output exists")
    parser.add_argument("--dry-run", action="store_true", help="Print the generation plan only")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ifeval_ja pipeline (Issue #104)")
    sub = parser.add_subparsers(dest="command", required=True)

    gen_parser = sub.add_parser("generate", help="Generate model responses (GPU)")
    _add_common_args(gen_parser)
    _add_generate_args(gen_parser)

    score_parser = sub.add_parser("score", help="Score generated responses (CPU)")
    _add_common_args(score_parser)

    all_parser = sub.add_parser("all", help="Generate then score")
    _add_common_args(all_parser)
    _add_generate_args(all_parser)

    return parser


def run_generate(args: argparse.Namespace) -> None:
    results = run_generation(
        config_path=args.config,
        models=args.models,
        resume_from=args.resume_from,
        limit=args.limit,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(f"status={results['status']}")
    for item in results["plan"]:
        print(f"  [{item['name']}] hf_path={item['hf_path']} num_prompts={item['num_prompts']}")
    for run in results.get("runs", []):
        print(f"  -> {run['model']}: {run['status']} ({run['count']} generations)")


def run_score(args: argparse.Namespace) -> None:
    results = run_scoring(config_path=args.config, models=args.models)
    print_summary(results)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        run_generate(args)
    elif args.command == "score":
        run_score(args)
    elif args.command == "all":
        run_generate(args)
        run_score(args)


if __name__ == "__main__":
    main()
