"""Quick held-out perplexity evaluation helpers (Issue #29 lead-in).

Two pieces:

* ``build_heldout``: stream ``wikipedia_ja`` past the documents already
  consumed for training (default: the first 100000, matching the CPT
  mixture) and save the next slice as a small JSONL held-out set.
* ``measure_ppl``: load a causal LM and compute a token-count-weighted
  perplexity over that held-out set.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Any

import torch

from lfm25_ja.data.clean import _read_jsonl, _write_jsonl, normalize_nfkc
from lfm25_ja.data.download import download_corpus, load_corpus_config

logger = logging.getLogger(__name__)

DEFAULT_SKIP = 100_000
DEFAULT_TAKE = 200


def build_heldout(
    config_path: str | Path,
    out_path: str | Path,
    skip: int = DEFAULT_SKIP,
    take: int = DEFAULT_TAKE,
) -> Path:
    """Stream ``wikipedia_ja`` from ``config_path``, skip the first ``skip``
    documents (already used for training), and write the next ``take`` as a
    ``{"text": ...}`` JSONL held-out set at ``out_path``.

    Text is NFKC-normalized (see :func:`lfm25_ja.data.clean.normalize_nfkc`).
    """
    config = load_corpus_config(config_path)
    corpora: list[dict[str, Any]] = config.get("corpora", [])
    entry = next((c for c in corpora if c.get("name") == "wikipedia_ja"), None)
    if entry is None:
        raise ValueError(f"corpus config {config_path!r} has no 'wikipedia_ja' entry")
    cache_dir = config.get("cache_dir", "data/raw")

    dataset = download_corpus(entry, cache_dir=cache_dir, streaming=True)
    rows = itertools.islice(iter(dataset), skip, skip + take)
    docs = [{"text": normalize_nfkc(row["text"])} for row in rows if row.get("text")]
    if not docs:
        raise ValueError(
            f"No documents produced from wikipedia_ja after skipping {skip} "
            f"(take={take}); check the corpus config and skip/take values."
        )

    _write_jsonl(out_path, docs)
    logger.info("Wrote %d held-out documents to %s", len(docs), out_path)
    return Path(out_path)


def measure_ppl(
    model_path: str | Path,
    heldout_path: str | Path,
    max_docs: int | None = None,
    max_length: int = 2048,
) -> dict[str, Any]:
    """Compute a token-count-weighted perplexity of ``model_path`` over the
    held-out JSONL at ``heldout_path``.

    Returns ``{"ppl": float, "n_docs": int, "n_tokens": int}``.
    """
    heldout_file = Path(heldout_path)
    if not heldout_file.is_file():
        raise FileNotFoundError(f"heldout file not found: {heldout_path}")
    docs = _read_jsonl(heldout_file)
    if not docs:
        raise ValueError(f"heldout file is empty: {heldout_path}")
    if max_docs is not None:
        docs = docs[:max_docs]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001 - re-raised with model context below
        raise RuntimeError(f"Failed to load tokenizer from {model_path!r}: {exc}") from exc
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with model context below
        raise RuntimeError(f"Failed to load model from {model_path!r}: {exc}") from exc
    model.eval()

    total_weighted_loss = 0.0
    total_tokens = 0
    n_docs = 0
    with torch.no_grad():
        for doc in docs:
            text = doc.get("text", "")
            if not text:
                continue
            encoded = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            )
            input_ids = encoded["input_ids"].to(model.device)
            n_tok = input_ids.shape[1]
            if n_tok < 2:
                # A causal LM needs >= 2 tokens to have any predicted position.
                continue
            out = model(input_ids=input_ids, labels=input_ids)
            loss = float(out.loss.detach().cpu())
            n_pred_tokens = n_tok - 1
            total_weighted_loss += loss * n_pred_tokens
            total_tokens += n_pred_tokens
            n_docs += 1

    if total_tokens == 0:
        raise ValueError(f"No scorable tokens found in heldout set: {heldout_path}")

    mean_loss = total_weighted_loss / total_tokens
    ppl = float(torch.exp(torch.tensor(mean_loss)))
    return {"ppl": ppl, "n_docs": n_docs, "n_tokens": total_tokens}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Quick held-out PPL eval (Issue #29 lead-in)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_p = subparsers.add_parser(
        "build-heldout", help="Build a held-out JSONL from wikipedia_ja"
    )
    build_p.add_argument("--config", required=True, help="Path to configs/data/corpus.yaml")
    build_p.add_argument("--out", required=True, help="Output JSONL path")
    build_p.add_argument(
        "--skip", type=int, default=DEFAULT_SKIP, help="Documents to skip (already trained on)"
    )
    build_p.add_argument("--take", type=int, default=DEFAULT_TAKE, help="Documents to keep")

    ppl_p = subparsers.add_parser("ppl", help="Measure perplexity of a model on a held-out set")
    ppl_p.add_argument("--model", required=True, help="HF model id or local path")
    ppl_p.add_argument("--heldout", required=True, help="Held-out JSONL path")
    ppl_p.add_argument("--max-docs", type=int, default=None)
    ppl_p.add_argument("--max-length", type=int, default=2048)
    ppl_p.add_argument("--json", default=None, help="Optional path to also write the JSON result")

    args = parser.parse_args()

    if args.command == "build-heldout":
        out = build_heldout(args.config, args.out, skip=args.skip, take=args.take)
        print(f"Held-out set written to {out}")
    elif args.command == "ppl":
        result = measure_ppl(
            args.model, args.heldout, max_docs=args.max_docs, max_length=args.max_length
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.json:
            json_path = Path(args.json)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
