"""Scratch driver for K2-1 (Issue #130): build the cpt-D corpus without
materializing the full English Wikipedia (which twice crashed WSL2 via
prepare_data's uniform non-streaming load). wikipedia_en is streamed with a
sample cap instead; wikipedia_ja_japan_subset and aozora are loaded in full
(both already proven to work). Reuses the same library functions prepare.py
uses, just orchestrated by hand for this one production run.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, "/home/usr/lfm25-ja-k2/src")

from lfm25_ja.data.clean import _read_jsonl, _write_jsonl, clean_corpus, render_stats_report
from lfm25_ja.data.download import download_corpus, load_corpus_config
from lfm25_ja.data.mix import mix_corpora, render_mix_report
from lfm25_ja.data.wikipedia_ja_japan_subset import (
    _extract_text_title_rows,
    filter_matching_documents,
    oversample_documents,
)
from lfm25_ja.data.prepare import _extract_text_rows

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_cptD")

CONFIG_PATH = "configs/data/corpus_cptD.yaml"
OUT_DIR = Path("data/processed_cptD")
EVAL_TEXTS_PATH = "datasets/eval/jkb/eval_texts.jsonl"
EN_SAMPLE_LIMIT = 60000

config = load_corpus_config(CONFIG_PATH)
cache_dir = config["cache_dir"]
clean_cfg = config["clean"]
mix_cfg = config["mix"]
by_name = {e["name"]: e for e in config["corpora"]}

eval_texts = [d["text"] for d in _read_jsonl(EVAL_TEXTS_PATH) if "text" in d]

corpus_stats = {}
docs_by_lang: dict[str, list[dict]] = {}

# 1. wikipedia_ja_japan_subset -- full scan (already proven not to crash).
entry = by_name["wikipedia_ja_japan_subset"]
logger.info("=== wikipedia_ja_japan_subset: download ===")
ds = download_corpus(entry, cache_dir=cache_dir, streaming=False)
logger.info("=== wikipedia_ja_japan_subset: extract+filter ===")
docs = _extract_text_title_rows(ds, sample_limit=None)
docs = filter_matching_documents(docs)
logger.info("matched %d docs pre-clean", len(docs))
clean_docs, stats = clean_corpus(docs, clean_cfg, eval_texts=eval_texts)
clean_docs = oversample_documents(clean_docs, weight=entry["filter"].get("oversample_weight", 3))
corpus_stats["wikipedia_ja_japan_subset"] = {"downloaded_count": len(docs), "clean": stats, "final_after_oversample": len(clean_docs)}
for d in clean_docs:
    docs_by_lang.setdefault(d.get("language", "other"), []).append(d)
logger.info("wikipedia_ja_japan_subset done: %d -> %d clean -> %d after oversample", len(docs), stats["output_count"], len(clean_docs))

# 2. aozora -- full scan (small corpus, already cached).
entry = by_name["aozora"]
logger.info("=== aozora: download ===")
ds = download_corpus(entry, cache_dir=cache_dir, streaming=False)
logger.info("=== aozora: extract+clean ===")
docs = _extract_text_rows(ds, sample_limit=None)
clean_docs, stats = clean_corpus(docs, clean_cfg, eval_texts=eval_texts)
corpus_stats["aozora"] = {"downloaded_count": len(docs), "clean": stats}
for d in clean_docs:
    docs_by_lang.setdefault(d.get("language", "other"), []).append(d)
logger.info("aozora done: %d -> %d clean", len(docs), stats["output_count"])

# 3. wikipedia_en -- STREAMED with a sample cap to avoid materializing all 6.4M rows.
entry = by_name["wikipedia_en"]
logger.info("=== wikipedia_en: streaming download (sample_limit=%d) ===", EN_SAMPLE_LIMIT)
ds = download_corpus(entry, cache_dir=cache_dir, streaming=True)
docs = _extract_text_rows(ds, sample_limit=EN_SAMPLE_LIMIT)
clean_docs, stats = clean_corpus(docs, clean_cfg, eval_texts=eval_texts)
corpus_stats["wikipedia_en"] = {"downloaded_count": len(docs), "clean": stats}
for d in clean_docs:
    docs_by_lang.setdefault(d.get("language", "other"), []).append(d)
logger.info("wikipedia_en done: %d -> %d clean", len(docs), stats["output_count"])

# Mix.
logger.info("=== mixing ===")
mixed_docs, mix_stats = mix_corpora(docs_by_lang, ratios=mix_cfg["ratios"], seed=mix_cfg["seed"], unit=mix_cfg.get("unit", "documents"))

OUT_DIR.mkdir(parents=True, exist_ok=True)
_write_jsonl(OUT_DIR / "mixture.jsonl", mixed_docs)

report_lines = ["# K2 cpt-D data preparation report (Issue #130 / #123)", ""]
for name, s in corpus_stats.items():
    report_lines.append(f"## {name}")
    report_lines.append("")
    report_lines.append(render_stats_report(s["clean"]))
    report_lines.append("")
report_lines.append(render_mix_report(mix_stats))
(OUT_DIR / "prepare_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

logger.info("DONE: %d mixed docs -> %s", len(mixed_docs), OUT_DIR / "mixture.jsonl")
