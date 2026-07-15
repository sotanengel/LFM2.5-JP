"""sft-002 mix end-to-end pipeline tests (Issue #105).

Every HF loader is mocked at the network boundary (``download_*_raw``,
patched on ``lfm25_ja.data.prepare_sft_mix`` where they're imported into
scope) -- this is a smoke test of the orchestration (loaders -> format
synthesis -> mix -> report), not of any individual loader's HF integration
(those live in their own test modules).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml
from lfm25_ja.data.prepare_sft_mix import prepare_sft_mix

from lfm25_ja.data.clean import _read_jsonl

_ICHIKARA_RAW = [
    {"text": f"ichikara質問{i}", "output": f"ichikara回答{i}です。"} for i in range(6)
]

_LLM_JP_RAW = [
    {
        "conversations": [
            {"from": "human", "value": f"llmjp質問{i}"},
            {"from": "gpt", "value": "- 項目1\n- 項目2\n- 項目3"},
        ]
    }
    for i in range(6)
] + [
    {
        "conversations": [
            {"from": "human", "value": f"llmjp質問b{i}"},
            {"from": "gpt", "value": "東京タワーは有名な観光地です。"},
        ]
    }
    for i in range(6)
]

_AYA_RAW = [
    {"inputs": f"aya質問{i}", "targets": f"aya回答{i}です。", "language_code": "jpn"}
    for i in range(6)
] + [
    {"inputs": f"skip{i}", "targets": f"skip{i}", "language_code": "eng"} for i in range(3)
]


def _write_config(tmp_path: Path) -> Path:
    config = {
        "mix": {
            "seed": 42,
            "output_path": str(tmp_path / "mix_002.jsonl"),
            "stats_report": str(tmp_path / "phase3_sft002_mix_stats.md"),
            "components": {
                "ichikara": {"source": "ichikara", "n_samples": None},
                "llm_jp_instruct": {"source": "llm_jp_instruct", "n_samples": 5},
                "aya_ja": {"source": "aya_ja", "n_samples": 5},
            },
            "format_constraints": {
                "seed": 42,
                "targets": {"bullet_count": 2, "keyword": 2},
                "polite_form_sources": ["ichikara", "aya_ja"],
            },
        }
    }
    config_path = tmp_path / "mix_002.yaml"
    config_path.write_text(yaml.dump(config, allow_unicode=True), encoding="utf-8")
    return config_path


@patch("lfm25_ja.data.prepare_sft_mix.download_aya_raw")
@patch("lfm25_ja.data.prepare_sft_mix.download_llm_jp_instruct_raw")
@patch("lfm25_ja.data.prepare_sft_mix.download_ichikara_raw")
def test_prepare_sft_mix_end_to_end_smoke(
    mock_ichikara, mock_llm_jp, mock_aya, tmp_path: Path
) -> None:
    mock_ichikara.return_value = _ICHIKARA_RAW
    mock_llm_jp.return_value = _LLM_JP_RAW
    mock_aya.return_value = _AYA_RAW

    config_path = _write_config(tmp_path)
    stats = prepare_sft_mix(config_path)

    # ichikara: all 6 taken (target=None). llm_jp/aya: capped at 5 each.
    assert stats["mix"]["components"]["ichikara"]["selected"] == 6
    assert stats["mix"]["components"]["llm_jp_instruct"]["selected"] == 5
    assert stats["mix"]["components"]["aya_ja"]["selected"] == 5

    # format constraints: bullet_count=2 (from llm_jp bullet rows), keyword=2
    # (from llm_jp keyword rows) -- both reachable from the fixtures above.
    assert stats["format_counts"].get("bullet_count") == 2
    assert stats["format_counts"].get("keyword") == 2
    assert stats["format_total"] == 4

    assert stats["total"] == 6 + 5 + 5 + 4

    rows = _read_jsonl(stats["output_path"])
    assert len(rows) == stats["total"]
    for row in rows:
        assert "messages" in row
        assert row["messages"][0]["role"] == "user"
        assert row["messages"][-1]["role"] == "assistant"

    report_path = Path(stats["report_path"])
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "sft-002 mix stats report" in report
    assert "bullet_count" in report
    assert "ichikara" in report


@patch("lfm25_ja.data.prepare_sft_mix.download_aya_raw")
@patch("lfm25_ja.data.prepare_sft_mix.download_llm_jp_instruct_raw")
@patch("lfm25_ja.data.prepare_sft_mix.download_ichikara_raw")
def test_prepare_sft_mix_origin_tags_are_preserved(
    mock_ichikara, mock_llm_jp, mock_aya, tmp_path: Path
) -> None:
    mock_ichikara.return_value = _ICHIKARA_RAW
    mock_llm_jp.return_value = _LLM_JP_RAW
    mock_aya.return_value = _AYA_RAW

    config_path = _write_config(tmp_path)
    stats = prepare_sft_mix(config_path)

    rows = _read_jsonl(stats["output_path"])
    origins = {row["origin"] for row in rows}
    # Format-constrained rows inherit their *source* row's origin tag (Issue
    # #105 provenance requirement), so "format" itself is a mix-component
    # name (see stats["mix"]["components"]) but never an origin value.
    assert origins <= {"ichikara", "llm_jp_instruct", "aya_ja"}
    assert "ichikara" in origins
