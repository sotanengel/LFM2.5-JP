#!/usr/bin/env python3
"""Create all Epic and sub-issues for LFM2.5-JP project."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class IssueSpec:
    title: str
    body: str
    labels: list[str]
    milestone: str
    parent_epic_title: str | None = None


EPICS: list[IssueSpec] = [
    IssueSpec(
        title="Epic: Phase 0 環境検証とベースライン確立",
        body="""## 概要
RTX 3060 Ti (8GB) で LFM2.5-1.2B の層限定 QLoRA 学習が安定動作することを確認し、ベースライン評価スコアを確立する。

## ゲート条件
- [ ] 層限定 QLoRA 学習が安定動作
- [ ] ベースラインスコアが公表値と大きく乖離しない（評価ハーネス検証）

## 参照
- lfm2_5-ja-plan.md Phase 0
""",
        labels=["epic", "phase-0"],
        milestone="Phase 0: Environment",
    ),
    IssueSpec(
        title="Epic: Phase 1 データパイプライン検証",
        body="""## 概要
学習前にデータ品質を固める。ダウンロード・クリーニング・ChatML変換・コンタミチェック・350Mパイロット学習。

## ゲート条件
- [ ] テスト全通過
- [ ] パイロット perplexity 単調改善
- [ ] 生成テキストに文字化け・テンプレート崩れなし

## 参照
- lfm2_5-ja-plan.md Phase 1
""",
        labels=["epic", "phase-1"],
        milestone="Phase 1: Data Pipeline",
    ),
    IssueSpec(
        title="Epic: Phase 2 継続事前学習 (CPT)",
        body="""## 概要
日本語言語知識の底上げ。cpt-A/B/C の A/B 検証で CPT 採用可否を判断。

## ゲート条件
- [ ] cpt-B が cpt-A/C を JMMLU + 日本語 perplexity で上回る場合のみ CPT 採用

## 参照
- lfm2_5-ja-plan.md Phase 2
""",
        labels=["epic", "phase-2"],
        milestone="Phase 2: CPT",
    ),
    IssueSpec(
        title="Epic: Phase 3 SFT",
        body="""## 概要
日本語指示追従・対話品質の作り込み。sft-001〜005 の ablation。

## ゲート条件
- [ ] J-MIFEval / MT-Bench-ja が Phase 0 ベースライン超え
- [ ] 敬語・自然さの目視評価で明確な破綻なし

## 参照
- lfm2_5-ja-plan.md Phase 3
""",
        labels=["epic", "phase-3"],
        milestone="Phase 3: SFT",
    ),
    IssueSpec(
        title="Epic: Phase 4 DPO",
        body="""## 概要
日本語の自然さ・丁寧さ・安全性の微調整。選好最適化。

## ゲート条件
- [ ] MT-Bench-ja 向上かつ llm-jp-eval ±1pt 以内

## 参照
- lfm2_5-ja-plan.md Phase 4
""",
        labels=["epic", "phase-4"],
        milestone="Phase 4: DPO",
    ),
    IssueSpec(
        title="Epic: Phase 5 統合評価・量子化・リリース",
        body="""## 概要
LoRA マージ、GGUF 量子化、推論速度計測、モデルカード作成。

## 参照
- lfm2_5-ja-plan.md Phase 5
""",
        labels=["epic", "phase-5"],
        milestone="Phase 5: Release",
    ),
]


def _body(
    parent: str,
    summary: str,
    artifacts: list[str],
    ac: list[str],
    test_cmd: str,
    ref: str,
) -> str:
    artifacts_md = "\n".join(f"- `{a}`" for a in artifacts)
    ac_md = "\n".join(f"- [ ] {a}" for a in ac)
    return f"""## Parent
- {parent}

## 概要
{summary}

## 成果物
{artifacts_md}

## 受け入れ条件
{ac_md}

## テスト
```bash
{test_cmd}
```

## 参照
- {ref}
"""


SUB_ISSUES: list[IssueSpec] = [
    # Phase 0
    IssueSpec(
        title="[Phase0] リポジトリ基盤スキャフォールド",
        body=_body(
            "Epic: Phase 0 環境検証とベースライン確立",
            "pyproject.toml, README, Makefile, .gitignore, .env.example, src/lfm25_ja パッケージ骨格",
            ["pyproject.toml", "README.md", ".gitignore", ".env.example", "Makefile", "src/lfm25_ja/__init__.py"],
            ["make setup が成功", "pip install -e . が可能"],
            "make setup",
            "lfm2_5-ja-plan.md §2",
        ),
        labels=["phase-0", "infrastructure"],
        milestone="Phase 0: Environment",
        parent_epic_title="Epic: Phase 0 環境検証とベースライン確立",
    ),
    IssueSpec(
        title="[Phase0] CI・pre-commit・GA セキュリティ",
        body=_body(
            "Epic: Phase 0 環境検証とベースライン確立",
            "GitHub Actions CI, pre-commit, pinact/zizmor",
            [".github/workflows/ci.yml", ".pre-commit-config.yaml"],
            ["PR で lint+pytest (GPU なし) が緑", "pinact/zizmor が CI に含まれる"],
            "make lint && make test",
            "lfm2_5-ja-plan.md §2",
        ),
        labels=["phase-0", "infrastructure"],
        milestone="Phase 0: Environment",
        parent_epic_title="Epic: Phase 0 環境検証とベースライン確立",
    ),
    IssueSpec(
        title="[Phase0] 設定駆動基盤",
        body=_body(
            "Epic: Phase 0 環境検証とベースライン確立",
            "YAML config から全設定をロード",
            ["configs/base.yaml", "src/lfm25_ja/utils/config.py"],
            ["YAML 1枚から全設定ロード", "config hash 取得可能"],
            "pytest tests/test_config.py",
            "lfm2_5-ja-plan.md §2",
        ),
        labels=["phase-0", "infrastructure"],
        milestone="Phase 0: Environment",
        parent_epic_title="Epic: Phase 0 環境検証とベースライン確立",
    ),
    IssueSpec(
        title="[Phase0] 再現性・メモリユーティリティ",
        body=_body(
            "Epic: Phase 0 環境検証とベースライン確立",
            "seed 固定と VRAM ログ",
            ["src/lfm25_ja/utils/seed.py", "src/lfm25_ja/utils/memory.py"],
            ["seed 固定で同一出力", "VRAM ログ関数が動作"],
            "pytest tests/test_seed.py",
            "lfm2_5-ja-plan.md §2",
        ),
        labels=["phase-0", "infrastructure"],
        milestone="Phase 0: Environment",
        parent_epic_title="Epic: Phase 0 環境検証とベースライン確立",
    ),
    IssueSpec(
        title="[Phase0] スモークテスト（推論 + 層限定 QLoRA 20 step）",
        body=_body(
            "Epic: Phase 0 環境検証とベースライン確立",
            "4bit ロード推論と 20 step 学習で loss 低下確認",
            ["scripts/00_smoke_test.sh", "src/lfm25_ja/train/callbacks.py", "src/lfm25_ja/train/layer_select.py", "tests/test_train_smoke.py"],
            ["4bit ロード推論 OK", "20 step で loss 低下", "VRAM ピーク記録"],
            "make smoke-test / pytest tests/test_train_smoke.py",
            "lfm2_5-ja-plan.md Phase 0",
        ),
        labels=["phase-0", "train"],
        milestone="Phase 0: Environment",
        parent_epic_title="Epic: Phase 0 環境検証とベースライン確立",
    ),
    IssueSpec(
        title="[Phase0] OOM プロービング",
        body=_body(
            "Epic: Phase 0 環境検証とベースライン確立",
            "seq_len x batch x LoRA rank の格子探索",
            ["src/lfm25_ja/utils/memory.py", "experiments/reports/phase0_memory.md"],
            ["seq{1024,2048,4096} x batch{1,2,4} x r{16,64} の最大構成を文書化"],
            "make probe-memory",
            "lfm2_5-ja-plan.md Phase 0",
        ),
        labels=["phase-0", "infrastructure"],
        milestone="Phase 0: Environment",
        parent_epic_title="Epic: Phase 0 環境検証とベースライン確立",
    ),
    IssueSpec(
        title="[Phase0] ベースライン評価",
        body=_body(
            "Epic: Phase 0 環境検証とベースライン確立",
            "Instruct / JP-202606 の llm-jp-eval ベースライン",
            ["src/lfm25_ja/eval/run_llm_jp_eval.py", "configs/eval/llm_jp_eval.yaml", "scripts/50_eval_all.sh"],
            ["スコア表を EXPERIMENT_LOG.md に記録"],
            "make eval-baseline",
            "lfm2_5-ja-plan.md Phase 0",
        ),
        labels=["phase-0", "eval"],
        milestone="Phase 0: Environment",
        parent_epic_title="Epic: Phase 0 環境検証とベースライン確立",
    ),
    IssueSpec(
        title="[Phase0] 実験台帳初期化",
        body=_body(
            "Epic: Phase 0 環境検証とベースライン確立",
            "EXPERIMENT_LOG.md テンプレート",
            ["experiments/EXPERIMENT_LOG.md"],
            ["run 名規則あり", "必須記録項目テンプレートあり"],
            "N/A (docs review)",
            "lfm2_5-ja-plan.md §5",
        ),
        labels=["phase-0", "docs"],
        milestone="Phase 0: Environment",
        parent_epic_title="Epic: Phase 0 環境検証とベースライン確立",
    ),
]

# Phase 1 sub-issues
SUB_ISSUES.extend(
    [
        IssueSpec(
            title="[Phase1] HF データセット取得",
            body=_body(
                "Epic: Phase 1 データパイプライン検証",
                "HF datasets からコーパス取得",
                ["src/lfm25_ja/data/download.py"],
                ["llm-jp-corpus 等をダウンロード可能", "キャッシュパスを config で指定"],
                "pytest tests/test_data_pipeline.py -k download",
                "lfm2_5-ja-plan.md §3",
            ),
            labels=["phase-1", "data"],
            milestone="Phase 1: Data Pipeline",
            parent_epic_title="Epic: Phase 1 データパイプライン検証",
        ),
        IssueSpec(
            title="[Phase1] クリーニングパイプライン",
            body=_body(
                "Epic: Phase 1 データパイプライン検証",
                "NFKC, 言語判定, MinHash, 制御文字除去, 長さフィルタ",
                ["src/lfm25_ja/data/clean.py"],
                ["全必須フィルタ実装", "統計レポート出力"],
                "pytest tests/test_data_pipeline.py -k clean",
                "lfm2_5-ja-plan.md §3",
            ),
            labels=["phase-1", "data"],
            milestone="Phase 1: Data Pipeline",
            parent_epic_title="Epic: Phase 1 データパイプライン検証",
        ),
        IssueSpec(
            title="[Phase1] データ混合",
            body=_body(
                "Epic: Phase 1 データパイプライン検証",
                "JA:EN 比率制御",
                ["src/lfm25_ja/data/mix.py"],
                ["config で混合比率指定可能", "再現可能な shuffle"],
                "pytest tests/test_data_pipeline.py -k mix",
                "lfm2_5-ja-plan.md §3",
            ),
            labels=["phase-1", "data"],
            milestone="Phase 1: Data Pipeline",
            parent_epic_title="Epic: Phase 1 データパイプライン検証",
        ),
        IssueSpec(
            title="[Phase1] ChatML 変換",
            body=_body(
                "Epic: Phase 1 データパイプライン検証",
                "ChatML 形式変換と loss マスク検証",
                ["src/lfm25_ja/data/format_chat.py"],
                ["特殊トークン位置正しい", "decode で目視確認可能"],
                "pytest tests/test_data_pipeline.py -k chatml",
                "lfm2_5-ja-plan.md §3",
            ),
            labels=["phase-1", "data"],
            milestone="Phase 1: Data Pipeline",
            parent_epic_title="Epic: Phase 1 データパイプライン検証",
        ),
        IssueSpec(
            title="[Phase1] データ準備スクリプト",
            body=_body(
                "Epic: Phase 1 データパイプライン検証",
                "make data / 10_prepare_data.sh",
                ["scripts/10_prepare_data.sh", "Makefile (data target)"],
                ["make data で end-to-end 前処理完了"],
                "make data",
                "lfm2_5-ja-plan.md §2",
            ),
            labels=["phase-1", "data"],
            milestone="Phase 1: Data Pipeline",
            parent_epic_title="Epic: Phase 1 データパイプライン検証",
        ),
        IssueSpec(
            title="[Phase1] データパイプラインテスト",
            body=_body(
                "Epic: Phase 1 データパイプライン検証",
                "フォーマット崩れ・特殊トークン混入検出",
                ["tests/test_data_pipeline.py"],
                ["全データパイプラインの回帰テスト"],
                "pytest tests/test_data_pipeline.py",
                "lfm2_5-ja-plan.md §2",
            ),
            labels=["phase-1", "data"],
            milestone="Phase 1: Data Pipeline",
            parent_epic_title="Epic: Phase 1 データパイプライン検証",
        ),
        IssueSpec(
            title="[Phase1] 評価セットコンタミチェック",
            body=_body(
                "Epic: Phase 1 データパイプライン検証",
                "n-gram 重複検査",
                ["src/lfm25_ja/data/clean.py (contamination check)"],
                ["評価セットとの重複レポート出力", "ゲート条件として必須"],
                "pytest tests/test_data_pipeline.py -k contam",
                "lfm2_5-ja-plan.md §3",
            ),
            labels=["phase-1", "data"],
            milestone="Phase 1: Data Pipeline",
            parent_epic_title="Epic: Phase 1 データパイプライン検証",
        ),
        IssueSpec(
            title="[Phase1] 350M パイロット学習",
            body=_body(
                "Epic: Phase 1 データパイプライン検証",
                "350M + 1% データ 1 epoch パイロット",
                ["configs/cpt/cpt_350m_pilot.yaml", "src/lfm25_ja/train/train_cpt.py"],
                ["loss カーブ健全", "生成サンプル正常"],
                "make train-cpt CONFIG=configs/cpt/cpt_350m_pilot.yaml",
                "lfm2_5-ja-plan.md Phase 1",
            ),
            labels=["phase-1", "train"],
            milestone="Phase 1: Data Pipeline",
            parent_epic_title="Epic: Phase 1 データパイプライン検証",
        ),
    ]
)

# Phase 2
SUB_ISSUES.extend(
    [
        IssueSpec(
            title="[Phase2] CPT 学習スクリプト",
            body=_body("Epic: Phase 2 継続事前学習 (CPT)", "packed causal LM + QLoRA", ["src/lfm25_ja/train/train_cpt.py"], ["CPT 学習ループ実装", "config 駆動"], "pytest tests/test_train_smoke.py", "lfm2_5-ja-plan.md Phase 2"),
            labels=["phase-2", "train"], milestone="Phase 2: CPT", parent_epic_title="Epic: Phase 2 継続事前学習 (CPT)",
        ),
        IssueSpec(
            title="[Phase2] CPT 設定ファイル",
            body=_body("Epic: Phase 2 継続事前学習 (CPT)", "CPT YAML configs", ["configs/cpt/cpt_1.2b_qlora.yaml", "configs/cpt/cpt_350m_pilot.yaml"], ["1.2B QLoRA config 完備"], "N/A", "lfm2_5-ja-plan.md §2"),
            labels=["phase-2", "infrastructure"], milestone="Phase 2: CPT", parent_epic_title="Epic: Phase 2 継続事前学習 (CPT)",
        ),
        IssueSpec(
            title="[Phase2] 実験 cpt-A（CPT なし対照）",
            body=_body("Epic: Phase 2 継続事前学習 (CPT)", "CPT なし → 直接 SFT 対照群", ["configs/cpt/cpt_a_skip.yaml"], ["EXPERIMENT_LOG エントリ手順文書化"], "N/A", "lfm2_5-ja-plan.md Phase 2"),
            labels=["phase-2", "train"], milestone="Phase 2: CPT", parent_epic_title="Epic: Phase 2 継続事前学習 (CPT)",
        ),
        IssueSpec(
            title="[Phase2] 実験 cpt-B（Base + 日本語 CPT）",
            body=_body("Epic: Phase 2 継続事前学習 (CPT)", "Base + 日本語 CPT ~1B tokens", ["configs/cpt/cpt_b_base_ja.yaml"], ["200M トークン中間評価チェックポイント"], "make train-cpt", "lfm2_5-ja-plan.md Phase 2"),
            labels=["phase-2", "train"], milestone="Phase 2: CPT", parent_epic_title="Epic: Phase 2 継続事前学習 (CPT)",
        ),
        IssueSpec(
            title="[Phase2] 実験 cpt-C（公式 JP ベース SFT のみ）",
            body=_body("Epic: Phase 2 継続事前学習 (CPT)", "公式 1.2B-JP ベース", ["configs/cpt/cpt_c_jp_base.yaml"], ["ベースモデル切替 config"], "N/A", "lfm2_5-ja-plan.md Phase 2"),
            labels=["phase-2", "train"], milestone="Phase 2: CPT", parent_epic_title="Epic: Phase 2 継続事前学習 (CPT)",
        ),
        IssueSpec(
            title="[Phase2] 忘却監視 quick_eval",
            body=_body("Epic: Phase 2 継続事前学習 (CPT)", "英語 MMLU サブセット監視", ["src/lfm25_ja/eval/quick_eval.py"], ["英語ベンチ常時監視"], "python -m lfm25_ja.eval.quick_eval", "lfm2_5-ja-plan.md Phase 2"),
            labels=["phase-2", "eval"], milestone="Phase 2: CPT", parent_epic_title="Epic: Phase 2 継続事前学習 (CPT)",
        ),
        IssueSpec(
            title="[Phase2] CPT 実行スクリプト",
            body=_body("Epic: Phase 2 継続事前学習 (CPT)", "make train-cpt", ["scripts/20_train_cpt.sh"], ["make train-cpt 動作"], "make train-cpt", "lfm2_5-ja-plan.md §2"),
            labels=["phase-2", "train"], milestone="Phase 2: CPT", parent_epic_title="Epic: Phase 2 継続事前学習 (CPT)",
        ),
    ]
)

# Phase 3
SUB_ISSUES.extend(
    [
        IssueSpec(title="[Phase3] SFT 学習スクリプト", body=_body("Epic: Phase 3 SFT", "TRL SFTTrainer ラッパ", ["src/lfm25_ja/train/train_sft.py"], ["SFT 学習実装"], "make train-sft", "lfm2_5-ja-plan.md Phase 3"), labels=["phase-3", "train"], milestone="Phase 3: SFT", parent_epic_title="Epic: Phase 3 SFT"),
        IssueSpec(title="[Phase3] SFT 設定群", body=_body("Epic: Phase 3 SFT", "SFT YAML configs", ["configs/sft/sft_1.2b_qlora_r16.yaml", "configs/sft/sft_1.2b_qlora_r64.yaml"], ["r16/r64 config 完備"], "N/A", "lfm2_5-ja-plan.md §2"), labels=["phase-3", "infrastructure"], milestone="Phase 3: SFT", parent_epic_title="Epic: Phase 3 SFT"),
        IssueSpec(title="[Phase3] sft-001 ichikara のみ", body=_body("Epic: Phase 3 SFT", "最小基準実験", ["configs/sft/sft_001_ichikara.yaml"], ["ichikara のみ / r=16 / 2 epoch"], "make train-sft", "lfm2_5-ja-plan.md Phase 3"), labels=["phase-3", "train"], milestone="Phase 3: SFT", parent_epic_title="Epic: Phase 3 SFT"),
        IssueSpec(title="[Phase3] sft-002 データ拡大", body=_body("Epic: Phase 3 SFT", "データ配合拡大", ["configs/sft/sft_002_mix.yaml"], ["+llm-jp instruct, +Aya-ja"], "make train-sft", "lfm2_5-ja-plan.md Phase 3"), labels=["phase-3", "train"], milestone="Phase 3: SFT", parent_epic_title="Epic: Phase 3 SFT"),
        IssueSpec(title="[Phase3] sft-003 LoRA rank / target modules 比較", body=_body("Epic: Phase 3 SFT", "r16 vs r64, 全線形 vs attention", ["configs/sft/sft_003_ablation.yaml"], ["ablation config 群"], "make train-sft", "lfm2_5-ja-plan.md Phase 3"), labels=["phase-3", "train"], milestone="Phase 3: SFT", parent_epic_title="Epic: Phase 3 SFT"),
        IssueSpec(title="[Phase3] sft-004 lr / epoch スイープ", body=_body("Epic: Phase 3 SFT", "lr/epoch スイープ", ["configs/sft/sft_004_sweep.yaml"], ["eval loss 反転監視"], "make train-sft", "lfm2_5-ja-plan.md Phase 3"), labels=["phase-3", "train"], milestone="Phase 3: SFT", parent_epic_title="Epic: Phase 3 SFT"),
        IssueSpec(title="[Phase3] sft-005 合成データ追加", body=_body("Epic: Phase 3 SFT", "大モデル生成パイプライン", ["scripts/generate_synthetic.py"], ["合成データ生成・混入"], "N/A", "lfm2_5-ja-plan.md Phase 3"), labels=["phase-3", "data"], milestone="Phase 3: SFT", parent_epic_title="Epic: Phase 3 SFT"),
        IssueSpec(title="[Phase3] SFT 実行スクリプト", body=_body("Epic: Phase 3 SFT", "make train-sft", ["scripts/30_train_sft.sh"], ["make train-sft 動作"], "make train-sft", "lfm2_5-ja-plan.md §2"), labels=["phase-3", "train"], milestone="Phase 3: SFT", parent_epic_title="Epic: Phase 3 SFT"),
    ]
)

# Phase 4
SUB_ISSUES.extend(
    [
        IssueSpec(title="[Phase4] DPO 学習スクリプト", body=_body("Epic: Phase 4 DPO", "TRL DPOTrainer, seq_len=1024", ["src/lfm25_ja/train/train_dpo.py"], ["DPO 学習実装"], "make train-dpo", "lfm2_5-ja-plan.md Phase 4"), labels=["phase-4", "train"], milestone="Phase 4: DPO", parent_epic_title="Epic: Phase 4 DPO"),
        IssueSpec(title="[Phase4] DPO 設定", body=_body("Epic: Phase 4 DPO", "beta スイープ config", ["configs/dpo/dpo_1.2b_qlora.yaml"], ["beta {0.05,0.1,0.3} config"], "N/A", "lfm2_5-ja-plan.md §2"), labels=["phase-4", "infrastructure"], milestone="Phase 4: DPO", parent_epic_title="Epic: Phase 4 DPO"),
        IssueSpec(title="[Phase4] 選好データ構築", body=_body("Epic: Phase 4 DPO", "LLM-as-judge", ["src/lfm25_ja/eval/judge.py"], ["選好ペア構築パイプライン"], "pytest tests/test_judge.py", "lfm2_5-ja-plan.md Phase 4"), labels=["phase-4", "eval"], milestone="Phase 4: DPO", parent_epic_title="Epic: Phase 4 DPO"),
        IssueSpec(title="[Phase4] DPO 実行スクリプト", body=_body("Epic: Phase 4 DPO", "make train-dpo", ["scripts/40_train_dpo.sh"], ["make train-dpo 動作"], "make train-dpo", "lfm2_5-ja-plan.md §2"), labels=["phase-4", "train"], milestone="Phase 4: DPO", parent_epic_title="Epic: Phase 4 DPO"),
    ]
)

# Phase 5
SUB_ISSUES.extend(
    [
        IssueSpec(title="[Phase5] LoRA マージ", body=_body("Epic: Phase 5 統合評価・量子化・リリース", "LoRA マージ", ["src/lfm25_ja/merge_export/merge_lora.py"], ["マージ後モデル保存"], "python -m lfm25_ja.merge_export.merge_lora", "lfm2_5-ja-plan.md Phase 5"), labels=["phase-5", "export"], milestone="Phase 5: Release", parent_epic_title="Epic: Phase 5 統合評価・量子化・リリース"),
        IssueSpec(title="[Phase5] GGUF 変換・量子化", body=_body("Epic: Phase 5 統合評価・量子化・リリース", "Q8_0 / Q4_K_M", ["src/lfm25_ja/merge_export/export_gguf.py"], ["GGUF 変換成功"], "python -m lfm25_ja.merge_export.export_gguf", "lfm2_5-ja-plan.md Phase 5"), labels=["phase-5", "export"], milestone="Phase 5: Release", parent_epic_title="Epic: Phase 5 統合評価・量子化・リリース"),
        IssueSpec(title="[Phase5] 量子化後精度劣化測定", body=_body("Epic: Phase 5 統合評価・量子化・リリース", "perplexity + 主要タスク", ["experiments/reports/phase5_quant.md"], ["劣化レポート"], "make eval-baseline", "lfm2_5-ja-plan.md Phase 5"), labels=["phase-5", "eval"], milestone="Phase 5: Release", parent_epic_title="Epic: Phase 5 統合評価・量子化・リリース"),
        IssueSpec(title="[Phase5] 推論速度ベンチ", body=_body("Epic: Phase 5 統合評価・量子化・リリース", "3060 Ti / CPU tok/s", ["scripts/bench_inference.sh"], ["速度計測レポート"], "./scripts/bench_inference.sh", "lfm2_5-ja-plan.md Phase 5"), labels=["phase-5", "eval"], milestone="Phase 5: Release", parent_epic_title="Epic: Phase 5 統合評価・量子化・リリース"),
        IssueSpec(title="[Phase5] モデルカード", body=_body("Epic: Phase 5 統合評価・量子化・リリース", "HF model card", ["MODEL_CARD.md"], ["ライセンス・データ出典・再現手順"], "N/A", "lfm2_5-ja-plan.md Phase 5"), labels=["phase-5", "docs"], milestone="Phase 5: Release", parent_epic_title="Epic: Phase 5 統合評価・量子化・リリース"),
        IssueSpec(title="[Phase5] Phase 5 最終レポート", body=_body("Epic: Phase 5 統合評価・量子化・リリース", "最終比較レポート", ["experiments/reports/phase5_final.md"], ["公式 JP 版との比較表"], "N/A", "lfm2_5-ja-plan.md Phase 5"), labels=["phase-5", "docs"], milestone="Phase 5: Release", parent_epic_title="Epic: Phase 5 統合評価・量子化・リリース"),
    ]
)


def gh_json(cmd: list[str]) -> Any:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout) if result.stdout.strip() else {}


def get_milestone_number(title: str) -> int | None:
    milestones = gh_json(["gh", "api", "repos/sotanengel/LFM2.5-JP/milestones", "--paginate"])
    for m in milestones:
        if m["title"] == title and m["state"] == "open":
            return m["number"]
    return None


def create_issue(spec: IssueSpec) -> int:
    cmd = [
        "gh",
        "issue",
        "create",
        "--title",
        spec.title,
        "--body",
        spec.body,
        "--label",
        ",".join(spec.labels),
    ]
    if spec.milestone:
        ms = get_milestone_number(spec.milestone)
        if ms:
            cmd.extend(["--milestone", spec.milestone])
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    url = result.stdout.strip()
    number = int(url.rsplit("/", 1)[-1])
    print(f"Created #{number}: {spec.title}")
    return number


def link_sub_issue(parent: int, child: int) -> None:
    subprocess.run(
        [
            "gh",
            "api",
            f"repos/sotanengel/LFM2.5-JP/issues/{parent}/sub_issues",
            "-f",
            f"sub_issue_id={child}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> int:
    existing = gh_json(["gh", "issue", "list", "--limit", "100", "--json", "title,number"])
    existing_titles = {i["title"] for i in existing}

    epic_numbers: dict[str, int] = {}

    for epic in EPICS:
        if epic.title in existing_titles:
            num = next(i["number"] for i in existing if i["title"] == epic.title)
            print(f"Skip existing epic #{num}: {epic.title}")
            epic_numbers[epic.title] = num
            continue
        num = create_issue(epic)
        epic_numbers[epic.title] = num

    for sub in SUB_ISSUES:
        if sub.title in existing_titles:
            print(f"Skip existing: {sub.title}")
            continue
        child = create_issue(sub)
        if sub.parent_epic_title and sub.parent_epic_title in epic_numbers:
            link_sub_issue(epic_numbers[sub.parent_epic_title], child)

    print(f"Done. Epics: {len(epic_numbers)}, Sub-issues defined: {len(SUB_ISSUES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
