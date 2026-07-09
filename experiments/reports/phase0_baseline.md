# Phase 0 ベースライン評価レポート (Issue #14)

実施日: 2026-07-09 / 実施環境: WSL2 Ubuntu 26.04 + RTX 3060 Ti (torch 2.13.0+cu130)

## 評価条件(凍結)

| 項目 | 値 |
|---|---|
| ハーネス | llm-jp-eval **v2.1.5**(commit 2cf4573)+ llm-jp-eval-inference (transformers モジュール) |
| データセット | jcommonsenseqa, jmmlu, jnli, jsem, jsick, jsquad, jsts, niilc(8 種、test split) |
| サンプル数 | **各 100 件**(max_num_samples=100) |
| few-shot | **4-shot**(llm-jp-eval デフォルト) |
| 生成 | greedy(do_sample=false)、bf16、batch 4、tokenizer max_length 4096 |
| プロンプト | llm-jp-eval 標準形式(チャットテンプレート**不使用** = base モデル形式) |

再現手順: llm-jp-eval の `configs/lfm25_config.yaml` + `eval_configs/lfm25_baseline.yaml`(WSL `~/llm-jp-eval`)。
結果 JSON: WSL `~/llm-jp-eval/local_files/results/result_baseline-*.json`

## スコア表(exact_match、特記なき限り)

| タスク | LFM2.5-1.2B-Instruct | LFM2.5-1.2B-JP-202606 | 備考 |
|---|---|---|---|
| **AVG(カテゴリ平均)** | **0.334** | **0.387** | JP が上回る |
| jcommonsenseqa | 0.54 | **0.78** | |
| jmmlu | **0.42** | 0.22 | ⚠️ JP は ool 55%(下記) |
| jnli | 0.34 | **0.45** | |
| jsem | 0.00 | 0.00 | ⚠️ 両モデル ool 100%(下記) |
| jsick | 0.38 | **0.49** | |
| jsquad (EM / char_f1) | 0.41 / 0.672 | **0.49 / 0.747** | |
| jsts (pearson / spearman) | 0.285 / 0.288 | **0.678 / 0.659** | |
| niilc (EM / char_f1) | 0.06 / 0.207 | **0.13 / 0.283** | |

ool = out-of-label 率(出力がラベル形式に一致しなかった割合)。

## 解釈と注意(重要)

1. **jsem は両モデルで ool=100%** — 出力がラベル集合に全く一致せず 0 点。ハーネスのプロンプト/回答抽出形式とモデル出力形式の不整合であり、モデル能力の測定になっていない。**jsem は比較から除外して解釈すること**
2. **jmmlu の JP モデルは ool=55%** — 出力の過半がラベル形式に不一致(Instruct は ool 2%)。JP モデルの jmmlu 0.22 は形式問題で大きく押し下げられており、公表値(JMMLU 54.19)との乖離の主因。**このスコアで JP モデルの知識を判断しないこと**
3. **公表値との直接比較は不可**: 100 サンプル(±10pt 級のノイズ)、4-shot 標準プロンプト(チャットテンプレート不使用)、タスクサブセットのため。**本表の用途は「同一条件での相対比較の基準線」**であり、Phase 2 以降の自作モデルはこの同一パイプラインで比較する
4. 形式問題(ool)を除けば、**JP-202606 が 6/8 タスクで Instruct を上回る**(特に jsts +0.39、jcommonsenseqa +0.24)— 日本語特化の効果は本パイプラインでも観測でき、ハーネスとしての妥当性を確認

## Phase 0 ゲート判定

- 評価パイプラインが end-to-end で動作し、モデル間の既知の傾向(JP 特化版が日本語タスクで優位)を再現 → **ハーネス検証としては通過**
- ただし jsem・jmmlu(JP)の形式不整合は、プロンプト形式または回答抽出パターンの調整が必要 → フォローアップ Issue 参照
