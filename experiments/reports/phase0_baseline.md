# Phase 0 ベースライン評価レポート (Issue #14 / #66)

実施日: 2026-07-09(初回) / **2026-07-13 に jsem・jmmlu(JP) の ool 修正再実行(Issue #66)**
実施環境: WSL2 Ubuntu 26.04 + RTX 3060 Ti (torch 2.13.0+cu130)

## 評価条件(凍結)

| 項目 | 値 |
|---|---|
| ハーネス | llm-jp-eval **v2.1.5**(commit 2cf4573)+ llm-jp-eval-inference (transformers モジュール) |
| データセット | jcommonsenseqa, jmmlu, jnli, jsem, jsick, jsquad, jsts, niilc(8 種、test split) |
| サンプル数 | **各 100 件**(max_num_samples=100) |
| few-shot | **4-shot**(llm-jp-eval デフォルト) |
| 生成 | greedy(do_sample=false)、bf16、batch 4、tokenizer max_length 4096 |
| プロンプト | llm-jp-eval 標準形式(チャットテンプレート**不使用** = base モデル形式) |
| jsem / jmmlu の回答抽出 | Issue #66 で修正(次節参照): output_length 拡大 + custom 正規表現(最終結論を優先抽出) |

再現手順: `make eval-baseline`(`lfm25_ja.eval.run_llm_jp_eval` が WSL `~/llm-jp-eval` +
`~/llm-jp-eval-inference` に対して dump → inference → evaluate を実行、
`configs/eval/llm_jp_eval.yaml` の `dataset_info_overrides` を
`eval_configs/lfm25_baseline.yaml` として書き出す)。
結果 JSON: WSL `~/llm-jp-eval/local_files/results/result_baseline-*.json`

## Issue #66: jsem・jmmlu(JP) の ool(out-of-label)修正

初回実施(2026-07-09)で jsem が両モデル ool 100%、jmmlu の JP-202606 が ool 55% と
判明(下記「旧スコア」参照)。実際の生成テキスト(records)を確認した根本原因:

1. **抽出正規表現の不一致**: jsem は llm-jp-eval 既定の `AnswerPatternId.ANSWER_TAGS_JP`
   (`<answer></answer>` タグ必須)だが、両モデルとも生ラベル(`yes`)や `\boxed{yes}` 形式で
   答えるだけでタグを使わない。jmmlu は `AnswerPatternId.CHOICE_ONLY_JP` で、データセット側の
   `output_length` がわずか jsem=15 / jmmlu=1 トークンに固定されており、`\boxed{...}` 形式で
   答える癖のある JP-202606 モデルは最初のトークン `\` だけで打ち切られる。
2. **出力長予算の不足**(1 回目修正後もなお jsem(JP) 37%・jmmlu(JP) 15% が残存): 両モデルとも
   ラベルを出す前に長い日本語の説明・途中式を書く傾向があり、拡大後の予算(jsem=48,
   jmmlu=16)でもラベルに到達できず打ち切られていた。
3. **途中で選択肢に言及するリスク**: 長い説明の途中で選択肢を検討・棄却する記述
   (例:「選択肢A: ...これは誤りである」)が先に出ると、先頭マッチの正規表現が誤って拾う。

修正内容(`configs/eval/llm_jp_eval.yaml` の `dataset_info_overrides`、
`src/lfm25_ja/eval/run_llm_jp_eval.py` の `DATASET_INFO_OVERRIDES`):
- `output_length` を jsem=160 / jmmlu=200 に拡大(config の `generation.max_new_tokens=256` の範囲内)
- `answer_pattern_id: custom` + 先頭に貪欲な `.*` を付けた正規表現
  (`(?s).*(yes|no|unknown|undef)` / `(?s).*\b([ABCD])\b`)で、
  本文中で**最後に出現するラベル**(=モデルの結論)を優先的に抽出

### ool 改善(100 件、再実行後)

| タスク | LFM2.5-1.2B-Instruct(旧→新) | LFM2.5-1.2B-JP-202606(旧→新) |
|---|---|---|
| jsem | 100% → **0%** | 100% → **1%** |
| jmmlu | 2% → **1%** | 55% → **5%** |

受け入れ条件(jsem・jmmlu(JP)の ool < 10%)を両モデルとも満たす。

## スコア表(exact_match、特記なき限り)— 2026-07-13 修正後

| タスク | LFM2.5-1.2B-Instruct | LFM2.5-1.2B-JP-202606 | 備考 |
|---|---|---|---|
| **AVG(カテゴリ平均)** | 0.355 | **0.469** | JP が上回る |
| jcommonsenseqa | 0.54 | **0.78** | |
| jmmlu | **0.39** | 0.34 | ool 1% / 5%(旧: 0.42 / 0.22、JP は ool 55%で過小評価) |
| jnli | 0.34 | **0.46** | |
| jsem | 0.50 | **0.60** | ool 0% / 1%(旧: 両者 0.00、ool 100%で測定不能) |
| jsick | 0.40 | **0.49** | |
| jsquad (EM / char_f1) | 0.37 / 0.67 | **0.53 / 0.758** | |
| jsts (pearson / spearman) | 0.302 / 0.334 | **0.667 / 0.659** | |
| niilc (EM / char_f1) | 0.06 / 0.187 | **0.18 / 0.332** | |

ool = out-of-label 率(出力がラベル形式に一致しなかった割合)。jsem・jmmlu 以外は Issue #66 の
修正対象外(既に ool 5% 未満だった)だが、dump のプロンプトが再生成されているため 100 件中
数件の greedy 出力差による ±数 pt の変動を含む(下記注意 3 参照)。

## 解釈と注意(重要)

1. **jsem・jmmlu(JP) の ool は Issue #66 で解消**(上記参照)。jsem は両モデルとも
   ラベルが正しく測定でき、JP-202606(0.60)が Instruct(0.50)を上回る ——
   日本語特化の効果が今回初めて観測できた。jmmlu は JP-202606 の ool が 55%→5% に
   下がり、実力に近い 0.34 が得られた(なお Instruct 0.39 にはまだ僅差で劣後。
   公表 JMMLU 54.19 との乖離は下記 3 の条件差によるもので残る)
2. **公表値との直接比較は不可**: 100 サンプル(±10pt 級のノイズ)、4-shot 標準プロンプト
   (チャットテンプレート不使用)、タスクサブセットのため。**本表の用途は「同一条件での
   相対比較の基準線」**であり、Phase 2 以降の自作モデルはこの同一パイプラインで比較する
3. **jsem/jmmlu 以外のタスクにも数 pt の変動がある**(例: niilc EM 0.13→0.18)。
   これは dump 時のプロンプト再生成(`dataset_info_overrides` 変更で prompts ディレクトリの
   ハッシュが変わる)と 100 件サンプルのノイズによるもので、Issue #66 の修正対象ではない
4. **JP-202606 が 6/8 タスクで Instruct を上回る**(特に jsts +0.36、jcommonsenseqa +0.24、
   今回追加で jsem +0.10)— 日本語特化の効果は本パイプラインでも観測でき、ハーネスとしての
   妥当性を確認

## Phase 0 ゲート判定

- 評価パイプラインが end-to-end で動作し、モデル間の既知の傾向(JP 特化版が日本語タスクで優位)を再現 → **ハーネス検証としては通過**
- jsem・jmmlu(JP)の形式不整合 → **Issue #66 で解消**(ool < 10% を両モデルで達成、上記参照)
