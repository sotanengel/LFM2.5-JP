# Phase 3 SFT lr/epoch スイープ(sft-004)

Issue #36。本ドキュメントは学習・評価・生成テキスト目視の実測記録である。

## 背景

sft-003 層 ablation(#33 #35、PR #86〜#100)で次が記録されている
(`experiments/reports/phase3_sft_layer_ablation.md`):

1. **train loss と llm-jp-eval AVG の順位がほぼ完全に逆転**した。
2. **5 アーム全て**が pre-SFT base `LiquidAI/LFM2.5-1.2B-JP-202606`(AVG 0.469)を下回った。
3. 生成テキスト 288 件の目視検査(検証A)では、劣化は可変 params 数に比例した
   3 段階(様式ドリフト → エコー/反復ループ → 事実崩壊)であり、文法・流暢性は
   全アームで崩壊ゼロと記録された。
4. chat_template 不一致仮説は generated diff=0 で棄却済み(PR #88)。

検証A時点で最有力とされた仮説は、「学習強度(lr 1e-4 × 2 epoch × ichikara
2,903 例の単一様式データ)が可変 params 数に対して過剰」である。sft-004 は
データと層構成を固定し、**lr × epoch のみ**を振ってこの仮説を直接計測する
ために設定された(データ多様化 #34 sft-002 より先に 1 変数ずつ切り分ける方針)。

## 目的と成功条件

| 項目 | 内容 |
|---|---|
| 目的 | base AVG を維持できる学習強度の動作点の有無を計測する |
| 成功条件(受け入れ) | llm-jp-eval AVG ≥ 0.469(`LiquidAI/LFM2.5-1.2B-JP-202606`)を満たすセルの有無 |
| 判定規則(計画 Step 5) | 動作点あり → 該当 (layer, lr, epoch) を Phase 4 条件候補として記録 / 動作点なし(全セル < 0.469) → 計画どおり fable5 への次手相談材料とする |

## 実験設定

固定条件は sft-003 と同一: ichikara 2,903 例、bf16、paged_adamw_8bit、
max_seq_len 1024、cosine / warmup 0.1。差分は `learning_rate` と
`num_train_epochs(=1)` のみ。epoch 軸を揃えるため、1e-4 セルも sft-003 の
2 epoch 結果を流用せず epoch=1 で新規実行した。

| セル | 層 | lr | epoch | config | 出力ディレクトリ |
|---|---|---|---:|---|---|
| A | [6..9] | 1e-5 | 1 | `configs/sft/sft_004_L6-9_lr1e-5.yaml` | `outputs/sft-004/sft-004-L6-9-lr1e-5` |
| B | [6..9] | 3e-5 | 1 | `configs/sft/sft_004_L6-9_lr3e-5.yaml` | `outputs/sft-004/sft-004-L6-9-lr3e-5` |
| C | [6..9] | 1e-4 | 1 | `configs/sft/sft_004_L6-9_lr1e-4.yaml` | `outputs/sft-004/sft-004-L6-9-lr1e-4` |
| D | L9 | 1e-5 | 1 | `configs/sft/sft_004_L9_lr1e-5.yaml` | `outputs/sft-004/sft-004-L9-lr1e-5` |
| E | L9 | 3e-5 | 1 | `configs/sft/sft_004_L9_lr3e-5.yaml` | `outputs/sft-004/sft-004-L9-lr3e-5` |
| F | L9 | 1e-4 | 1 | `configs/sft/sft_004_L9_lr1e-4.yaml` | `outputs/sft-004/sft-004-L9-lr1e-4` |

参照点(評価のみ、学習なし):

| 名称 | 出典 |
|---|---|
| base-jp202606 | `LiquidAI/LFM2.5-1.2B-JP-202606`(本スイープと同条件で再評価) |
| sft-003 L9 / [6..9] | lr 1e-4 × 2 epoch(過剰強度参照点。新規学習なし、既報 AVG を併記) |

### 実行メタデータ

| 項目 | 値 |
|---|---|
| 学習実行日時 | 2026-07-15T07:12:57〜07:55:10+09:00 |
| 評価実行日時 | 2026-07-15T08:35:22〜08:54:23+09:00(`SFT004_EVAL_OK`、elapsed=1141s) |
| commit | `477813d`(configs PR #101) |
| 学習ログ | WSL `~/sft_004_all.log` |
| 評価ログ | WSL `~/sft_004_eval.log` |
| 評価 config | `configs/eval/llm_jp_eval_sft004.yaml`(凍結 8 タスク × 100 件 × 4-shot × greedy、`dataset_info_overrides`・`apply_chat_template=false` は sft-003 評価と同一方針) |
| 要約 JSON | `outputs/eval/baseline_summary.json` |
| ハード | RTX 3060 Ti 8GB(WSL2) |

## 結果

### 学習(train loss / VRAM / 所要)

Trainer の `train_loss`(epoch 全体平均)と、ランナーが吐く
`SFT run finished: loss <開始> -> <終端>` を併記。`--no-checkpoints` のため
`trainer_state.json` は残っていない。

| セル | 可変 params | trainable% | 開始→終端 loss | train_loss | mean_token_acc | reserved VRAM | elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|
| D L9 lr1e-5 | 67,119,104 | 5.735% | 1.9090 → 1.8279 | 1.878 | 0.5587 | 3.3 GiB | 393 s |
| E L9 lr3e-5 | 67,119,104 | 5.735% | 1.9076 → 1.7945 | 1.837 | 0.5631 | 3.3 GiB | 395 s |
| F L9 lr1e-4 | 67,119,104 | 5.735% | 1.8997 → 1.7763 | 1.828 | 0.5575 | 3.3 GiB | 401 s |
| A [6..9] lr1e-5 | 262,178,944 | 22.402% | 1.9080 → 1.8010 | 1.839 | 0.5647 | 3.6 GiB | 455 s |
| B [6..9] lr3e-5 | 262,178,944 | 22.402% | 1.9023 → 1.7772 | 1.815 | 0.5625 | 3.6 GiB | 443 s |
| C [6..9] lr1e-4 | 262,178,944 | 22.402% | 1.8605 → 1.8276 | 1.887 | 0.5463 | 3.6〜3.9 GiB | 446 s |

全 6 本 `===RUN_OK===`。学習 wall 時間合計は約 42 分
(07:12:57〜07:55:10)。allocated は全セルで 2.2 GiB とログ記録。

### llm-jp-eval AVG(判定指標)

閾値: **AVG ≥ 0.469**。本評価の base 再測は **0.4693**。

| 順位 | モデル | AVG | vs 0.469 | 判定 |
|---:|---|---:|---:|---|
| — | base-jp202606 | **0.4693** | +0.0003 | 基準線 |
| 1 | D sft004-L9-lr1e-5 | 0.4653 | −0.0037 | 未達 |
| 2 | A sft004-L6-9-lr1e-5 | 0.4433 | −0.0257 | 未達 |
| 3 | B sft004-L6-9-lr3e-5 | 0.4407 | −0.0283 | 未達 |
| 4 | E sft004-L9-lr3e-5 | 0.4253 | −0.0437 | 未達 |
| 5 | F sft004-L9-lr1e-4 | 0.4187 | −0.0503 | 未達 |
| 6 | C sft004-L6-9-lr1e-4 | 0.3640 | −0.1050 | 未達 |

**動作点: 該当なし**(全 6 セルが AVG < 0.469)。

### タスク別スコア

| モデル | AVG | jmmlu EM | jmmlu ool | jcqa EM | jnli EM | jsem EM | jsem ool | jsick EM | jsquad EM | jsquad F1 | jsts P | jsts S | niilc EM | niilc F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base-jp202606 | 0.4693 | 0.34 | 0.05 | 0.78 | 0.46 | 0.60 | 0.01 | 0.49 | 0.53 | 0.7579 | 0.6668 | 0.6591 | 0.18 | 0.3321 |
| L9-lr1e-5 | 0.4653 | 0.38 | 0.08 | 0.75 | 0.58 | 0.59 | 0.05 | 0.53 | 0.53 | 0.7561 | 0.6199 | 0.6076 | 0.10 | 0.2653 |
| L6-9-lr1e-5 | 0.4433 | 0.33 | 0.11 | 0.73 | 0.56 | 0.51 | 0.07 | 0.51 | 0.54 | 0.7593 | 0.5178 | 0.5929 | 0.09 | 0.2416 |
| L6-9-lr3e-5 | 0.4407 | 0.42 | 0.06 | 0.66 | 0.60 | 0.42 | 0.05 | 0.46 | 0.52 | 0.7279 | 0.4662 | 0.5354 | 0.11 | 0.2736 |
| L9-lr3e-5 | 0.4253 | 0.26 | 0.20 | 0.71 | 0.67 | 0.46 | 0.22 | 0.57 | 0.50 | 0.7343 | 0.6258 | 0.6479 | 0.09 | 0.2656 |
| L9-lr1e-4 | 0.4187 | 0.29 | 0.19 | 0.64 | 0.67 | 0.46 | 0.16 | 0.50 | 0.51 | 0.7237 | 0.5217 | 0.5236 | 0.11 | 0.2977 |
| L6-9-lr1e-4 | 0.3640 | 0.26 | 0.17 | 0.44 | 0.66 | 0.37 | 0.16 | 0.56 | 0.52 | 0.6737 | 0.4516 | 0.4527 | 0.07 | 0.2196 |

### sft-003 参照点との対比(既報)

sft-003 は lr=1e-4 × **2 epoch**(本スイープの F/C と同 lr・異 epoch)。

| 条件 | AVG |
|---|---:|
| base (sft-003 当時 / 本スイープ再測) | 0.469 / 0.4693 |
| sft-003 L9 (1e-4 × 2ep) | 0.396 |
| sft-004 L9 (1e-4 × 1ep) | 0.4187 |
| sft-004 L9 (1e-5 × 1ep) | 0.4653 |
| sft-003 [6..9] (1e-4 × 2ep) | 0.324 |
| sft-004 [6..9] (1e-4 × 1ep) | 0.3640 |
| sft-004 [6..9] (1e-5 × 1ep) | 0.4433 |

同一層で比較すると、epoch を 2→1 に減らした場合および lr を下げた場合で
AVG は sft-003 同条件より高い。いずれも 0.469 には達していない。

### AVG と学習強度の関係(実測順)

同一層内では lr が高いほど AVG が低い:

- L9: 1e-5 (0.4653) > 3e-5 (0.4253) > 1e-4 (0.4187)
- [6..9]: 1e-5 (0.4433) > 3e-5 (0.4407) > 1e-4 (0.3640)

同一 lr では L9 の AVG が [6..9] 以上(1e-5 / 1e-4)。3e-5 のみ [6..9]
(0.4407) > L9 (0.4253)。

train_loss 順位(低いほど学習データ適合が強い)と AVG 順位は一致しない。
例: L9-lr1e-4 の train_loss 1.828 は L9 系列で最低だが AVG は L9 系列最低、
L9-lr1e-5 の train_loss 1.878 は L9 系列で最高だが AVG は全セル最高。

## 生成テキスト目視(検証A 同枠)

### 手順

- 元データ: `~/llm-jp-eval-inference/inference-modules/transformers/outputs/baseline-{sft004*,basejp202606}/<task>.eval-generated.json`
- 抽出: 8 タスク × 7 アーム × 先頭 6 件 = 336 件(全文は
  `scratchpad/generated_samples_sft004.md`)
- 追加計測: 各アーム全 800 件(8×100)について、出力末尾の few-shot 漏れ
  (`</example` / `/examples>` / `### 入力` / `### 応答` / `</例`)件数を計数

### few-shot 漏れ件数(800 件あたり)

| アーム | few-shot 漏れ |
|---|---:|
| base | 24 |
| L9-lr1e-5 | 16 |
| L9-lr3e-5 | 10 |
| L9-lr1e-4 | 34 |
| L6-9-lr1e-5 | 30 |
| L6-9-lr3e-5 | 48 |
| L6-9-lr1e-4 | **159** |

### jcommonsenseqa: 同一サンプル横断(先頭 6 件)

質問例 sample[0] gold=`2`(マザーボード):

| アーム | generated |
|---|---|
| base | `2` |
| L9-lr1e-5 | `2` |
| L9-lr3e-5 | `2` |
| L9-lr1e-4 | `2` |
| L6-9-lr1e-5 | `2` |
| L6-9-lr3e-5 | `2` |
| L6-9-lr1e-4 | `1` |

sample[0..5] では、L6-9-lr1e-4 の sample[0]=`1` と sample[5] の相違を除き、
弱〜中強度セルは base と同じ 1 桁数字応答だった。sft-003 検証Aで記録された
`答` / 質問語エコー(`電子` 等)はこの先頭 6 件には現れなかった。

jcqa 100 件の先頭トークン分布:

| アーム | 0 | 1 | 2 | 3 | 4 | 非数字 |
|---|---:|---:|---:|---:|---:|---:|
| base | 18 | 16 | 28 | 16 | 22 | 0 |
| L9-lr1e-5 | 16 | 16 | 27 | 17 | 24 | 0 |
| L6-9-lr1e-4 | 9 | 29 | 24 | 26 | 12 | 0 |

### jsts: 抜粋

sample[0] gold=`0.0`:

| アーム | generated(短縮) |
|---|---|
| base | `<answer>0.0</answer>` |
| L9-lr1e-5 | `<answer>0.0</answer>` |
| L9-lr3e-5 | `<answer>1.6</answer>` |
| L9-lr1e-4 | `<answer>0.0</answer>` |
| L6-9-lr1e-5 | `<answer>0.0</answer>` |
| L6-9-lr3e-5 | `<answer>0.0</answer>` |
| L6-9-lr1e-4 | `<answer>2.0</answer>` |

中〜高強度では `</examples>` / `</example_N>` が答えの直後に続く例が複数あった
(例: L6-9-lr3e-5 sample[2] `<answer>4.0</answer>\n</examples>`、
L6-9-lr1e-4 sample[3] `<answer>3.0</answer>\n</example_4>`)。

### niilc: 抜粋

sample[0] gold=`東芝`(初のノート PC メーカー):

| アーム | generated(短縮) |
|---|---|
| base | `<answer>IBM</answer>` |
| L9-lr1e-5 | `<answer>IBM</answer>` |
| L9-lr3e-5 | `<answer>NEC</answer>` |
| L9-lr1e-4 | `<answer>富士通</answer>` |
| L6-9-lr1e-5 | `<answer>NEC</answer>` |
| L6-9-lr3e-5 | `<answer>NEC</answer>` |
| L6-9-lr1e-4 | `<answer>ファクシミリ社</answer>\n</example_1>` |

sample[3] gold=`1964年`(東京オリンピック開催年):

| アーム | generated(短縮) |
|---|---|
| base | `<answer>2021</answer>` |
| L9-lr1e-5 | `<answer>2020</answer>` の後に別質問の偽 few-shot が続く |
| L9-lr3e-5 | `<answer>2020</answer>\n<answer>2021</answer>` |
| L9-lr1e-4 | `<answer>2021年</answer>` |
| L6-9-lr1e-5 | `<answer>2020</answer>` の後に別質問の偽 few-shot が続く |
| L6-9-lr3e-5 | `<answer>2021</answer>` |
| L6-9-lr1e-4 | `<answer>1964</answer>` |

L6-9-lr1e-4 の niilc sample[6] 付近では、`</例_N>` のあとに
「ペルーカタツムリ」などの連続した架空 QA が生成された(全文は
`generated_samples_sft004.md` および元 JSON)。

### jsem / 反復: 抜粋

- L9-lr1e-4 jsem[42]: `yes` の後に前提・仮説の復唱が続き、末尾が
  `いいいいいい…` の長ランになった。
- L6-9-lr1e-4 jsem[47]: `前提→仮説→仮説→前提→…` の矢印反復が続いた。

### jmmlu: 様式

base 自身が `正解は A です\n\n説明：…` 形式を出す例がある。L9-lr1e-5 も同型の
前置き+説明を出す例があり、sft-003 検証Aで問題化した「短 output_length 下の
前置き打ち切り」単体とは、本設定(`jmmlu output_length=200`)上は切り分けが必要。

### 目視で観測された事実の要約(評価コメントなし)

1. 全セルで日本語の文法崩壊・文字化けは、検査した抜粋範囲では確認されなかった。
2. sft-003 検証Aの重いモード(jcqa の質問語エコー、年号の長列挙)は、本スイープの
   弱〜中強度セル先頭サンプルには再現しなかった。
3. 強度最大セル(L6-9 × 1e-4)では few-shot 漏れが 159/800 と突出し、架空固有名や
   ラベル矢印反復が記録された。
4. AVG 最高の L9 × 1e-5 は、jcqa 先頭トークン分布・多数の short-form 応答で
   base に最も近い外形を示した。

## 動作点判定

計画の成功条件「AVG ≥ 0.469」を 6 セルへ適用した結果:

- **動作点に該当するセル: なし**
- 閾値からの最小差: L9 × 1e-5(AVG 0.4653、差 −0.0037)
- 閾値からの最大差: [6..9] × 1e-4(AVG 0.3640、差 −0.1050)

計画 Step 5 の分岐表に照らすと、本結果は「動作点なし」側に入る。

## 記録上の次アクション参照

Issue #36 / 実施計画 Step 5 に書かれた分岐(本報告の解釈ではなく計画原文の参照):

- 動作点あり → 該当セルの (layer, lr, epoch) を Phase 4 DPO の学習条件として確定
- **動作点なし → fable5 エージェントに次手を相談**(入力材料: 6 セルのスコア表・
  生成サンプル・sft-003 検証A・仮説候補)

本レポートおよび `scratchpad/generated_samples_sft004.md`、
`outputs/eval/baseline_summary.json`、各 `result_baseline-sft004*_*.json` が
その入力材料に相当する。

## 参照

- Issue #36
- Configs PR #101(`477813d`)
- sft-003 層 ablation: `experiments/reports/phase3_sft_layer_ablation.md`
- 評価ハーネス修正: Issue #66 / PR #85、chat_template 検証 PR #88
- 評価設定: `configs/eval/llm_jp_eval_sft004.yaml`
- 実験台帳: `experiments/EXPERIMENT_LOG.md` rows 023〜029
