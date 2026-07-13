# 検証 B: ゼロショット + チャット形式評価

Issue #89 / #90 / #91 / #92。

## 問題意識

凍結条件(8 タスク × 100 件 × **4-shot** × greedy)は in-context learning を測る。
SFT が鍛えるのは単発インストラクション追従であり、few-shot とは別スキル。
検証 A / スポット検証では 4-shot 上で `apply_chat_template` 有無の generated
diff=0 だったが、例示模倣の可能性がある。ゼロショット + ChatML なら別挙動が
観測されるはず、という仮説で再評価した。

## 仮説

- SFT アーム AVG が base を上回る → SFT は本来のタスクで機能している
- 同順で下回る → SFT 自体が下流性能を毀損している

## 評価条件

| 項目 | 値 |
|---|---|
| harness | llm-jp-eval v2.1.5 |
| shots | **0** (`num_few_shots: 0`) |
| chat | **`apply_chat_template: true`** |
| samples | 100 / タスク |
| decoding | greedy (`do_sample=false`) |
| config | [`configs/eval/llm_jp_eval_zeroshot_chat.yaml`](../../configs/eval/llm_jp_eval_zeroshot_chat.yaml) |
| harness dump config | ラッパ生成 `~/llm-jp-eval/configs/lfm25_config_generated.yaml`(Issue #90) |
| prompts | `prompts_SyMdutvgb8TWmqkyyBIlYw==`(ゼロショット確認済み: jnli に `<examples>` なし) |
| 実行 | 2026-07-14 06:54–07:19 JST、約 25 分(7 モデル) |

## スコア表(llm-jp-eval 公式集計)

| アーム | jmmlu | jcommonsenseqa | jnli | jsem | jsick | jsquad | jsts(pearson) | niilc | **AVG** | vs base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **base(JP-202606)** | 0.34 | 0.74 | 0.00 | 0.60 | 0.00 | 0.233 | 0.000 | 0.188 | **0.296** | — |
| L6 | 0.39 | 0.59 | 0.01 | 0.07 | 0.00 | 0.071 | 0.000 | 0.149 | **0.215** | **-0.081** |
| Instruct(参考) | 0.39 | 0.50 | 0.00 | 0.45 | 0.00 | 0.050 | 0.000 | 0.008 | **0.212** | -0.084 |
| [6, 9] | 0.32 | 0.50 | 0.03 | 0.31 | 0.04 | 0.048 | 0.000 | 0.025 | **0.193** | **-0.103** |
| L9(sft-001) | 0.24 | 0.49 | 0.01 | 0.39 | 0.03 | 0.052 | 0.000 | 0.067 | **0.183** | **-0.113** |
| [6..9] | 0.18 | 0.23 | 0.00 | 0.31 | 0.00 | 0.000 | 0.000 | 0.010 | **0.105** | **-0.191** |
| full | 0.14 | 0.12 | 0.00 | 0.11 | 0.00 | 0.000 | 0.000 | 0.000 | **0.059** | **-0.237** |

順位(AVG): **base > L6 > Instruct > [6,9] > L9 > [6..9] > full**

### ool(抜粋)

構造化ラベル系(jnli / jsick / jsts)は **全モデルで ool ≈ 90–100%**
(ゼロショットでは few-shot 例示がないため `<answer>` / ラベル形式への収束が
ほぼ全滅)。相対比較への交絡はあるが、**base も含め同条件**なのでアーム間
比較は成立する。

| アーム | jmmlu | jnli | jsick | jsquad | jsts | niilc |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.05 | 1.00 | 1.00 | 0.58 | 0.97 | 0.35 |
| L6 | 0.02 | 0.91 | 0.97 | 0.88 | 1.00 | 0.47 |
| L9 | 0.27 | 0.93 | 0.95 | 0.92 | 0.93 | 0.76 |
| [6,9] | 0.07 | 0.95 | 0.89 | 0.93 | 1.00 | 0.91 |
| [6..9] | 0.24 | 1.00 | 0.99 | 1.00 | 0.97 | 0.98 |
| full | 0.10 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Instruct | 0.01 | 1.00 | 1.00 | 0.93 | 1.00 | 0.89 |

## 判定

**仮説は棄却。** ゼロショット + ChatML でも **5 SFT アーム全てが base(0.296)を下回り**、
順位も 4-shot 評価(L6 > [6,9] > L9 > [6..9] > full)とほぼ同型。

→ 「測り方が few-shot だから SFT が不利」という説明では逆転しない。
**SFT 自体が本データ/本ハイパーパラでの下流構造化出力を毀損している**、
という解釈が残る。

補足:
- Instruct も base より低い(0.212)。ゼロショット llm-jp-eval は ChatML 既製品にも
  厳しいが、それでも SFT アームは Instruct と同帯〜下位。
- jmmlu / jcommonsenseqa では L6 が base に近い/部分勝ち(jmmlu 0.39 > base 0.34)が、
  AVG を押し上げるには足りない。

## 4-shot 結果との対比

| 条件 | base | 最良 SFT | full |
|---|---:|---:|---:|
| 4-shot / no chat(sft-003-llmjpeval-020) | 0.469 | L6 0.410 | 0.089 |
| **0-shot / chat(本検証)** | **0.296** | **L6 0.215** | **0.059** |

絶対値はゼロショットで全体に低下(例示依存タスクの壊滅が主因)だが、
相対順位と「全アームが base 劣後」の結論は不変。

## ハーネス配線の確認(Issue #90)

- プロジェクト YAML の `num_few_shots: 0` → `lfm25_config_generated.yaml` に反映
- 凍結 `~/llm-jp-eval/configs/lfm25_config.yaml`(`num_few_shots: 4`)は未変更
- `apply_chat_template: true` は推論 YAML に反映済み
- dump で新 prompts ハッシュディレクトリが生成された(4-shot と非衝突)

## 示唆

1. Phase 3 ゲートに 4-shot AVG だけを使う判断を、ゼロショットで覆す材料にはならない
2. 層選択は引き続き「train loss 単独では不可」。人手対話品質の確認が次の優先
3. sft-002(データ拡大)に進む場合も、下流ゲートとして本ゼロショット設定を
   並行モニタしてよい(配線は再利用可)

## 結果ファイル

| アーム | result JSON |
|---|---|
| L9 | `result_baseline-sft003l9_20260714_065743.json` |
| L6 | `result_baseline-sft003l6_20260714_070050.json` |
| [6,9] | `result_baseline-sft003l6l9_20260714_070433.json` |
| [6..9] | `result_baseline-sft003l69_20260714_070820.json` |
| full | `result_baseline-sft003full_20260714_071145.json` |
| base | `result_baseline-basejp202606_20260714_071524.json` |
| Instruct | `result_baseline-instruct_20260714_071914.json` |

いずれも `~/llm-jp-eval/local_files/results/`。
