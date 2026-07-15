# Phase 3 sft-002 データ多様化 + format 保持混合(学習 + IFEval 再評価)

Issue #105。Issue #104 の決定規則(全 SFT セル base +1pt 以下 → sft-002 必須化)を受け、
データ多様化 + format 制約混合の SFT データを構築・学習し、Issue #104 の IFEval ハーネスで
6 モデル再評価した。**結果は base -18pt で、事前固定の決定規則により「データ側の根本改善に
戻る」が確定**。失敗機序(off-policy スタイルシフト)と次段方針は後述。

## 設計(Fable5 コンサル、agentId `a2cdb09320bf6f720`)

### 採用: 能力ルーティング型ハイブリッド(Fable5 D 案の簡略版)

- **機械的制約(char_count / bullet_count / format_json / keyword)**: 上流 Q/A のうち既存回答が
  verifier を通過するものを選抜し、制約プロンプトを機械付与(選抜方式)
- **polite_form**: aya-ja / ichikara のみから選抜(llm-jp 系は敬体比率が低いため除外)
- **`format_markdown_table` / `numeric_only` は意図的除外**: 7 verifier 中 5 のみ訓練し、
  残り 2 をホールドアウト汎化プローブとする(train-on-test 回避)
- Fable5 の原推奨は「機械的制約 = base 自己蒸留(on-policy)」だったが、実装単純化のため
  選抜方式に簡略化した。**この簡略化が結果的に主要な敗因になった**(§失敗機序参照)

### データ混合(PR #107、`configs/data/mix_002.yaml`)

| 種別 | ソース | 件数 | 備考 |
|---|---|---:|---|
| 既存 | ichikara-instruction-003 | 2,903(全数) | baseline |
| 追加 | llm-jp/oasst1-21k-ja | 4,000 / 21,164 | ⚠️ Issue #105 記載の `llm-jp/llm-jp-instruct-v1` は **HF Hub に存在しない**(API 確認)。`llm-jp/llm-jp-instructions` は実在するが 1,000 rows のみで不足。apache-2.0 の oasst1-21k-ja に fallback |
| 追加 | CohereForAI/aya_dataset(ja) | 1,500 / 6,259 | apache-2.0、`language_code=="jpn"` |
| 追加 | format 制約付き | 1,500 | char 350 / bullet 300 / json 300 / polite 350 / keyword 200 |
| **合計** | | **9,903** | format 比率 15.1%(受け入れ条件 15〜25% 内) |

統計詳細: [phase3_sft002_mix_stats.md](phase3_sft002_mix_stats.md)。
**訓練データの平均応答長 284.1 文字 / 中央値 200**(この時点で base 出力平均 175.8 との
乖離に気づくべきだった — §教訓参照)。

### 学習(凍結条件、sft-004 A と bit-identical、データのみ差替)

| 項目 | 値 |
|---|---|
| モデル | LiquidAI/LFM2.5-1.2B-JP-202606 |
| 層 / lr / epoch | L9 単層 / 1e-5 / 1ep(sft-004 A 動作点) |
| batch / grad_accum | 1 / 4(base.yaml 継承。Issue #105 本文は grad_accum=1 と記載しているが、sft-004 も同じ base.yaml 継承で 4 のため**比較条件としては同一**) |
| steps / 所要 | 2,476 steps / 23.8 min(RTX 3060 Ti WSL2、`--no-checkpoints`) |
| VRAM | allocated 2.2 / reserved 3.7 GiB |
| train_loss | 平均 1.706(sft-004 A(ichikara 単独)より高いのはデータ多様化の自然な帰結) |

## 結果(prompt_strict acc、100 件、6 モデル)

生成は既存 5 モデル分を冪等スキップし sft002-mix のみ新規(ハーネス排他確認済み)。

| モデル | prompt_strict | Δ base | 依頼 | 敬語 | 要約 | 質問 |
|---|---:|---:|---:|---:|---:|---:|
| **base(JP-202606)** | **0.950** | 0 | 0.960 | 0.920 | 0.920 | 1.000 |
| sft004-L9-lr1e-5 | 0.890 | -0.060 | 0.880 | 0.840 | 0.840 | 1.000 |
| sft004-L9-lr3e-5 | 0.810 | -0.140 | 0.880 | 0.800 | 0.640 | 0.920 |
| sft004-L6-9-lr1e-5 | 0.770 | -0.180 | 0.800 | 0.800 | 0.680 | 0.800 |
| **sft002-mix(本実験)** | **0.770** | **-0.180** | 0.760 | 0.760 | 0.600 | 0.960 |
| sft003-L9(強学習参照) | 0.560 | -0.390 | 0.560 | 0.680 | 0.400 | 0.600 |

### verifier 別(sft002-mix、訓練対象 verifier との対応)

| verifier | 訓練件数 | sft002-mix | base | Δ |
|---|---:|---:|---:|---:|
| char_count(n=21) | 350 | 0.667 | 1.000 | **-33pt** |
| bullet_count(n=15) | 300 | 0.667 | 1.000 | **-33pt** |
| format_json(n=10) | 300 | 1.000 | 1.000 | 0 |
| polite_form(n=23) | 350 | 0.652 | 0.783 | -13pt |
| keyword(n=21) | 200 | 0.857 | 1.000 | -14pt |
| format_markdown_table(n=10、**除外・汎化プローブ**) | 0 | 1.000 | 1.000 | 0 |
| numeric_only(n=10、**除外・汎化プローブ**) | 0 | 1.000 | 1.000 | 0 |

**訓練した verifier ほど劣化する逆説的な結果**(構造制約 format_json とホールドアウト 2 種は
無傷、冗長性が直撃する char/bullet が最大劣化)。

### 定量スポット

- char_count 制約付き 21 プロンプト: base 平均 56.5 文字・超過 0 件 / sft002-mix 平均 65.1
  文字・**超過 6 件**
- 質的サンプル(ifja-001「100 文字以内で」): base ~40 文字通過 / sft002-mix ~150 文字超過
  (メリットを 11 項目列挙する冗長応答)
- 全 100 プロンプトの平均応答長は base 175.8 / sft002-mix 178.7 とほぼ同一 — 劣化は
  「全体が長くなった」のではなく**制約プロンプト下での遵守が壊れた**(テール violation)

## 決定規則の機械適用(事前固定、Issue #105 §評価)

- ✗ base +3pt 以上(Phase 4 DPO 進行): **不該当**(-18pt)
- ✗ base ±3pt(混合レシピ再調整 or sft-006): **不該当**
- ✓ **base -3pt 以下 → データ側の根本改善に戻る**: **該当・確定**

## 失敗機序(Fable5 解釈コンサル、agentId `a8e6268738816df07`)

**主因 = off-policy スタイルシフト**。証拠 3 点:

1. 劣化が冗長性の直撃する char_count / bullet_count に集中し、構造制約(format_json)と
   ホールドアウト 2 種は無傷
2. 質的サンプルで「緩い逸脱」でなく桁違いの超過(100 字制限に対し ~150 字)
3. 同一学習条件・データのみ差替の sft004-L9-lr1e-5 が -6pt に対し本件 -18pt — 追加 -12pt は
   データ起因と切り分け可能

制約プロンプトを既存応答に後付けした選抜合成は「制約文言 → 遵守」の因果を弱くしか教えず、
85% を占める off-policy な長文スタイル事前分布(oasst1 会話系、平均 284 文字)に負けた。
buffer 0〜20 の緩さは副次要因。

### 教訓(次段セッションへの引き継ぎ)

1. **学習前ガード指標の必須化**: 「訓練データ平均応答長 vs base 出力平均応答長」を混合構築の
   受け入れ条件に組み込む(今回 284.1 vs 175.8 の乖離を事前検出できたはず)
2. verifier 通過は on-distribution を意味しない(スタイル分布の一致とは別物)
3. format 15% は「加法的に効く」という楽観は誤り — スタイル事前分布は比率でなく総量で支配する
4. 自己蒸留の on-policy 性はレシピの核心であり、実装簡略化で落としてよい要素ではなかった

## 次段方針(Issue 起票済み、詳細はそちら)

Fable5 推奨は **base 自己蒸留の復活(on-policy rejection sampling)**: base 自身に制約プロンプト
K=4〜8 サンプル生成 → verifier strict + タイトマージン選抜(char_count は上限の 60〜90% のみ
採択、buffer 廃止)→ 3,000〜5,000 例、oasst1/aya は全除外。加えてユーザー提案の
**Qwen3-Swallow 8B(ローカル環境あり)による上位モデル蒸留**を候補化(polite_form の base
天井 0.783 を超えうる唯一の経路)。

## 明示的却下オプション(再燃防止、Fable5 追認済み)

1. **データ増加継続**(oasst1/aya を 20k 規模へ拡大) — スタイルシフトを増幅するだけ
2. **format 比率のみ引き上げ**(30〜50%、後付け合成のまま) — off-policy 性と緩 buffer が残る
   症状対処
3. **lr さらに下げる / epoch 削減** — no-op への漸近。最良でも base 0.950 に戻るだけで
   +3pt に到達不能
4. **多層 FT への再挑戦(L6-9 系)** — 既に -18pt で棄却済み

## 実装物一覧

- 実装 PR: #107(マージ済み) — loaders(`llm_jp_instruct.py` / `aya_ja.py`)、
  `format_constraints.py`、`mix_sft.py`、`prepare_sft_mix.py`、configs、scripts、tests 361 green
- データ: WSL `~/lfm25-ja/data/processed/sft/mix_002.jsonl`(9,903 rows、seed=42 で再現可能)
- モデル: WSL `~/lfm25-ja/outputs/sft-002-mix/`
- 評価: WSL `~/lfm25-ja/outputs/eval/ifeval_ja/sft002-mix/`(generations / scores / aggregate)
- 統計: [phase3_sft002_mix_stats.md](phase3_sft002_mix_stats.md)

## 受け入れ条件(Issue #105)チェック

- [x] `configs/sft/sft_002_mix.yaml` 完成(PR #107)
- [x] `data/processed/sft/mix_002.jsonl` 生成(9,903 rows)
- [x] format 制約付きサンプルが混合の 15〜25%(15.1%)
- [x] sft-002-mix モデルの学習完了(1 epoch、23.8 min)
- [x] Issue #104 IFEval ハーネスで 6 モデル再評価完了
- [x] 決定規則を機械適用し 3 択のいずれかで結論(**-3pt 以下 → データ側の根本改善**)
- [x] `experiments/reports/phase3_sft002_mix.md` 完成(本ドキュメント)
- [x] EXPERIMENT_LOG.md に 2 rows 追加

## Fable5 相談履歴(トレーサビリティ)

- 設計(format 制約構築方針、D 案 = 能力ルーティング型ハイブリッド): agentId `a2cdb09320bf6f720`
- 結果解釈(off-policy スタイルシフト機序 + 次段 = 自己蒸留復活): agentId `a8e6268738816df07`
- 実装委任(sonnet): agentId `abeb2060e653be84b`
