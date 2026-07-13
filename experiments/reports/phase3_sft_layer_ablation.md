# Phase 3 SFT 層 ablation(sft-001 + sft-003)

Issue #33 / #35。Phase 2 層プロファイリング(deci スイープで L9 ppl 最良・
中央帯 L6-L9 が全て有効、両端は fit しない)の知見を Phase 3 SFT で検証し、
どの層構成が最も効率良く SFT 学習できるかを実測する。

## 実験設定

**共通条件**(sft-001 = 単層 L9 アームと同一に統一、公平な層 ablation):

| 項目 | 値 |
|---|---|
| ベースモデル | `LiquidAI/LFM2.5-1.2B-JP-202606`(Phase 2 ゲート決定) |
| データ | ichikara-instruction-003 batch-001(2,903 例) |
| エポック | 2 |
| 学習率 | 1.0e-4 (cosine, warmup 0.1) |
| optimizer | paged_adamw_8bit |
| 精度 | bf16 + gradient_checkpointing |
| batch | 1 × grad_accum 1(SFTConfig デフォルト) |
| max_seq_len | 1024 |
| step 数 | 1,452(全アーム同一)|

**アーム構成**(可変層のみ違い):

| アーム | 可変層 | 可変 params 数 | 可変率 | config |
|---|---|---|---|---|
| L9(=sft-001) | [9] | ~72M | 6-7% | `configs/sft/sft_1.2b_layerft_L9.yaml` |
| L6 | [6] | ~72M | 6-7% | `configs/sft/sft_1.2b_layerft_L6.yaml` |
| [6,9] | [6, 9] | ~144M | 12-13% | `configs/sft/sft_1.2b_layerft_L6L9.yaml` |
| [6..9](中央4層) | [6, 7, 8, 9] | ~288M | 24% | `configs/sft/sft_1.2b_layerft_L6-9.yaml` |
| full | 全 16 層 | 1,036M | 88.5% | `configs/sft/sft_1.2b_layerft_full.yaml` |

L9 アームは sft-001 の完走モデル(`outputs/sft-001-ichikara/`)を流用し、
再実行していない(#33 受け入れ条件)。sft-003 の他 4 アームは `feat/sft-003-arms`
(commit `20a5073`)で 2026-07-13 に順次実行。

## 結果

### 損失・精度(checkpoint-1452 = 2 epoch 終端)

| アーム | 開始 loss | 終端 loss | epoch2 平均 loss | 終端 mean_token_acc | 所要 |
|---|---:|---:|---:|---:|---:|
| L6 | 1.9038 | 1.3831 | 1.3491 | 0.6880 | 14 min |
| L9(sft-001) | 1.9063 | 1.2645 | 1.2761 | 0.7238 | ~24 min †|
| full(参照) | 1.8657 | 1.1473 | 1.1965 | 0.7240 | 33 min |
| [6, 9] | 1.9010 | 1.1034 | 1.1029 | 0.7471 | 18 min |
| **[6..9]** | **1.8918** | **0.9904** | **1.0084** | **0.7612** | 21 min |

† sft-001 は初回 checkpoint 破損によるクラッシュ + 再開を挟んでいるため所要時間は
実効的な学習時間より長い。純粋な学習速度は他アームと同等(1.9-2.0 it/s)。

**順位(epoch2 平均 loss、低いほど良い):**
1. **L6..9(中央 4 層)0.9904** ← 全体最良
2. [6, 9] 1.1034
3. full 1.1473
4. L9 1.2645
5. L6 1.3831

### VRAM ピーク(reserved、`--report-to []` + paged_adamw_8bit)

| アーム | allocated | reserved | 備考 |
|---|---:|---:|---|
| L6 / L9 | 2.20 GiB | 3.30 GiB | 単層 |
| [6, 9] | 2.20 GiB | 3.60 GiB | +0.30 GiB |
| [6..9] | 2.20 GiB | 3.90 GiB | +0.30 GiB |
| full | 2.20 GiB | 5.20 GiB | +1.30 GiB |

allocated(実使用量)は全アームで 2.20 GiB。paged_adamw_8bit が optimizer state
を CPU にページングし、gradient_checkpointing が activation を再計算するため、
可変層が全 16 層に増えても実物理 VRAM の増加は限定的(1.30 GiB)。RTX 3060 Ti
8 GB で全アームが余裕を持って収まった。

## 考察

### 1. 中央 4 層 [6..9] が全アーム中で最良

**フル FT にも epoch2 平均 loss で 0.20 差(1.20 vs 1.01)勝った**。可変 params 数は
フル FT の 27.8%(288M vs 1,036M)に過ぎない。Phase 2 CPT の層プロファイリングで
「中央帯 L6-L9 のみが ppl 改善に寄与、両端は悪化」だった知見が SFT でも一貫して
再現している。フル FT では両端層(特に埋め込み層に近い最終付近)の可変化が
SFT loss 収束を阻害している可能性が高い。

### 2. 層数の効果(単層 → 2 層 → 4 層)

L9(単層)1.26 → [6,9](2 層)1.10 → [6..9](4 層)0.99 と、中央帯内で層を増やすと
単調に改善する。L6+L9 の 2 層で単層 L9 より 12% 改善、さらに 4 層に増やすと
追加で 10% 改善。中央帯内の層は互いに補完的で、複数層の同時 FT が有効。

### 3. フル FT の伸び悩み

全 16 層可変(88.5%)にしても [6..9](24%)より悪い。単純な過学習ではなく、
「無関係あるいは悪影響の層まで動かしてしまう」ことによる loss surface の悪化が
仮説。Phase 2 プロファイリングで単層 L0(埋込直後)L11〜L14(最終層側)が全て
CPT ppl を悪化させたことと整合する。

### 4. 単層 L9 > 単層 L6

Phase 2 deci プロファイリングでは L9 ppl 8.876 vs L6 8.898 で L9 が単層最良
(プローブ指標では L6 が最良、順位入替)。SFT では両者の差がより明確に開き、
L9(1.26)が L6(1.38)を明確に上回った。中央帯内でも位置による寄与差がある。

### 5. 効率性

VRAM ピークとエポック時間で見た「学習効率」は [6..9] が明確に優位:
- フル FT の 27.8% の可変 params 数
- フル FT の 75% の実行時間(21 min vs 33 min)
- フル FT の 75% の reserved VRAM(3.90 GiB vs 5.20 GiB)
- **かつ loss は 20% 改善**

Phase 4 DPO 以降の学習でも「中央 4 層のみ可変」を採用するのが合理的。

## llm-jp-eval 評価

Issue #66 の修正(PR #85、マージ済み)後、5 アーム全モデルを凍結条件(ベース
`LiquidAI/LFM2.5-1.2B-JP-202606` の Phase 0 baseline と同一: 8 タスク × 100 件 ×
4-shot × greedy、`configs/eval/llm_jp_eval.yaml` の `dataset_info_overrides` を
そのまま流用)で評価した。2026-07-13 実行、実行時間は 5 アーム合計で約 17 分
(dump 共有 + 各アーム推論 3-4 分 + 評価数秒、事前見積もりの 30-60 分/モデルより
大幅に短かった)。

### スコア表(exact_match 系、AVG は llm-jp-eval 側集計)

| アーム | jmmlu | jcommonsenseqa | jnli | jsem | jsick | jsquad | jsts(pearson) | niilc | **AVG** | train loss 順位 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L6 | 0.39 | 0.67 | 0.57 | 0.02 | 0.52 | 0.53 | 0.589 | 0.09 | **0.410** | 5 位(最悪) |
| [6, 9] | 0.32 | 0.64 | 0.47 | 0.30 | 0.51 | 0.53 | 0.640 | 0.10 | **0.403** | 2 位 |
| L9(sft-001) | 0.24 | 0.69 | 0.48 | 0.30 | 0.57 | 0.51 | 0.664 | 0.09 | **0.396** | 4 位 |
| [6..9](中央4層) | 0.18 | 0.40 | 0.65 | 0.31 | 0.48 | 0.46 | 0.000 | 0.10 | **0.324** | 1 位(最良) |
| full | 0.14 | 0.16 | 0.07 | 0.16 | 0.14 | 0.01 | 0.000 | 0.01 | **0.089** | 3 位 |
| *(参考)JP-202606 base(SFT 前)* | 0.34 | 0.78 | 0.46 | 0.60 | 0.49 | 0.53 | 0.667 | 0.18 | **0.469** | - |
| *(参考)Instruct(既製品)* | - | - | - | - | - | - | - | - | **0.355** | - |

- ベース(SFT 前、AVG 0.469)との差分: L6 **-0.059**、[6,9] **-0.066**、L9 **-0.073**、
  [6..9] **-0.145**、full **-0.380**。**5 アーム全てが SFT 前ベースを下回った**。
- ool(out-of-label、括弧内は主な要因タスク)は full が突出: jnli 84%・jsquad 89%・
  niilc 85%・jsts 56%(ほぼ全崩壊)。[6..9] も jmmlu 24%・jsem 17%・jsquad 7%、
  jsts は pearson/spearman ともに 0.000(定数出力に収束したとみられる)。L6 は
  ool がほぼ 0 で出力自体は健全だが jsem exact_match が 0.02 まで崩れている。

### train loss 順位との一致/不一致

**明確な逆転**が見られた。train loss(epoch2 平均、低いほど良い)の順位は
[6..9] 0.99 > [6,9] 1.10 > full 1.15 > L9 1.26 > L6 1.38 だったが、llm-jp-eval
AVG の順位は **L6 > [6,9] > L9 > [6..9] > full** とほぼ完全に反転している。
特に train loss で最良だった [6..9] と full が、8 タスク横断の few-shot 評価では
下位 2 位に沈んだ。

### 考察: 評価条件と SFT 学習形式の不一致という重大な留保

上記の逆転は「中央層 FT が下流タスクを損なう」という結論を裏付けるものでは
**ない可能性が高い**。決定的な交絡要因を発見した:

- SFT 学習(`train_sft.py` / `format_chat.py`)はトークナイザの
  `apply_chat_template` を使い、`<|im_start|>system ... <|im_start|>user ...`
  という ChatML 形式(各アーム出力の `chat_template.jinja` で確認)でデータを
  構築している。
- 一方、今回の llm-jp-eval 実行は Phase 0 baseline との比較可能性を保つため
  `apply_chat_template: false`(`run_llm_jp_eval.build_inference_config` の
  固定値、`configs/eval/llm_jp_eval.yaml` 由来)のまま、生の few-shot 連結
  プロンプトを投げている。
- つまり**5 アームとも、学習時に期待させたプロンプト形式と異なる形式で評価
  されている**。可変 params 数が多い(=ベースモデルの分布からより遠くへ
  動いた)アームほど、この形式不一致の影響を強く受けたと考えられる:
  full(88.5% 可変)で ool が壊滅的(jnli 84%/jsquad 89%/niilc 85%)、
  [6..9](24% 可変)も部分的に崩壊、L6/L9(単層、6-7% 可変)は相対的に
  ベースモデルの生成挙動を保持できたため影響が小さかった、という説明が
  ool の傾向(可変層が多いアームほど ool が高い)と整合する。
- したがって**この結果表を「中央層 FT は下流タスクに悪い」と解釈するのは
  時期尚早**。真に問うべきは「ChatML 形式で評価した場合にどうなるか」であり、
  これは未実施。

### Phase 3 ゲート判定への示唆

- **今回の凍結条件(apply_chat_template=false)での比較は、SFT モデルの
  真の性能差を測れていない可能性が高いため、この表単体では層構成の優劣を
  決定づけない**。
- 一方で「SFT 後は全アームで base を下回った」という事実自体は、
  apply_chat_template の有無に関わらず、chat 形式を学習していない
  few-shot 評価枠組みとの相性問題を示している可能性が高く、実運用
  (チャット形式での推論)での性能を測るには **`apply_chat_template: true`
  での再評価が必須**。
- train loss の順位([6..9] 最良)は学習の収束状況としては引き続き有効な
  指標だが、下流タスク性能のゲートとしては本評価では確定できなかった。

## 次のステップ

1. ~~**llm-jp-eval による Phase 3 ゲート評価**~~ **完了(2026-07-13)**。ただし
   上記の apply_chat_template 不一致により結果の解釈は保留。
2. **再評価(優先)**: `configs/eval/llm_jp_eval.yaml` 相当の設定で
   `apply_chat_template: true`(ChatML)にして 5 アームを再実行し、真の
   下流タスク影響を確認する。`build_inference_config` の固定値を config 駆動に
   するか、SFT モデル専用の eval config を新設する必要がある。
3. **sft-002(データ拡大、Issue #34)**: 最良層構成 [6..9] をベースに
   llm-jp instruct + Aya-ja を混合したデータで再学習
4. **推論による人手比較**: sft-001 と sft-003 [6..9] の生成品質差を実例で確認
5. **Phase 4 DPO**: 選択層 [6..9] のまま DPO に進む前提で選好データ収集開始

## 参考

- Phase 2 層プロファイリング: `experiments/reports/phase2_layer_profiling.md`
- Phase 2 ゲート判定: `experiments/reports/phase2_gate_and_next_steps.md` §4.1
- sft-001 実装 PR: #83、sft-003 configs PR: #86
- 実行時ドライバ: `scratchpad/run_sft_003.sh`
- 結果集計スクリプト: `scratchpad/summarize_sft003.py`, `scratchpad/vram_by_arm.py`
