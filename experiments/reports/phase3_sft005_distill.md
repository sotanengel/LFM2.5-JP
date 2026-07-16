# Phase 3 sft-005: 蒸留選抜データ SFT と IFEval 評価(Issue #109)

- 実施日: 2026-07-16
- 関連: Issue #109(親)/ #110 #111 #112(sub)/ PR #113(実装)
- 先行: sft-002-mix(Issue #105、base -18pt で決定規則発火 → データ側根本改善へ)
- 実装 commit: `d76806d`(branch `claude/github-issue-109-0b1a67`)

## TL;DR

**sft-005-distill = IFEval prompt_strict 0.900(base 0.950 比 -5.0pt)→ 事前固定の決定規則
「base -3pt 以下」が発火。蒸留アプローチを棄却し、「SFT スキップして Phase 4 直行
(base を DPO 初期値に)」の是非の協議に進む。**

ただし内訳は全面敗北ではない: sft-002 の主劣化(bullet/char 各 -33pt)はほぼ解消し、
**polite_form は 0.826 と base の天井(0.783)を初めて突破**。同一凍結条件でデータのみ
差し替えた系列は 0.770(sft-002-mix)→ 0.890(sft-004 ichikara)→ **0.900(sft-005)** と
単調改善したが、base 0.950 には届かなかった。

## 1. データ(mix_005.jsonl)

### 来歴(トレーサビリティ)

- 蒸留応答は**事前生成 CSV**(`datasets/sft/sft005_distill_candidateB_prompts.csv`、4,000 行、
  前セッション成果物)を**ユーザーの明示指示によりそのまま採用**。本セッションでの新規モデル
  推論生成は行っていない。
- CSV 生成スクリプト(前セッションの `gen.py`)は 125 トピックの文章バンクからの
  **規則ベース構成**であり、Issue #109 候補 B に記載の Qwen3-Swallow 8B の推論出力**ではない**。
  したがって教師モデルのライセンス上の制約は発生しない(参考: ローカル環境の
  `tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2` は Apache-2.0 であることを確認済み。8B 教師の
  実推論蒸留は未実施のまま)。
- CSV md5: `a80af9d80bdba0695eafee5d409e82ef`(git LF 正規化後。元ファイル CRLF では
  `235f98fa48129d809a1d29a5ca5c4958`)。

### 選抜+ガード(詳細: [phase3_sft005_distill_stats.md](phase3_sft005_distill_stats.md))

- 評価 harness と同一 verifier(凍結)で strict 選抜 + char 系はタイトマージン 60〜90% 帯:
  採択 **3,221 / 4,000**。主な棄却は polite_form 612 件(凍結 verifier が「〜くださいませ」
  文末を非敬体判定)、format_json 同一 detail+response 重複 155 件、compound max=90 の
  評価値衝突 12 件。
- **応答長ガード(ハードゲート)PASS**: 蒸留部分 mean 152.6 字 ∈ [140.8, 211.2]
  (base 出力平均 176 字 ±20%、#105 の教訓の機械化)。
- **評価非重複(ハードゲート)PASS**: char_count 値集合の重複なし、topic の評価プロンプト
  出現なし。ホールドアウト 2 種(markdown_table / numeric_only)は #105 から継続して訓練除外。
- 混合: 蒸留 3,221 + ichikara 全数 2,903 = **6,124 行**(oasst1 / aya-ja は全除外)、seed=42。
  参考: mix 全体平均 228.4 字(ichikara 部分 312.4 字)。sft-002-mix の 284 字より短いが、
  ガード対象(蒸留部分)以外の ichikara は sft-004 と同一データ。

## 2. 学習(凍結条件、データのみ差替)

| 項目 | 値 |
|---|---|
| モデル | LiquidAI/LFM2.5-1.2B-JP-202606 |
| 可変層 | L9 単層(67.1M params、5.7%) |
| lr / epoch | 1e-5 / 1ep(sft-004 A・sft-002-mix と同一動作点) |
| batch / grad_accum | 1 / 4(base.yaml 継承) |
| steps / 時間 | 1,531 steps / 683.5 s(11.4 min) |
| train_loss | 1.659 |
| VRAM | allocated 2.2 / reserved 3.5 GiB |
| その他 | `--no-checkpoints`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

## 3. IFEval 評価(7 モデル、100 プロンプト、greedy)

既存 6 モデルの generations は冪等スキップ、sft005-distill のみ新規生成(全体 ~2 分)。

| モデル | prompt_strict | Δ base |
|---|---:|---:|
| **base(JP-202606)** | **0.950** | 0 |
| **sft005-distill** | **0.900** | **-5.0pt** |
| sft004-L9-lr1e-5 | 0.890 | -6.0pt |
| sft004-L9-lr3e-5 | 0.810 | -14.0pt |
| sft004-L6-9-lr1e-5 | 0.770 | -18.0pt |
| sft002-mix | 0.770 | -18.0pt |
| sft003-L9 | 0.560 | -39.0pt |

### verifier 別(instruction_strict、sft005 vs base)

| verifier | base | sft002-mix | sft005 | 備考 |
|---|---:|---:|---:|---|
| bullet_count (n=15) | 1.000 | 0.667 | **1.000** | sft-002 の -33pt を完全回復 |
| char_count (n=21) | 1.000 | 0.667 | **0.810** | **主劣化源**(下記) |
| format_json (n=10) | 1.000 | 1.000 | 1.000 | |
| format_markdown_table (n=10) | 1.000 | 1.000 | 1.000 | ホールドアウト無傷 |
| numeric_only (n=10) | 1.000 | 1.000 | 1.000 | ホールドアウト無傷 |
| keyword (n=21) | 1.000 | 0.857 | 0.905 | |
| **polite_form (n=23)** | 0.783 | 0.652 | **0.826** | **base 天井を初突破**(+4.3pt) |

カテゴリ別: 敬語 0.96(base 0.92)/ 質問 0.96(1.00)/ 依頼 0.92(0.96)/ 要約 0.76(0.92)。

### 失敗ケースの特徴づけ(sft005 10 件 vs base 5 件、共通 2 件)

sft005 が新規に落とした 8 件の内訳:

1. **char_count 精度(4 件)**: `ifja-005` は「50 字**以上** 100 字以内」を 31 字で**下回り**
   (短応答バイアスの裏目)、`ifja-010`/`ifja-022` は上限 +3〜5 字の僅差超過、`ifja-018` は
   原文 2 文をほぼ echo して 60 字制限を 16 字超過。
2. **文体バイアス(2 件)**: `ifja-032` は「常体で」指示に敬体が混入(訓練蒸留分の敬体比率
   由来の bleed)。`ifja-067` は敬体メールだが署名行「株式会社△△」が verifier の免除
   パターン外で strict 落ち。
3. **keyword(2 件)**: `ifja-066` は指定語「人工知能」を「AI技術」に言い換えて欠落、
   `ifja-070` は禁止語「値上げ」を要約中に使用。

一方で base の失敗 5 件中 3 件(`ifja-031`/`047`/`058`)は sft005 で解消(主に敬語系)。

**機序の解釈**: 評価非重複のため char 制約の訓練 N を 85 以上・「以内」型のみに限定した
副作用で、評価の小さい N(50〜70)と「以上」型下限制約が**訓練分布外**になった。
タイトマージン(上限の 60〜90%)+短応答傾向は下限制約と正面衝突する。つまり残余 -5pt の
少なくとも半分は「規則ベース合成データの被覆漏れ」であり、sft-002 型のスタイルシフト
(平均応答長の乖離)は応答長ガードにより再発していない。

### 統計ノート

100 件二項統計の目安は ±5pt。McNemar 型の不一致対は base のみ失敗 3 vs sft005 のみ失敗 8
(二項両側 p ≈ 0.23)であり、**-5.0pt は統計的には有意でない**。ただし決定規則は事前固定の
点推定で機械適用する(#109 記載どおり)。

## 4. 決定規則の機械適用(事前固定、変更不可)

- 判定: sft005 prompt_strict 0.900 − base 0.950 = **-5.0pt ≤ -3pt**
- → **第 3 分岐が発火: 蒸留アプローチ自体を棄却し、「SFT をスキップして Phase 4 直行
  (base を DPO 初期値に)」の是非を協議する。**

### 協議材料(次セッションでユーザー/Fable5 と協議)

Phase 4 直行を支持する材料:
- 同一凍結動作点でデータを 3 世代再設計(sft-002 → 004 → 005)しても全て base 未達。
  base 0.950 に対し SFT で +3pt を稼ぐ余地は構造的に残り 5pt しかなく、SFT の期待値が低い。
- sft-005 で「学習で base に足すより、base の高い事前性能を壊さないこと」がボトルネックだと
  再確認された。DPO(選好最適化)は base を初期値にでき、劣化リスクの管理がしやすい。

慎重材料(棄却規則への註記):
- -5pt は統計的に非有意(上記)。また劣化の主因は蒸留方式そのものではなく
  規則ベース合成の被覆漏れ(小 N・下限制約の欠落)と特定できている。
- **polite_form は base 天井を突破**(0.783 → 0.826)しており、「verifier 通過データを
  選抜して弱点カテゴリだけ教える」機構自体は機能実証された。
- Issue #109 候補 B の本来形(Qwen3-Swallow 8B の**実推論**による教師蒸留)は
  未実施のまま棄却される点に注意(本データは規則ベース合成であり、候補 A/B いずれの
  「実際に制約下で生成された応答」でもない)。
- 選抜・ガード・非重複のインフラ(PR #113)は Phase 4 の DPO ペア構築
  (chosen = verifier 通過 / rejected = 違反応答)にそのまま転用可能。

## 5. 成果物

- 実装: PR #113(`distill_select.py` + configs + tests 421 green)
- データ: WSL `data/processed/sft/mix_005.jsonl`(6,124 行)+ 選抜 stats report
- モデル: WSL `outputs/sft-005-distill/`(final のみ、checkpoints なし)
- 評価: WSL `outputs/eval/ifeval_ja/sft005-distill/`(generations / scores / aggregate)
- EXPERIMENT_LOG: rows 026–027
