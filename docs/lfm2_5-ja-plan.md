# LFM2.5-JA プロジェクト：リポジトリ設計と実験計画

## 決定事項(2026-07-08)

- 「特定の層のみを可変にしてパラメータ調整」は、**選択層のフルパラメータ学習**（LoRA/QLoRA は不使用）と解釈する。モデルは bf16 でロードし、全パラメータを freeze した上で config で指定した層（`model.model.layers[i]`）のみ `requires_grad=True` にする。
- それに伴い、config 名を `*_qlora*.yaml` → `*_layerft*.yaml` に変更する（Issue #25 / #32 / #40 参照）。
- Phase 5 の「LoRA マージ」は「最終モデルの保存・HF 形式エクスポート」に変更する（Issue #43）。

## 決定事項(2026-07-19): 目的の再定義

これまで「日本語特化」の定義は暗黙のうちに「日本語応答スタイルの強化」（instruction following / 敬語 / IFEval 系スコア）と解釈され、Phase 3（SFT）・Phase 4（dpo-001）はその前提で構築された。Issue #120 でこの前提を見直し、目的を以下のとおり再定義する：「**日本語特化 = 日本関連の知識がオフライン上でも概ね担保できること**」。

変更点：

- **主目標の変更**：クローズドブック（RAG なし）での日本関連知識のカバレッジを主軸に据える。
- **ガード指標への降格**：IFEval / llm-jp-eval のスタイル系スコア（polite_form・char_count 等を含む）は主目標から外し、「劣化していないことを確認するための非劣化ガード」として位置づけ直す。
- **継続保持する資産**：
  - CPT（継続事前学習）パイプラインを知識注入の主レバーとして継続使用する。
  - dpo-001-b005 は無劣化チェックポイント（base ±0.0pt, v1.1 採点）として保持する。
  - Swallow 8B を事実性 judge として継続利用する。
- **新ロードマップ（Issue #120 配下の sub-issue）**：
  - #121 (K0): 日本知識ベンチ（JKB）データセットの構築
  - #122 (K1): 既存資産（CPT/SFT/DPO 成果物）の知識軸での再評価
  - #123 (K2): 知識注入のための継続事前学習（cpt-D）
  - #124 (K3): 事実性を対象にした DPO
  - #125 (K4): オフライン RAG の追加（1.2B のチューニングを尽くした後の拡張）
- **既存フェーズの位置づけ**：Phase 3（SFT）・Phase 4（dpo-001）で行ったチューニングはスタイル軸の最適化に限定されたものであり、成果物自体は削除しない。ただしその役割は「主目標」から「非劣化ガード用の参照資産」へ再分類する。

日本語特化 LFM2.5 モデルを RTX 3060 Ti（VRAM 8GB）環境で段階的に構築・検証するための設計文書。

---

## 1. 前提とハードウェア制約

### 1.1 対象モデル

| 候補 | パラメータ数 | 用途 |
|---|---|---|
| `LiquidAI/LFM2.5-1.2B-Instruct` / `-Base` | 1.2B | 本命モデル |

- アーキテクチャ: LFM2 系ハイブリッド（LIV 畳み込み 10 層 + GQA 6 層、計 16 層）、32K コンテキスト
- チャットテンプレート: ChatML 類似（`<|im_start|>` / `<|im_end|>`）
- ライセンス: LFM Open License v1.0（年商 $10M 未満は商用利用可。**Apache ではない**点に注意し、成果物配布時にライセンス条項を確認すること）
- 公式 JP 版のベンチマーク（JMMLU 54.19 / J-MIFEval 79.08 / J-GSM8K 62.20 前後）が「超えるべき or 比較すべき基準線」になる

### 1.2 RTX 3060 Ti (8GB) での VRAM 見積もり

| 手法 | モデル | 概算 VRAM | 可否 |
|---|---|---|---|
| フル FT (bf16 + AdamW) | 1.2B | ~19GB+ | ❌ 不可 |
| フル FT (bf16 + AdamW 8bit + grad ckpt) | 350M | ~5–6GB | ✅ 可 |
| LoRA (bf16 ベース) | 1.2B | ~7–8GB | ⚠️ ギリギリ（seq 短めなら可） |
| QLoRA (NF4 4bit + LoRA) | 1.2B | ~4–5GB | ✅ 推奨 |
| 推論 (GGUF Q4_K_M) | 1.2B | ~1GB | ✅ 余裕 |

**基本方針**

- 学習は 「特定の層（1~複数）のみを可変にしてパラメータ調整」を標準とする
  - https://arxiv.org/pdf/2607.01232の研究結果を参考にすること
- シーケンス長は Phase ごとに 1024 → 2048 → 4096 と段階的に拡張し、OOM 境界を最初に計測する
- 大規模データ処理（トークナイズ・フィルタリング）は CPU/ディスク側で完結させ、GPU は学習と評価に専念させる

---

## 2. リポジトリ設計

```
lfm25-ja/
├── README.md
├── pyproject.toml              # uv / pip 管理（torch, transformers, peft, trl, bitsandbytes, datasets）
├── Makefile                    # make setup / data / train-sft / eval など定型操作
├── .env.example                # HF_TOKEN, WANDB_API_KEY
│
├── configs/                    # ★ 実験＝config ファイル 1 枚、を原則とする
│   ├── base.yaml               # 共通設定（モデル名、seed、精度、ログ先）
│   ├── cpt/                    # 継続事前学習 (Continued Pre-Training)
│   │   ├── cpt_350m_pilot.yaml
│   │   └── cpt_1.2b_qlora.yaml
│   ├── sft/
│   │   ├── sft_1.2b_qlora_r16.yaml
│   │   └── sft_1.2b_qlora_r64.yaml
│   ├── dpo/
│   │   └── dpo_1.2b_qlora.yaml
│   └── eval/
│       └── llm_jp_eval.yaml
│
├── src/lfm25_ja/
│   ├── data/
│   │   ├── download.py         # HF datasets からの取得
│   │   ├── clean.py            # 正規化・重複除去・品質フィルタ
│   │   ├── mix.py              # データ配合比の制御（JA:EN 比率など）
│   │   └── format_chat.py      # ChatML 形式への変換・トークン数統計
│   ├── train/
│   │   ├── train_cpt.py        # 継続事前学習（packed causal LM）
│   │   ├── train_sft.py        # TRL SFTTrainer ラッパ
│   │   ├── train_dpo.py        # TRL DPOTrainer ラッパ
│   │   └── callbacks.py        # VRAM 監視・生成サンプルログ
│   ├── eval/
│   │   ├── run_llm_jp_eval.py  # llm-jp-eval / lm-eval-harness 呼び出し
│   │   ├── quick_eval.py       # 学習中の軽量評価（perplexity + 数十問）
│   │   └── judge.py            # LLM-as-a-judge（日本語自然さ・指示追従）
│   ├── merge_export/
│   │   ├── merge_lora.py       # LoRA マージ
│   │   └── export_gguf.py      # llama.cpp 変換・量子化
│   └── utils/
│       ├── memory.py           # OOM プロービング、VRAM ログ
│       └── seed.py
│
├── scripts/                    # 1 コマンド実行用シェル
│   ├── 00_smoke_test.sh
│   ├── 10_prepare_data.sh
│   ├── 20_train_cpt.sh
│   ├── 30_train_sft.sh
│   ├── 40_train_dpo.sh
│   └── 50_eval_all.sh
│
├── data/                       # .gitignore 対象
│   ├── raw/  ├── processed/  └── mixtures/
├── outputs/                    # checkpoints, LoRA adapters, GGUF（.gitignore）
├── experiments/                # ★ 実験台帳
│   ├── EXPERIMENT_LOG.md       # 全 run の 1 行サマリ（ID / config / 結果 / 結論）
│   └── reports/                # Phase ごとの詳細レポート
└── tests/
    ├── test_data_pipeline.py   # フォーマット崩れ・特殊トークン混入の検出
    └── test_train_smoke.py     # 10 step だけ回して loss が下がるか
```

### 設計原則

1. **config 駆動**：ハイパーパラメータはすべて YAML。コード変更なしに実験を再現できる
2. **実験台帳の一元化**：W&B（またはローカル TensorBoard）+ `EXPERIMENT_LOG.md` の二重記録。run 名は `{phase}-{model}-{手法}-{連番}`（例: `sft-1.2b-qlora-r16-003`）
4. **評価の固定**：評価データ・few-shot・生成パラメータを config で凍結し、Phase 間で比較可能にする

---

## 3. データセット計画

| 用途 | 候補データ | 備考 |
|---|---|---|
| CPT（日本語知識） | llm-jp-corpus, Wikipedia-ja, CC-100 ja / OSCAR ja サブセット, 青空文庫 | 品質フィルタ後に 1〜5B トークン規模を目標。忘却防止に英語を 10–20% 混合 |
| SFT | ichikara-instruction, llm-jp インストラクションデータ, Aya-ja, 自作合成データ（大モデルで生成） | ライセンス確認必須。ChatML に統一 |
| DPO | 日本語選好データ（hh-rlhf-ja 等）+ 自作（自モデル出力を judge で順位付け） | 品質 > 量。数千〜数万ペアで十分 |
| 評価 | llm-jp-eval（JMMLU, JSQuAD, JCommonsenseQA, J-GSM8K 等）, Japanese MT-Bench | **学習データへの混入（コンタミ）チェックを必ず実施** |

前処理パイプライン（`clean.py`）の必須項目：NFKC 正規化、言語判定、MinHash 近似重複除去、機種依存文字・制御文字除去、極端な短文/長文除去、評価セットとの n-gram 重複検査。

---

## 4. 段階的実験計画

各 Phase に **ゲート条件（次に進む基準）** を設ける。失敗したら前 Phase に戻る。

### Phase 0：環境検証とベースライン確立（〜3 日）

**目的**：8GB で何ができるかの実測と、比較の物差しづくり。

- [ ] 環境構築（CUDA / bitsandbytes / TRL / PEFT の動作確認）
- [ ] `00_smoke_test.sh`：LFM2.5-1.2B を 4bit ロードし推論、次に 「特定の層（1~複数）のみを可変にしてパラメータ調整」 で 20 step 学習して loss 低下と VRAM ピークを記録
- [ ] OOM プロービング：seq_len {1024, 2048, 4096} × batch {1, 2, 4} × r {16, 64} の格子で最大構成を実測 → `experiments/reports/phase0_memory.md`
- [ ] ベースライン評価：`LFM2.5-1.2B-Instruct`（無改造）と `LFM2.5-1.2B-JP-202606` を llm-jp-eval で評価し、スコア表を凍結

**ゲート**：「特定の層（1~複数）のみを可変にしてパラメータ調整」 学習が安定動作し、ベースラインスコアが公表値と大きく乖離していないこと（評価ハーネス自体の検証）。

### Phase 1：データパイプライン検証（〜1 週間）

**目的**：学習を始める前にデータ品質を固める。データ起因の失敗が最も高コスト。

- [ ] 各コーパスのダウンロード・クリーニング・統計出力（トークン数分布、言語比率、重複率）
- [ ] ChatML 変換の単体テスト（特殊トークンの位置、loss マスクの正しさを decode して目視確認）
- [ ] 評価セットとのコンタミチェック
- [ ] **パイロット学習**：350M + データ 1% で 1 epoch 回し、loss カーブが健全か・生成サンプルが壊れていないかを確認

**ゲート**：テスト全通過、パイロットで perplexity が単調改善、生成テキストに文字化け・テンプレート崩れなし。

### Phase 2：継続事前学習（CPT）（1〜3 週間）

**目的**：日本語の言語知識を底上げする。ただし **1.2B-JP-202606 が既に存在するため、まず「CPT が本当に必要か」を A/B で検証する**。

| 実験 ID | 内容 |
|---|---|
| cpt-A | CPT なし → 直接 SFT（Phase 3 へ）※対照群 |
| cpt-B | 1.2B-Base + 日本語 CPT（「特定の層（1~複数）のみを可変にしてパラメータ調整」, ~1B トークン）→ SFT |
| cpt-C | 公式 1.2B-JP をベースに SFT のみ（実務的な近道） |

- 8GB では 1B トークンの CPT に数日〜1 週間以上かかるため、**200M トークンで中間評価**し、日本語 perplexity と JMMLU の改善傾向が出なければ打ち切る
- 破滅的忘却の監視：英語ベンチ（MMLU サブセット）を quick_eval に含める

**ゲート**：cpt-B が cpt-A / cpt-C を日本語 perplexity + JMMLU で明確に上回る場合のみ CPT を採用。そうでなければ cpt-C ルートに切り替え（工数削減）。

### Phase 3：SFT（1〜2 週間）

**目的**：日本語の指示追従・対話品質を作り込む。ここが体感品質への寄与が最も大きい。

段階的 ablation（1 実験 = 1 変数）：

1. **sft-001**: ichikara のみ / 単層 L9（選択層フル FT）/ 2 epoch（最小構成の基準。Phase 2 層プロファイリングの最良層を採用し、sft-003 の「単層 L9」アームを兼ねる）
2. **sft-002**: データ配合を拡大（+llm-jp instruct, +Aya-ja）
3. **sft-003**: 可変層の数・位置比較（単層 L9 / 単層 L6 / 2 層 [6,9] / 中央 4 層 [6..9] / フル FT 参照。Phase 2 層プロファイリングの中央帯知見を転用、端の層・L15 は比較群から除外。詳細: `experiments/reports/phase2_gate_and_next_steps.md` §4.1）
4. **sft-004**: 学習率・epoch 数のスイープ（過学習の兆候＝eval loss 反転を監視）
5. **sft-005**: 合成データ追加（大モデルで日本語タスク特化データを生成・蒸留）

評価：quick_eval（学習中）→ llm-jp-eval + Japanese MT-Bench（各 run 終了後）→ 上位 2 run のみ人手で 30 プロンプト目視比較。

**ゲート**：J-MIFEval / MT-Bench-ja がベースライン（Phase 0 の表）を上回ること。敬語・自然さの目視評価で明確な破綻がないこと。

### Phase 4：選好最適化（DPO）（〜1 週間）

**目的**：日本語の自然さ・丁寧さ・安全性の微調整。

- SFT ベストモデルの出力ペアから選好データを構築（judge モデルでスコアリング）
- DPO は beta {0.05, 0.1, 0.3} をスイープ。「特定の層（1~複数）のみを可変にしてパラメータ調整」 では reference モデルもロードするため **seq_len を 1024 に落として VRAM を確保**
- 過剰最適化（応答が冗長化・定型化する現象）を MT-Bench-ja と目視で監視

**ゲート**：MT-Bench-ja 向上 かつ llm-jp-eval のタスクスコアが劣化していない（±1pt 以内）。劣化するなら DPO をスキップして SFT モデルを最終版とする判断もあり。

### Phase 5：統合評価・量子化・リリース（〜1 週間）

- [ ] LoRA マージ → フル評価（llm-jp-eval 全タスク、公式 JP 版との最終比較表）
- [ ] GGUF 変換（Q8_0 / Q4_K_M）と量子化後の精度劣化測定（perplexity + 主要タスク）
- [ ] 3060 Ti / CPU での推論速度計測（tok/s）
- [ ] モデルカード作成（データ出典・ライセンス・制限事項・再現手順）

---

## 5. 実験管理と再現性

- **記録必須項目**：config ハッシュ、git commit、データバージョン、seed、VRAM ピーク、所要時間、全評価スコア、生成サンプル 5 件
- **seed**：主要実験は seed 3 種で分散確認（350M パイロットのみ。1.2B は計算資源上 1 seed + 慎重な判断）
- **失敗の記録**：OOM 条件・発散した lr なども `EXPERIMENT_LOG.md` に残す（同じ穴に落ちない）
- **週次レビュー**：ゲート条件に照らして go / no-go / pivot を判断し、レポートに 3 行で結論を書く

## 6. 主要リスクと対策

| リスク | 対策 |
|---|---|
| 8GB での学習速度が遅く反復回数が稼げない | 350M での先行検証を徹底。1.2B は「勝ち筋の確認」にのみ使う |
| LFM2 アーキテクチャがツール（TRL/PEFT/llama.cpp）で未対応・不具合 | Phase 0 のスモークテストで全経路（学習→マージ→GGUF→推論）を最初に貫通させる |
| 破滅的忘却（英語・コード能力の喪失） | 英語データ 10–20% 混合、英語ベンチの常時監視 |
| 評価セットのコンタミで見かけ上のスコア向上 | Phase 1 で n-gram 重複検査を必須ゲート化 |
| 公式 JP 版を超えられない | 目標を「汎用日本語で勝つ」から「特定ドメイン（自分のユースケース）で勝つ」に再設定できるよう、ドメイン評価セットを早期に用意 |
