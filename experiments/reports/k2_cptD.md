# K2 知識注入 CPT (cpt-D) — コーパス構築 + centi パイロット (Issue #123 / #130 / #131)

K1(#122、`experiments/reports/k1_asset_reassessment.md`)で確定したワースト10セル
(生活・慣習 3 セル・言語 advanced・政治・制度 advanced・歴史 advanced・食文化
advanced+standard・地域・観光 standard・科学技術・産業 standard)を直接ターゲットにした
CPT コーパスを構築した。本レポートは K2-1(#130)の成果物(コーパス分野被覆・語数統計)。

## パイプライン構成

- **実装**: `src/lfm25_ja/data/wikipedia_ja_japan_subset.py`(`WORST10_CELL_KEYWORDS` +
  `filter_matching_documents` + `oversample_documents`)、`prepare.py` への `filter:
  type: keyword_oversample` 配線(TDD、tests/test_wikipedia_ja_japan_subset.py +
  tests/test_data_pipeline.py)
- **設定**: `configs/data/corpus_cptD.yaml`
  - `wikipedia_ja_japan_subset`: wikipedia_ja(20231101.ja)からワースト10セルの
    キーワード(タイトル or 本文の部分一致)にマッチする記事を抽出し、**3 倍
    オーバーサンプル**
  - `aozora`: 既存 aozora コーパスをそのまま追加(歴史・伝統文化の補強)
  - `wikipedia_en`: 忘却防止用 15% 英語混合(下記「実行上の注記」参照)
  - `mix.ratios`: ja 85% / en 15%(既存 corpus.yaml 継承)
  - **汚染ガード適用**: `datasets/eval/jkb/eval_texts.jsonl`(504 件)を
    `clean.contamination` の n-gram 重複検査に通した(該当 0 件)

## 重要なバグ修正: オーバーサンプルが dedup で相殺される問題

最初の実行(単一コーパステスト)で、`filter_and_weight_documents` が
**clean_corpus(MinHash dedup)より前**にオーバーサンプル(3 倍複製)していたため、
複製されたほぼ同一文書を dedup が正しく重複除去し、**オーバーサンプルが実質的に
無効化されていた**(133,401 入力 → 43,951 出力 ≈ オーバーサンプル前のユニーク
マッチ数)ことが判明。

**修正**: フィルタ(マッチング + `matched_cells` タグ付け、複製なし)は
clean_corpus **より前**、オーバーサンプル(複製)は clean_corpus **より後**に
実行する順序に変更(`filter_matching_documents` / `oversample_documents` に分離)。
修正後は dedup 除去率が 66.68% → 0.03%(14 件)に改善し、オーバーサンプルが
最終コーパスに正しく反映されることを確認した。

## 実行上の注記: WSL2 クラッシュと wikipedia_en のストリーミング化

`prepare_data`(全コーパス一括、非ストリーミング)で **wikipedia_en(638 万記事)の
train split 生成中に WSL2 が 2 回連続でクラッシュ**(`Wsl/Service/.../E_UNEXPECTED`、
`wsl --shutdown` で復旧)。フルの英語 Wikipedia 全件をメモリに展開する処理が
トリガーになったと推定される。

**回避策**: `wikipedia_en` のみ `streaming=True` + `sample_limit=60000` で読み込む
scratch ドライバ(`scratchpad/build_cptD_corpus.py`、WSL 実行環境
`~/lfm25-ja-k2` に配置、再現可能)に切り替えて実行し、クラッシュなく完走した。
mix 段階で 15% 比率に必要な分(実績 25,975 件)しか使われないため、6 万件の
サンプルで十分だった。`prepare.py` 本体はコーパスごとに streaming/sample_limit を
分けられない(グローバル 1 フラグ)ため、本番はこの scratch ドライバを正式な
再現手順として扱う。将来 `prepare.py` にコーパス単位の `streaming`/`sample_limit`
override を追加する価値はあるが、K2 の完了条件には含めない(スコープ外、必要なら
別 issue 化)。

## コーパス統計

### コーパス別クリーニング結果

| コーパス | 入力 | 出力(clean 後) | 除去率(dedup) |
|---|---|---|---|
| wikipedia_ja_japan_subset | 44,467(ユニークマッチ) | 43,951 | 0.03%(14 件) |
| aozora | 16,951 | 15,976 | 0.20%(32 件) |
| wikipedia_en(60k サンプル) | 60,000 | 59,152 | 0.53%(314 件) |

wikipedia_ja_japan_subset はオーバーサンプル(×3)後に mix 段階へ投入(43,951 → 131,853)。

### 最終 mixture(`data/processed_cptD/mixture.jsonl`)

- 総文書数: **173,171**(ja: 147,196 / en: 25,975、実測比率 85.00% / 15.00%)
- 文字数: min=50 max=50,000 mean=5,191.8 median=2,459
- 総文字数: **約 8.99 億文字**(トークン数は packed cache 構築時に実測)

### ワースト10セル別カバレッジ(文書数・文字数、複数セルにマッチする文書は両方でカウント)

| セル | 文書数 | 文字数 |
|---|---|---|
| 地域・観光_standard | 29,694 | 148,934,325 |
| 生活・慣習_standard | 26,254 | 176,601,217 |
| 歴史_advanced | 22,533 | 75,220,794 |
| 生活・慣習_advanced | 17,403 | 105,677,321 |
| 科学技術・産業_standard | 13,880 | 88,531,402 |
| 政治・制度_advanced | 11,590 | 70,175,111 |
| 食文化_standard | 8,327 | 52,623,748 |
| 食文化_advanced | 6,559 | 34,815,506 |
| 生活・慣習_core | 1,821 | 17,058,201 |
| 言語_advanced | 1,611 | 7,843,527 |

全 10 セルで非ゼロのカバレッジを確保。ただし件数には **キーワード誤マッチ(偽陽性)を
含む**点に注意(次節)。

## 既知の限界: キーワード部分一致による偽陽性

ワースト10セルのキーワードは単純な部分文字列一致(タイトル or 本文)であるため、
汎用的な語(例: 「世界遺産」「国民年金」「健康保険」「彼岸」)は本題と無関係な記事
(例: 映画監督の経歴中の「TBS『THE世界遺産』」という番組名への言及)にもマッチする。
実際にサンプル確認で「地域・観光_standard」「生活・慣習_standard」に偽陽性が
含まれることを確認した。これはコーパスの平均的な密度を薄める方向に働くが、
真陽性記事(冠婚葬祭・日本国憲法・郷土料理等の専門記事)自体は正しく含まれている。

**K2-2 centi パイロットでの確認事項**として引き継ぐ: もし JKB 実測でワースト10
セルの改善が乏しければ、キーワードマッチの精度向上(タイトルのみマッチに限定、
複数キーワード共起要求、本文中の出現位置制限等)をコーパス再構築の見直し候補とする。

## 完了条件(#130)

- [x] `configs/data/corpus_cptD.yaml` 追加(wikipedia_ja_japan_subset + aozora限定 + wikipedia_en 15%)
- [x] 新規抽出モジュールにユニットテスト付き(tests/test_wikipedia_ja_japan_subset.py、tests/test_data_pipeline.py)
- [x] 汚染フィルタ適用済みで prepare 実行可能(JKB eval_texts.jsonl 使用、重複 0 件)
- [x] ワースト10セル対応カテゴリの一覧が config/本レポートに明記されている

## K2-2: centi パイロット (Issue #131)

### 環境トラブルと恒久対応(重要、他セッションにも影響)

centi 実行時に **CUDA `device not ready` エラーが全 5 回連続で再現**(WSL2 再起動・
Windows 本体再起動を挟んでも解消せず)。fable5 エージェントに相談し原因を確定:

- **根本原因**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` と **VRAM 物理容量
  (8GB)超過の組み合わせが WSL2 の dxgkrnl パラバーチャライゼーション層で壊れている**
  (NVIDIA driver 610.47 / CUDA 13.3)。expandable_segments は CUDA VMM API を使い、
  通常の cudaMalloc とは別の residency パスを通るため、VRAM を超えた瞬間に失敗する
  (`dmesg` に `dxgkio_make_resident: Ioctl failed: -12` = ENOMEM が対応)。
- **対策**: 学習実行時は `unset PYTORCH_CUDA_ALLOC_CONF`(空にする)。この変更のみで
  同一configが1回で通過した。
- **副次原因**: `.wslconfig` の `memory=29GB`(直近変更、32GB機の91%)が host 側メモリを
  枯渇させ、K2-1 の WSL クラッシュ3回の一因だったと判明。`memory=24GB`(75%目安)に
  修正・反映済み。
- 詳細な恒久対応・診断手順は `C:\Users\na-g-\.claude\CLAUDE.md`(本PCコーディング標準)
  の「WSL でよくあるミス」節に追記済み(次回セッション以降で同じ調査時間を消費しない
  ため)。

### 実行内容

- packed_cache の設計(全173,171文書を先にトークナイズしてから1/100に間引く)は
  centi でも巨大なトークナイズバッファを生成しWSLクラッシュの一因となったため、
  centi 専用に**事前に間引いた入力ファイル**(shuffle済みmixtureの先頭1,200文書、
  `data/processed_cptD/mixture_centi.jsonl`)を使う変則構成(`configs/cpt/
  cpt_1.2b_layerft_cptD_L9_centi_manual.yaml`、`--package full` で実行)で代替。
  本来の `--package centi` はフルコーパスの事前トークナイズを要するため deci/final
  (#132)着手前に packed_cache のストリーミング化を検討する必要がある(スコープ外・
  次アクションに記載)。
- 668 packed sequences(seq_len 6144)、L9単層(trainable 5.7%)、1 epoch、167 steps、
  約22分で完走。

### 学習結果(健全性チェック)

- **train loss: 3.4519 → 2.4610**、単調減少・発散なし・grad_norm 安定(0.5〜0.8 台で
  推移、cpt-B や phase2 の L15 で見られた「loss 上昇 = 適合不能」パターンは**出現せず**
- **定性チェック(base vs centi、ワースト10セル対象の質問2問)**:
  - 「国民年金制度」: base は要領を得ない反復気味の説明。centi は「1948年制定」
    「厚生年金保険」「国民皆保険制度」等、より具体的で構成された説明(制定年は不正確だが
    方向性は改善)
  - 「四十九日とは」: **base は完全に破綻**(謎の四択問題を捏造し「結婚式の翌日」という
    無関係な誤答を返す)。**centi は仏教行事として一貫した説明**(「亡くなった人の…
    納骨…」で細部はやや不正確だが、base のような支離滅裂さは無い)
  - **cpt-B で見られたエコー・反復ループ・文法崩壊は centi では確認されず**

### Decision gate 判定(centi → deci 進行可否)

**PASS**。centi 規模(1,200文書・1 epoch)で: (1) loss 発散なし (2) 生成が流暢で
instruction-following の崩壊兆候なし (3) ワースト10セル対象トピックで base より
むしろ具体的・一貫した応答が観測された。K1/#123 の本決定規則(JKB数値ゲート)は
deci/final 相当の規模で JKB v1 を実測して初めて機械判定できるため、centi 段階では
「健全性ゲート」(発散・崩壊がないこと)のみで判定する契約どおり、**deci へ進行可**
と結論する。

## K2-3: deci 学習 + 全指標ゲート (Issue #132 / #137 / #138)

### インフラ前提(実施中に確定)

- packed_cache: ストリーミング pack(#140) + **deci/centi は文書先間引き + int32 圧縮**(#142)。
  初回フル tokenize は WSL で RSS〜20GB に達し中断 → 文書先間引き後は max_docs=17317 /
  10047 sequences、RSS〜3.5GB で安定。
- 学習ラッパー `scripts/22_train_cpt_d.sh` で `unset PYTORCH_CUDA_ALLOC_CONF`。
- C: ディスク逼迫(空き〜64MB)で最終 `training_args.bin` が 0 バイトになったが、
  `model.safetensors`(2.18 GiB, 148 tensors)と `checkpoint-2512` は健全。Temp/Windows
  `.venv` 削除で C: 空き〜11GB 確保後、WSL 再起動して検証済み。

### deci 学習結果

| 項目 | 値 |
|---|---|
| config | `configs/cpt/cpt_1.2b_layerft_cptD_L9.yaml` `--package deci` |
| run dir | `~/lfm25-ja-k2/outputs/cpt-1.2b-layerft-cptD-L9-deci-L9` |
| packed | 10,047 sequences (max_docs=17,317 / mixture 173,171 の 1/10) |
| steps | 2,512 / 2,512 (1 epoch) |
| runtime | 5h43m (`train_runtime` 20,620s) |
| train_loss | 初期〜3.62 → 最終付近〜2.26、**mean train_loss 2.399** |
| VRAM | allocated〜2.2 GiB / reserved〜8.3 GiB |
| 健全性(loss) | 発散なし・grad_norm 0.5〜0.7 台で安定 |

### 全指標実測(中間決定ゲート)

| 指標 | base (凍結) | cpt-D deci | 判定 |
|---|---|---|---|
| JKB v1 全体 | 49.9% (199/399) | **0.0% (0/399)** | FAIL (≥60%) |
| JKB 12 分野 base−3pt | — | **全 12 分野で −22〜−75pt** | FAIL |
| IFEval prompt_strict | 0.950 | **0.060** | FAIL (≥0.920) |
| llm-jp-eval AVG | 0.469 | **0.012** | FAIL (≥0.459) |

機械適用: `scripts/eval_k2_gate.py` →
`outputs/eval/k2_gate/cptD_deci_verdict.json` = **FAIL**(4 条件すべて未達)。

生成観察: JKB 短答が英語断片・反復・記号列に崩壊。IFEval もほぼ非遵守。
centi 定性チェック(国民年金・四十九日)で「崩壊なし」と見えたが、**deci 規模の
定量評価では cpt-B 以上の全面崩壊**が確定。

### Decision gate 判定(deci → final 進行可否)

**FAIL。final は実行しない**(Issue #132 / #139 スキップ)。cpt-D は **棄却**。

### 原因分析と RAG(K4, #125)へ委ねる範囲

1. **機序仮説**: 知識密度コーパス × 中央層フル FT を deci(〜1/10 文書)まで伸ばすと、
   短答 QA / 指示追従の出力分布が破壊される(loss は下がるが用途指標が全滅)。
   centi(1,200 文書・変則 manual)の定性健全性は **スケールアップの予測子として不十分**。
2. **cpt-D パラメータ経路は主レバーとして不採用**。日本知識のオフライン担保は
   **K4 オフライン RAG(#125)** に委ねる:
   - 事実・統計・制度・地誌・年号など **検索で答えが確定する知識**
   - JKB ワーストセル(生活・慣習 / 言語 / 食文化 advanced 等)の根拠パッセージ
3. **保持する資産**: base JP-202606、dpo-001-b005(様式・知識とも無劣化)、K0 JKB ベンチ、
   packed_cache 省メモリ化・`22_train_cpt_d.sh` の環境知見。
4. **次段**: K3(#124)は base または dpo-001-b005 を初期値に事実性 DPO。知識ギャップは
   K4 RAG 設計へ。cpt-D final 再挑戦はコーパス/強度の抜本見直しが前提(本 Issue 外)。

## 次アクション

- #132 / #123 を cpt-D 棄却・K4 委譲でクローズ判断
- K4(#125) でオフライン RAG の索引・取得・引用仕様を起こす
- K3(#124) 着手時の初期値候補は base / dpo-001-b005(cpt-D は除外)

