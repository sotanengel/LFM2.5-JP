# K2 知識注入 CPT (cpt-D) — コーパス構築 (Issue #123 / #130)

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

## 次アクション

- K2-2(#131): `configs/cpt/cpt_1.2b_layerft_cptD_L9.yaml`(base=JP-202606、L9単層)で
  centi パイロットを実行し、健全性(cpt-B のような instruction-following 崩壊が
  ないか)を確認してから deci/final の decision gate へ進む
- `data/processed_cptD/mixture.jsonl` は WSL 実行環境(`~/lfm25-ja-k2/data/processed_cptD/`)
  に生成済み。再現は `scratchpad/build_cptD_corpus.py`(同ディレクトリ)を参照
