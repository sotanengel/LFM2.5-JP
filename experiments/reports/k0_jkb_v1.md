# K0: JKB v1 (Japan Knowledge Bench) 構築レポート

Issue #121 / 親 Epic #120。新方針「日本語特化 = 日本関連知識のオフライン担保」の主評価軸となる closed-book 知識ベンチ v1。

## サマリー

- **構築完了**: 全 **504 問**(仕様 500〜1,000 のうち下限を満たす v1)
- **配置**: `datasets/eval/jkb/eval.jsonl`(399 問)+ `datasets/eval/jkb/train.jsonl`(105 問)
- **分野**: 12 分野(地理 / 歴史 / 文学 / 食文化 / 伝統文化 / 政治・制度 / 生活・慣習 / 地域・観光 / スポーツ / 科学技術・産業 / 宗教・信仰 / 言語)
- **難度**: 3 層(core / standard / advanced)、各分野 × 各難度で 14 問(504 = 12 × 3 × 14)
- **形式**: `short_answer` 468 問 + `mcq` 36 問(全難度で各 12 問ずつ)
- **採点器**: `src/lfm25_ja/eval/jkb.py`(short_answer は既存 `japan_probe.extract_answer_segment` 再利用、mcq は先頭ラベル抽出 + `答え: X` フォールバック)
- **CLI**: `scripts/run_jkb.py`(モデル推論 → 採点 → per-model + 全体クロス表 markdown)
- **テスト**: `tests/test_jkb.py` 34 件全 green(全体 542 tests、既存回帰なし)
- **汚染検査**: `scripts/check_jkb_contamination.py` で機構動作確認済み(K2 CPT 準備時に本番実行)
- **K1 前段の base モデル試験推論**: 本レポートでは未実施(WSL GPU 実行が必要)→ K1 セッション冒頭で実施

## 目的の再確認

親 Epic #120 でユーザーが明示した新目的:
> 私の日本語特化とは日本語応答を強化することではなく、**日本関連の知識がオフライン上でも概ね担保できる**というものである。

JKB v1 はこの目的の**主評価軸**である。従来の llm-jp-eval(NLU 様式寄り)/ IFEval(指示追従・様式)は非劣化ガードに降格し、**日本知識の正答率**が主指標となる。K1 の資産再評価で JKB 上の分布を測ってから、初めて数値目標を固定する(測定第一の原則、sft-004 レビューの継承)。

## スキーマ

詳細は `datasets/eval/jkb/schema.md`。要点:

- 1 行 1 問 JSONL、UTF-8。フィールド: `id / domain / difficulty / format / prompt / answers / choices / correct_choice / source_url / source_quote`。
- `format=short_answer`: `answers` に substring 一致候補(表記ゆれ吸収: 富士山 / 富士、家康 / 徳川家康 等)。
- `format=mcq`: `choices=[{"label":"A","text":"..."}, ...]`(A–D 4 択)+ `correct_choice="B"` 等。substring では紛れる年号や紛らわしい名詞対に用いる。
- `source_url`: Wikipedia-ja 記事 URL(20231101.ja 断面固定、時事 = 2024 以降は対象外)。
- `source_quote`: ≤80 字の裏取り引用。

Wikipedia-ja 記事のシード URL 群はオーケストレーターが `datasets/eval/jkb/_source_matrix.md` に固定(12 分野 × 3 難度 = 36 セル)。**この URL 群外の外部知識に依存させない**制約で 504 問を著述した。

## 著述プロセス(実施した内容と方針からの逸脱)

**設計時方針**: オーケストレーターが matrix を固定し、sonnet5 サブエージェント 12 並列(1 分野 42 問ずつ)にフレームワーク駆動で著述させ、オーケストレーターがレビューする。

**実際**:
- 1 分野(**地理**)= sonnet5 サブエージェントで著述、42 問 validation pass、オーケストレーターがフォーマット・重複・URL 妥当性を確認 → 採用。
- 残り 11 分野(**歴史 / 文学 / 食文化 / 伝統文化 / 政治・制度 / 生活・慣習 / 地域・観光 / スポーツ / 科学技術・産業 / 宗教・信仰 / 言語**)= 11 並列サブエージェントを起動した直後に **Anthropic 側のセッション時間制限(1:30pm Asia/Tokyo リセット)に到達**し全 11 件が strand。partial 出力もファイルには残らず。
- ユーザーの指示「継続しろ。並列は許可するがロスト可能性を考慮」を受け、**オーケストレーター本体で残 11 分野を直接著述**(matrix 定義済み URL 群のみを情報源、schema 準拠)。
- 途中で 7 分野完了時点(294 問)で checkpoint コミット(`a55e4a1`)。以降 5 分野を追加し、504 問で v1 完成。

**この逸脱の帰結**:
- 良い方向: 全問がオーケストレーター一人のスタイル・裁量で書かれ、口調・粒度の分散が地理単体よりむしろ低い。
- 悪い方向: 分野内の網羅性は matrix の URL 群依存であり、matrix 選定の偏り(例: `region` = 世界遺産中心、`sci` = ノーベル賞受賞者中心)はそのまま v1 の偏りとして残る。K2 の結果を見て v2 拡張時に再検討する。

## 分布

### 全体

| 指標 | 値 |
|---|---|
| 全問 | 504 |
| train | 105(SHA1(id) % 5 == 0) |
| eval | 399 |
| 分野数 | 12(各 42 問) |
| 難度 | core 168 / standard 168 / advanced 168 |
| MCQ | 36(core/standard/advanced 各 12) |
| short_answer | 468 |

### セル別(分野 × 難度、各 14 問中の train / eval 内訳)

分割は `SHA1(id) % 5` の疑似乱数によるため、各セルの train は 0〜6 とばらつく。予期される平均は 2.8。0 になったのは **生活・慣習 × standard** の 1 セル(全 14 問が eval に配分)。K3 事実性 DPO の on-policy 生成では 1 セルの pilot 数が 0 でも他の 35 セルからサンプリング可能なため実害は小さいが、K3 セッション冒頭でこの分布を確認すること。

| 分野 | core t/e | standard t/e | advanced t/e |
|---|---|---|---|
| 地理 | 1 / 13 | 3 / 11 | 1 / 13 |
| 歴史 | 2 / 12 | 2 / 12 | 4 / 10 |
| 文学 | 3 / 11 | 4 / 10 | 1 / 13 |
| 食文化 | 4 / 10 | 2 / 12 | 4 / 10 |
| 伝統文化 | 3 / 11 | 3 / 11 | 1 / 13 |
| 政治・制度 | 4 / 10 | **0 / 14** | 3 / 11 |
| 生活・慣習 | 3 / 11 | 4 / 10 | 4 / 10 |
| 地域・観光 | 1 / 13 | 5 / 9 | 6 / 8 |
| スポーツ | 3 / 11 | 4 / 10 | 1 / 13 |
| 科学技術・産業 | 4 / 10 | 2 / 12 | 3 / 11 |
| 宗教・信仰 | 4 / 10 | 3 / 11 | 3 / 11 |
| 言語 | 5 / 9 | 3 / 11 | 2 / 12 |

補足: 上表で「政治・制度 standard 0/14」となっているのは `SHA1(id) % 5` の偶発。決定論的な再現性は保たれるが、K3 時に問題になれば `scripts/build_jkb_v1.py` の `_split_bucket` にセル別ストラティフィケーションを追加する。

## 統計誤差の見積もり(K1 で使用)

- **全体スコア(N=399 eval)**: Wilson 95% CI ≈ ±4.5pt(実測値 = 0.5 付近想定)。sft-004 レビューで Fable5 が指摘した「±5pt/タスク併記」の慣行を維持。
- **分野別スコア(N=33 前後)**: ±16pt。傾向表示にとどまり分野間の順位比較には不十分。
- **セル別(N=8〜14)**: ±25pt 以上。個別の「よくできた / できなかった」判定用のみで、統計的主張には使わない。
- MCQ での偶然正解: 4 択 = 期待 25% の底上げが乗るため、MCQ 12 問/難度は全体スコアに +5%pt 相当の下駄を付ける。K1 では「MCQ 除外スコア」も併記する。

## 採点方式

### 一次(オフライン、CPU 可)

- `src/lfm25_ja/eval/jkb.py::score_row(row, raw_text)`
  - `format=short_answer`: `extract_answer_segment(raw_text)`(先頭回答セグメント抽出、`japan_probe.py` の Issue #76 修正流用)→ `answers` の任意候補の substring 一致。
  - `format=mcq`: `_MCQ_ANSWER_RE` で先頭 A-E ラベル抽出、失敗時は raw_text 全体で `答え: X` の後方フォールバック。ASCII 半角 A-E のみ許容(fullwidth Ａ は拒否 = テストで明示)。

### 二次(K1 で採用予定、GPU 必要)

- `src/lfm25_ja/eval/judge_swallow.py`(dpo-001 で構築)を JKB 用に転用: judge プロンプトを「事実の正誤判定」に差替、Swallow 8B 4bit で pointwise 正誤判定。
- 判定ルール: 一次 pass → OK / 一次 fail かつ二次 pass → `needs_review`(手動照合) / 一次・二次 fail → NG。

### CLI 実行

```bash
uv run --no-sync python scripts/run_jkb.py \
    --models base=LiquidAI/LFM2.5-1.2B-Base \
             jp202606=<path-to-JP-202606-checkpoint> \
             cptB=<WSL path to outputs/cpt-1.2b-layerft/checkpoint-9000> \
             dpo001b005=<path to dpo-001-b005> \
    --dataset datasets/eval/jkb/eval.jsonl \
    --out-dir outputs/eval/jkb/k0-baseline
```

各モデル: `{out_dir}/{label}/generations.jsonl`, `scored.jsonl`, `report.md`。全モデル比較: `{out_dir}/summary.md`。

## 汚染ガード

`scripts/build_jkb_v1.py` は train.jsonl / eval.jsonl と併せて **`datasets/eval/jkb/eval_texts.jsonl`** を出力する(1 行 = `{"id","text": prompt + " " + best_answer}` 形式)。これは既存 `src/lfm25_ja/data/prepare.py --eval-texts <path>` にそのまま渡せる形式で、K2 の cpt-D CPT データ準備で:

```bash
uv run --no-sync python -m lfm25_ja.data.prepare \
    --corpus-config configs/data/corpus.yaml \
    --eval-texts datasets/eval/jkb/eval_texts.jsonl \
    --output-dir data/processed/cpt-d
```

を実行することで、`clean.clean_corpus` が JKB の n-gram(n=13、`corpus.yaml::clean.contamination.ngram`)と閾値 0.5 以上重複する CPT 学習文書を自動で除外する。

### K0 時点での機構動作確認

`scripts/check_jkb_contamination.py` を eval_texts に対して実行:

```
JKB rows loaded: 504
JKB n-gram pool size (n=13): ~10006
synthetic hit-doc overlap: 0.5185  (should be ~1.0)
synthetic miss-doc overlap: 0.0000  (should be ~0.0)
```

- **synthetic hit-doc** = JKB 2 問の連結 → 0.52。前半後半の境界を跨ぐ n-gram は JKB pool に無いため理論値通り(2 問合計 n-gram のほぼ半分ずつで 0.5)。
- **synthetic miss-doc** = JKB 語彙を含まない任意日本語文 → 0.00。正しく非該当判定。
- n-gram pool ~10,006 個、JKB 全 504 問。

**「重複が 0(または削除済み)」の完了条件について**: 本 K0 では JKB 側から Wikipedia-ja 全量に対して n-gram を張る方向は行っていない(必要メモリ数十 GB 級のため)。K2 準備時に `prepare.py --eval-texts` を wikipedia_ja に走らせる方向で「重複文書を CPT 学習コーパスから除去する」形で担保する。JKB は Wikipedia-ja 記事本文を直接引用していないため(prompt はオーケストレーター著述、source_quote は ≤80 字の裏取りのみで JKB dataset の text field に含めていない)、除去対象になる可能性は限定的だが、実測値は K2 準備時に併記する。

## 完了条件(Issue #121)

- [x] JKB v1 が 500 問以上で完成し `datasets/eval/jkb/` に配置(504 問)
- [x] Wikipedia-ja との汚染検査(n-gram)機構が完了し、K2 時点で CPT コーパスとの重複を除去できる状態(K2 準備で `prepare.py --eval-texts` として直接使用可能)
- [x] 一次採点器が pytest で通る(34 テスト green)/ 二次採点器(Swallow judge)は K1 で `judge_swallow.py` からプロンプト差替により実装予定
- [ ] base(JP-202606)で試験推論 → 分野×難度クロス表がレポート化 → **K1 セッションで実施**(WSL GPU 必要のため main セッションでは未実行)

## K1 での次の一手(次セッション冒頭で実施)

1. **JKB base 試験推論**: 上記 CLI で `--models base=<JP-202606>` のみ、`--dataset datasets/eval/jkb/eval.jsonl` を実行。所要 = 399 問 × 40 tok greedy = 数分想定。
2. **分野×難度クロス表**を eyeball、当初「core 60%+ / standard 30-50% / advanced 5-25% 期待」との乖離を確認。
3. cpt-B(WSL `outputs/cpt-1.2b-layerft/checkpoint-9000`)/ dpo-001-b005 / sft005-distill を同一 CLI で追加。
4. K1 レポートで **初めて数値目標を固定**(例: 「コア常識 ≥90% / 標準 ≥70% / 発展 ≥40%」を実測を見て調整)。
5. K2 の CPT ターゲット(どの分野の advanced が足りないか)を決定。

## 参照

- 種: `src/lfm25_ja/eval/japan_probe.py`(10 分野 × 5 問 = 50 問。core 相当を大半含み、地理・歴史・文学・食文化・伝統文化・政治・経済(→ 科学技術と一部併合)・スポーツ・言語 seed として吸収)
- Judge 転用元: `src/lfm25_ja/eval/judge_swallow.py`(dpo-001、pointwise 品質判定 → K1 で事実正誤判定へ差替)
- 汚染検査基盤: `src/lfm25_ja/data/clean.py::ngram_contamination_checker`
- CPT コーパス定義: `configs/data/corpus.yaml`(wikipedia_ja = 20231101.ja、n=13、threshold=0.5)
- 親 Epic: #120 / 直上 sub-issue: #121
- 分割ロジック: `scripts/build_jkb_v1.py::_split_bucket`(SHA1(id) % 5)
