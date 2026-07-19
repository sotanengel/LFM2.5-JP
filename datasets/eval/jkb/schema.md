# JKB v1 (Japan Knowledge Bench) — スキーマと構築基準

**Issue #121 / 親 Epic #120。K0 の主成果物**。closed-book で「日本関連の知識」を測る評価ベンチ。K1（資産再評価）/ K2（知識注入 CPT）/ K3（事実性 DPO）全ての基盤となる。

- **問題数**: v1 で **500 問（train 100 / eval 400）**。K2 まで進めた段階で v2 拡張判断。
- **分野**: 12 分野 × 難度 3 層 × ~14 問/セル ≒ 504 問。
- **スコープ**: **Wikipedia 2023-11 断面で正解が固定できるもの限定**（`configs/data/corpus.yaml` の `wikipedia_ja` = `20231101.ja` に一致）。時事（2024 以降）は対象外。
- **分割**: `train.jsonl` (K3 事実性 DPO の on-policy 用) と `eval.jsonl` (K1/K2 評価用) を最初から物理分離。両者は domain × difficulty 分布を揃える。分割は問題 ID を SHA1(id) % 5 で mod 分割し 1/5 を train、残り 4/5 を eval に。

## JSONL 形式（1 行 1 問）

```jsonc
{
  "id": "jkb-geo-core-001",             // 一意 ID。<slug>-<domain-code>-<difficulty>-<seq> 形式
  "domain": "地理",                      // 12 分野のいずれか
  "difficulty": "core",                 // core | standard | advanced
  "format": "short_answer",             // short_answer | mcq
  "prompt": "日本で一番高い山は何ですか?",  // 質問文（1-shot fewshot は runner 側で付与）
  "answers": ["富士山", "富士"],          // 短答: substring 一致で正答扱いする候補リスト
  "choices": null,                      // mcq のみ: [{ "label": "A", "text": "..." }, ...]
  "correct_choice": null,               // mcq のみ: "A" 等
  "source_url": "https://ja.wikipedia.org/wiki/富士山",  // 正解検証に使った Wikipedia-ja 記事
  "source_quote": "富士山（ふじさん）は、日本の...標高3,776m..." // URL 中の裏取り引用（≤80文字）
}
```

- **short_answer**: `answers` に完全一致でなく substring 候補を並べる（表記ゆれ吸収: 富士山 / 富士、家康 / 徳川家康 等）。
- **mcq**: 採点頑健性のためのバックアップ形式。難問（数字比較・年号一発など substring が効きにくい問）で採用。
- 追加禁止フィールド: 個人名の生年月日詳細、住所、電話番号、政治的意見。

## 12 分野

| コード | 分野 | 例 |
|---|---|---|
| geo | 地理 | 山岳・河川・湖沼・気候・地形・都道府県境 |
| hist | 歴史 | 事件・人物・年号・時代区分・政変 |
| lit | 文学 | 作品・作者・時代・文学賞・書き出し |
| food | 食文化 | 料理・食材・調味料・地域名産・作法 |
| trad | 伝統文化 | 芸能・工芸・祭事・武道・年中行事 |
| pol | 政治・制度 | 内閣・国会・地方自治・司法・憲法 |
| life | 生活・慣習 | 祝日・行政手続き・冠婚葬祭・マナー |
| region | 地域・観光 | 世界遺産・国立公園・温泉・名所 |
| sport | スポーツ | プロ野球・大相撲・柔道・剣道・オリンピック |
| sci | 科学技術・産業 | 発明・企業・研究・ノーベル賞・宇宙 |
| relig | 宗教・信仰 | 神道・仏教宗派・寺社・神話 |
| lang | 言語 | 表記体系・敬語・詩型・古文・書道 |

## 難度 3 層

- **core**（コア常識）: 日本人ならほぼ全員知る（都道府県、代表的歴史人物、祝日、寿司、富士山、内閣総理大臣の呼称レベル）。base モデルでも 60%+ 期待。
- **standard**（標準）: 義務教育レベル（内閣制度の役割、伝統行事の由来、地方の代表産業、有名文学作品）。base モデルで 30-50% 期待。
- **advanced**（発展）: 専門・地域固有（旧国名 → 現県名、地方行政区分の細目、特定地域の民俗、専門用語）。base モデルで 5-25% 期待。

**基準**: 「そのモデルが日本語話者として通用する必要最低ライン = core が概ね正答」「日本の中学生程度の教養 = standard が概ね正答」。K1 の目標水準はこの実測後に固定（測定第一の原則）。

## 著述方針（オーケストレーター駆動）

**Fable5 委任は不使用。設計・著述の判断はオーケストレーター（Claude 本体）が担い、実装（scorer/runner/tests）と個別バッチの下書き（sonnet5 サブエージェント）を委任する。**

- **候補問生成フロー**:
  1. domain × difficulty のマトリクス（12 × 3 × ~14）を先に固定
  2. 各セルにつきオーケストレーターが Wikipedia-ja のシード記事群 URL を選定
  3. sonnet5 サブエージェントに「1 セル分 (14 問)」を構造化フレームワーク（URL 群 + 例題 2 問 + 難度定義 + 出力スキーマ）で依頼
  4. 全件をオーケストレーターがレビュー（answers 候補の抜け、URL の妥当性、difficulty 判定）
  5. Wikipedia-ja 本文で正解検証（`source_quote` に引用を保存し、URL と一緒に記録）

- **禁則**: 単純な `sonnet5 に 500 問生成させる` は禁止（品質担保不能）。1 バッチ ≤ 20 問、必ずセルごと（domain × difficulty）に分離してフレームワーク駆動。

## 汚染ガード

- 既存 `src/lfm25_ja/data/clean.py::ngram_contamination_checker`（n=13 の文字 n-gram）を CPT 学習コーパスとの重複除去に流用。
- `prompt + " " + best_answer` を JKB 側の n-gram プールに投入し、CPT の各コーパス（wikipedia_ja / cc100_ja / aozora）と重複率を測定。閾値超過（≥0.5 相当）の問題は書き換え候補としてマーク。
- train/eval 分離は最初から。K3 の on-policy DPO 生成でも eval に触れないよう物理分離（別ファイル・別ローダ）。

## 採点

- **一次**: substring / normalized-answer 一致（`japan_probe.py::extract_answer_segment` を再利用し、モデルの最初の回答セグメントを切り出してから `answers` 候補と照合）。
- **二次**: Swallow 8B judge（`src/lfm25_ja/eval/judge_swallow.py` 転用、事実正誤判定用にプロンプト差替）で自由記述の正誤を判定。
- **判定ルール**: 一次 pass → OK / 一次 fail かつ二次 pass → 手動再確認 (`needs_review`) / 一次・二次 fail → NG。
- **集計**: (domain, difficulty) の 36 セル × モデルで正答率を出す。K1 で cpt-B / dpo-001-b005 / sft005-distill と並べる。

## MCQ 形式の採点

- MCQ の場合、モデル出力から `_MCQ_ANSWER_RE = re.compile(r"^[\s（(]*([A-E])[\s)）:：.]*")` で先頭のラベルを抽出。マッチしなければ後方に「答え: <label>」パターンを探索。それでもなければ NG。
- MCQ 難問例: 「日本国憲法が施行された西暦は次のうちどれか? A: 1945 B: 1946 C: 1947 D: 1948」→ substring では 1945 と 1947 を両方拾うため、MCQ 化して混同を避ける。

## 統計誤差

- N=500 全体で 1 セル ≒ 14 問。標準誤差 ±13%pt/セル（Wilson 95%）と非常にラフ。集計は全体 (N=500 → ±4pt) と domain 別 (N≒42 → ±15pt) までを主に。cell 別は「傾向表示」用にとどめる。
- K1 レポートでは全体と domain 別に統計誤差を必ず併記する（sft-004 レビューでの Fable5 指摘の踏襲）。

## 参照

- 種: `src/lfm25_ja/eval/japan_probe.py` (10 分野 × 5 問 = 50 問。core 相当を大半含む。JKB v1 に組み込み)
- 汚染検査: `src/lfm25_ja/data/clean.py::ngram_contamination_checker` (n=13)
- Judge 転用元: `src/lfm25_ja/eval/judge_swallow.py`
- 親 Epic: #120（新方針: 日本語特化 = 日本知識のオフライン担保）
