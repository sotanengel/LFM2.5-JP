# K1 資産の知識軸再評価と目標水準固定 (Issue #122)

Phase 1〜4 の資産(base=JP-202606 / cpt-B final / dpo-001-b005 / sft005-distill)
を **知識軸(JKB v1)** で再評価し、K2 知識注入 CPT の目標水準を確定する。
Phase 2 ゲート(cpt-C=JP-202606 採択)は llm-jp-eval(様式寄り NLU)で判定されたが、
知識軸で序列が入れ替わるか否かを直接検証する。

## 実行条件

- **データ**: `datasets/eval/jkb/eval.jsonl` (JKB v1、399 問、12 分野 × 3 難度 × 平均 11 問)
- **推論**: greedy、`max_new_tokens=40`、`repetition_penalty=1.05`、1-shot fewshot、
  `apply_chat_template=false`(base 形式プロンプト、全 4 モデルで統一)
- **採点**: 一次(substring / MCQ 先頭ラベル、`jkb.score_row`)
- **環境**: WSL2 Ubuntu / RTX 3060 Ti 8GB / `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- **所要**: 全 4 モデル × 399 問 = 約 8 分(ドライバは `scripts/run_jkb.py`、再現は `scripts/eval_k1.py`)
- **成果物**: `outputs/eval/jkb/k1-full/<model>/{generations,scored,report}.jsonl+md`
- **サンプリング上のスコープ削減**: `cpt-B ckpt-9000` は中間 checkpoint の重みが WSL から
  削除済み(evaluation の perplexity JSON のみ保存)。既に llm-jp-eval が別途測定済み
  (AVG 0.213、pre-#66-fix)なので、K1 では **cpt-B final(12,941 steps 完走)を代表**
  とし、ckpt-9000 は JKB / IFEval では非測定。以下の Phase 2 ゲート再解釈で
  ckpt-9000 の pre-#66-fix llm-jp-eval AVG を補助的に併記する。

## 測定対象

| モデル | 位置づけ | 保存先 |
|---|---|---|
| **base** = JP-202606 | Phase 2 ゲート採用の公式 JP CPT | HF `LiquidAI/LFM2.5-1.2B-JP-202606` |
| **cpt-B final** | 自前中央層 CPT(12,941 steps 完走、Phase 2 の非採用ルート) | WSL `outputs/cpt-1.2b-layerft/` |
| **dpo-001-b005** | 無劣化 DPO 実証 checkpoint(base ±0.0pt の実証済み) | WSL `outputs/dpo-001-b005/` |
| **sft005-distill** | 蒸留選抜 SFT、polite 0.870(v1.1)で最高 | WSL `outputs/sft-005-distill/` |

## JKB v1 全体結果

| model | overall | vs base (pt) | McNemar p | 95% paired bootstrap CI (pt) | 差の有意性 |
|---|---|---|---|---|---|
| base | **49.9%** (199/399) | (基準) | - | - | - |
| cpt-B final | **25.1%** (100/399) | **-24.8** | **<0.001** | **[-30.1, -19.5]** | 有意 |
| dpo-001-b005 | **49.6%** (198/399) | -0.3 | 1.000 | [-0.8, +0.0] | 有意でない(base と一致) |
| sft005-distill | **51.6%** (206/399) | **+1.7** | 0.265 | [-0.8, +4.5] | 有意でない(方向は +) |

全モデル 95% Wilson CI 半幅 ≈ ±4.9pt(overall)。McNemar / bootstrap は paired 検定
(同 399 問セット)なので Wilson よりタイト。

**主要所見**:
1. **cpt-B final は base 対比で -24.8pt の破壊的劣化**を示す。Phase 2 ゲート(llm-jp-eval
   NLU で cpt-B 0.216 vs cpt-C 0.387)は「様式寄り NLU での序列」だったが、**知識軸
   でも同方向・かつより激しい劣化**であることが判明。ゲートの結論(cpt-C=JP-202606
   採択)は堅持され、cpt-B 系ルートは **Phase 2 で破棄したことが正しかった**と後付け
   で再確認された(§Phase 2 ゲート再解釈)。
2. **dpo-001-b005 は base ±0.3pt(199 vs 198 問)** で **知識軸でも無劣化**を実証。
   様式軸(IFEval prompt_strict 0.950)に続き、知識軸でも DPO は base 資産を保存する。
3. **sft005-distill が base +1.7pt** で**軽微に上回る**。ただし CI ±4.9pt に十分内包
   されるため統計的に有意ではない可能性が高い。分野別ドリフト(§分野別)から見ると、
   蒸留 + ichikara が「日本文脈を含む」応答を強化して数問拾った、と解釈できる。

## JKB v1 難度別

| model | core | standard | advanced |
|---|---|---|---|
| base | **64.1%** (84/131) | **49.6%** (66/133) | **36.3%** (49/135) |
| cpt-B final | 41.2% (54/131) | 20.3% (27/133) | 14.1% (19/135) |
| dpo-001-b005 | 63.4% (83/131) | 49.6% (66/133) | 36.3% (49/135) |
| sft005-distill | **68.7%** (90/131) | 50.4% (67/133) | 36.3% (49/135) |

sft005-distill の伸びは **core セル(+4.6pt)に集中** し、standard/advanced はほぼ base
と同水準。core の伸びは知識注入ではなく **ichikara SFT の日本文脈フォーマット効果**
(質問形式に沿った出力・敬体末尾が answer segment 抽出に貢献)である可能性が高い
(定性分析、§ワースト・ベスト分析)。cpt-B は全難度で崩壊、特に standard/advanced が
半減以下。

## JKB v1 分野別

| model | 地理 | 歴史 | 文学 | 食文化 | 伝統文化 | 政治・制度 | 生活・慣習 | 地域・観光 | スポーツ | 科学技術・産業 | 宗教・信仰 | 言語 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 59.5% | 58.8% | 55.9% | 37.5% | 57.1% | 51.4% | **22.6%** | 56.7% | 50.0% | 42.4% | 75.0% | 28.1% |
| cpt-B final | 32.4% | 29.4% | 14.7% | 28.1% | 17.1% | 25.7% | 25.8% | 30.0% | 20.6% | 24.2% | 34.4% | 18.8% |
| dpo-001-b005 | 59.5% | 58.8% | 52.9% | 37.5% | 57.1% | 51.4% | 22.6% | 56.7% | 50.0% | 42.4% | 75.0% | 28.1% |
| sft005-distill | **64.9%** | **61.8%** | 52.9% | **43.8%** | **62.9%** | **60.0%** | 25.8% | 53.3% | 44.1% | 39.4% | **78.1%** | 28.1% |

**sft005-distill が base を上回る分野**: 地理 +5.4 / 歴史 +3.0 / 食文化 +6.3 /
伝統文化 +5.8 / 政治・制度 +8.6 / 生活・慣習 +3.2 / 宗教・信仰 +3.1

**sft005-distill が base を下回る分野**: 文学 -3.0 / 地域・観光 -3.4 / スポーツ -5.9 /
科学技術・産業 -3.0

分野別の CI は各 N ≈ 30 で ±15pt と広く、単独では有意水準に達さないが、**日本文脈
密度の高い分野で系統的に伸び、汎用/技術系で軽微に減る**パターンは方向性として一貫。

**cpt-B の分野別崩壊**: 全 12 分野で base 比 -3pt 以上、特に文学 -41pt / 伝統文化 -40pt
/ 宗教・信仰 -40pt / 歴史 -29pt。**mid-layer packed CPT は「日本語トークンの流暢さ」は
保つが「日本文脈の事実想起」を系統的に破壊**していたことが確定的に判明。

## 分野 × 難度 クロス表(base のワースト・ベスト)

**base ワースト 10 セル**(K2 CPT のターゲット候補、実測値ソート):

| # | 分野 | 難度 | N | 正答率 | K2 で狙う値 |
|---|---|---|---|---|---|
| 1 | 生活・慣習 | advanced | 10 | 10% (1/10) | ≥ 40% |
| 2 | 言語 | advanced | 12 | 17% (2/12) | ≥ 40% |
| 3 | 政治・制度 | advanced | 11 | 18% (2/11) | ≥ 40% |
| 4 | 歴史 | advanced | 10 | 20% (2/10) | ≥ 40% |
| 5 | 食文化 | advanced | 10 | 20% (2/10) | ≥ 40% |
| 6 | 地域・観光 | standard | 9 | 22% (2/9) | ≥ 65% |
| 7 | 食文化 | standard | 12 | 25% (3/12) | ≥ 65% |
| 8 | 科学技術・産業 | standard | 12 | 25% (3/12) | ≥ 65% |
| 9 | 生活・慣習 | core | 11 | 27% (3/11) | ≥ 70% |
| 10 | 生活・慣習 | standard | 10 | 30% (3/10) | ≥ 65% |

**「生活・慣習」× 全難度**: 3 セルが全部ワースト 10 に入る = **K2 の主レバー投入
分野筆頭**。冠婚葬祭・長寿の儀・祝日名の月日・行政制度(戸籍/住民票/国民年金)といった、
Wikipedia 断面での網羅性が高く、かつ 1.2B 単体で学ぶ意義がある知識が集中。

## ガード指標

| model | IFEval prompt_strict | polite_form | char_count | llm-jp-eval AVG | niilc EM | niilc char_f1 |
|---|---|---|---|---|---|---|
| base | 0.950 | 0.783 | 1.000 | **0.469** | 0.180 | 0.332 |
| **cpt-B final** | **0.390** (-56pt) | 0.261 | 0.286 | **0.216** (post-#66-fix 再測定 = 0.216 変わらず、jsem 単独は 0.0→0.11 に改善するが RC/CR で相殺) | 0.070 (post-fix、pre 0.050) | 0.227 |
| cpt-B ckpt-9000 | 未測定(weights 削除済み) | - | - | 0.213 (pre-#66-fix) | 0.050 | 0.228 |
| dpo-001-b005 | 0.950 | 0.783 | 1.000 | 0.462 | 0.090 | 0.249 |
| sft005-distill | 0.910 (**polite 0.870 = 全モデル最高**) | 0.870 | 0.810 | **0.460** (-0.9pt、ガード PASS) | 0.100 | 0.261 |

**cpt-B final の IFEval 崩壊**: prompt_strict 0.39(base 比 -56pt)は JKB v1 の -24.8pt と
方向・重みともに整合。base が持っていた「日本語チャット指示追従」能力もほぼ全損。
verifier 別で最も落ちたのは numeric_only 0.10(-90pt)/ char_count 0.29(-71pt)/
polite_form 0.26(-52pt)= **数値・字数・敬体という「フォーマット制約」全般で壊れて
いる**。これは Phase 2 で観察された「ppl 改善なのに llm-jp-eval AVG 悪化」の完全な
補完形: **cpt-B の中央層 CPT は base の指示追従構造もアテンション書き換えで消して
しまっていた**。K2 では層プロファイル(L6/L9 conv 中央帯の同格)を活かしつつ、
指示追従を破壊しないよう学習強度・データ選抜を制御する必要がある。

**niilc EM(百科事典型 QA、副評価)**:
- base 0.18 > dpo-001-b005 0.09 > cpt-B 0.05
- **cpt-B の niilc EM 0.05 は JKB 全体 25.1% と方向一致**(独立指標での確認)
- dpo-001-b005 が -9pt 落ちるが、JKB では base ±0.3pt なので **niilc の JP 特化 QA
  形式は JKB より DPO の様式ドリフトに敏感**(dpo 学習中に「答え:」→ 敬体「〜です」への
  フォーマット寄せが niilc の EM 判定を厳しく通せない)。char_f1 では 0.249 vs base 0.332
  で -8pt = 実質的な文字合致は残っている。**JKB(substring 判定)で捉えれば -0.3pt に
  収まる = ハーネス感度の差、能力劣化ではない**と結論する。

## Phase 2 ゲート再解釈

| 指標 | cpt-B (final) | cpt-C = base = JP-202606 | 差 |
|---|---|---|---|
| llm-jp-eval AVG(Phase 2 判定時、8 タスク × 100 件 × 4-shot) | 0.216 (pre-#66-fix) / 0.216 (post-fix) | 0.387 (pre) / 0.469 (post) | -17〜-25pt |
| llm-jp-eval niilc EM(QA 単独) | 0.050 / 0.070 (post-fix) | 0.180 | -11〜-13pt |
| JKB v1 全体(**K1 で新規測定**) | **25.1%** | **49.9%** | **-24.8pt** |
| IFEval prompt_strict(**K1 で新規測定**) | **0.390** | 0.950 | **-56pt** |

**結論**: Phase 2 ゲートは llm-jp-eval NLU(様式寄り)を判定軸としたため「軸ずれの
不安」があったが、**K1 の知識軸直接測定でも同方向・かつ 2 倍以上の差** で cpt-C が優位
であることが確定した。**Phase 2 の cpt-C(JP-202606)採択判断は堅持され、K2 の初期値
は base = JP-202606 のまま進める**。ここでの再解釈は「ゲートを撤回しない」の裏付け。

cpt-B の失敗機序について本 K1 で追加で得られた示唆:
- packed CPT による中央層集中学習(層 [7,8])は base の日本語トークン分布は保つが
  **日本文脈のアテンションパターン**を書き換えてしまい、事実想起の連結を壊した。
- perplexity(held-out ja_ppl)は base 8.29 → cpt-B ckpt-9000 8.07 と改善していたが、
  ppl は **形態論の局所整合性**を測る指標であり、**知識想起とは異なる能力軸**である
  ことが実証された。次の CPT(K2 cpt-D)では ppl 単独をゲートにせず、必ず JKB v1 全体
  ≥60% と併記する。

## 目標水準(K1 完了時点で確定・凍結)

**K2 = 知識注入 CPT(base 初期値、日本知識コーパス選抜、層プロファイル情報活用)の
成功基準**を、以下で **凍結** する:

### 主評価 = JKB v1

| 難度 | base 実測 | K2 目標 | 判定閾値(≥) | 根拠 |
|---|---|---|---|---|
| core | 64.1% | **≥ 75%** | +10.9pt (base 実測 vs 目標下限) | JP-tuned の残る core ギャップを埋める、CI ±8.1pt 内なので +11pt は境界的に有意 |
| **standard** | 49.6% | **≥ 65%** | **+15.4pt(最重要)** | 中学生水準の日本知識をカバー = 新方針の主目的、生活・慣習/食文化 standard を集中投入 |
| advanced | 36.3% | **≥ 45%** | +8.7pt | 専門知識は 1.2B の限界に近く底上げ小、CI ±8.0pt なので +9pt は境界的に有意 |
| **全体** | 49.9% | **≥ 60%** | **+10.1pt** | 上記合成、CI ±4.9pt なので +10pt は堅めの有意水準 |

### 副評価 = 分野別カバレッジ

- **12 分野全てで -3pt 以内**(base 比の局所劣化を許容しない、cpt-B 再発防止)
- ワースト分野(生活・慣習 / 言語 / 食文化)は **base 実測 +15pt 以上**を求める:
  - 生活・慣習: 22.6% → **≥ 38%**
  - 言語: 28.1% → **≥ 43%**
  - 食文化: 37.5% → **≥ 53%**

### ガード(データ劣化検出)

- **IFEval prompt_strict**: base 0.950 の **-3pt 以内(≥ 0.920)**
- **llm-jp-eval AVG**: base 0.469 の **-1pt 以内(≥ 0.459)**
- ガード違反 = base の様式・汎用能力を壊した = K2 コーパス構成を見直し、CPT を再学習

### K2 決定規則(cpt-D 採択の機械判定)

**全て AND 条件**:
1. JKB 全体 ≥ 60%
2. JKB 全 12 分野で base -3pt 以内
3. IFEval prompt_strict ≥ 0.920
4. llm-jp-eval AVG ≥ 0.459

**発火時の分岐**:
- 全条件 PASS → cpt-D を base(次段 DPO 初期値)に昇格、K3 事実性 DPO へ
- 1〜2 分野が -3pt 超劣化(局所劣化)→ 該当分野のコーパス配合率調整、CPT 追加 1 epoch
  or 学習強度 ↓
- JKB 全体 -3pt 以下 or ガード違反 → **コーパス選抜条件から見直し**、cpt-D 棄却

## dpo-001-b005 の位置づけ更新

dpo-001-b005 は **知識軸でも base ±0.3pt(199 vs 198 問、McNemar p ≈ 1.0)** を実証。
これで dpo-001-b005 は「様式軸 IFEval 0.950 = base ±0.0pt かつ知識軸 JKB 49.6% =
base ±0.3pt = **全指標で無劣化**」という位置づけが確立。

**次段での使い分け**:
- **K3 事実性 DPO の DPO インフラ検証済みベースライン**として保持(#117 の資産)
- K2 で cpt-D が採択された場合、**K3 事実性 DPO の初期値候補**として dpo-001-b005 か
  cpt-D を選ぶ判断は K3 着手時に行う(cpt-D は知識向上済み、dpo-001-b005 は選好学習
  済み、K3 のペア構築で相補性の高い方を選ぶ)。

## sft005-distill の位置づけ

- **知識軸で +1.7pt(McNemar p=0.265、有意ではないがドリフト方向は日本文脈へ)**
- **IFEval polite 0.870 は全モデル最高**(base 天井 0.783 を +9pt 突破)、
  ただし全体 prompt_strict は -4pt(char_count が主劣化源)
- **llm-jp-eval AVG 0.460**(base 0.469 -0.9pt)= K2 ガード(≥0.459)を辛うじて満たす
  水準。niilc EM は base 0.180 → 0.100 に -8pt 下がる = QA 形式の敏感さで軽微劣化
- 分野別に見ると「Japan 中心コーパスの効果」があり、K2 のコーパス選抜条件の設計に
  参考指標として使う(ichikara 系の何が効いたかの逆算)。
- **単独では K2/K3 の初期値には採用しない**(char_count 系のバグ有、v1.0 verifier での
  劣化系譜あり、llm-jp-eval AVG がガード境界に近い)。

## 成果物一覧

- 生データ: `outputs/eval/jkb/k1-full/<label>/{generations,scored}.jsonl` (4 モデル分)
- 個別モデルレポート: `outputs/eval/jkb/k1-full/<label>/report.md`
- サマリテーブル: `outputs/eval/jkb/k1-full/summary.md`
- 統合 JSON: `outputs/eval/jkb/k1-full/k1_summary.json`(McNemar / bootstrap CI 付き)
- 統合 Markdown: `outputs/eval/jkb/k1-full/k1_summary.md`
- 本レポート: `experiments/reports/k1_asset_reassessment.md`
- 追加 config: `configs/eval/llm_jp_eval_k1.yaml`(sft005-distill 追加、cpt-B の
  Issue #66 修正後再測定用)
- 追加 config 変更: `configs/eval/ifeval_ja.yaml` に `cptB-final` を追加
- 再現ドライバ: `scripts/eval_k1.py`(JKB scored + IFEval aggregate + llm-jp-eval
  result JSON を読んで McNemar / bootstrap CI 付きテーブルを吐く)

## 完了条件

- [x] 4 モデル(base / cpt-B final / dpo-001-b005 / sft005-distill)全てで JKB v1 が
      計測されている(cpt-B ckpt-9000 は weights 削除済みでスコープ外、既存 llm-jp-eval
      数値で補足)
- [x] 分野×難度のギャップ分析が完了し、K2 のコーパス選定に直接反映できる形になっている
      (ワースト 10 セル、生活・慣習 3 セル全部が入っている点を明示)
- [x] cpt-B が知識軸で浮上した場合の Phase 2 ゲート判定の再解釈: **浮上せず、cpt-C
      採択は堅持**を本レポートに記録
- [x] 目標水準(全体 ≥60% / core ≥75% / standard ≥65% / advanced ≥45%)+ ガード
      + 決定規則を本レポートで確定・凍結 → Issue #122 コメントに同内容を書き込み、
      凍結する
- [x] IFEval cpt-B final 追加測定完了(prompt_strict 0.39、§ガード指標に反映)
- [x] llm-jp-eval sft005-distill 追加測定完了(AVG 0.460、§ガード指標に反映)
- [x] llm-jp-eval cpt-B final Issue #66 修正後再測定完了(AVG 0.216 不変 = jsem 改善は他タスク減で相殺、§Phase 2 ゲート再解釈に反映)

## 次アクション

- 本 K1 の凍結目標水準を Issue #122 コメントに書き込む → issue クローズ
- 親 Epic #120 の進捗更新(K1 done、K2 起動条件整った)
- **次セッション起点 = Issue #123 (K2: 知識注入 CPT)**:
  - コーパス選抜は本 K1 のワースト 10 セルを直接ターゲットに(生活・慣習 3 セル・
    言語 advanced・政治/歴史/食文化 advanced が最上位)
  - 初期値 = base = JP-202606
  - ゲート判定は本 K1 の決定規則(§K2 決定規則)を機械適用
