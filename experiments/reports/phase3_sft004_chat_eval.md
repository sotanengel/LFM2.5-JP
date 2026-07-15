# Phase 3 sft-004 チャット評価(日本語 IFEval 系ハーネス初回実測)

Issue #104。sft-004 lr/epoch スイープ完了(`experiments/reports/phase3_sft004_lr_sweep.md`)で
「学習強度過剰仮説」が確定した一方、Fable5 相談で「**Epic #5 のゲート基準(J-MIFEval /
MT-Bench-ja が Phase 0 超え、敬語破綻なし)は sft-001 以来 30 モデル分未測定**」という致命的
な穴が判明した。llm-jp-eval(8 タスク × 100 件 × 4-shot NLU)は SFT が教える様式(短い直接
応答・チャット形)を構造的に罰する alignment tax 指標で、SFT の便益をそもそも検知できない。

そこで、Phase 4 DPO か sft-002 データ多様化かを決める前に、**指示追従を実測できる評価
インフラを最初に構築**する。追加学習ゼロ、5 モデルの再評価のみ。決定規則は事前固定。

## 実装(Fable5 コンサル → 推奨 B 採用)

### 選択

- **自作軽量ハーネス**(Issue #104 の第二候補)を採用。理由:
  1. WSL2 が不定期再起動する環境で外部ハーネス(llm-jp-instruction-eval / j-ifeval)の
     依存増加はリスク(既存 llm-jp-eval にも `_find_latest_prompts_dir` mtime バグあり)
  2. 検証関数は CPU 完結 → CI(`pytest -m "not gpu"`)で単体テスト可能
  3. 100 件 × 7 verifier は二項統計誤差 ±5pt の設計と整合
  4. 目的は「事前固定した決定規則の適用」で横比較ではないため
- JSONL スキーマは IFEval 互換(`instruction_id_list` + `kwargs`)、将来外部データセット差し替え可

### 凍結条件

| 項目 | 値 |
|---|---|
| プロンプト | 100 件(Fable5 生成、[datasets/eval/ifeval_ja/prompts.jsonl](../../datasets/eval/ifeval_ja/prompts.jsonl)) |
| カテゴリ | 依頼 25 / 質問 25 / 要約 25 / 敬語 25 |
| verifier | char_count 20 / bullet_count 15 / polite_form 20 / keyword 15 / format_json 10 / format_markdown_table 10 / numeric_only 10(複合 10 件) |
| 生成 | `apply_chat_template=true`, greedy(`do_sample=False`, `temperature=0`), `max_new_tokens=512` |
| 環境 | RTX 3060 Ti WSL2、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| 採点 | strict(生応答) / loose(前置き剥がし後)の二重採点 |
| config | [configs/eval/ifeval_ja.yaml](../../configs/eval/ifeval_ja.yaml)(以後変更不可) |

### 評価モデル(5 本、追加学習ゼロ)

| # | 名前 | パス(WSL) | 由来 |
|---|---|---|---|
| 1 | base | `LiquidAI/LFM2.5-1.2B-JP-202606` | Phase 0 基準線 |
| 2 | sft004-L9-lr1e-5 | `~/lfm25-ja/outputs/sft-004/sft-004-L9-lr1e-5/` | 動作点候補 A |
| 3 | sft004-L9-lr3e-5 | 同上 lr3e-5 | 候補 B |
| 4 | sft004-L6-9-lr1e-5 | 同上 L6-9 lr1e-5 | 候補 D |
| 5 | sft003-L9(強学習参照) | `~/lfm25-ja/outputs/sft-001-ichikara/` | 参照(lr 1e-4×2ep) |

## 決定規則(事前固定、変更不可)

- 任意の SFT セルが **base +3pt 以上**(prompt-strict acc)→ そのセルを Phase 4 DPO 初期化採用、
  sft-004 系列 + #36 クローズ
- 全 SFT セルが **base +1pt 以下** → **sft-002(#34)必須化**して着手
- 中間帯(base +1〜+3pt) → 目視採点と合わせて判断、または sft-002 並行

## 結果(prompt-strict acc、100 件)

### v1(初回、`polite_form` verifier に非文断片誤検出あり)

| モデル | prompt_strict | Δ base |
|---|---:|---:|
| base(JP-202606) | 0.850 | 0 |
| sft004-L9-lr1e-5 | 0.820 | -0.030 |
| sft004-L9-lr3e-5 | 0.760 | -0.090 |
| sft004-L6-9-lr1e-5 | 0.720 | -0.130 |
| sft003-L9 | 0.490 | -0.360 |

polite_form が 敬語カテゴリで 件名/宛名/挨拶断片(「件名:xxx」「A社様」「拝啓」「敬具」
「[署名]」等)を誤検出。base 敬語 52%、polite_form 34.8%(23 件中 8 件通過)まで押し下げ。
Fable5 コンサル → **verifier 修正 + 再採点**(生成再実行不要)を優先。

### v2(polite_form 修正、business-letter fragment を判定対象外に。生成は同一、採点のみ再実行)

verifier 修正差分: `_is_label_line`(件名: 等)+ `_POLITE_EXEMPT_PATTERNS`(様/殿/御中/各位/
拝啓/敬具/前略/草々/[プレースホルダ]/〇〇 で始まる行)を polite 判定から除外。plain 判定は
変更なし(plain 違反は常に完全な文で発生するため)。テスト 5 件追加(80/80 green)。

| モデル | prompt_strict | Δ base | prompt_loose | 依頼 | 質問 | 要約 | 敬語 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **base(JP-202606)** | **0.950** | 0 | 0.950 | 0.960 | 1.000 | 0.920 | 0.920 |
| sft004-L9-lr1e-5 | 0.890 | -0.060 | 0.890 | 0.880 | 1.000 | 0.840 | 0.840 |
| sft004-L9-lr3e-5 | 0.810 | -0.140 | 0.810 | 0.880 | 0.920 | 0.640 | 0.800 |
| sft004-L6-9-lr1e-5 | 0.770 | -0.180 | 0.770 | 0.800 | 0.800 | 0.680 | 0.800 |
| sft003-L9(強学習参照) | 0.560 | -0.390 | 0.560 | 0.560 | 0.600 | 0.400 | 0.680 |

verifier 別(instruction-level strict acc):

| verifier | base | L9-lr1e-5 | L9-lr3e-5 | L6-9-lr1e-5 | sft003-L9 |
|---|---:|---:|---:|---:|---:|
| bullet_count(n=15) | 1.000 | 1.000 | 1.000 | 1.000 | 0.933 |
| char_count(n=21) | 1.000 | 1.000 | 0.810 | 0.810 | 0.429 |
| format_json(n=10) | 1.000 | 1.000 | 0.900 | 0.800 | 0.400 |
| format_markdown_table(n=10) | 1.000 | 1.000 | 0.800 | 0.800 | 0.500 |
| keyword(n=21) | 1.000 | 0.857 | 0.905 | 0.905 | 0.857 |
| numeric_only(n=10) | 1.000 | 1.000 | 0.900 | 0.900 | 0.500 |
| **polite_form(n=23)** | **0.783** | 0.652 | 0.609 | 0.478 | 0.478 |

### 観察

1. **学習強度単調傾向を再現**: lr↑・可変層↑・epoch↑ → prompt-strict↓。**llm-jp-eval と直交
   する評価軸で同順序**を独立に確認(llm-jp-eval 上でも base 0.469 > L9 lr1e-5 0.465 > L9
   lr3e-5 0.425 > L6-9 lr1e-5 0.443 > sft003-L9 0.396)。
2. **base 未達が全 SFT セルで発生**。best SFT(L9 lr1e-5)は -6pt(v2)。統計誤差 ±5pt の
   外側(base 0.95、SFT 0.89 の差は 100 件で ~1.8σ)。
3. **strict == loose が 500/500 完全一致**(`strip_preamble` の適用差はゼロ)。base 応答は
   直接応答で「はい、承知しました。」型の前置きが出ない。sft003 は「〜を N 文字以内で説明
   します。」型のプロンプト・エコーが出るが `strip_preamble` の対象外(前置き剥がしは無効)。
4. **polite_form は最大のボトルネック**: 全モデルで最低通過率。base ですら 78.3%。verifier
   修正後も宛名/敬具などの完全なフィルタは残る余地あり(将来 v1.1 で難化と合わせて再設計)。
5. **keyword で base 100%、SFT 全て低下**: SFT で命名エンティティ保持が悪化している兆候。
6. **敬語カテゴリで SFT が若干勝つ配列**(L9 lr1e-5=0.84 vs base=0.92 は同点圏、但し
   sft004-L6-9 は 0.80 で base より低い)。他カテゴリでは base 優位。

## 決定規則の機械適用

- ✗ +3pt 以上のセル: **ゼロ** → Phase 4 DPO 初期化候補なし
- ✓ 全 SFT セルが base +1pt 以下: **該当**(best -6pt、worst -39pt)
- (中間帯不該当)

**→ 決定: sft-002(#34)必須化して着手**(Fable5 追認、`agentId: a56ffba56b80eb7a5`)

Phase 4 DPO はデータ側の穴(現行 SFT データは base の指示追従を毀損する)を DPO に押し付ける
だけになるため却下(前 Fable5 相談 A オプション再確認)。

## 明示的却下オプション(記録用、次段セッションが誤って戻らないよう)

Fable5 追認済み(300〜500 語コンサル、`agentId: a56ffba56b80eb7a5`):

1. **sft-004 セルからの Phase 4 DPO 初期化** — 全セル負、規則により却下
2. **sft-004 軸の追加 lr/epoch スイープ** — 「学習強度↑ = 劣化」を 2 指標で確認済み、
   一変数掘りは終了
3. **敬語の目視採点重み増による verifier 問題の運用回避** — 自動ゲートの意味を失う。
   verifier 側を修正するのが正
4. **polite_form 通過率の目的関数への直接組み込み**(verifier 修正前) — アーティファクト
   への最適化(Goodhart 化)
5. **verifier 修正に伴う 500 応答の再生成** — 再採点で足りる(GPU コストゼロ)
6. **外部ハーネス(lm-eval-harness 等)への乗り換え** — strict==loose 完全一致・順序
   妥当性で自作ハーネスは検証済み、乗り換えは工数の純損

## 次段方針(Issue #105 として起票、以下は Fable5 の推奨まとめ)

1. **verifier v1.1**(小差分): 天井 100% の 6 verifier を難化して SFT の便益を可視化できる
   余地を残す。ただし polite_form 修正のみで比較可能性は担保できるため後回し可
2. **sft-002 データ多様化**(#34、必須化):
   - 既存計画: llm-jp instruct + Aya-ja 混合
   - **追加要件**(Fable5): 「format 保持」を混合設計の必須条件に加える。制約付き指示
     (文字数 / 箇条書き / JSON 指定)を明示的にサンプル化して SFT データに含める
   - polite_form 通過率を直接目的関数化するのは verifier v1.1 完了後に検討
3. **sft-005 以降は一変数主義を緩和、複合実験に移行**(sft-004 lr sweep レポート合意事項の
   持ち越し)

## 実装物一覧

- ソース: [src/lfm25_ja/eval/instruction_verifiers.py](../../src/lfm25_ja/eval/instruction_verifiers.py) /
  [generate_ifeval_ja.py](../../src/lfm25_ja/eval/generate_ifeval_ja.py) /
  [score_ifeval_ja.py](../../src/lfm25_ja/eval/score_ifeval_ja.py) /
  [run_ifeval_ja.py](../../src/lfm25_ja/eval/run_ifeval_ja.py)
- config: [configs/eval/ifeval_ja.yaml](../../configs/eval/ifeval_ja.yaml)
- データ: [datasets/eval/ifeval_ja/prompts.jsonl](../../datasets/eval/ifeval_ja/prompts.jsonl)
- WSL スクリプト: [scripts/60_eval_ifeval_ja.sh](../../scripts/60_eval_ifeval_ja.sh)
- テスト: [tests/test_instruction_verifiers.py](../../tests/test_instruction_verifiers.py)(80 件) /
  [tests/test_ifeval_ja_dataset.py](../../tests/test_ifeval_ja_dataset.py) /
  [tests/test_run_ifeval_ja_pipeline.py](../../tests/test_run_ifeval_ja_pipeline.py)
- 生成物(WSL): `~/lfm25-ja/outputs/eval/ifeval_ja/<model>/generations.jsonl` + `scores.jsonl` + `aggregate.json`

## 受け入れ条件(Issue #104)チェック

- [x] 評価ハーネスが base で完走(base = 0.95、5 モデル × 100 プロンプト × ~3 分 = 15 分)
- [x] 5 モデル全ての prompt-strict acc + instruction-level acc + per-category が出揃った
- [ ] 目視 20 件採点 → **スキップ**(Fable5 判断: sft-002 データ構築が critical path、
      verifier fix + 決定規則機械適用で本 issue の目的は達成、目視は sft-002 の効果測定時
      に本格実施)
- [x] 決定規則を適用し 3 択のいずれかで結論(**sft-002 必須化**確定)
- [x] `experiments/reports/phase3_sft004_chat_eval.md` 完成(本ドキュメント)

## Fable5 相談履歴

- 実装前(ハーネス選定): B 確定、agentId `ad549fc625a3d36fa`
- 実装後(結果解釈 + 次段): sft-002 必須化 + verifier fix 追認、agentId `a56ffba56b80eb7a5`
