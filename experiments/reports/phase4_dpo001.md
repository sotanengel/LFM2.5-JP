# Phase 4 dpo-001: rule+judge 選好ペア DPO と IFEval 評価(Issue #117)

- 実施日: 2026-07-18〜19
- 関連: Issue #115(設計・決定規則)/ #117(実行)/ PR #116(実装)/ 本レポートの PR
- 実行 commit: `b14d463`〜(branch `claude/issue-117-dpo-001-run`)

## TL;DR

**最良アーム dpo-001-b005 = IFEval prompt_strict 0.950 = base と完全同点(±0.0pt)。**
事前固定の決定規則では「±3pt → 微調整 1 回まで」が発火し、微調整(verifier v1.1 で
ペア再構築した dpo-001r-b005)も 0.940 で改善なし → **+3pt(≥0.98)未達で dpo-001 は
プロモーションなし、協議へ**。

ただし内実は Phase 3 SFT 全系列(-3〜-39pt)と対照的で、**DPO は一切の劣化なしに
base と同点**(全 verifier で base 同等、ホールドアウト無傷)。また評価の過程で
**凍結 verifier の敬体判定に 3 つの盲点(でした/ましょう/ませ)と署名行の
誤検出を発見・修正(v1.1)し、全 12 モデルを再採点**した。v1.1 では
**sft005-distill が polite_form 0.870 で全モデル最高**(base 天井 0.783 を大幅突破)
という新事実も判明し、次期方針の主材料になる。

## 1. 選好データ構築(Phase G/V/J/P)

### プール(P0、実行済みは前セッション)

5,461 プロンプト = Source A 3,988(蒸留 CSV、評価衝突 12 除外)+ Source B 853
(被覆漏れ: 小 N「以内」/「以上」型/min-max/常体指示)+ Source C 620
(joryu バンク層化、詳細: [phase4_dpo001_pool_stats.md](phase4_dpo001_pool_stats.md))。

### Phase G: base on-policy 生成(GPU)

- K=4 一括: 21,844 サンプル / 1h50m(実測 ~200 tok/s 実効、VRAM 2.3 GiB)
- **K=6 へ top-up**: +10,922 / ~55min(冪等 (prompt_id,k) resume の純増ラウンド)
- 計 32,766 サンプル

### Phase V: rule 判定(CPU、数秒)と収率の推移

| 判定条件 | rule_pass | ペア成立可能プロンプト |
|---|---:|---:|
| K=4、v1.0 verifier | 12,112 / 21,844 | 777 / 5,461 |
| K=6、fence 修正後 | 19,536 / 32,766 | 983 |
| K=6、**verifier v1.1** | 21,722 / 32,766 | **976** |

- **fence バグ**: format_json の detail チェックが ```json フェンスを剥がさず全 466
  プロンプトを全滅させていた(凍結 verifier のフェンス抽出に整合させ修正)。
  修正後も base は英語キー(name/classification)を使いがちで、日本語キー指定の
  遵守は依然低い(= 正しい rejected 供給源)
- カテゴリ別成立可能(v1.1): polite_form 397 / char_count 312 / no_constraint 248 /
  format_json 15 / その他 4。keyword・start_word・compound 等は base がほぼ全通過で
  ペア不成立(all-pass)

### Phase J: LLM judge(vLLM AWQ-INT4、8B)

- 対象 16,104 サンプル(ペア成立可能プロンプト全数 + polite/no_constraint の品質選別)
- **78 分で完走**(concurrency 8、~3.4 判定/s)。score null は 11 件のみ(0.07%)
- サーバ: `scripts/41_serve_judge_vllm.sh`(WSL ネイティブ、VRAM 7.2 GiB、フェーズ排他)

### Phase P: ペア確定

| 版 | ペア数 | 内訳(polite/char/open/json/他) | chosen 長ガード |
|---|---:|---|---|
| v0(旧 verifier) | 976 | 399/312/246/15/4 | mean 187.7 → PASS |
| **v1.1(採用)** | **970** | 393/312/246/15/4 | mean 192.2 → PASS |

v0 の polite ペアは「実は敬体の応答」(くださいませ/でした/ましょう/署名行)を
rejected に含む汚染信号だった。評価非重複ハードゲートは両版クリーン。
詳細: [phase4_dpo001_pairs_stats.md](phase4_dpo001_pairs_stats.md)。

## 2. DPO 学習(4 アーム)

L9 単層(67.1M、5.7%)/ lr 5e-6 / 1ep / batch1×accum4 / `--no-checkpoints` /
`precompute_ref_log_probs=True`(参照モデル非常駐)。各アーム **~4 分**、
VRAM allocated 2.2 / reserved 3.3 GiB。

| アーム | ペア | beta | DPO loss |
|---|---|---:|---|
| dpo-001-b005 | v0 976 | 0.05 | 0.697 → 0.632 |
| dpo-001-b01 | v0 976 | 0.1 | 0.695 → 0.586 |
| dpo-001-b03 | v0 976 | 0.3 | 0.697 → 0.460 |
| dpo-001r-b005(微調整) | **v1.1 970** | 0.05 | 0.691 → 0.662 |

## 3. verifier v1.1(測定補正、#104 の「再採点のみ」先例に準拠)

初回採点で dpo 3 アームが polite_form 0.652(-13pt)に見えたが、反転 3 件の目視で
すべて verifier の盲点と判明:

1. 「誠に申し訳ございません**でした**」→ でした(です の過去形)が敬体語尾リストに無い
2. 「歩んでまいり**ましょう**」→ ましょう(ます の意向形)が無い
3. 「幹事 〇〇」→ ひらがなを含まない署名行(述語なし)が免除されない

v1.1 = 敬体語尾に でした/ましょう/ませ を追加 + ひらがな無し行の免除。
**生成は不変、全 12 モデルを再採点**(全モデルに等しく適用、plain 判定は対称に厳格化)。

## 4. IFEval 結果(12 モデル、v1.1 採点)

| モデル | prompt_strict | Δ base | polite_form | 参考: v1.0 採点 |
|---|---:|---:|---:|---:|
| **base(JP-202606)** | **0.950** | 0 | 0.783 | 0.950 |
| **dpo-001-b005** | **0.950** | **±0.0** | 0.783 | 0.920 |
| dpo-001-b01 | 0.940 | -1.0 | 0.783 | 0.910 |
| dpo-001-b03 | 0.940 | -1.0 | 0.739 | 0.920 |
| dpo-001r-b005(微調整) | 0.940 | -1.0 | 0.783 | — |
| sft004-L9-lr1e-5 | 0.920 | -3.0 | 0.783 | 0.890 |
| **sft005-distill** | 0.910 | -4.0 | **0.870(全モデル最高)** | 0.900 |
| sft004-L9-lr3e-5 | 0.840 | -11.0 | 0.739 | 0.810 |
| sft002-mix | 0.790 | -16.0 | 0.739 | 0.770 |
| sft004-L6-9-lr1e-5 | 0.790 | -16.0 | 0.565 | 0.770 |
| sft003-L9 | 0.600 | -35.0 | 0.652 | 0.560 |

- dpo アームの verifier 別: char/bullet/keyword/format_json/ホールドアウト 2 種
  すべて base 同等(b005 は完全一致)。**DPO は何も壊していない**
- base 自身は v1.1 でも 0.950 のまま = base の 5 失敗は本物(polite 4 + keyword 系 1)
- **sft005-distill の polite 0.870** は旧 verifier が「くださいませ」文面を
  誤って減点していたことによる過小評価の是正(0.783 天井は実は突破されていた)

## 5. llm-jp-eval 回帰(dpo-001-b005、凍結 8 タスク×100 件×4-shot)

- base AVG 0.4693(row 014)に対し、dpo-001-b005 AVG = **0.4620(-0.73pt)**
  (判定基準: ≥ 0.459)→ **PASS(回帰なし)**
- タスク別デルタ: jmmlu +7.0pt / jsick +1.0pt / jcommonsenseqa・jsem ±0 /
  jnli -3.0pt / jsquad -1.0pt / jsts -5.2pt / niilc -8.4pt — n=100 の
  ノイズ帯(±5pt/タスク)内で双方向に散っており系統的劣化なし

## 6. 決定規則の機械適用(#115 で事前固定)

1. 最良アーム(b005)= base ±0.0pt → 「+3pt 以上」不成立、「±3pt」成立
2. 許容された微調整 1 回(dpo-001r: v1.1 ペアで再構築・再学習)を実施 → 0.940、改善なし
3. **結論: +3pt(≥0.98)未達。dpo-001 はプロモーションなし、次アクションは協議**
   (次期方針は Issue #118(仮)にまとめる)

統計註記: 100 件二項 ±5pt。b005 と base は 100 プロンプト中 95 で一致判定
(discordant 5/5 対称)であり、統計的に完全同等。

## 7. 考察(次期方針の材料)

1. **DPO の安全性は実証**: SFT 全系列が -3〜-39pt 劣化したのに対し、DPO は
   base 性能を完全保持。「壊さない」学習手段は確立した
2. **しかし +3pt の壁**: base 0.950 の残余 5 失敗は polite 系の難ケースに集中。
   970 ペア / 1ep / L9 の弱い介入では動かなかった(b005 の出力は base と
   ほぼ同分布)。beta を上げる(b03)と polite が悪化する兆候
3. **sft005-distill の再評価**: v1.1 で polite 0.870(base +8.7pt)が判明。
   総合 0.910 に留まるのは char_count 0.810 が原因。つまり
   「polite は sft005、char は base/dpo が強い」— 能力が相補的
4. **測定の天井**: IFEval 100 問で base 0.950。+3pt は残り 5 問中 3 問の反転を
   要求し、±5pt ノイズ帯の中の勝負になっている。評価の解像度(問題数・難度)
   自体を上げないと、これ以上の最適化は測定不能
5. verifier は「先に難化してから最適化する」が正順(sft-002/dpo-001 とも
   verifier 盲点がデータ品質と測定の双方を汚した)

## 8. 成果物

- 実装差分: verifier v1.1 + fence 修正 + K=6 + dpo-001r/eval configs(本 PR)
- データ: WSL `data/processed/dpo/`(generations 32,766 / verdicts / judgments 16,104 /
  dpo_pairs.jsonl 970 + dpo_pairs_v0.jsonl 976)
- モデル: WSL `outputs/dpo-001-b005` / `dpo-001-b01` / `dpo-001-b03` / `dpo-001r-b005`
- 評価: WSL `outputs/eval/ifeval_ja/`(12 モデル)+ `outputs/eval/dpo-001/`(llm-jp-eval)
- EXPERIMENT_LOG: rows 028–030
