# K3 事実性 DPO — JKB on-policy 選好学習 (Issue #124 / #145)

- 実施日: 2026-07-20
- 関連: Issue #124(設計) / #145(再編ハンドオフ) / サブ Issue #146–#151
- 実行ブランチ: `feature/k3-facts-dpo-issue-145`
- WSL 実行環境: `~/lfm25-ja-k2` (RTX 3060 Ti 8GB)

## TL;DR

**K3 事実性 DPO は 3 beta 点すべて K3 決定ゲート FAIL。プロモーションなし。**

| アーム | JKB eval | vs base 49.9% | ゲート ≥52.9% | IFEval | llm-jp AVG | ゲート |
|---|---:|---:|---|---:|---:|---|
| k3-facts-b005 | **51.6%** | +1.7pt | FAIL | 0.940 | 0.458 | FAIL |
| k3-facts-b01 | 51.1% | +1.2pt | FAIL | 0.940 | 0.442 | FAIL |
| k3-facts-b03 | 50.6% | +0.7pt | FAIL | 0.940 | 0.458 | FAIL |

ガード条件: IFEval prompt_strict **≥ 0.920** は全アーム PASS (base 0.950 比 -1pt 以内)。
llm-jp-eval AVG **≥ 0.459** は b005/b03 が 0.458 で境界 FAIL、b01 は 0.442 で明確 FAIL。

**方針どおり**: 日本知識の +3pt 未達 → K3 非採用。次段は K4 RAG (#125) へ委譲。

## 背景と設計

cpt-D (K2) 棄却後、K3 は **base (`LiquidAI/LFM2.5-1.2B-JP-202606`) 単独起点**で
JKB 事実正答を on-policy DPO する実験。K2 の cpt-D 依存は削除。

- **pref プール**: JKB train 分割 105 問 (`datasets/eval/jkb/train.jsonl`)
- **評価**: JKB eval 399 問 (`eval.jsonl`) — train/eval ID 物理分離で非重複ゲート
- **judge**: Swallow 8B factual テンプレ (参照解 + source_quote 照合、1–5 採点)
- **DPO**: L9 単層、lr 5e-6、1ep、dpo-001 と同ハイパラ、beta {0.05, 0.1, 0.3}

## WSL クリーンアップ (Phase 2-0)

`scratchpad/cleanup_wsl_k2.sh` 実行済み。cpt-D deci 重み、`data/processed_cptD`、
k2 評価中間物を削除。K1 参照資産 (`~/lfm25-ja/outputs/eval/jkb/k1-full/base/` 等) は保持。

## 選好データ構築 (P0 → G → V → J → P)

| Phase | 出力 | 件数 |
|---|---|---:|
| P0 pref_prompts_jkb | `data/processed/dpo_k3/pref_prompts.jsonl` | 105 |
| G pref_generate (base, K=8) | `generations.jsonl` | 840 |
| V pref_verify_facts | `verdicts.jsonl` | 840 (rule_pass 427 / degenerate 313) |
| J judge_swallow (factual) | `judgments.jsonl` | 840 (~6 min, vLLM 8B) |
| P pref_pairs_facts | `dpo_pairs.jsonl` | **848** |

ペア内訳: pass_fail 143 / rule_pass_fail 633 / pass_pass_gap 72。
目標 500 組を大幅に上回る。length_guard base_mean=53 (JKB 短答向け)。

## DPO 学習

| アーム | 出力 | 学習時間 (実測) |
|---|---|---:|
| k3-facts-b005 | `outputs/k3-facts-b005` | ~85 s |
| k3-facts-b01 | `outputs/k3-facts-b01` | ~85 s |
| k3-facts-b03 | `outputs/k3-facts-b03` | ~85 s |

各 ~2.2 GiB。`unset PYTORCH_CUDA_ALLOC_CONF` 適用 (`scripts/43_train_dpo_k3_facts.sh`)。

## 評価

### JKB eval (399 問)

最良 **k3-facts-b005: 51.6% (206/399)** — base 49.9% 比 +1.7pt だが +3pt ゲート (52.9%) 未達。
beta 増加に伴い JKB は単調低下 (51.6 → 51.1 → 50.6%)。

分野×難度の傾向 (b005): advanced セルは依然低い (生活・慣習 advanced 10%、言語 advanced 17%)。
K1 ワースト 10 セルへの改善は限定的。

### IFEval

全 beta **prompt_strict = 0.940** (base 0.950 比 -1pt、ガード PASS)。

### llm-jp-eval

| アーム | AVG | vs base 0.469 | ガード ≥0.459 |
|---|---:|---:|---|
| k3-facts-b005 | 0.458 | -1.1pt | FAIL (境界) |
| k3-facts-b01 | 0.442 | -2.7pt | FAIL |
| k3-facts-b03 | 0.458 | -1.1pt | FAIL |

結果 JSON: `~/llm-jp-eval/local_files/results/result_baseline-k3factsb*.json`

## K3 決定ゲート

`scripts/eval_k3_gate.py` + `src/lfm25_ja/eval/k3_decision.py`

3 条件 AND: JKB ≥52.9% / IFEval ≥0.920 / llm-jp ≥0.459

**全アーム verdict = FAIL** (詳細: [k3_gate_artifacts/](k3_gate_artifacts/))

## 所見

1. **事実 DPO は cpt-D より遥かに健全**: JKB 0% 崩壊ではなく +0.7〜+1.7pt の小幅改善。
   IFEval も dpo-001 同等の無劣化帯 (0.940)。
2. **+3pt ゲート未達**: 105 問 on-policy ペア 848 組でも eval 399 問全体の +3pt には届かず。
   rule_pass_fail ペア (633) が主供給 = 一次 substring 正誤の差分学習が中心で、
   judge 4–5 vs 1–2 の高品質ペア (143) は相対的に少ない。
3. **llm-jp 回帰**: b005/b03 は AVG 0.458 でガード 0.459 を 0.001 だけ下回る境界 FAIL。
   知識特化 DPO が汎化ベンチマークをわずかに毀損する可能性。
4. **次段**: K4 RAG (#125) — closed-book(K3) vs RAG-on の JKB eval 差分で
   ワースト 10 セル (生活・慣習、言語 advanced 等) を直接ターゲット。

## 再現手順

```bash
# WSL ~/lfm25-ja-k2
bash scratchpad/cleanup_wsl_k2.sh          # 初回のみ
bash scripts/42_pref_k3_facts_pipeline.sh  # P0→G→V→J→P
bash scripts/43_train_dpo_k3_facts.sh      # beta 3 点
# JKB / IFEval / llm-jp-eval → eval_k3_gate.py
```

テスト: `uv run pytest tests/test_pref_prompts_jkb.py tests/test_pref_verify_facts.py \
  tests/test_judge_swallow_facts.py tests/test_pref_pairs_facts.py tests/test_k3_decision.py -q`
