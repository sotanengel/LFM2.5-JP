# Experiment Log

全 run の 1 行サマリを記録する台帳。W&B / TensorBoard と二重記録する。

## Run 命名規則

`{phase}-{model}-{method}-{seq}` 例: `sft-1.2b-layerft-L15-003`

## 必須記録項目

| 項目 | 説明 |
|---|---|
| run_id | 上記命名規則 |
| config_hash | `lfm25_ja.utils.config.config_hash()` |
| git_commit | `git rev-parse HEAD` |
| data_version | データセット名 + revision |
| seed | 乱数 seed |
| vram_peak | ピーク VRAM (bytes / human) |
| duration | 所要時間 |
| scores | 評価スコア（タスク別） |
| samples | 生成サンプル 5 件 |

## ログ

| run_id | phase | config | commit | vram_peak | scores | conclusion |
|---|---|---|---|---|---|---|
| _pending_ | phase0-baseline | configs/eval/llm_jp_eval.yaml | - | - | - | ベースライン評価待ち |
| smoke-1.2b-layerft-L8-001 | phase0-smoke | base.yaml + override `tuning.trainable_layer_indices: [8]` | e864a94 | 2,800,482,304 B (2.61 GiB) | loss 18.47 → 7.26 (20 step, seq 512, batch 1, seed 42) | 2026-07-09 実施。層 FT(第 9 層 = index 8)が RTX 3060 Ti で安定動作。デスクトップアプリが VRAM 7.8 GiB 占有中でも WDDM 退避で問題なし。学習部所要 ~4.1 s。Phase 0 スモークゲート通過 |

## 失敗記録

OOM 条件・発散 lr などもここに残す（同じ失敗を繰り返さないため）。

| date | condition | error | action |
|---|---|---|---|
| - | - | - | - |
