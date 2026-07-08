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
| probe-1.2b-layerft-L8-seq-002 | phase0-oom-probe | base.yaml + L8 + smoke_test.max_seq_len sweep | e864a94 | 下表参照 | seq_len 別 peak: 1024=3.02GiB / 2048=3.84 / 4096=5.50 / 5120=6.33 / 6144=7.16 / 7168=7.99 / 8192=8.82 / 16384=15.58 / 20480=19.00 / 22528=20.72(GiB)、24576/32768 で CUDA OOM | 2026-07-09 実施(#57 の seq 軸)。**ハード限界 seq=22528**(WDDM が RAM へスピルするため物理 8GB 超でも動くが低速)。**実用限界(物理 VRAM 内)は seq≈7168**(7.99GiB でほぼ満杯)、**推奨は 6144 以下**(7.16GiB、デスクトップ併用の余裕込み)。step 時間: 5120=1.82s / 6144=2.32s / 7168=2.93s(batch1, grad ckpt, 4step 平均) |

## 失敗記録

OOM 条件・発散 lr などもここに残す（同じ失敗を繰り返さないため）。

| date | condition | error | action |
|---|---|---|---|
| 2026-07-09 | 1.2B 層FT(L8)・batch 1・grad ckpt・seq_len 24576/32768 | CUDA out of memory(割り当て失敗) | seq_len 上限を 22528 とする。ただし物理 VRAM(8GB)に収まるのは seq≈7168 まで(超過分は WDDM が RAM にスピルし低速化)。実学習は seq 6144 以下を推奨 |
