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
| baseline-1.2b-llmjpeval-007 | phase0-baseline | llm-jp-eval v2.1.5 / 8 タスク × 100 件 × 4-shot × greedy(reports/phase0_baseline.md) | fbcadfc | 推論 peak ~3 GiB | **AVG: Instruct 0.334 / JP-202606 0.387**。jcqa 0.54/0.78、jmmlu 0.42/0.22(JP は ool 55%)、jsquad EM 0.41/0.49、jsts pearson 0.29/0.68、niilc EM 0.06/0.13、jsem 両者 0(ool 100%) | 2026-07-09 実施(#14)。WSL2 上で end-to-end 動作。JP 特化版が 6/8 タスクで優位 = ハーネス妥当性確認。⚠️ jsem 全滅・jmmlu(JP)の ool 55% は形式不整合でありモデル能力ではない(フォローアップ Issue)。公表値との絶対比較は不可(100 件・base 形式プロンプト)。**本表を Phase 2 以降の相対比較の基準線として凍結** |
| smoke-1.2b-layerft-L8-001 | phase0-smoke | base.yaml + override `tuning.trainable_layer_indices: [8]` | e864a94 | 2,800,482,304 B (2.61 GiB) | loss 18.47 → 7.26 (20 step, seq 512, batch 1, seed 42) | 2026-07-09 実施。層 FT(第 9 層 = index 8)が RTX 3060 Ti で安定動作。デスクトップアプリが VRAM 7.8 GiB 占有中でも WDDM 退避で問題なし。学習部所要 ~4.1 s。Phase 0 スモークゲート通過 |
| probe-1.2b-layerft-L8-seq-002 | phase0-oom-probe | base.yaml + L8 + smoke_test.max_seq_len sweep | e864a94 | 下表参照 | seq_len 別 peak: 1024=3.02GiB / 2048=3.84 / 4096=5.50 / 5120=6.33 / 6144=7.16 / 7168=7.99 / 8192=8.82 / 16384=15.58 / 20480=19.00 / 22528=20.72(GiB)、24576/32768 で CUDA OOM | 2026-07-09 実施(#57 の seq 軸)。**ハード限界 seq=22528**(WDDM が RAM へスピルするため物理 8GB 超でも動くが低速)。**実用限界(物理 VRAM 内)は seq≈7168**(7.99GiB でほぼ満杯)、**推奨は 6144 以下**(7.16GiB、デスクトップ併用の余裕込み)。step 時間: 5120=1.82s / 6144=2.32s / 7168=2.93s(batch1, grad ckpt, 4step 平均) |
| probe-1.2b-layerft-L8-9-seq-003 | phase0-oom-probe | base.yaml + `trainable_layer_indices: [8, 9]` | e864a94 | 6144: 7.158 GiB / 7168: 7.993 GiB | 1 層(L8)比で peak **+2 MiB**、step 時間 +4%(6144: 2.41s) | 2026-07-09 実施。可変層を中間 2 層に増やしても VRAM ピークは実質不変。ピークは backward 序盤(活性値が最も残る時点)で発生し、中間層の勾配(bf16 ~128MB)はその後に確保されるため。optimizer state は paged_adamw_8bit が CPU へページング。**seq 6144 で 2 層 FT は問題なし** |
| probe-1.2b-layerft-realgrid-004 | phase0-oom-probe | base.yaml memory_probe(`--real`, 全 attend 系列) | (PR #63) | reports/phase0_memory.md 参照 | 全 attend(packed 学習相当): 1024×b1=3.0 / 2048×b1=4.5 / 2048×b2=6.7 / **2560×b1=5.66 / 3072×b1=7.08** / 4096×b1=10.7(スピル)GiB。4096×b2 以上は CUDA OOM。可変層 1→2 の差は +0.1 GiB 未満 | 2026-07-09 実施(#57 完了)。**⚠️ probe-002/003 と smoke-001 の値は「ほぼ pad のダミーデータ」によるもので packed 実学習を大幅に過小評価**(4096×b1: pad 5.5 vs 全attend 10.7 GiB)。原因は Windows 版 torch(2.12.1+cu126)に flash/mem-efficient SDP カーネルが無く、全 attend 時に math バックエンド(N² 実体化)へ落ちるため(SDPBackend 明示指定実験で確定)。**packed CPT の実効上限: seq 3072×batch1(7.08GiB)、安全推奨: 2048(batch2 で 6.7GiB)**。6144 決定は撤回し config を 2048 に修正 |

| probe-1.2b-layerft-wsl2-005 | phase0-oom-probe | WSL2 Ubuntu 26.04 + torch 2.13.0+cu130(検証用 venv) | (PR #63) | 4096×b1=5.48 / **6144×b1=7.12** / 8192×b1=8.76 GiB(全 attend) | flash SDP カーネル動作 OK(GQA 対応)。6144: 2.15 s/step | 2026-07-09 実施。**WSL2 の Linux 版 torch では flash SDP が使え、全 attend 4096 が 10.67 → 5.48 GiB に半減。seq 6144×b1 が物理 8GB 内(7.12 GiB)に収まる**。Windows ネイティブの N² 問題の根本解決を実証。学習環境の WSL2 移行を推奨 |

| wsl2-migration-006 | phase0-env | WSL2 Ubuntu 26.04 の `~/lfm25-ja` + gpu extra(torch 2.13.0+cu130) | (PR #65) | grid: reports/phase0_memory.md | CPU テスト 85 件緑 / GPU スモーク loss 2.32→0.013(peak 2.61 GiB, 5.1 steps/s)/ 実測グリッド 20/24 成功。物理 8GB 内: 1024×b4=5.5 / 2048×b2=5.5 / 4096×b1=5.5 / **6144×b1=7.1** GiB | 2026-07-09 実施(#64)。リポジトリ環境一式を WSL2 で構築・検証。flash SDP によりメモリは seq に対しほぼ線形(N² ペナルティ解消)。**学習環境を WSL2 に正式移行、max_seq_len=6144 を採用**。Windows ネイティブは開発・CPU テスト用(GPU 学習は 2048 制限) |

## 失敗記録

OOM 条件・発散 lr などもここに残す（同じ失敗を繰り返さないため）。

| date | condition | error | action |
|---|---|---|---|
| 2026-07-09 | 1.2B 層FT(L8)・batch 1・grad ckpt・seq_len 24576/32768 | CUDA out of memory(割り当て失敗) | seq_len 上限を 22528 とする。ただし物理 VRAM(8GB)に収まるのは seq≈7168 まで(超過分は WDDM が RAM にスピルし低速化)。実学習は seq 6144 以下を推奨 → **probe-004 で撤回(pad データによる過小評価)** |
| 2026-07-09 | VRAM 計測にほぼ pad のダミーデータを使用(smoke-001, probe-002/003) | packed 実学習メモリを最大 2 倍過小評価 → seq 6144 と誤決定 | **計測は必ず全 attend 系列で行う**(この環境は flash/mem-efficient SDP カーネル非搭載で math SDPA の N² 実体化が発生)。全 attend 実測に基づき max_seq_len=2048 に修正(上限は 3072)。根本解決は WSL2/Linux 化(flash SDP 入り torch) |
