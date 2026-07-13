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

| cpt-350m-layerft-pilot-008 | phase1-pilot | configs/cpt/cpt_350m_pilot.yaml + corpus_pilot.yaml(reports/phase1_pilot.md) | d8fd05b | 693 MiB | loss 5.72 → 4.88(1 epoch、~86 万 tok、216 steps、108 s)。生成サンプル正常(日本語・英語とも崩れなし) | 2026-07-09 実施(#23 の GPU 実走)。prepare end-to-end(aozora+wikitext、混合 ja85:en15 ぴったり)→ 350M 層 FT(可変 5.18%)。loss 単調改善・文字化けなし。**Phase 1 ゲート通過**。本番 CPT は prepare の streaming 対応が課題(レポート備考) |

| cpt-1.2b-layerft-centi-009 | phase2-cpt-B | configs/cpt/cpt_1.2b_layerft.yaml `--package centi`(1/100) | 8a63f78 | allocated 2.2 GiB / reserved 8.3 GiB | loss 2.393 → 2.299(平均 2.236、130 steps・~3.2M tok・17 分、7.9 s/step)。生成: 日本語が Wikipedia 調で流暢・崩れなし、英語も無事 | 2026-07-09 実施(#27 の予行)。データ: wikipedia_ja+en streaming 100k 行 → packed 51,763 系列(~318M tok、キャッシュ保存済み)。可変層 [7,8](arXiv:2607.01232 中央 k=2、10.93%)。grad_norm 0.7〜1.0 で安定、lr 1e-4 適正。**フルラン Go 判定**。出力は outputs/cpt-1.2b-layerft-centi に退避(--package が run_name を変えないため。恒久修正は要対応) |
| cpt-1.2b-layerft-full-010 | phase2-cpt-B | 同上 `--package full`(reports/phase2_cpt_b_verification.md) | 8a63f78 | reserved ~8.3 GiB | train loss 2.39 → 2.04〜2.11、grad_norm 0.55〜0.7 で全区間安定。held-out ja_ppl: base 8.29 → ckpt-9000 **8.07**。日本知識プローブ: 自動 25/25/28、**人手(先頭優先 0–1) base 27.25 > ckpt-5000=ckpt-9000 24.95**。ckpt-8000 以降で四国 4 県を正答(base は誤答) | 2026-07-09 17:49 開始 → **2026-07-10 に 9,864/12,941 steps(76%、~245M tok)でユーザー判断により停止**(発散なし)。チェックポイント 9 個を全数検証: 発散なし、ppl は初期悪化後に回復。知識は局所改善と科学技術等の悪化が混在し、人手合計では base 優位。自動 28 は選択肢列挙の偽陽性込み。**cpt-B 代表 = ckpt-9000**(ppl/健全性)。ゲートは llm-jp-eval |
| cpt-b-final-reeval-011 | phase2-gate | Stage V(reports/phase2_cpt_b_final_eval.md) | 2e9f508 | - | held-out ppl: base 9.12 → final **8.36**(単調改善・12000 以降収束)。プローブ v2(採点修正版): 25→25→25→**26**/50(旧採点の ckpt9000=28 は **3 点過大評価**と確定 — 人手検証と整合)。llm-jp-eval AVG: **cptb-base 0.235 / 9000 0.213 / final 0.216** vs cpt-C 0.387 / Instruct 0.334 | 2026-07-11 実施(#27/#76、完走モデル)。**評価軸で逆転**: ppl は改善する一方、llm-jp-eval は CPT 前より低下(jsts 負相関化、jsick/jnli/niilc 低下)= 生 wiki CPT が few-shot 構造化出力への適合を毀損。例外: jmmlu 0.05(ool77%)→0.27(ool13%)。**ゲート判定: cpt-B は cpt-C(0.387)に全面劣後 → 設計書ルールどおり cpt-C ルート切替を推奨**(層プロファイリング #75 の結果も加味して最終決定) |
| layer-profiling-centi-012 | phase2-layer-sweep | 16 単層 × centi + [7,8] 参照(reports/phase2_layer_profiling.md) | 2e9f508 | reserved ~8.3 GiB/本 | Δppl vs base 9.12: **中央帯 L6〜L9 のみ改善〜中立**(L6 −0.020 最良)、両端は悪化(L0 +0.33、L11〜14 +0.09〜0.12)。プローブ: L9=30 最良。**L15 は loss 上昇で適合不能** | 2026-07-11〜12 実施(#75)。detach ドライバが agent 死後も完走(中断耐性の実証)。**論文の中央集中パターンがハイブリッド LFM2.5 でも再現**。位置 > 層タイプ。単層 > 2 層(小予算)。deci 候補: L6/L7/L9 + L8(タイプ代表) |
| layer-profiling-deci-013 | phase2-layer-sweep | L6/L7/L8/L9 × deci(1/10 ≈ 32M tok)(reports/phase2_layer_profiling.md) | e47c1b6 | reserved ~8.3 GiB/本 | ppl: **L9 8.876(最良)**/ L6 8.898 / L7 8.919 / L8 8.964(base 9.1215)。プローブ: L6=30 最良 / L9=27 / L7=26 / L8=25。~3.2h/本 | 2026-07-12 実施(#75 完了)。**中央帯 4 層すべてが明確に改善(Δppl −0.16〜−0.25)、conv 3 層(9,6,7)> attn(8)が確定**。帯内 1 位は centi(L6)⇔ deci(L9)で入替 = L6/L9 は同格の最有力。プローブは centi+deci 計 1,000 問を手動照合し**過大評価ゼロ**を確認(過小評価 2 パターンは全層一律で順位に影響なし)。**Phase 2 ゲート: cpt-C ルート切替を確定**(reports/phase2_gate_and_next_steps.md)。層知見は Phase 3 sft-003(#35)へ転用 |
| eval-harness-format-fix-014 | phase0-baseline-fix | configs/eval/llm_jp_eval.yaml `dataset_info_overrides`(reports/phase0_baseline.md 更新版) | (fix/eval-harness-format, Issue #66) | 推論 peak ~4.6 GiB | ool: jsem Instruct 100%→**0%** / JP-202606 100%→**1%**、jmmlu Instruct 2%→**1%** / JP-202606 55%→**5%**(受け入れ条件 <10% 達成)。AVG: Instruct 0.334→0.355 / JP-202606 0.387→**0.469**。jsem exact_match: 両者 0.00→0.50/**0.60**。jmmlu exact_match: Instruct 0.42→0.39 / JP-202606 0.22→0.34 | 2026-07-13 実施。根本原因: jsem は ANSWER_TAGS_JP(`<answer>` タグ必須、モデル未使用)、jmmlu は output_length=1 トークン固定(JP-202606 の `\boxed{}` 癖で即打ち切り)。**2 段階の修正**: (1) custom 正規表現 + output_length 適度拡大(48/16)でも jsem(JP)37%・jmmlu(JP)15%が残存 → (2) output_length を jsem=160/jmmlu=200 にさらに拡大し、正規表現の先頭に貪欲 `.*` を付けて長い chain-of-thought の**最終結論**(先頭で言及・棄却された選択肢ではなく)を優先抽出。`run_llm_jp_eval.py` を実 v2 パイプライン(dump→llm-jp-eval-inference→evaluate)呼び出しに全面書き換え。実行中に別プロセスが同一 WSL harness ディレクトリへ並行書き込みしているのを検出・停止(共有ファイルの競合を回避) |
| sft-001-ichikara-L9-015 | phase3-sft-001 | configs/sft/sft_001_ichikara.yaml(reports/phase3_sft_layer_ablation.md) | c57a8ed | allocated 2.2 / reserved 3.3 GiB | loss 1.906 → **1.265**(2 epoch 完走、epoch2 平均 1.276、mean_token_acc 0.724)。step 1452/1452(ichikara 2,903 例 × 2 epoch)。~24 min(途中 checkpoint-600 で WSL2 仮想ディスク I/O エラーによりクラッシュ、WSL 再起動 → checkpoint-500 自動再開で完走) | 2026-07-13 実施(#33 完了)。sft-003 の「単層 L9」アームを兼ねる。破損した checkpoint は `checkpoint-600.broken/` に退避。runbook: docs/sft_training_runbook.md |
| sft-003-1.2b-L6-016 | phase3-sft-layer-ablation | configs/sft/sft_1.2b_layerft_L6.yaml(reports/phase3_sft_layer_ablation.md) | 20a5073 | allocated 2.2 / reserved 3.3 GiB | loss 1.904 → 1.383(epoch2 平均 1.349、acc 0.688)。step 1452、14 min | 2026-07-13 実施(#35)。単層 L6。**5 アーム中 5 位**。中央帯内でも L6 は L9 より SFT loss 収束が悪い(Phase 2 プローブでは L6 が最良の順位入替と対照的) |
| sft-003-1.2b-L6L9-017 | phase3-sft-layer-ablation | configs/sft/sft_1.2b_layerft_L6L9.yaml(reports/phase3_sft_layer_ablation.md) | 20a5073 | allocated 2.2 / reserved 3.6 GiB | loss 1.901 → 1.103(epoch2 平均 1.103、acc 0.747)。step 1452、18 min | 2026-07-13 実施(#35)。2 層 [6, 9]。5 アーム中 2 位。単層 L9(1.276)から 13% 改善、単層 L6(1.349)から 18% 改善。中央帯内で層を増やす効果が明確 |
| sft-003-1.2b-L6-9-018 | phase3-sft-layer-ablation | configs/sft/sft_1.2b_layerft_L6-9.yaml(reports/phase3_sft_layer_ablation.md) | 20a5073 | allocated 2.2 / reserved 3.9 GiB | loss 1.892 → **0.990**(epoch2 平均 **1.008**、acc **0.761**)。step 1452、21 min | 2026-07-13 実施(#35)。**中央 4 層 [6..9]、全アーム中最良**。フル FT(1.197、88.5%可変)を epoch2 平均で 20% 上回る。可変 params は 288M(24%)、reserved VRAM は full の 75%、所要時間も 75%。**Phase 4 DPO 以降は [6..9] 採用を推奨**(reports/phase3_sft_layer_ablation.md §考察) |
| sft-003-1.2b-full-019 | phase3-sft-layer-ablation | configs/sft/sft_1.2b_layerft_full.yaml(reports/phase3_sft_layer_ablation.md) | 20a5073 | allocated 2.2 / reserved 5.2 GiB | loss 1.866 → 1.147(epoch2 平均 1.197、acc 0.724)。step 1452、33 min。trainable_params 1,036M / 88.53% | 2026-07-13 実施(#35 参照アーム)。全 16 層可変。**[6..9] より劣後**(1.197 vs 1.008、20% 悪化)。両端層(Phase 2 で悪化群と判明した L0/L11-L14)を含めたことで loss surface が悪化した仮説と整合。20 step の VRAM プローブ結果と本ランで reserved 5.2 GiB 一致、RTX 3060 Ti 8GB で安全実行 |

## 失敗記録

OOM 条件・発散 lr などもここに残す（同じ失敗を繰り返さないため）。

| date | condition | error | action |
|---|---|---|---|
| 2026-07-09 | 1.2B 層FT(L8)・batch 1・grad ckpt・seq_len 24576/32768 | CUDA out of memory(割り当て失敗) | seq_len 上限を 22528 とする。ただし物理 VRAM(8GB)に収まるのは seq≈7168 まで(超過分は WDDM が RAM にスピルし低速化)。実学習は seq 6144 以下を推奨 → **probe-004 で撤回(pad データによる過小評価)** |
| 2026-07-09 | VRAM 計測にほぼ pad のダミーデータを使用(smoke-001, probe-002/003) | packed 実学習メモリを最大 2 倍過小評価 → seq 6144 と誤決定 | **計測は必ず全 attend 系列で行う**(この環境は flash/mem-efficient SDP カーネル非搭載で math SDPA の N² 実体化が発生)。全 attend 実測に基づき max_seq_len=2048 に修正(上限は 3072)。根本解決は WSL2/Linux 化(flash SDP 入り torch) |
| 2026-07-13 | llm-jp-eval-inference の `inference.py` に `--generation_config.do_sample=false` / `--generation_config='{"do_sample":false}'` を CLI 引数で渡そうとした | `generation_config` は `transformers.GenerationConfig`(プレーンな pydantic モデルではない)のため、`model`/`tokenizer` と違いドット記法の CLI 上書きが存在せず、JSON 文字列渡しも pydantic のバリデータが `dict` しか受け付けず失敗(即座に returncode 2 で終了、GPU に一切触れない) | **generation_config は必ず YAML ファイルの nested mapping として `--config <path>` 経由で渡す**(元の Issue #14 baseline 実行時の `baseline_instruct.yaml` 等と同じ方式)。`run_llm_jp_eval.build_inference_config` がこの YAML を生成する |
| 2026-07-13 | 同じ WSL `~/lfm25-ja-fix66` ワークツリー・同じ harness ディレクトリ(`~/llm-jp-eval`, `~/llm-jp-eval-inference`)に対して、検証用の一時設定(jsem/jmmlu のみ・広い output_length)を流している最中に、別プロセス(`run_full_baseline3.sh`、既定 config 使用)が並行実行されているのを `ps -ef` で発見 | 両プロセスが `eval_configs/lfm25_baseline.yaml` と `<run_name>_generated.yaml`(推論設定)を同名で上書きし合うため、一方または両方の生成結果が意図しない設定で汚染されるリスク | **同一 harness ディレクトリに対する eval パイプラインは同時に 1 つしか走らせない**。着手前に必ず `ps -ef \| grep -E 'run_llm_jp_eval\|inference.py\|evaluate_llm'` で既存プロセスの有無を確認し、競合を見つけたら該当プロセスツリーを kill してから自分の実行を継続する |
