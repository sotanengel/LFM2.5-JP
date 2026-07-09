# LFM2.5-JA

日本語特化 [LFM2.5](https://huggingface.co/LiquidAI) モデルを RTX 3060 Ti（VRAM 8GB）環境で段階的に構築・検証するリポジトリです。

## ハードウェア前提

- GPU: NVIDIA RTX 3060 Ti (8GB)
- 学習: 選択層フルパラメータ FT（bf16 ロード、選択層のみ `requires_grad=True`、他は freeze）を標準
- シーケンス長: Phase ごとに 1024 → 2048 → 4096 と段階拡張

## 対象モデル

| モデル | 用途 |
|---|---|
| `LiquidAI/LFM2.5-1.2B-Instruct` | SFT / DPO ベース |
| `LiquidAI/LFM2.5-1.2B-Base` | CPT ベース |
| `LiquidAI/LFM2.5-1.2B-JP-202606` | ベースライン比較 |

**ライセンス注意**: LFM モデルは LFM Open License v1.0 です（Apache ではありません）。配布時は条項を確認してください。

## クイックスタート

```bash
# CPU 依存（CI と同じ）
make setup

# GPU 学習用（.venv のみに CUDA 版 torch を入れる。システム CUDA Toolkit は不要）
make setup-gpu

# CPU テスト
make test

# GPU テスト（CUDA 必須）
make test-gpu
make smoke-test
```

### GPU セットアップについて

- **NVIDIA ドライバ**は既に入っていれば十分（RTX 3060 Ti 確認済み）
- **CUDA Toolkit をシステムに入れる必要はない** — PyTorch wheel に CUDA 12.6 ランタイム同梱
- `make setup-gpu` はプロジェクトの `.venv` だけを変更し、グローバル Python や PATH は汚さない
- GPU コマンドは `uv run --no-sync` 経由で CPU 版 torch への巻き戻しを防ぐ

### WSL2（GPU 学習の推奨環境。Issue #64）

Windows ネイティブの torch は flash / mem-efficient SDP カーネルを持たず、GQA 構成では
mem-efficient も使えないため、packed 学習（全トークン attend）時に math バックエンド
（N² 実体化）へ落ちる。この制約で 1.2B 層 FT の実効 `max_seq_len` は 2048 に制限される
（実測: 2048×batch1=4.5GiB / 4096×batch1=10.7GiB でスピル）。

WSL2（Ubuntu）上の Linux 版 torch（2.13.0+cu130 で検証済み）では flash SDP が動作し、
全 attend seq 6144×batch1 が 7.12 GiB で物理 8GB 内に収まることを実測済み
（`experiments/EXPERIMENT_LOG.md` probe-005）。**GPU 学習は WSL2 上での実行を前提とする。**

手順:

```bash
# 1. リポジトリを WSL 側の ext4 に clone する（/mnt/c 上は I/O が遅いため避ける）
git clone <repo-url> ~/lfm25-ja
cd ~/lfm25-ja

# 2. venv 作成 + GPU 込みで依存インストール
uv venv --python 3.11
uv sync --extra dev --extra gpu

# 3. CPU テスト
uv run pytest -m "not gpu"

# 4. GPU 認識確認
nvidia-smi
```

- HF キャッシュ・`data/`・`outputs/` は WSL の ext4 側に置くこと（I/O が速い）。
  `/mnt/c` 上の既存 HF キャッシュを `HF_HOME` で再利用することも可能だが読み込みは遅め。
- `make setup-gpu`（`scripts/setup_gpu.ps1`）は Windows 専用。WSL では
  `make setup-gpu-linux`（`uv sync --extra dev --extra gpu` のみ）で足りる —
  Linux 版 torch は CUDA ランタイム同梱で追加セットアップ不要。

### Windows

Makefile を主入口としています。[make](https://gnuwin32.sourceforge.net/packages/make.htm) または WSL を使用してください。`scripts/*.sh` は WSL / Git Bash 用ラッパーです。

CPU でのテスト・開発は従来どおり Windows ネイティブで可能。**GPU 学習を Windows ネイティブで行う場合**は
flash SDP カーネル非搭載により `max_seq_len` を 2048 に下げること（`configs/base.yaml` 参照）。

## ディレクトリ構成

```
configs/          # 実験設定（YAML）
src/lfm25_ja/     # データ・学習・評価コード
scripts/          # シェルラッパー
experiments/      # 実験台帳・レポート
tests/            # pytest
data/             # データ（gitignore）
outputs/          # チェックポイント（gitignore）
```

## Makefile ターゲット

| ターゲット | 説明 |
|---|---|
| `make setup` | 開発依存をインストール |
| `make setup-gpu` | GPU 依存をインストール（Windows ネイティブ、PowerShell 経由） |
| `make setup-gpu-linux` | GPU 依存をインストール（WSL2 / Linux、`uv sync` のみ） |
| `make lint` | ruff 静的解析 |
| `make test` | CPU テスト（GPU マーカー除外） |
| `make test-gpu` | GPU テスト |
| `make smoke-test` | bf16 推論 + 選択層フルパラメータ FT 20 step |
| `make probe-memory` | OOM 格子探索 |
| `make eval-baseline` | llm-jp-eval ベースライン |

## 環境変数

`.env.example` を `.env` にコピーして設定:

- `HF_TOKEN` — Hugging Face 認証
- `WANDB_API_KEY` — 実験追跡（任意）

## 開発

```bash
make lint
make test
pre-commit run --all-files
```

## 参照

- 計画書: [`docs/lfm2_5-ja-plan.md`](docs/lfm2_5-ja-plan.md)
- GitHub Issues: Phase 0〜5 の Epic / サブ Issue
