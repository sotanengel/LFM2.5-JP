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

### Windows

Makefile を主入口としています。[make](https://gnuwin32.sourceforge.net/packages/make.htm) または WSL を使用してください。`scripts/*.sh` は WSL / Git Bash 用ラッパーです。

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
