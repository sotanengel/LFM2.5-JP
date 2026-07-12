# SFT学習ジョブ 実行手順(runbook)

Issue #31(`train_sft.py` 実装)・#33(sft-001 実行)関連。WSL2上で SFT 学習ジョブを起動・監視・再開するための手順を記す。

## 前提

- 本番学習環境は WSL2 Ubuntu の `~/lfm25-ja`(ext4)。Windows 側は開発・CPU テスト用。
- `wsl.exe` 経由のインラインコマンドはシェル変数が壊れる(展開が Windows 側で起きる/空になる)ため、**必ずスクリプトファイルを書いてから実行する**。
- WSL 内で `nohup` を使ってもデタッチ実行にはならない(親シェル終了で死ぬ)。バックグラウンド実行は PowerShell 側の `Start-Process` を使う。

## 1. WSL疎通確認

学習ジョブ起動前に WSL が正常応答することを確認する。

```powershell
wsl -d Ubuntu -- echo ok
```

`ok` が返らない場合(コマンドが無応答・`exit=1`・`E_UNEXPECTED` 等)は WSL インスタンスが壊れている可能性が高い。特に「チェックポイント保存中の `Input/output error (os error 5)`」でジョブが落ちた場合は WSL2 の仮想ディスク側の問題であることが多く、以下で再起動してから再開する。

```powershell
wsl --shutdown
# 数秒待ってから再確認
wsl -d Ubuntu -- echo ok
```

`wsl --shutdown` は WSL 上で動いている全プロセス(学習ジョブ含む)を停止する。実行中のジョブがある場合は影響範囲を必ずユーザーに確認してから実施すること。

## 2. 学習スクリプトの用意

wsl.exe 経由のインラインコマンドではなく、スクリプトファイルを用意して実行する。

```bash
#!/usr/bin/env bash
set -uo pipefail
unset VIRTUAL_ENV   # Windows側のVIRTUAL_ENVがWSLに伝播するため必須
cd ~/lfm25-ja
echo "START_MARKER $(date)"
uv run --no-sync python -m lfm25_ja.train.train_sft --config configs/sft/<config>.yaml
echo "EXIT_CODE_IS $?"
echo "END_MARKER $(date)"
```

`<config>` は `configs/sft/` 配下の対象 config(例: `sft_001_ichikara`)に置き換える。`--layers` / `--no-checkpoints` / `--output-root` オプションは `train_sft.py` の CLI 引数として上書き可能([train_sft.py](../src/lfm25_ja/train/train_sft.py) 参照)。

`make train-sft`(`CONFIG=configs/sft/xxx.yaml make train-sft`)でも同等に起動できるが、デタッチ実行や引数上書きが必要な場合は上記スクリプト形式を使う。

## 3. デタッチ実行

WSL 側の `nohup` は使わず、PowerShell から `Start-Process` でバックグラウンド起動しログをリダイレクトする。

```powershell
Start-Process wsl.exe -ArgumentList "-d","Ubuntu","--","bash","/mnt/c/path/to/run_sft.sh" `
  -RedirectStandardOutput "sft_train.log" -RedirectStandardError "sft_train.err.log" -NoNewWindow
```

## 4. 進捗確認

```powershell
Get-Content sft_train.log -Tail 20
Get-Content sft_train.log -Wait   # tail -f 相当
```

ログに `EXIT_CODE_IS 0` と `END_MARKER` が出力されれば正常終了。`sft_train.err.log` にトレースバックが出ている場合は失敗しているので原因を確認する。

## 5. 中断からの自動再開

`train_sft.py` は `resolve_resume_checkpoint()`(`train_cpt.py` 由来、`train_sft.py` から import)により `output_dir` 内の最新 `checkpoint-*` を自動検出し、`trainer.train(resume_from_checkpoint=...)` に渡す。

- `output_dir` が存在しない、または `checkpoint-*` が一つもない場合 → 最初から学習
- `--no-checkpoints` 指定時は常に最初から学習(中間チェックポイントを保存しないため)
- それ以外は最新チェックポイントから再開

**したがって、途中で落ちたジョブは同じ config・同じコマンドで再実行するだけで自動的に再開される。** config やコマンドを変更する必要はない。

### 既知の障害事例(2026-07-13, sft-001実行時)

sft-001(ichikara, 単層L9)実行中、step 600/1452(41%)のチェックポイント保存中に

```
safetensors._safetensors_rust.SafetensorError: Error while serializing: I/O error: Input/output error (os error 5)
```

で失敗し、以後 `wsl.exe` 経由の全コマンドが無応答になった。原因は WSL2 仮想ディスクのI/Oエラーで、`wsl --shutdown` によるWSLインスタンス再起動で復旧した。同一 config での再実行により、破損したチェックポイントの一つ前の正常なチェックポイントから自動再開される。
