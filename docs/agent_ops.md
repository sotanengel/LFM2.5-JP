# エージェント運用 runbook(再発防止集)

このプロジェクトを触る AI エージェント / 人間向けの、**過去に何度も踏んだ落とし穴と回避策**の
集約 doc。実験ログ側の失敗記録(`experiments/EXPERIMENT_LOG.md` §失敗記録)を、実行者視点で
再利用しやすい形に整理する。

---

## Python 実行(Windows / WSL の使い分け)

### ❌ ダメなパターン(2026-07-15 実測)

- `python -c '...'` → Windows Store 版 `python.exe` が `0x80070003` で launch fail
- `.venv/Scripts/python.exe` → 作業サブエージェントが cleanup で消していることがある
- `uv run python ...`(worktree 内)→ `uv sync` が torch 2.4GiB のダウンロードを試み、
  ホストのディスク空き不足で失敗(`os error 112`)

### ✅ 正解パターン

**CPU テスト / スクリプト実行は必ず WSL Ubuntu の `~/lfm25-ja/.venv` を使う**:

```bash
wsl -d Ubuntu -- bash -lc "cd ~/lfm25-ja && unset VIRTUAL_ENV && uv run --no-sync python ..."
```

- `unset VIRTUAL_ENV`: Windows 側の VIRTUAL_ENV が伝播すると uv が誤動作する
- `uv run --no-sync`: 既存 venv を使うだけで、依存解決を試みない(**必須**)

**Windows worktree のソースを WSL 側 venv で回したい場合**(worktree 側で編集 → CPU テスト):

```bash
WORKTREE=/mnt/c/Users/na-g-/LFM2.5-JP/.claude/worktrees/<branch>
wsl -d Ubuntu -- bash -lc "
  cd ~/lfm25-ja
  unset VIRTUAL_ENV
  PYTHONPATH=$WORKTREE/src uv run --no-sync pytest $WORKTREE/tests/test_xxx.py -v
"
```

または後述の「scratchpad script 経由」方式を使う(変数展開の落とし穴回避)。

---

## wsl.exe 経由でコマンドを渡すとき

### ❌ ダメなパターン(2026-07-15 実測)

```bash
# 変数が空で渡る
wsl -d Ubuntu -- bash -lc "WORKTREE=/mnt/c/... && cp \$WORKTREE/foo /home/x/"
# → 'cp: cannot stat '/foo':' となる(\$ の外側 shell 展開タイミングを外す)
```

### ✅ 正解パターン

**wsl.exe に渡す複数行 / 変数入りコマンドは必ずスクリプトファイルを経由する**:

1. スクリプトを scratchpad に書く(絶対パス、`set -e` 推奨):
   ```bash
   # C:\Users\...\scratchpad\my_script.sh
   #!/usr/bin/env bash
   set -e
   WORKTREE=/mnt/c/Users/na-g-/LFM2.5-JP/.claude/worktrees/xxx
   cp "$WORKTREE/foo" "$HOME/lfm25-ja/"
   ```

2. wsl.exe で `bash /mnt/c/.../my_script.sh` を呼ぶ:
   ```bash
   wsl -d Ubuntu -- bash -lc "bash /mnt/c/.../scratchpad/my_script.sh"
   ```

なお `bash -lc "..."`(login shell)を経由するのは、`~/.bashrc` / `~/.profile` を source
して uv 等の PATH を通すため(必須。素の `bash /path/to/script.sh` だと `uv: command not
found` になる、`docs/sft_training_runbook.md` §runbook 参照)。

---

## transformers `apply_chat_template` の罠

### ❌ ダメなパターン(2026-07-15 実測 / 2026-07-12 実装時にも同種)

```python
input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)
model.generate(input_ids, ...)  # → AttributeError in inputs_tensor.shape
```

新しい transformers では `apply_chat_template(tokenize=True)` がデフォルトで
`BatchEncoding`(dict-like)を返す。`.to(device)` は動くが、`model.generate()` 内で
`.shape` アクセスが失敗する。既存 [src/lfm25_ja/data/format_chat.py:54](../src/lfm25_ja/data/format_chat.py:54)
にも同種の落とし穴が記録済み(SFT 用の別のケースだが原因は同じ)。

### ✅ 正解パターン

```python
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=False,  # ← 必須
).to(model.device)
```

**新しくチャット template を使うコードを書くたびに `return_dict=False` を明示する**。
`format_chat.py` の実装を必ず参照すること。

---

## サブエージェント(sonnet / Fable5)委任時の注意

### ❌ ダメなパターン

- サブエージェントの「〜が実行中です、後で確認します」的な進捗報告を鵜呑みにする
  → 実は何も起動していない、ということが過去にあった(`experiments/EXPERIMENT_LOG.md`
  失敗記録 2026-07-12 起点)
- 同一 worktree に並列で複数エージェントを走らせる(ブランチ切り替えで作業ツリー内容が
  変わり、片方の作業が別ブランチに紛れ込む)

### ✅ 正解パターン

- サブエージェントが「実行中」と言ったら、必ず `ps aux | grep <keyword>` で実プロセスを
  確認してから信頼する
- 並列サブエージェントは `isolation: "worktree"` で別ワークツリーに分離する
- Fable5(model=fable)は novelty / 判断コンサル用途に限定(コード実装は sonnet に委任)
- サブエージェントに委任する場合、**プロンプトは cold-start 前提でフル context を渡す**
  (経緯・関連ファイルパス・成功条件・落とし穴 tips までパック)

---

## 評価パイプラインの排他性

**同一 harness ディレクトリに対する eval パイプラインは同時に 1 つしか走らせない**
(`experiments/EXPERIMENT_LOG.md` §失敗記録 2026-07-13)。着手前に:

```bash
wsl -d Ubuntu -- bash -lc "ps -ef | grep -E 'run_llm_jp_eval|inference.py|evaluate_llm|run_ifeval_ja|generate_ifeval_ja' | grep -v grep"
```

で既存プロセスの有無を確認。競合を見つけたら該当プロセスツリーを kill してから自分の実行
を継続する。

---

## WSL2 の不定期再起動対策

WSL2 は高 I/O 負荷で不定期に再起動する(1 日 6 回発生歴)。

- 長時間ジョブは中断しても再開できるよう、生成 / 採点を分離(生成 JSONL は冪等スキップ機能付き
  → [src/lfm25_ja/eval/generate_ifeval_ja.py](../src/lfm25_ja/eval/generate_ifeval_ja.py)
  の `_existing_generation_count` 参照)
- SFT 学習は途中 checkpoint から自動再開(`docs/sft_training_runbook.md`)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を launcher script で必ず設定
  (RTX 3060 Ti WDDM の VRAM 断片化対策、`experiments/reports/phase3_sft004_lr_sweep.md` §182
  参照)

---

## 参照

- `docs/sft_training_runbook.md` — SFT 学習 runbook
- `docs/lfm2_5-ja-plan.md` — プロジェクト全体計画
- `experiments/EXPERIMENT_LOG.md` §失敗記録 — 実施日別の失敗ログ(本 doc の source of truth)
