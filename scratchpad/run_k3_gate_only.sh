#!/usr/bin/env bash
# Build JKB aggregate.json from existing generations and run K3 gate.
set -euo pipefail
cd "$HOME/lfm25-ja-k2"
export PATH="$HOME/.local/bin:$PATH"

for ARM in k3-facts-b005 k3-facts-b01 k3-facts-b03; do
  uv run python - <<PY
import json
from pathlib import Path
from lfm25_ja.eval import jkb

arm = "${ARM}"
rows = jkb.load_jkb_jsonl("datasets/eval/jkb/eval.jsonl")
gen = Path(f"outputs/eval/jkb/k3-facts/{arm}/generations.jsonl")
raw = {}
for line in gen.read_text(encoding="utf-8").splitlines():
    if line.strip():
        e = json.loads(line)
        raw[e["id"]] = e["response"]
agg = jkb.aggregate(rows, raw)
gate_agg = {"overall": agg["overall"], "by_domain": agg["by_domain"]}
out = Path(f"outputs/eval/jkb/k3-facts/{arm}/aggregate.json")
out.write_text(json.dumps(gate_agg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(arm, agg["overall"]["accuracy"])
PY
done

for ARM in b005 b01 b03; do
  NAME="k3-facts-${ARM}"
  case "$ARM" in
    b005) LLMJP="/home/usr/llm-jp-eval/local_files/results/result_baseline-k3factsb005_20260720_164848.json" ;;
    b01)  LLMJP="/home/usr/llm-jp-eval/local_files/results/result_baseline-k3factsb01_20260720_165220.json" ;;
    b03)  LLMJP="/home/usr/llm-jp-eval/local_files/results/result_baseline-k3factsb03_20260720_165544.json" ;;
  esac
  if [[ ! -f "$LLMJP" ]]; then
    echo "ERROR: missing llm-jp result for $NAME: $LLMJP" >&2
    exit 1
  fi
  uv run python scripts/eval_k3_gate.py \
    --jkb "outputs/eval/jkb/k3-facts/${NAME}/aggregate.json" \
    --ifeval "outputs/eval/ifeval_ja/k3_facts/${NAME}/aggregate.json" \
    --llmjp "$LLMJP" \
    --out-json "outputs/eval/k3_gate/${NAME}_verdict.json" \
    --out-md "outputs/eval/k3_gate/${NAME}_verdict.md" || true
done
