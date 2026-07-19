"""CLI smoke tests for scripts/eval_k2_gate.py (Issue #132 / #138)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_eval_k2_gate_cli_writes_verdict(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "eval_k2_gate.py"

    domains = {
        n: {"n": 10, "correct": 5, "accuracy": 0.5}
        for n in [
            "地理",
            "歴史",
            "政治・制度",
            "経済",
            "生活・慣習",
            "食文化",
            "伝統文化",
            "言語",
            "文学",
            "地域・観光",
            "スポーツ",
            "科学技術・産業",
        ]
    }
    jkb_base = {"overall": {"n": 100, "correct": 50, "accuracy": 0.5}, "by_domain": domains}
    jkb_cand = {
        "overall": {"n": 100, "correct": 65, "accuracy": 0.65},
        "by_domain": {
            k: {"n": 10, "correct": 5, "accuracy": 0.5} for k in domains
        },
    }
    ifeval = {"prompt_strict_acc": 0.93}
    llmjp = {"AVG": 0.47}

    base_path = tmp_path / "jkb_base.json"
    cand_path = tmp_path / "jkb_cand.json"
    ifeval_path = tmp_path / "ifeval.json"
    llmjp_path = tmp_path / "llmjp.json"
    out_json = tmp_path / "gate.json"
    out_md = tmp_path / "gate.md"
    for path, payload in (
        (base_path, jkb_base),
        (cand_path, jkb_cand),
        (ifeval_path, ifeval),
        (llmjp_path, llmjp),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--jkb-candidate",
            str(cand_path),
            "--jkb-base",
            str(base_path),
            "--ifeval",
            str(ifeval_path),
            "--llmjp",
            str(llmjp_path),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    verdict = json.loads(out_json.read_text(encoding="utf-8"))
    assert verdict["pass"] is True
    assert "PASS" in out_md.read_text(encoding="utf-8")
