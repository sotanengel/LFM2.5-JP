"""Shell wrapper smoke checks."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def test_train_cpt_b_script_exists_and_uses_lf_line_endings() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "21_train_cpt_b.sh"
    content = script.read_bytes()
    assert content.startswith(b"#!/usr/bin/env bash\n")
    assert b"\r\n" not in content
    assert b"--package" in content
    assert b"data/processed_phase2/mixture.jsonl" in content


def test_train_cpt_b_script_has_valid_bash_syntax() -> None:
    bash = shutil.which("bash")
    if bash is None:
        return
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "21_train_cpt_b.sh"
    # Git Bash on Windows mangles Windows paths; Linux/WSL CI validates syntax.
    if "\\" in str(script):
        return
    subprocess.run([bash, "-n", str(script)], check=True)
