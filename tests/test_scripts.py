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


def test_train_cpt_b_script_accepts_deci_package() -> None:
    # packed_cache.PACKAGES includes "deci" (see Issue #75 layer profiling
    # follow-up); the launcher's validation must accept it too.
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "21_train_cpt_b.sh"
    content = script.read_text(encoding="utf-8")
    assert "full | centi | deci" in content


def test_train_cpt_script_accepts_deci_package() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "20_train_cpt.sh"
    content = script.read_text(encoding="utf-8")
    assert "full|centi|deci" in content


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


def test_train_sft_script_exists_and_uses_lf_line_endings() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "30_train_sft.sh"
    content = script.read_bytes()
    assert content.startswith(b"#!/usr/bin/env bash\n")
    assert b"\r\n" not in content
    assert b"lfm25_ja.train.train_sft" in content
    assert b"configs/sft/" in content


def test_train_sft_script_has_valid_bash_syntax() -> None:
    bash = shutil.which("bash")
    if bash is None:
        return
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "30_train_sft.sh"
    # Git Bash on Windows mangles Windows paths; Linux/WSL CI validates syntax.
    if "\\" in str(script):
        return
    subprocess.run([bash, "-n", str(script)], check=True)
