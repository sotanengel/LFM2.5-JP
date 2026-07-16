"""DPO training tests: TRL DPOTrainer wrapper (Issue #115 / #39 / #40 / #42).

Covers config loading/validation, run-name composition, layer-selection
(reusing ``lfm25_ja.train.train_sft.resolve_trainable_layer_indices``),
preference-dataset construction (dropping ``meta``, rendering ``prompt``
through the chat template while leaving ``chosen``/``rejected`` as raw
response text -- see ``build_dpo_prompt`` docstring for why), resume
resolution (reused from ``train_cpt``), a ``DPOConfig`` construction smoke
test pinned to the installed trl API, and a fast CPU-only dry-run smoke test
that a few training steps reduce loss while only exercising cached
("precomputed") reference log-probs -- mirroring trl's
``precompute_ref_log_probs=True`` + ``ref_model=None`` contract without
needing a real HF download.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lfm25_ja.data.clean import _write_jsonl
from lfm25_ja.train.train_dpo import (
    build_dpo_dataset,
    build_dpo_prompt,
    build_dpo_run_name,
    ensure_disk_backed,
    resolve_resume_checkpoint,
    run_dpo,
)
from lfm25_ja.utils.config import load_config, load_project_config, merge_configs

# ---------------------------------------------------------------------------
# Mock tokenizers
# ---------------------------------------------------------------------------


class MockNoTemplateTokenizer:
    """Deterministic whitespace tokenizer with no ``apply_chat_template``
    attribute, so ``build_dpo_prompt`` exercises the ``to_chatml()`` fallback
    (mirrors ``MockSFTTokenizer`` in ``tests/test_train_sft.py``).
    """

    def __init__(self) -> None:
        self.eos_token = "<eos>"
        self.eos_token_id = 0

    def __call__(self, text: str, **kwargs) -> dict[str, list[int]]:
        tokens = text.split()
        return {"input_ids": list(range(len(tokens)))}


class MockChatTemplateTokenizer:
    """Stand-in HF tokenizer exposing ``apply_chat_template`` so
    ``build_dpo_prompt`` exercises the real-tokenizer branch.
    """

    def __init__(self) -> None:
        self.eos_token = "<eos>"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        assert tokenize is False, "build_dpo_prompt must request a string, not token ids"
        assert add_generation_prompt is True
        rendered = "".join(f"[{m['role']}]{m['content']}" for m in messages)
        return rendered + "[gen]"


# ---------------------------------------------------------------------------
# build_dpo_prompt
# ---------------------------------------------------------------------------


def test_build_dpo_prompt_uses_apply_chat_template_when_available() -> None:
    tokenizer = MockChatTemplateTokenizer()
    result = build_dpo_prompt("hello there", tokenizer)
    assert result == "[user]hello there[gen]"


def test_build_dpo_prompt_falls_back_to_chatml_without_apply_chat_template() -> None:
    tokenizer = MockNoTemplateTokenizer()
    result = build_dpo_prompt("hello there", tokenizer)
    assert result == "<|im_start|>user\nhello there<|im_end|>\n<|im_start|>assistant\n"


# ---------------------------------------------------------------------------
# build_dpo_dataset (dpo_pairs.jsonl -> {"prompt","chosen","rejected"} dataset)
# ---------------------------------------------------------------------------


def test_build_dpo_dataset_reads_jsonl_and_drops_meta(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "dpo_pairs.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "prompt": "hello there",
                "chosen": "hi friend",
                "rejected": "go away",
                "meta": {"source": "unit-test"},
            }
        ],
    )
    tokenizer = MockNoTemplateTokenizer()
    dataset = build_dpo_dataset(str(jsonl_path), tokenizer)

    assert len(dataset) == 1
    row = dataset[0]
    assert set(row.keys()) == {"prompt", "chosen", "rejected"}
    assert row["prompt"] == "<|im_start|>user\nhello there<|im_end|>\n<|im_start|>assistant\n"
    assert row["chosen"] == "hi friend"
    assert row["rejected"] == "go away"


def test_build_dpo_dataset_multiple_rows(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "dpo_pairs.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {"prompt": "a", "chosen": "b", "rejected": "c", "meta": {}},
            {"prompt": "d", "chosen": "e", "rejected": "f", "meta": {}},
        ],
    )
    dataset = build_dpo_dataset(str(jsonl_path), MockNoTemplateTokenizer())
    assert len(dataset) == 2


def test_ensure_disk_backed_yields_cache_files(tmp_path: Path) -> None:
    # trl 1.7.0 precompute_ref_log_probs requires a disk-backed dataset (its
    # map(new_fingerprint=...) cache round-trip); Dataset.from_list is
    # in-memory (cache_files empty) and must be converted.
    datasets = pytest.importorskip("datasets")
    ds = datasets.Dataset.from_list([{"prompt": "a", "chosen": "b", "rejected": "c"}])
    assert not ds.cache_files
    backed = ensure_disk_backed(ds, tmp_path / "_dataset")
    assert backed.cache_files
    assert backed[0] == {"prompt": "a", "chosen": "b", "rejected": "c"}


def test_build_dpo_dataset_empty_input_raises(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "empty.jsonl"
    jsonl_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        build_dpo_dataset(str(jsonl_path), MockNoTemplateTokenizer())


# ---------------------------------------------------------------------------
# build_dpo_run_name (identical rule to build_sft_run_name; DPO has no
# data-package axis either)
# ---------------------------------------------------------------------------


def test_build_dpo_run_name_no_override_is_prefix() -> None:
    assert build_dpo_run_name("dpo-001-b005", [9], layers_overridden=False) == "dpo-001-b005"


def test_build_dpo_run_name_with_override_includes_layers() -> None:
    assert build_dpo_run_name("dpo-001-b005", [6, 9], layers_overridden=True) == "dpo-001-b005-L6-9"


# ---------------------------------------------------------------------------
# resolve_resume_checkpoint (re-exported from train_cpt; exercised here so the
# DPO CLI's --no-checkpoints / auto-resume wiring is covered directly)
# ---------------------------------------------------------------------------


def test_resolve_resume_checkpoint_no_checkpoints_flag_is_always_none(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    (output_dir / "checkpoint-500").mkdir(parents=True)
    assert resolve_resume_checkpoint(output_dir, no_checkpoints=True) is None


def test_resolve_resume_checkpoint_missing_output_dir_is_none(tmp_path: Path) -> None:
    assert resolve_resume_checkpoint(tmp_path / "does-not-exist", no_checkpoints=False) is None


def test_resolve_resume_checkpoint_returns_latest_checkpoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    (output_dir / "checkpoint-500").mkdir(parents=True)
    (output_dir / "checkpoint-1000").mkdir(parents=True)
    result = resolve_resume_checkpoint(output_dir, no_checkpoints=False)
    assert result == str(output_dir / "checkpoint-1000")


# ---------------------------------------------------------------------------
# DPOConfig construction -- pinned to the installed trl API (Issue #115: "trl
# のバージョンを最初に確認し...DPOConfig の引数名は版差が大きい"). This runs
# against the real `trl` package (available in the WSL venv the project tests
# run under) but never loads a model or dataset, so it stays fast and needs
# no HF download.
# ---------------------------------------------------------------------------


def test_dpo_config_accepts_wired_arguments(monkeypatch, tmp_path: Path) -> None:
    trl = pytest.importorskip("trl")
    from lfm25_ja.train.train_dpo import build_dpo_training_args

    # This test only checks that build_dpo_training_args maps our config dict
    # onto the installed trl DPOConfig/TrainingArguments API correctly -- it
    # never actually trains. transformers.TrainingArguments.__post_init__
    # rejects bf16=True on hardware without a bf16-capable GPU (CI runners
    # have no GPU at all, unlike the CUDA-equipped WSL box these tests were
    # authored against), so the wiring check itself would otherwise be
    # hardware-dependent. Pin is_torch_bf16_gpu_available() to True so this
    # test exercises the precision="bf16" -> bf16=True mapping on any runner.
    monkeypatch.setattr("transformers.training_args.is_torch_bf16_gpu_available", lambda: True)

    training_cfg = {
        "beta": 0.05,
        "learning_rate": 5e-6,
        "num_train_epochs": 1,
        "max_length": 1024,
        "max_prompt_length": 512,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
    }
    args = build_dpo_training_args(
        training_cfg, output_dir=str(tmp_path / "out"), no_checkpoints=True, precision="bf16"
    )

    assert isinstance(args, trl.DPOConfig)
    assert args.beta == pytest.approx(0.05)
    assert args.learning_rate == pytest.approx(5e-6)
    assert args.max_length == 1024
    assert args.precompute_ref_log_probs is True
    # trl default (keep_start): "keep_end" is deprecated in trl 1.7.0 and
    # slated for removal in v2.0.0, so build_dpo_training_args deliberately
    # doesn't set it (see its docstring).
    assert args.truncation_mode == "keep_start"
    assert args.save_strategy == "no"


# ---------------------------------------------------------------------------
# run_dpo(dry_run=True) -- CPU-only smoke test, no HF download
# ---------------------------------------------------------------------------


def test_run_dpo_dry_run_loss_decreases_and_summary_consistent(tmp_path: Path) -> None:
    dpo_config = tmp_path / "dpo_test.yaml"
    dpo_config.write_text(
        """
model_name: dummy/does-not-matter
tuning:
  trainable_layer_indices: [1]
training:
  beta: 0.1
""",
        encoding="utf-8",
    )
    result = run_dpo(str(dpo_config), dry_run=True)

    assert result["final_loss"] < result["initial_loss"]
    assert len(result["losses"]) >= 2
    summary = result["trainable_summary"]
    assert 0 < summary["trainable_params"] < summary["total_params"]
    assert summary["trainable_pct"] == pytest.approx(
        summary["trainable_params"] / summary["total_params"] * 100.0
    )


def test_run_dpo_dry_run_supports_all_layers(tmp_path: Path) -> None:
    single_layer_config = tmp_path / "dpo_single_layer.yaml"
    single_layer_config.write_text(
        """
model_name: dummy/does-not-matter
tuning:
  trainable_layer_indices: [1]
""",
        encoding="utf-8",
    )
    all_layers_config = tmp_path / "dpo_full_ft.yaml"
    all_layers_config.write_text(
        """
model_name: dummy/does-not-matter
tuning:
  trainable_layer_indices: all
""",
        encoding="utf-8",
    )
    single_result = run_dpo(str(single_layer_config), dry_run=True)
    all_result = run_dpo(str(all_layers_config), dry_run=True)

    assert all_result["final_loss"] < all_result["initial_loss"]
    single_summary = single_result["trainable_summary"]
    all_summary = all_result["trainable_summary"]
    assert all_summary["trainable_params"] == pytest.approx(4 * single_summary["trainable_params"])
    assert all_summary["trainable_pct"] > single_summary["trainable_pct"]


def test_run_dpo_dry_run_uses_precomputed_reference_log_probs(monkeypatch, tmp_path: Path) -> None:
    """The dry run must compute reference log-probs exactly once, before any
    optimizer step, and never re-run a second ("live") reference forward pass
    afterward -- this is the CPU-only analogue of trl's
    `precompute_ref_log_probs=True` + `ref_model=None` contract (see
    `lfm25_ja.train.train_dpo._run_dry_run_dpo` docstring).
    """
    import lfm25_ja.train.train_dpo as train_dpo_mod

    calls: list[str] = []
    original = train_dpo_mod._sequence_logps

    def _tracking_sequence_logps(model, input_ids, completion_mask):
        calls.append("call")
        return original(model, input_ids, completion_mask)

    monkeypatch.setattr(train_dpo_mod, "_sequence_logps", _tracking_sequence_logps)

    dpo_config = tmp_path / "dpo_test.yaml"
    dpo_config.write_text(
        """
model_name: dummy/does-not-matter
tuning:
  trainable_layer_indices: [1]
""",
        encoding="utf-8",
    )
    result = train_dpo_mod.run_dpo(str(dpo_config), dry_run=True, max_steps=5)

    # 2 reference calls (chosen + rejected) precomputed once up front, plus
    # 2 policy calls (chosen + rejected) per training step.
    assert calls.count("call") == 2 + 2 * 5
    assert result["final_loss"] < result["initial_loss"]


# ---------------------------------------------------------------------------
# config loading: fixture config merges over base.yaml
# ---------------------------------------------------------------------------


def test_dpo_fixture_config_merges_over_base() -> None:
    root = Path(__file__).resolve().parents[1]
    base_cfg = load_project_config("base.yaml")
    dpo_cfg = load_config(root / "tests" / "fixtures" / "dpo_test.yaml")
    merged = merge_configs(base_cfg, dpo_cfg)

    assert merged["model_name"] == "dummy/does-not-matter"
    assert merged["tuning"]["trainable_layer_indices"] == [9]
    assert merged["training"]["beta"] == pytest.approx(0.1)
    assert merged["training"]["learning_rate"] == pytest.approx(5.0e-6)
    assert merged["training"]["max_length"] == 1024
    assert merged["training"]["max_prompt_length"] == 512
    # not overridden by the fixture config -> inherited from base.yaml
    assert merged["training"]["per_device_train_batch_size"] == 1
    assert merged["dataset"]["train_path"] == "tests/fixtures/dpo_test_pairs.jsonl"
