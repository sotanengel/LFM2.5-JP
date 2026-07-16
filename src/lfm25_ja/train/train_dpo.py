"""Direct Preference Optimization (DPO) via `trl.DPOTrainer`, config-driven
(Issue #115 / #39 / #40 / #42).

Trains via the same layer-selective full-parameter fine-tuning as CPT/SFT
(see ``lfm25_ja.train.layer_select``): the policy model is loaded in bf16,
every parameter is frozen, and only ``tuning.trainable_layer_indices`` is
unfrozen -- no LoRA/PEFT adapters are used. Phase 3's safe operating point is
a single layer (L9); see ``configs/dpo/dpo_001_beta*.yaml``.

Training data is a JSONL of ``{"prompt": str, "chosen": str, "rejected": str,
"meta": {...}}`` rows (the ``dpo_pairs.jsonl`` contract produced by the
sibling preference-data pipeline). ``chosen``/``rejected`` are response text
only, with no chat template applied; ``build_dpo_prompt`` renders ``prompt``
through the chat template (with ``add_generation_prompt=True``) so that the
installed trl release's non-conversational path -- which builds training
sequences via plain string concatenation ``example["prompt"] +
example["chosen"]`` (see ``trl.trainer.dpo_trainer.DPOTrainer._prepare_dataset``)
-- produces a well-formed ChatML sequence without needing conversational
(list-of-message-dicts) columns.

8GB VRAM constraint: no reference model is ever loaded. ``ref_model=None`` +
``DPOConfig(precompute_ref_log_probs=True)`` computes reference log-probs
once, up front, from the policy model's initial weights (under
``torch.no_grad()``, before any optimizer step -- see
``trl.trainer.dpo_trainer.DPOTrainer.compute_ref_log_probs``: with no
separate ``ref_model``, it runs ``self.model`` itself and never touches a
second copy afterward). This project's installed trl release (1.7.0) has no
separate ``max_prompt_length`` argument on ``DPOConfig`` (older trl releases
did); see ``build_dpo_training_args`` for how that config key's *intent* is
still honored.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lfm25_ja.data.clean import _read_jsonl
from lfm25_ja.data.format_chat import to_chatml
from lfm25_ja.train.callbacks import LossTrackerCallback, VramMonitorCallback
from lfm25_ja.train.layer_select import select_trainable_layers, trainable_param_summary
from lfm25_ja.train.train_cpt import parse_layer_indices, resolve_resume_checkpoint
from lfm25_ja.train.train_sft import build_sft_run_name, resolve_trainable_layer_indices
from lfm25_ja.utils.config import load_config, load_project_config, merge_configs
from lfm25_ja.utils.memory import get_vram_usage, reset_peak_memory
from lfm25_ja.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Chat-template opening tag for the assistant turn, appended after the
# rendered user turn so `prompt + chosen`/`prompt + rejected` (the installed
# trl release's non-conversational concatenation) forms a well-formed ChatML
# sequence. Mirrors `lfm25_ja.data.format_chat._ASSISTANT_TAG` (kept as a
# local literal rather than importing that private symbol).
_ASSISTANT_GENERATION_TAG = "<|im_start|>assistant\n"


def build_dpo_prompt(prompt_text: str, tokenizer: Any) -> str:
    """Render a raw user-turn prompt string into a chat-templated string
    ending right at the assistant turn's opening tag (i.e. as if
    ``add_generation_prompt=True``), returned as plain text (not token ids).

    Uses the tokenizer's own ``apply_chat_template`` when available (the real
    HF tokenizer path), requesting ``tokenize=False`` so the return value is
    a string, not token ids -- trl's ``DPOTrainer`` does its own tokenization
    of the ``prompt``/``chosen``/``rejected`` columns (see module docstring),
    so applying the chat template here must stop at rendering text.
    Falls back to ``format_chat.to_chatml`` + the assistant tag for
    tokenizers with no ``apply_chat_template`` (e.g. the mock tokenizers used
    in tests).
    """
    messages = [{"role": "user", "content": prompt_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return to_chatml(messages) + _ASSISTANT_GENERATION_TAG


def build_dpo_dataset(jsonl_path: str | Path, tokenizer: Any) -> Any:
    """Read a ``dpo_pairs.jsonl`` (``{"prompt", "chosen", "rejected", "meta"}``
    rows) and build a ``datasets.Dataset`` with only ``prompt``/``chosen``/
    ``rejected`` columns (``meta`` is dropped; trl's ``DPOTrainer`` has no use
    for it and it isn't a valid model input).

    ``prompt`` is rendered through :func:`build_dpo_prompt`; ``chosen``/
    ``rejected`` are passed through unchanged (response text only, no
    template applied -- the dpo_pairs.jsonl contract).
    """
    from datasets import Dataset

    docs = _read_jsonl(jsonl_path)
    if not docs:
        raise ValueError(f"No examples found in {jsonl_path!r}")
    rows = [
        {
            "prompt": build_dpo_prompt(doc["prompt"], tokenizer),
            "chosen": doc["chosen"],
            "rejected": doc["rejected"],
        }
        for doc in docs
    ]
    return Dataset.from_list(rows)


def ensure_disk_backed(dataset: Any, cache_dir: str | Path) -> Any:
    """Round-trip a ``datasets.Dataset`` through ``save_to_disk``/
    ``load_from_disk`` so it is arrow-file-backed (``dataset.cache_files``
    non-empty).

    trl 1.7.0's ``precompute_ref_log_probs=True`` path requires this: its
    ``_precompute_ref_logps`` calls ``dataset.map(new_fingerprint=...)`` and
    then re-reads the ``cache-<fingerprint>.arrow`` file that map is expected
    to write -- but ``datasets`` only writes map cache files for disk-backed
    datasets, so an in-memory ``Dataset.from_list`` fails with
    ``FileNotFoundError`` on the cache path (found in the Issue #115 smoke
    run). ``cache_dir`` should live under the run's ``output_dir`` so the
    arrow copy is cleaned up with the run.
    """
    from datasets import load_from_disk

    cache_dir = Path(cache_dir)
    dataset.save_to_disk(str(cache_dir))
    reloaded = load_from_disk(str(cache_dir))
    assert reloaded.cache_files, "load_from_disk must yield a disk-backed dataset"
    return reloaded


def build_dpo_run_name(prefix: str, layer_indices: list[int], layers_overridden: bool) -> str:
    """Compose the DPO run name / output subdirectory.

    Identical rule to :func:`lfm25_ja.train.train_sft.build_sft_run_name`:
    DPO has no data-package axis either (no packed-cache full/centi/deci
    split), so the naming rule is reused as-is rather than reimplemented.
    """
    return build_sft_run_name(prefix, layer_indices, layers_overridden)


def build_dpo_training_args(
    training_cfg: dict[str, Any],
    output_dir: str,
    no_checkpoints: bool,
    precision: str | None,
) -> Any:
    """Build the ``trl.DPOConfig`` for this run, pinned to the installed trl
    release (1.7.0 at the time of writing; verify via ``python -c "import
    trl; print(trl.__version__)"`` in the WSL venv before touching this
    function -- ``DPOConfig``'s argument names vary a lot across trl
    releases).

    Notable version-specific mapping:

    - ``precompute_ref_log_probs=True``: always on (see module docstring --
      this is what lets us skip loading a second reference model).
    - ``max_length``: the *combined* prompt+completion token budget (trl
      1.7.0 has no separate ``max_prompt_length`` on ``DPOConfig``; that
      argument existed in older trl releases but was removed). We keep
      ``training.max_prompt_length`` in the YAML config schema for
      documentation / forward-compat with the Issue #115 spec, but it is not
      passed to ``DPOConfig`` directly under this trl version.
    - ``truncation_mode``: left at the trl default (``"keep_start"``,
      truncates the completion end when ``prompt+completion`` exceeds
      ``max_length``). trl 1.7.0's other option, ``"keep_end"`` (which would
      truncate the prompt start instead -- closer to old-trl's
      ``max_prompt_length`` intent), is already deprecated with a
      ``FutureWarning`` and documented for removal in trl v2.0.0 (see
      ``trl.trainer.dpo_config.DPOConfig.__post_init__``), so depending on it
      here would just trade one version-fragility for another. In practice
      ``configs/dpo/dpo_001_beta*.yaml`` sets ``max_length=1024`` for
      IFEval-style prompts with short completions, so this rarely triggers.
    """
    from trl import DPOConfig

    return DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 4)),
        num_train_epochs=float(training_cfg.get("num_train_epochs", 1)),
        max_steps=int(training_cfg.get("max_steps", -1)),
        learning_rate=float(training_cfg.get("learning_rate", 5e-6)),
        logging_steps=int(training_cfg.get("logging_steps", 5)),
        save_steps=int(training_cfg.get("save_steps", 100)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.0)),
        save_strategy="no" if no_checkpoints else "steps",
        report_to=[],
        fp16=False,
        bf16=precision == "bf16",
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        optim=str(training_cfg.get("optim", "paged_adamw_8bit")),
        beta=float(training_cfg.get("beta", 0.1)),
        max_length=int(training_cfg.get("max_length", 1024)),
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=int(
            training_cfg.get(
                "precompute_ref_batch_size", training_cfg.get("per_device_train_batch_size", 1)
            )
        ),
    )


# ---------------------------------------------------------------------------
# CPU-only dry run (no HF download / GPU / real trl.DPOTrainer)
# ---------------------------------------------------------------------------


class _TinyDPOStack(nn.Module):
    """Stand-in for `model.model.layers` used to exercise select_trainable_layers."""

    def __init__(self, dim: int, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim, bias=True) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _TinyDPOModel(nn.Module):
    """Small token-embedding + multi-layer + LM-head model mimicking the HF
    causal LM layout (`model.model.layers`) for the CPU dry_run path.

    Used only so dry_run can exercise the same freeze -> select layers ->
    sequence-logprob -> DPO loss pipeline as the real bf16 model, without
    downloading any weights.
    """

    def __init__(self, vocab_size: int = 32, dim: int = 8, n_layers: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.model = _TinyDPOStack(dim, n_layers)
        self.lm_head = nn.Linear(dim, vocab_size, bias=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = self.model(x)
        return self.lm_head(x)


def _sequence_logps(
    model: nn.Module, input_ids: torch.Tensor, completion_mask: torch.Tensor
) -> torch.Tensor:
    """Sum of per-token log-probabilities of the actual next token, restricted
    to the completion span (``completion_mask``), for a causal LM.

    Mirrors (in miniature) the token log-prob accumulation trl's real
    ``DPOTrainer`` does over ``completion_mask`` (see
    ``trl.trainer.dpo_trainer.DataCollatorForPreference`` for the equivalent
    mask it builds from ``prompt_ids``/``chosen_ids``/``rejected_ids``
    lengths).
    """
    logits = model(input_ids)[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = completion_mask[:, 1:].to(logits.dtype)
    logps = F.log_softmax(logits, dim=-1)
    token_logps = logps.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(-1)


def _dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Standard sigmoid DPO loss (Rafailov et al. 2023; trl's default
    ``loss_type="sigmoid"``)."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    return -F.logsigmoid(beta * logits).mean()


def _run_dry_run_dpo(cfg: dict[str, Any], max_steps: int = 20) -> dict[str, Any]:
    """CPU-only dry run: synthetic preference pairs through a tiny multi-layer
    model, exercising freeze -> select layers -> precompute reference
    log-probs once (before any optimizer step) -> DPO loss decreases, without
    any HF download or real ``trl.DPOTrainer``.

    The "precompute once, reuse forever" step below is the dry-run analogue
    of trl's ``precompute_ref_log_probs=True`` + ``ref_model=None`` contract:
    reference log-probs are computed under ``torch.no_grad()`` from the
    model's *initial* weights, cached, and never recomputed inside the
    training loop -- the model instance is never duplicated.
    """
    tuning = cfg.get("tuning", {})
    raw_indices = tuning.get("trainable_layer_indices", [1]) or [1]
    n_layers = 4

    model = _TinyDPOModel(vocab_size=32, dim=8, n_layers=n_layers)
    if raw_indices == "all":
        layer_indices = list(range(n_layers))
    else:
        # Map configured (possibly out-of-range for this tiny model) indices
        # into a valid range so the dry run stays fast while still
        # exercising the real select_trainable_layers() call path used
        # against the full model.
        layer_indices = sorted({idx % n_layers for idx in raw_indices})
    select_trainable_layers(model, layer_indices)
    trainable_summary = trainable_param_summary(model)

    vocab_size = 32
    seq_len = 8
    batch_size = 2
    prompt_len = seq_len // 2
    input_ids_chosen = torch.randint(0, vocab_size, (batch_size, seq_len))
    input_ids_rejected = torch.randint(0, vocab_size, (batch_size, seq_len))
    completion_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    completion_mask[:, prompt_len:] = 1

    # Precompute reference log-probs from the model's initial weights, once,
    # before any optimizer step -- see docstring above.
    model.eval()
    with torch.no_grad():
        ref_chosen_logps = _sequence_logps(model, input_ids_chosen, completion_mask)
        ref_rejected_logps = _sequence_logps(model, input_ids_rejected, completion_mask)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=0.1)
    beta = float(cfg.get("training", {}).get("beta", 0.1))

    losses: list[float] = []
    for _ in range(max_steps):
        policy_chosen_logps = _sequence_logps(model, input_ids_chosen, completion_mask)
        policy_rejected_logps = _sequence_logps(model, input_ids_rejected, completion_mask)
        loss = _dpo_loss(
            policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "losses": losses,
        "trainable_summary": trainable_summary,
    }


def _load_dpo_config(config_path: str) -> dict[str, Any]:
    """Deep-merge a dpo config (e.g. configs/dpo/dpo_001_beta005.yaml) over base.yaml."""
    base_cfg = load_project_config("base.yaml")
    dpo_cfg = load_config(config_path)
    return merge_configs(base_cfg, dpo_cfg)


def run_dpo(
    config_path: str,
    dry_run: bool = False,
    layers: list[int] | None = None,
    no_checkpoints: bool = False,
    output_root: str | None = None,
    max_steps: int = 20,
) -> dict[str, Any]:
    """Run TRL-DPOTrainer-based direct preference optimization driven by
    ``config_path``.

    ``config_path`` is deep-merged over ``configs/base.yaml`` (dpo config
    wins). With ``dry_run=True``, no HF download happens: a tiny CPU model is
    trained on synthetic preference pairs (with reference log-probs
    precomputed once up front, mirroring ``precompute_ref_log_probs=True``)
    to exercise the layer-select + training loop, returning
    ``{"initial_loss", "final_loss", "losses", "trainable_summary"}``.
    ``max_steps`` only applies to the dry-run loop.

    ``layers``, when given, overrides ``tuning.trainable_layer_indices`` from
    the config (see :func:`lfm25_ja.train.train_cpt.parse_layer_indices` for
    the CLI form). It also changes the run name (see
    :func:`build_dpo_run_name`).

    ``no_checkpoints`` disables intermediate checkpoint saving
    (``save_strategy="no"``); only the final model is written via
    ``trainer.save_model``.

    ``output_root``, when given, overrides ``output_dir`` from the config as
    the root directory the run name is written under (default ``outputs``).
    """
    cfg = _load_dpo_config(config_path)
    set_seed(int(cfg.get("seed", 42)))

    if dry_run:
        return _run_dry_run_dpo(cfg, max_steps=max_steps)

    # Imported lazily so dry_run (and CPU-only test/CI environments) never
    # need a working HF download / GPU stack.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer

    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy model only -- no ref model is ever loaded (see module docstring:
    # ref_model=None + DPOConfig(precompute_ref_log_probs=True)).
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    tuning = cfg.get("tuning", {})
    layers_overridden = layers is not None
    raw_indices = list(layers) if layers_overridden else tuning.get("trainable_layer_indices", [])
    layer_indices = resolve_trainable_layer_indices(raw_indices, model)
    select_trainable_layers(model, layer_indices)
    summary = trainable_param_summary(model)
    logger.info("Trainable param summary: %s", summary)

    dataset_cfg = cfg.get("dataset", {})
    if "train_path" not in dataset_cfg:
        raise ValueError("dpo config must set dataset.train_path (see configs/dpo/*.yaml)")
    dataset = build_dpo_dataset(dataset_cfg["train_path"], tokenizer)

    sample_fraction = dataset_cfg.get("sample_fraction")
    if sample_fraction is not None:
        n = max(1, int(len(dataset) * float(sample_fraction)))
        dataset = dataset.select(range(n))

    training_cfg = cfg.get("training", {})
    logging_cfg = cfg.get("logging", {})
    run_name_prefix = logging_cfg.get("run_name_prefix", "dpo")
    run_name = build_dpo_run_name(run_name_prefix, layer_indices, layers_overridden)
    output_root_dir = (
        Path(output_root) if output_root is not None else Path(cfg.get("output_dir", "outputs"))
    )
    output_dir = str(output_root_dir / run_name)

    # See ensure_disk_backed docstring: required for trl 1.7.0's
    # precompute_ref_log_probs cache round-trip.
    dataset = ensure_disk_backed(dataset, Path(output_dir) / "_dataset")

    vram_cb = VramMonitorCallback()
    loss_cb = LossTrackerCallback()

    args = build_dpo_training_args(
        training_cfg,
        output_dir=output_dir,
        no_checkpoints=no_checkpoints,
        precision=cfg.get("precision"),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[vram_cb, loss_cb],
    )
    reset_peak_memory()
    # Resume from the latest checkpoint in output_dir when one exists (e.g. an
    # interrupted run being restarted); otherwise start fresh (see
    # resolve_resume_checkpoint for why the path is resolved explicitly
    # rather than passing resume_from_checkpoint=True).
    resume_checkpoint = resolve_resume_checkpoint(output_dir, no_checkpoints)
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(output_dir)

    losses = loss_cb.losses or [0.0, 0.0]
    peak = max(get_vram_usage()["max_allocated"], vram_cb.peak_bytes)
    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "losses": losses,
        "trainable_summary": summary,
        "peak_vram_bytes": peak,
        "output_dir": output_dir,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="LFM2.5-JA DPO training via trl.DPOTrainer (Issue #115 / #39)"
    )
    parser.add_argument("--config", required=True, help="Path to configs/dpo/*.yaml")
    parser.add_argument(
        "--dry-run", action="store_true", help="CPU-only dry run, no HF download / GPU required"
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated 0-based layer indices, e.g. '7,8'. Overrides "
        "tuning.trainable_layer_indices from the config and changes the run name.",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Disable intermediate checkpoint saving (save_strategy=no); only the "
        "final model is written via trainer.save_model",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root directory the run name is written under (default: outputs, "
        "or config output_dir). Use e.g. 'outputs/sweep' for sweep runs.",
    )
    args = parser.parse_args()

    layers = parse_layer_indices(args.layers) if args.layers is not None else None

    result = run_dpo(
        args.config,
        dry_run=args.dry_run,
        layers=layers,
        no_checkpoints=args.no_checkpoints,
        output_root=args.output_root,
    )
    print(
        f"DPO run finished: loss {result['initial_loss']:.4f} -> {result['final_loss']:.4f} "
        f"(trainable={result['trainable_summary']['trainable_pct']:.3f}%)"
    )
    if result["final_loss"] >= result["initial_loss"]:
        raise SystemExit("DPO run failed: loss did not decrease")


if __name__ == "__main__":
    main()
