"""dpo-001 LLM pointwise quality judge (Issue #115, Phase J, GPU).

Rule verifiers (Phase V, :mod:`lfm25_ja.data.pref_verify`) only check
instruction *compliance*. Among samples that already share the same
compliance verdict for a prompt, DPO still needs a chosen/rejected quality
ranking -- that's this module's job: a pointwise 1-5 quality score from
``tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2`` (NF4 4bit, bf16 compute, batched
inference, ``enable_thinking=False``), judging Japanese naturalness /
content accuracy / concision (and, for polite-form prompts, business-writing
appropriateness) -- explicitly *not* instruction compliance, which the rule
verifier already covers.

Judging is deliberately narrowed to samples that matter for pairing (see
:func:`select_judge_targets`): a prompt's samples are judged only if the
prompt has both a rule-pass and a rule-fail sample (a pair is potentially
formable and needs a quality ranking on both sides), or the prompt's category
is ``polite_form``/``no_constraint`` (categories where a bare rule-pass
doesn't by itself indicate a good *chosen* candidate -- tone/naturalness
still needs ranking even when every sample passes).

Output (``judgments.jsonl``, one row per judged (prompt_id, k)):
``{"prompt_id", "k", "score": <1-5 or null>, "reason": <str>}``. Idempotent
by (prompt_id, k) -- reruns after a WSL2 restart append only the still-
missing pairs (see :func:`_existing_judgment_keys`).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from lfm25_ja.data.clean import _read_jsonl
from lfm25_ja.utils.config import load_config

logger = logging.getLogger(__name__)

_JUDGE_PROMPT_TEMPLATE = (
    "あなたは日本語の文章品質を評価する採点者です。"
    "次のプロンプトと応答を読み、指示への適合性(文字数や書式などのルール遵守)は評価対象に含めず、"
    "日本語としての自然さ・内容の的確さ・簡潔さ"
    "(丁寧な文面が求められる場合は実務文面としての適切さ)のみを基準に、"
    "1〜5の整数で品質スコアを付けてください。\n"
    "応答は途中で切れている場合があります。その場合も、切れている箇所までの内容の品質で採点してください。\n"
    "\n"
    "# プロンプト\n"
    "{prompt}\n"
    "\n"
    "# 応答\n"
    "{response}\n"
    "\n"
    "以下の厳格なJSON形式のみで出力してください。他の説明文は一切含めないでください。\n"
    '{{"score": <1から5の整数>, "reason": "<採点理由を一文で>"}}\n'
)

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

_QUALITY_SELECT_CATEGORIES = frozenset({"polite_form", "no_constraint"})

_FACTUAL_JUDGE_PROMPT_TEMPLATE = (
    "あなたは日本の事実知識を評価する採点者です。"
    "次の質問に対する応答が、与えられた参照解・出典引用と照合して"
    "事実として正しいかを、1〜5の整数で採点してください。\n"
    "5=完全に正しい、4=ほぼ正しい(軽微な表現差のみ)、"
    "3=部分的に正しい/曖昧、2=重要な誤りあり、1=明らかな事実誤りまたは無関係。\n"
    "「わかりません」等の拒否応答は1点としてください。\n"
    "\n"
    "# 質問\n"
    "{question}\n"
    "\n"
    "# 参照解候補\n"
    "{reference_answers}\n"
    "\n"
    "# 出典引用\n"
    "{source_quote}\n"
    "\n"
    "# 応答\n"
    "{response}\n"
    "\n"
    "以下の厳格なJSON形式のみで出力してください。\n"
    '{{"score": <1から5の整数>, "reason": "<採点理由を一文で>"}}\n'
)


def build_judge_prompt(prompt: str, response: str) -> str:
    """Build the (Japanese) judge prompt for one (prompt, response) sample."""
    return _JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, response=response)


def build_factual_judge_prompt(
    question: str,
    reference_answers: list[str],
    source_quote: str,
    response: str,
) -> str:
    """Build the factual correctness judge prompt for K3 (Issue #124)."""
    refs = " / ".join(reference_answers) if reference_answers else "(MCQ: 参照解は採点時に質問文から判断)"
    return _FACTUAL_JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        reference_answers=refs,
        source_quote=source_quote or "(なし)",
        response=response,
    )


def parse_judge_output(text: str) -> dict[str, Any]:
    """Parse the judge model's raw text output into
    ``{"score": int|None, "reason": str}``.

    Tolerant of a leading/trailing preamble or a wrapping code fence around
    the JSON object (models sometimes add either) -- takes the first
    ``{...}`` block found via a greedy regex search. Never raises: any parse
    failure, missing ``score``, non-integer ``score``, or ``score`` outside
    1-5 all return ``score: None`` so callers can implement the "one retry,
    then null" policy purely by checking the returned score, not by catching
    exceptions."""
    match = _JSON_OBJ_RE.search(text or "")
    if not match:
        return {"score": None, "reason": ""}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"score": None, "reason": ""}
    if not isinstance(payload, dict):
        return {"score": None, "reason": ""}
    score = payload.get("score")
    reason = payload.get("reason", "")
    if not isinstance(score, int) or isinstance(score, bool) or not (1 <= score <= 5):
        return {"score": None, "reason": str(reason)}
    return {"score": score, "reason": str(reason)}


def select_judge_targets(
    prompts_by_id: dict[str, dict[str, Any]],
    generations: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select which generation rows (each ``{"prompt_id","k","response",...}``)
    need LLM judging -- see module docstring for the two inclusion rules.
    Samples with an empty (or whitespace-only) response are always skipped
    (nothing to score)."""
    pass_counts: dict[str, int] = {}
    fail_counts: dict[str, int] = {}
    for v in verdicts:
        counts = pass_counts if v["rule_pass"] else fail_counts
        counts[v["prompt_id"]] = counts.get(v["prompt_id"], 0) + 1

    targets = []
    for gen in generations:
        prompt_id = gen["prompt_id"]
        prompt_row = prompts_by_id.get(prompt_id)
        if prompt_row is None:
            continue
        response = gen.get("response", "")
        if not response.strip():
            continue

        has_pair_potential = (
            pass_counts.get(prompt_id, 0) >= 1 and fail_counts.get(prompt_id, 0) >= 1
        )
        quality_select_category = prompt_row["category"] in _QUALITY_SELECT_CATEGORIES
        if has_pair_potential or quality_select_category:
            targets.append(gen)
    return targets


def select_factual_judge_targets(
    prompts_by_id: dict[str, dict[str, Any]],
    generations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """K3 factual mode: judge every non-empty generation (all need scores)."""
    targets = []
    for gen in generations:
        if gen["prompt_id"] not in prompts_by_id:
            continue
        if not (gen.get("response") or "").strip():
            continue
        targets.append(gen)
    return targets


def _judge_prompt_for_row(
    prompt_row: dict[str, Any], response: str, mode: str
) -> str:
    if mode == "factual":
        detail = prompt_row.get("constraint_detail") or {}
        return build_factual_judge_prompt(
            question=prompt_row["prompt"],
            reference_answers=list(detail.get("answers") or []),
            source_quote=str(detail.get("source_quote") or ""),
            response=response,
        )
    return build_judge_prompt(prompt_row["prompt"], response)


def _existing_judgment_keys(output_path: Path) -> set[tuple[str, int]]:
    if not output_path.exists():
        return set()
    keys = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            keys.add((row["prompt_id"], row["k"]))
    return keys


def build_judge_request_body(
    judge_prompt: str,
    model: str,
    max_tokens: int,
    sampled: bool = False,
) -> dict[str, Any]:
    """Build the OpenAI-compatible /v1/chat/completions request body for the
    ``vllm_server`` backend. ``chat_template_kwargs.enable_thinking=False``
    keeps parity with the transformers backend's empty-think injection (vLLM
    forwards it into the model's chat template). First attempt is greedy;
    ``sampled=True`` is the parse-failure retry (a greedy retry on identical
    input would reproduce the identical unparseable output)."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": judge_prompt}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if sampled:
        body["temperature"] = 0.6
        body["top_p"] = 0.95
    else:
        body["temperature"] = 0.0
    return body


def _post_chat_completion(server_url: str, body: dict[str, Any], timeout: float = 120.0) -> str:
    """POST one chat completion to a vLLM OpenAI-compatible server and return
    the message content. stdlib-only (urllib) so the frozen ~/lfm25-ja venv
    needs no new dependency."""
    import urllib.request

    request = urllib.request.Request(
        server_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"] or ""


def _judge_one_via_server(
    server_url: str, judge_prompt: str, model: str, max_tokens: int
) -> dict[str, Any]:
    """Greedy attempt + one sampled retry, mirroring the transformers path."""
    text = _post_chat_completion(
        server_url, build_judge_request_body(judge_prompt, model, max_tokens)
    )
    parsed = parse_judge_output(text)
    if parsed["score"] is None:
        text = _post_chat_completion(
            server_url, build_judge_request_body(judge_prompt, model, max_tokens, sampled=True)
        )
        parsed = parse_judge_output(text)
    return parsed


def _judge_via_server(
    pending: list[dict[str, Any]],
    prompts_by_id: dict[str, dict[str, Any]],
    output_path: Path,
    file_mode: str,
    judge_mode: str,
    server_url: str,
    model: str,
    max_tokens: int,
    concurrency: int,
) -> int:
    """Score ``pending`` against a running vLLM server (continuous batching:
    concurrency comes from parallel HTTP requests, ordered writes keep the
    append-only idempotency contract)."""
    from concurrent.futures import ThreadPoolExecutor

    def _score(gen: dict[str, Any]) -> dict[str, Any]:
        prompt_row = prompts_by_id[gen["prompt_id"]]
        judge_prompt = _judge_prompt_for_row(
            prompt_row, gen.get("response", ""), judge_mode
        )
        return _judge_one_via_server(server_url, judge_prompt, model, max_tokens)

    judged = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(file_mode, encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for gen, parsed in zip(pending, pool.map(_score, pending)):
                f.write(
                    json.dumps(
                        {"prompt_id": gen["prompt_id"], "k": gen["k"], **parsed},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                judged += 1
                if judged % 200 == 0:
                    f.flush()
                    logger.info("judged %d/%d pending samples", judged, len(pending))
    return judged


def run_judge(
    config_path: str | Path,
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run Phase J: select judge targets, skip already-judged (prompt_id, k)
    pairs (unless ``force``), and score the rest with Qwen3-Swallow-8B.

    Two backends (config ``judge.backend``):

    - ``"transformers"`` (default): in-process NF4 4bit batched inference of
      the full-precision release (works with only the frozen ~/lfm25-ja venv).
    - ``"vllm_server"``: OpenAI-compatible requests against a locally running
      native-WSL vLLM server hosting the official AWQ-INT4 release (see
      ``scripts/41_serve_judge_vllm.sh``; faster, no docker). The server must
      be started first and must not share the GPU with the LFM model.

    ``--dry-run`` reports the plan without importing torch or touching a GPU.
    """
    config = load_config(config_path)
    cfg = config.get("judge", config)

    prompts_path = cfg["prompts_path"]
    generations_path = cfg["generations_path"]
    verdicts_path = cfg["verdicts_path"]
    output_path = Path(cfg["output_path"])
    backend = str(cfg.get("backend", "transformers"))
    judge_mode = str(cfg.get("mode", "quality"))
    if backend not in ("transformers", "vllm_server"):
        raise ValueError(f"judge.backend must be 'transformers' or 'vllm_server', got {backend!r}")
    model_path = cfg.get("model_path", "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2")
    batch_size = int(cfg.get("batch_size", 8))
    max_new_tokens = int(cfg.get("max_new_tokens", 96))
    server_url = str(cfg.get("server_url", "http://127.0.0.1:8100/v1"))
    server_concurrency = int(cfg.get("server_concurrency", 8))

    prompts = _read_jsonl(prompts_path)
    prompts_by_id = {p["id"]: p for p in prompts}
    generations = _read_jsonl(generations_path)
    verdicts = _read_jsonl(verdicts_path)

    if judge_mode == "factual":
        targets = select_factual_judge_targets(prompts_by_id, generations)
    else:
        targets = select_judge_targets(prompts_by_id, generations, verdicts)
    if limit is not None:
        targets = targets[:limit]

    existing_keys = set() if force else _existing_judgment_keys(output_path)
    pending = [t for t in targets if (t["prompt_id"], t["k"]) not in existing_keys]

    if dry_run:
        return {
            "status": "dry_run",
            "backend": backend,
            "model_path": model_path,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "total_targets": len(targets),
            "already_judged": len(targets) - len(pending),
            "pending": len(pending),
        }

    if not pending:
        logger.info("nothing pending: %d/%d targets already judged", len(targets), len(targets))
        return {"status": "skipped", "total_targets": len(targets), "judged": 0}

    if backend == "vllm_server":
        file_mode = "w" if (force or not output_path.exists()) else "a"
        judged = _judge_via_server(
            pending,
            prompts_by_id,
            output_path,
            file_mode,
            judge_mode,
            server_url=server_url,
            model=model_path,
            max_tokens=max_new_tokens,
            concurrency=server_concurrency,
        )
        return {
            "status": "executed",
            "backend": backend,
            "total_targets": len(targets),
            "judged": judged,
        }

    # Imported lazily (GPU/bitsandbytes stack not needed for target selection,
    # dry-run planning, or the CPU-only unit tests -- same pattern as
    # generate_ifeval_ja.generate_for_model).
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("loading judge model %s", model_path)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map="auto"
    )
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if (force or not output_path.exists()) else "a"
    judged = 0
    with output_path.open(mode, encoding="utf-8") as f:
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]
            judge_texts = []
            for gen in batch:
                prompt_row = prompts_by_id[gen["prompt_id"]]
                judge_prompt = _judge_prompt_for_row(
                    prompt_row, gen.get("response", ""), judge_mode
                )
                # tokenize=False avoids the apply_chat_template BatchEncoding
                # trap documented in docs/agent_ops.md -- batching is done
                # via a plain tokenizer(..., padding=True) call below.
                judge_texts.append(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": judge_prompt}],
                        add_generation_prompt=True,
                        tokenize=False,
                        enable_thinking=False,
                    )
                )

            encoded = tokenizer(judge_texts, return_tensors="pt", padding=True).to(model.device)
            input_len = encoded["input_ids"].shape[-1]
            with torch.no_grad():
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    use_cache=True,
                )

            for i, gen in enumerate(batch):
                raw_text = tokenizer.decode(output_ids[i][input_len:], skip_special_tokens=True)
                parsed = parse_judge_output(raw_text)
                if parsed["score"] is None:
                    # Retry must sample: a greedy retry on identical input
                    # reproduces the identical unparseable output.
                    with torch.no_grad():
                        retry_ids = model.generate(
                            input_ids=encoded["input_ids"][i : i + 1],
                            attention_mask=encoded["attention_mask"][i : i + 1],
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.6,
                            top_p=0.95,
                            use_cache=True,
                        )
                    retry_text = tokenizer.decode(
                        retry_ids[0][input_len:], skip_special_tokens=True
                    )
                    parsed = parse_judge_output(retry_text)
                f.write(
                    json.dumps(
                        {"prompt_id": gen["prompt_id"], "k": gen["k"], **parsed},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                judged += 1

    return {"status": "executed", "total_targets": len(targets), "judged": judged}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="dpo-001 Qwen3-Swallow-8B pointwise quality judge (Issue #115)"
    )
    parser.add_argument(
        "--config",
        default="configs/data/dpo_pairs_001.yaml",
        help="Path to configs/data/dpo_pairs_001.yaml",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only judge the first N targets")
    parser.add_argument("--force", action="store_true", help="Rejudge even if output exists")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan only")
    args = parser.parse_args()

    result = run_judge(args.config, limit=args.limit, force=args.force, dry_run=args.dry_run)
    print(result)


if __name__ == "__main__":
    main()
