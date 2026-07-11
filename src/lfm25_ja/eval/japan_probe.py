"""Japan-knowledge probe: 10 fields x 5 questions across arbitrary models
(Issue #76 -- promoted from a scratchpad script, with a scoring fix).

Scoring bug this module fixes: the original scratchpad scorer matched an
expected answer as a substring *anywhere* in the model's full greedy
continuation. When a model answered incorrectly but then rambled into a
self-generated multiple-choice enumeration (e.g. "岐阜県 A:立山 ...
C:滋賀県"), the correct answer sitting inside one of the *choices* would
count as a hit even though the model never actually answered correctly.
:func:`extract_answer_segment` isolates just the model's first answer
(cutting at the first self-generated question, choice-list marker, or
paragraph break, then at the first sentence terminator) so scoring only
looks at what the model actually asserted first.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

BASE_MODEL = "LiquidAI/LFM2.5-1.2B-Base"

FEWSHOT = "質問: 1年は何ヶ月ありますか?\n答え: 12ヶ月です。\n\n"

# (分野, 質問, 正解サブストリング候補のリスト)
QUESTIONS: list[tuple[str, str, list[str]]] = [
    ("地理", "日本で一番高い山は何ですか?", ["富士山", "富士"]),
    ("地理", "日本で一番北にある都道府県はどこですか?", ["北海道"]),
    ("地理", "琵琶湖がある都道府県はどこですか?", ["滋賀"]),
    ("地理", "日本の首都はどこですか?", ["東京"]),
    ("地理", "日本で一番長い川は何ですか?", ["信濃川"]),
    ("歴史", "江戸幕府を開いた人物は誰ですか?", ["徳川家康", "家康"]),
    ("歴史", "大政奉還を行った江戸幕府最後の将軍は誰ですか?", ["徳川慶喜", "慶喜"]),
    ("歴史", "鎌倉幕府を開いた人物は誰ですか?", ["源頼朝", "頼朝"]),
    ("歴史", "平安京に都を移した天皇は誰ですか?", ["桓武"]),
    ("歴史", "明治維新が始まった明治元年は西暦何年ですか?", ["1868"]),
    ("文学", "『源氏物語』の作者は誰ですか?", ["紫式部"]),
    ("文学", "『吾輩は猫である』の作者は誰ですか?", ["夏目漱石", "漱石"]),
    ("文学", "『雪国』を書いたノーベル賞作家は誰ですか?", ["川端康成", "川端"]),
    ("文学", "『枕草子』の作者は誰ですか?", ["清少納言"]),
    ("文学", "『羅生門』の作者は誰ですか?", ["芥川龍之介", "芥川"]),
    ("食文化", "味噌の主な原料となる豆は何ですか?", ["大豆"]),
    ("食文化", "日本酒の主な原料は何ですか?", ["米"]),
    ("食文化", "だしを取るのに使われる代表的な海藻は何ですか?", ["昆布"]),
    ("食文化", "梅干しの原料となる果実は何ですか?", ["梅"]),
    ("食文化", "そばの麺の主な原料は何ですか?", ["そば粉", "ソバ", "蕎麦"]),
    ("伝統文化", "祇園祭が行われる都市はどこですか?", ["京都"]),
    ("伝統文化", "大相撲の力士の最高位は何ですか?", ["横綱"]),
    ("伝統文化", "茶道で点てて飲む飲み物は何ですか?", ["抹茶"]),
    ("伝統文化", "七夕の行事は何月に行われますか?", ["7月", "七月"]),
    ("伝統文化", "歌舞伎で女性の役を演じる男性役者を何と呼びますか?", ["女形", "おやま"]),
    ("政治", "日本の国会を構成する二つの議院は参議院と何ですか?", ["衆議院"]),
    ("政治", "日本の内閣の長は何と呼ばれますか?", ["総理大臣", "首相"]),
    ("政治", "日本の都道府県はいくつありますか?", ["47", "四十七"]),
    ("政治", "日本国憲法が施行されたのは西暦何年ですか?", ["1947"]),
    ("政治", "日本国憲法において天皇は日本国の何と定められていますか?", ["象徴"]),
    ("経済", "日本の通貨単位は何ですか?", ["円"]),
    ("経済", "トヨタ自動車の本社がある都道府県はどこですか?", ["愛知"]),
    ("経済", "日本の中央銀行の名称は何ですか?", ["日本銀行", "日銀"]),
    ("経済", "東海道新幹線を運行している会社は何ですか?", ["JR東海", "東海旅客"]),
    ("経済", "日本最大の証券取引所がある都市はどこですか?", ["東京"]),
    ("科学技術", "iPS細胞の研究でノーベル賞を受賞した日本人は誰ですか?", ["山中伸弥", "山中"]),
    (
        "科学技術",
        "探査機はやぶさ2が試料を持ち帰った小惑星の名前は何ですか?",
        ["リュウグウ", "竜宮"],
    ),
    ("科学技術", "日本の宇宙航空研究開発機構の略称は何ですか?", ["JAXA"]),
    (
        "科学技術",
        "スーパーコンピュータ「富岳」を開発した研究機関はどこですか?",
        ["理化学研究所", "理研"],
    ),
    (
        "科学技術",
        "カミオカンデでのニュートリノ観測でノーベル賞を受賞したのは誰ですか?",
        ["小柴昌俊", "小柴"],
    ),
    ("スポーツ", "柔道の創始者は誰ですか?", ["嘉納治五郎", "嘉納"]),
    ("スポーツ", "プロ野球で通算868本塁打の世界記録を持つ選手は誰ですか?", ["王貞治", "王"]),
    ("スポーツ", "東京にある大相撲の本場所が行われる施設は何ですか?", ["国技館"]),
    ("スポーツ", "日本のプロサッカーリーグの名称は何ですか?", ["Jリーグ", "J1"]),
    ("スポーツ", "剣道の試合で使う竹製の道具は何ですか?", ["竹刀"]),
    ("言語", "日本語の表音文字のうち、ひらがなともう一つは何ですか?", ["カタカナ", "片仮名"]),
    ("言語", "五・七・五の十七音で作る日本の短い詩を何と呼びますか?", ["俳句"]),
    ("言語", "手紙の冒頭の「拝啓」に対応する結びの言葉は何ですか?", ["敬具"]),
    ("言語", "五・七・五・七・七の三十一音で作る日本の詩を何と呼びますか?", ["短歌", "和歌"]),
    ("言語", "日本語の文字のうち、漢字を崩して作られた表音文字は何ですか?", ["ひらがな", "平仮名"]),
]

# Cuts a raw generated continuation at the first self-generated follow-up
# question, multiple-choice marker (e.g. "A:"), or blank-line paragraph
# break -- whichever appears first.
_BOUNDARY_RE = re.compile(r"Question:|質問[:：]|\b[A-E]:|\n\s*\n")


def extract_answer_segment(text: str) -> str:
    """Cut a model's raw greedy continuation down to just its first answer.

    Stops at the first boundary matched by ``_BOUNDARY_RE``, then trims the
    remainder to its first sentence (up to and including the first "。").
    Scoring must only look at this segment -- see the module docstring for
    why matching anywhere in the full continuation produces false positives.
    """
    if not text:
        return ""
    match = _BOUNDARY_RE.search(text)
    segment = text[: match.start()] if match else text
    period_idx = segment.find("。")
    if period_idx != -1:
        segment = segment[: period_idx + 1]
    return re.sub(r"\s+", " ", segment).strip()


def score_answer(raw_text: str, answers: list[str]) -> bool:
    """True when any expected answer is a substring of ``raw_text``'s
    extracted answer segment (see :func:`extract_answer_segment`)."""
    segment = extract_answer_segment(raw_text)
    return any(a in segment for a in answers)


def _legacy_score(raw_text: str, answers: list[str]) -> bool:
    """Reproduce the original (pre-fix) scratchpad scoring: substring match
    anywhere before the model's next self-generated "質問:". Used only to
    compute the diff column in the report, showing where the fix changes a
    verdict."""
    cleaned = re.sub(r"\s+", " ", raw_text).strip()
    first = cleaned.split("質問:")[0].strip()
    return any(a in first for a in answers)


def _parse_model_specs(items: list[str]) -> list[tuple[str, str]]:
    """Parse ``--models label=path ...`` entries into ``(label, path)`` pairs."""
    specs: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--models entries must be label=path, got {item!r}")
        label, path = item.split("=", 1)
        if not label or not path:
            raise ValueError(f"--models entries must be label=path, got {item!r}")
        specs.append((label, path))
    return specs


def run_japan_probe(
    model_specs: list[tuple[str, str]],
    max_new_tokens: int = 40,
    fewshot: str = FEWSHOT,
) -> dict[tuple[str, int], str]:
    """Run every question in :data:`QUESTIONS` through each ``(label, path)``
    model and return the raw (un-scored) greedy continuations, keyed by
    ``(label, question_index)``.
    """
    if not model_specs:
        raise ValueError("model_specs must not be empty")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        raise RuntimeError(f"Failed to load tokenizer from {BASE_MODEL!r}: {exc}") from exc

    results: dict[tuple[str, int], str] = {}
    for label, path in model_specs:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            raise RuntimeError(f"Failed to load model {label!r} from {path!r}: {exc}") from exc
        model.eval()
        for i, (field, q, _answers) in enumerate(QUESTIONS):
            prompt = f"{fewshot}質問: {q}\n答え:"
            ids = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.05,
                )
            gen = tokenizer.decode(out[0][ids.input_ids.shape[1] :], skip_special_tokens=True)
            results[(label, i)] = gen
            logger.info("[%s] %d/%d %s: %s", label, i + 1, len(QUESTIONS), field, gen[:60])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def build_report(
    model_specs: list[tuple[str, str]],
    questions: list[tuple[str, str, list[str]]],
    raw_results: dict[tuple[str, int], str],
) -> str:
    """Build the markdown report: per-field score table, then a full answer
    table with a diff column ("旧O→新X") flagging where the old
    anywhere-substring scoring disagrees with the fixed
    :func:`extract_answer_segment`-based scoring, and a blank column for
    manual review.
    """
    labels = [label for label, _ in model_specs]
    fields: list[str] = []
    for field, _, _ in questions:
        if field not in fields:
            fields.append(field)

    # (label, idx) -> (segment, new_ok, old_ok)
    scored: dict[tuple[str, int], tuple[str, bool, bool]] = {}
    for (label, idx), raw in raw_results.items():
        _, _, answers = questions[idx]
        segment = extract_answer_segment(raw)
        new_ok = any(a in segment for a in answers)
        old_ok = _legacy_score(raw, answers)
        scored[(label, idx)] = (segment, new_ok, old_ok)

    lines = [
        "# 日本知識プローブ: 10 分野 x 5 問",
        "",
        "形式: 1-shot の 質問/答え 形式、greedy。採点: extract_answer_segment() "
        "で切り出した先頭の答えセグメントに対する想定解のサブストリング一致 "
        "(Issue #76: 選択肢列挙などから正解語を誤って拾う旧採点のバグを修正)。",
        "",
        "## 分野別スコア (正答数 / 5)",
        "",
        "| 分野 | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    totals = {label: 0 for label in labels}
    for field in fields:
        idxs = [i for i, (f, _, _) in enumerate(questions) if f == field]
        row = [field]
        for label in labels:
            n = sum(1 for i in idxs if scored.get((label, i), ("", False, False))[1])
            totals[label] += n
            row.append(str(n))
        lines.append("| " + " | ".join(row) + " |")
    total_row = [f"**合計 (/{len(questions)})**"] + [f"**{totals[label]}**" for label in labels]
    lines.append("| " + " | ".join(total_row) + " |")

    lines += ["", "## 全回答", ""]
    header = ["#", "分野", "質問", *labels, "旧判定との差分", "手動照合"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for i, (field, q, _answers) in enumerate(questions):
        cells = [str(i + 1), field, q]
        diff_notes = []
        for label in labels:
            segment, new_ok, old_ok = scored.get((label, i), ("", False, False))
            mark = "O" if new_ok else "X"
            cells.append(f"{mark} {segment[:45]}")
            if old_ok != new_ok:
                old_mark = "O" if old_ok else "X"
                diff_notes.append(f"{label}: 旧{old_mark}→新{mark}")
        diff_cell = "; ".join(diff_notes)
        lines.append("| " + " | ".join(cells) + f" | {diff_cell} |  |")

    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Japan-knowledge probe (Issue #76)")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="label=path pairs, e.g. base=LiquidAI/LFM2.5-1.2B-Base "
        "ckpt9000=outputs/.../checkpoint-9000",
    )
    parser.add_argument("--out", required=True, help="Output markdown report path")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    args = parser.parse_args()

    model_specs = _parse_model_specs(args.models)
    raw_results = run_japan_probe(model_specs, max_new_tokens=args.max_new_tokens)
    report = build_report(model_specs, QUESTIONS, raw_results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
