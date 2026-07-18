"""Pure-function rule verifiers for the ifeval_ja harness (Issue #104)."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Callable

# Sentence-final polite (敬体) forms. Anchored at the end of a (punctuation
# already stripped) sentence. Longer alternatives (でしょうか) are listed
# separately from their prefix (でしょう) since $-anchored alternation only
# matches when the sentence's actual tail equals that alternative.
# v1.1 (Issue #117 rescore): でした (past of です), ましょう (volitional of
# ます) and ませ (imperative of ます, e.g. くださいませ) added -- their
# absence false-flagged genuinely polite outputs; all 11 models rescored.
_POLITE_ENDINGS = re.compile(
    r"(です|でした|ます|ました|ましょう|ませ|ません|でしょうか|でしょう"
    r"|ください|下さい|ございます)$"
)

_BULLET_LINE = re.compile(r"^\s*[-・*]|^\s*\d+\.")

# Non-sentence fragments that survive 。 splitting but should NOT be judged
# for polite endings (business-email headers, addressees, salutations,
# placeholder brackets, formal-letter marks). Applied only for style=polite;
# plain-style judging stays strict because a genuine violation is always a
# full sentence.
_POLITE_EXEMPT_PATTERNS = [
    re.compile(r"[様殿][ 　]*$"),
    re.compile(r"(御中|各位|拝啓|敬具|前略|草々|拝復|敬白)[ 　]*$"),
    re.compile(r"^[ 　]*(拝啓|敬具|前略|草々|拝復|敬白|記|以上)[ 　]*$"),
    re.compile(r"^[ 　]*[\[［(（].*[\]］)）][ 　]*$"),
    re.compile(r"^[ 　]*〇[〇0-9A-Za-z]*"),
]


def _is_label_line(line: str) -> bool:
    """Header-style label like '件名:xxx' — colon within the first 15 chars."""
    m = re.search(r"[：:]", line)
    return m is not None and m.start() < 15


def _is_polite_exempt(line: str) -> bool:
    if _is_label_line(line):
        return True
    # v1.1 (Issue #117 rescore): a line with no hiragana at all (signatures /
    # role names like 幹事 〇〇, 担当 山田) has no predicate to judge -- any
    # real sentence needs hiragana for its verb or copula.
    if not re.search(r"[ぁ-ゖ]", line):
        return True
    return any(p.search(line) for p in _POLITE_EXEMPT_PATTERNS)

_PREAMBLE_PATTERNS = [
    r"^はい、?承知(いた)?しました[。!！]?\s*\n?",
    r"^承知(いた)?しました[。!！]?\s*\n?",
    r"^かしこまりました[。!！]?\s*\n?",
    r"^以下(の(とおり|通り))?[、,]?\s*\n?",
    r"^回答[:：]\s*",
    r"^お答え(いた)?します[。!！]?\s*\n?",
]

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?(.*?)\n?```$", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def _split_sentences(text: str) -> list[str]:
    """Split on 。！？, drop empty/whitespace-only fragments and bullet lines."""
    parts = re.split(r"[。!?！？]", text)
    sentences = []
    for part in parts:
        for line in part.splitlines():
            line = line.strip()
            if not line or _BULLET_LINE.match(line):
                continue
            sentences.append(line)
    return sentences


def verify_char_count(response: str, params: dict) -> tuple[bool, str]:
    mn = params.get("min")
    mx = params.get("max")
    if mn is None and mx is None:
        raise ValueError("verify_char_count requires 'min' and/or 'max' in params")
    length = len(_nfkc(response))
    if mn is not None and length < mn:
        return False, f"文字数 {length} が最小 {mn} 未満です"
    if mx is not None and length > mx:
        return False, f"文字数 {length} が最大 {mx} を超えています"
    return True, ""


def verify_bullet_count(response: str, params: dict) -> tuple[bool, str]:
    count_exact = params.get("count")
    mn = params.get("min")
    mx = params.get("max")
    if count_exact is None and mn is None and mx is None:
        raise ValueError("verify_bullet_count requires 'count' or 'min'/'max' in params")
    text = _nfkc(response)
    count = sum(1 for line in text.splitlines() if _BULLET_LINE.match(line))
    if count_exact is not None:
        if count != count_exact:
            return False, f"箇条書き数 {count} 件（期待: {count_exact} 件）"
        return True, ""
    if mn is not None and count < mn:
        return False, f"箇条書き数 {count} 件が最小 {mn} 件未満です"
    if mx is not None and count > mx:
        return False, f"箇条書き数 {count} 件が最大 {mx} 件を超えています"
    return True, ""


def verify_polite_form(response: str, params: dict) -> tuple[bool, str]:
    style = params.get("style")
    if style not in ("polite", "plain"):
        raise ValueError("verify_polite_form requires params['style'] in {'polite', 'plain'}")
    text = _nfkc(response)
    sentences = _split_sentences(text)
    if not sentences:
        return False, "検証対象の文が見つかりません"

    if style == "polite":
        # Exempt business-letter fragments (件名:xxx, A社様, 拝啓, [氏名], …) from
        # polite judging; they lack sentence predicates so the ending-check
        # gives false negatives. Applied only for polite style — plain-style
        # violations are always full sentences (Issue #104 rescore).
        judged = [s for s in sentences if not _is_polite_exempt(s)]
        if not judged:
            return False, "検証対象の敬体判定可能な文が見つかりません"
        offenders = [s for s in judged if not _POLITE_ENDINGS.search(s)]
        if offenders:
            return False, f"敬体でない文があります: {offenders[0]!r}"
        return True, ""

    # style == "plain": no sentence may end on a polite (敬体) form.
    offenders = [s for s in sentences if _POLITE_ENDINGS.search(s)]
    if offenders:
        return False, f"敬体の文があります: {offenders[0]!r}"
    return True, ""


def verify_keyword(response: str, params: dict) -> tuple[bool, str]:
    include = params.get("include") or []
    exclude = params.get("exclude") or []
    if not include and not exclude:
        raise ValueError("verify_keyword requires 'include' and/or 'exclude' in params")
    text = _nfkc(response)
    missing = [kw for kw in include if _nfkc(kw) not in text]
    if missing:
        return False, f"含むべきキーワードがありません: {missing}"
    present = [kw for kw in exclude if _nfkc(kw) in text]
    if present:
        return False, f"含んではいけないキーワードがあります: {present}"
    return True, ""


def verify_format_json(response: str, params: dict) -> tuple[bool, str]:
    text = response.strip()
    fence_match = _JSON_FENCE_RE.search(text)
    payload = fence_match.group(1).strip() if fence_match else text
    try:
        json.loads(payload)
    except json.JSONDecodeError as e:
        return False, f"JSON として解析できません: {e}"
    return True, ""


def verify_format_markdown_table(response: str, params: dict) -> tuple[bool, str]:
    min_rows = params.get("min_rows")
    lines = response.splitlines()
    row_re = re.compile(r"^\s*\|.*\|\s*$")
    sep_re = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?\s*$")

    sep_idx = None
    for i, line in enumerate(lines):
        if sep_re.match(line) and i > 0 and row_re.match(lines[i - 1]):
            sep_idx = i
            break
    if sep_idx is None:
        return False, "Markdown 表(ヘッダー行 + 区切り行)が見つかりません"

    data_rows = 0
    for line in lines[sep_idx + 1 :]:
        if row_re.match(line):
            data_rows += 1
        else:
            break

    if min_rows is not None and data_rows < min_rows:
        return False, f"表のデータ行数 {data_rows} が最小 {min_rows} 未満です"
    return True, ""


def verify_numeric_only(response: str, params: dict) -> tuple[bool, str]:
    # Deliberately skips NFKC: full-width digits must fail here, since NFKC
    # would silently "fix" them and defeat the point of this verifier.
    text = response.strip()
    allow_units = bool(params.get("allow_units"))
    if allow_units:
        m = re.fullmatch(r"([+-]?[0-9]+(?:\.[0-9]+)?)([^0-9]*)", text)
    else:
        m = re.fullmatch(r"[+-]?[0-9]+(?:\.[0-9]+)?", text)
    if not m:
        return False, f"半角数字のみの形式ではありません: {text!r}"
    return True, ""


def strip_preamble(response: str) -> str:
    """Strip common preambles and a wrapping code fence for loose scoring."""
    text = response
    changed = True
    while changed:
        changed = False
        for pattern in _PREAMBLE_PATTERNS:
            new_text = re.sub(pattern, "", text)
            if new_text != text:
                text = new_text
                changed = True
    text = text.strip()
    fence_match = _FENCE_RE.match(text)
    if fence_match:
        text = fence_match.group(1)
    return text.strip()


VERIFIERS: dict[str, Callable[[str, dict], tuple[bool, str]]] = {
    "char_count": verify_char_count,
    "bullet_count": verify_bullet_count,
    "polite_form": verify_polite_form,
    "keyword": verify_keyword,
    "format_json": verify_format_json,
    "format_markdown_table": verify_format_markdown_table,
    "numeric_only": verify_numeric_only,
}
