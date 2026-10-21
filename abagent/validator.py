"""Guardrails on the agent's memo.

1. Hallucination check: every number in the memo must be traceable to a tool output
   (allowing rounding, and fractions shown as percentages).
2. Consistency check: the decision must obey the hard rules (e.g. SRM -> never Ship).
3. Structure check: the memo has every required section.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .decision import Findings, consistency_violations
from .memo import REQUIRED_SECTIONS

_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMBER_RE = re.compile(r"(?<![\w.])[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_STRIP_RES = [
    re.compile(r"`[^`]*`"),                    # code spans (experiment ids, column names)
    re.compile(r"!?\[[^\]]*\]\([^)]*\)"),       # markdown images / links
    re.compile(r"^\s*\d+[.)]\s", re.MULTILINE),  # ordered-list markers
    re.compile(r"^#{1,6}\s.*$", re.MULTILINE),   # headings carry no data
]


@dataclass
class ValidationResult:
    passed: bool
    numbers_checked: int
    unsupported_numbers: list[str] = field(default_factory=list)
    unsupported_dates: list[str] = field(default_factory=list)
    rule_violations: list[str] = field(default_factory=list)
    missing_sections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        if self.passed:
            return (f"PASSED — {self.numbers_checked} numbers traced to tool outputs; "
                    "decision consistent with rules.")
        issues = []
        if self.unsupported_numbers:
            issues.append(f"unsupported numbers {self.unsupported_numbers}")
        if self.unsupported_dates:
            issues.append(f"unsupported dates {self.unsupported_dates}")
        issues.extend(self.rule_violations)
        if self.missing_sections:
            issues.append(f"missing sections {self.missing_sections}")
        return "FAILED — " + "; ".join(issues)


def _walk(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def reference_values(tool_outputs: Iterable[dict[str, Any]]) -> list[float]:
    """All numbers in the tool outputs, including numbers inside string fields."""
    values: list[float] = []
    for out in tool_outputs:
        for leaf in _walk(out):
            if isinstance(leaf, bool) or leaf is None:
                continue
            if isinstance(leaf, (int, float)) and math.isfinite(leaf):
                values.append(float(leaf))
            elif isinstance(leaf, str):
                text = _DATE_RE.sub(" ", leaf)
                values.extend(_parse(tok) for tok in _NUMBER_RE.findall(text))
    return values


def _parse(token: str) -> float:
    return float(token.replace(",", "").replace("−", "-"))


def extract_numbers(memo: str) -> list[str]:
    """Numeric tokens a reader would take as facts (dates handled separately)."""
    text = memo
    for rx in _STRIP_RES:
        text = rx.sub(" ", text)
    text = _DATE_RE.sub(" ", text)
    return _NUMBER_RE.findall(text)


def is_supported(token: str, refs: list[float]) -> bool:
    """A memo number is supported if some tool value rounds to it, either as-is or
    multiplied by 100 (fractions displayed as percentages)."""
    value = abs(_parse(token))
    decimals = len(token.split(".")[1]) if "." in token else 0
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(abs(r) - value) <= tol or abs(abs(r) * 100 - value) <= tol for r in refs)


def validate_memo(memo: str, decision: str, trace: list[dict[str, Any]]) -> ValidationResult:
    outputs = [call["result"] for call in trace]
    refs = reference_values(outputs)
    tokens = extract_numbers(memo)
    unsupported = [t for t in tokens if not is_supported(t, refs)]

    blob = json.dumps(outputs)
    bad_dates = [d for d in _DATE_RE.findall(_STRIP_RES[0].sub(" ", memo)) if d not in blob]

    rules = consistency_violations(decision, Findings.from_trace(trace))
    lower = memo.lower()
    missing = [s for s in REQUIRED_SECTIONS if s.lower() not in lower]
    return ValidationResult(
        passed=not (unsupported or bad_dates or rules or missing),
        numbers_checked=len(tokens),
        unsupported_numbers=unsupported,
        unsupported_dates=bad_dates,
        rule_violations=rules,
        missing_sections=missing,
    )
