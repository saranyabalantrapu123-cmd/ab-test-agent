"""The hallucination / consistency validator."""

from __future__ import annotations

import pytest

from abagent.agent import run_agent
from abagent.config import build_context
from abagent.validator import extract_numbers, is_supported, validate_memo


@pytest.fixture(scope="module")
def discount_run(dataset):
    df, _ = dataset
    return run_agent(build_context("exp_discount_nudge", df), reports_dir=None)


def test_templated_memo_passes(discount_run):
    v = discount_run.validation
    assert v.passed and v.numbers_checked > 20


def test_validator_catches_fabricated_number(discount_run):
    memo = discount_run.memo_markdown.replace(
        "## Risks", "## Risks\n- Revenue per user actually grew 47.3% in a hidden cohort.")
    v = validate_memo(memo, discount_run.decision, discount_run.trace)
    assert not v.passed
    assert v.unsupported_numbers == ["47.3"]


def test_validator_catches_fabricated_date(discount_run):
    memo = discount_run.memo_markdown + "\nThe test ended on 2026-12-25.\n"
    v = validate_memo(memo, discount_run.decision, discount_run.trace)
    assert v.unsupported_dates == ["2026-12-25"]


def test_validator_rejects_ship_on_guardrail_breach(discount_run):
    v = validate_memo(discount_run.memo_markdown, "Ship", discount_run.trace)
    assert not v.passed
    assert any("guardrail" in r for r in v.rule_violations)


def test_validator_rejects_ship_on_srm(dataset):
    df, _ = dataset
    run = run_agent(build_context("exp_autopay_banner", df), reports_dir=None)
    v = validate_memo(run.memo_markdown, "Ship", run.trace)
    assert any("SRM" in r for r in v.rule_violations)


def test_validator_flags_unknown_decision_and_missing_sections(discount_run):
    v = validate_memo("## Decision\nMaybe", "Maybe", discount_run.trace)
    assert v.rule_violations and "Guardrails" in v.missing_sections


def test_number_matching_rules():
    refs = [0.171234, 18_422.0]
    assert is_supported("+17.1", refs)       # fraction shown as a percentage
    assert is_supported("18,422", refs)      # thousands separator
    assert is_supported("0.17", refs)        # rounding
    assert not is_supported("17.3", refs)
    # list markers, code spans, links and dates are not treated as claims
    text = "1. See `exp_a1` and ![c](charts/x2.png) on 2026-05-04 with 12.5% lift"
    assert extract_numbers(text) == ["12.5"]
