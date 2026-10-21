"""Agent behaviour: SRM detection, SRM-first enforcement, decisions vs planted truth,
the validator repair loop, and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from abagent.agent import mock_policy, run_agent
from abagent.config import build_context, load_registry
from abagent.evaluate import evaluate
from abagent.llm import LLMResponse, MockLLM, ToolCallRequest, Usage
from abagent.tools import run_tool

EXPERIMENTS = list(load_registry())


def test_srm_detected_only_on_srm_experiment(dataset):
    df, _ = dataset
    for exp_id in EXPERIMENTS:
        res = run_tool(build_context(exp_id, df), "check_srm", {})
        assert res["srm_detected"] == (exp_id == "exp_autopay_banner"), exp_id


@pytest.mark.parametrize("exp_id", EXPERIMENTS)
def test_mock_agent_matches_ground_truth(dataset, exp_id, tmp_path):
    df, truth = dataset
    result = run_agent(build_context(exp_id, df), llm=MockLLM(mock_policy), reports_dir=tmp_path)
    assert result.decision == truth[exp_id]["expected_decision"]
    assert result.scope == truth[exp_id]["expected_scope"]
    assert result.validation.passed, result.validation.summary()
    assert [c["tool"] for c in result.trace][:2] == ["get_experiment_metadata", "check_srm"]
    assert Path(result.memo_path).exists()
    assert all(Path(p).exists() for p in result.charts.values())


def test_srm_experiment_is_never_shipped_even_with_apparent_lift(dataset):
    df, _ = dataset
    result = run_agent(build_context("exp_autopay_banner", df), reports_dir=None)
    assert result.decision == "Investigate"
    assert "SRM detected" in result.memo_markdown
    assert not any(c["tool"] == "segment_breakdown" for c in result.trace)


class ScriptedLLM:
    """Stands in for a real model: replays scripted responses and records what it saw."""

    offline = False
    model = "scripted"

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen: list[list[dict]] = []
        self.usage = Usage()

    def chat(self, messages, tools=None):
        self.seen.append([dict(m) for m in messages])
        return self.responses.pop(0)


def _final(decision: str, memo: str) -> LLMResponse:
    return LLMResponse(json.dumps({"decision": decision, "scope": "all", "memo_markdown": memo}))


MEMO_OK = """## Decision
Investigate
## Key numbers
Sample ratio mismatch detected (p < 0.001), 18,500 treatment users.
## Guardrails
Not evaluated.
## Segment insights
Not analysed.
## Risks
Invalid randomisation.
## Next steps
Audit logging.
"""


def test_harness_blocks_tools_before_srm(dataset):
    df, _ = dataset
    llm = ScriptedLLM([
        LLMResponse("Plan: jump straight to the effect.",
                    [ToolCallRequest("c1", "estimate_effect", {"metric": "conversion_rate"})]),
        LLMResponse(None, [ToolCallRequest("c2", "check_srm", {})]),
        _final("Investigate", MEMO_OK),
    ])
    result = run_agent(build_context("exp_autopay_banner", df), llm=llm, reports_dir=None)
    blocked = json.loads(llm.seen[1][-1]["content"])
    assert "check_srm must run" in blocked["error"]
    assert [c["tool"] for c in result.trace] == ["check_srm"]
    assert result.plan == "Plan: jump straight to the effect."
    assert result.validation.passed


def test_fabricated_number_triggers_one_repair_attempt(dataset):
    df, _ = dataset
    fabricated = MEMO_OK.replace("Audit logging.", "Audit logging; lift was really 47.3%.")
    llm = ScriptedLLM([
        LLMResponse(None, [ToolCallRequest("c1", "check_srm", {})]),
        _final("Investigate", fabricated),
        _final("Investigate", MEMO_OK),
    ])
    result = run_agent(build_context("exp_autopay_banner", df), llm=llm, reports_dir=None)
    feedback = llm.seen[2][-1]["content"]
    assert "failed validation" in feedback and "47.3" in feedback
    assert result.validation.passed


def test_ship_on_srm_fails_validation_after_repair(dataset):
    df, _ = dataset
    llm = ScriptedLLM([
        LLMResponse(None, [ToolCallRequest("c1", "check_srm", {})]),
        _final("Ship", MEMO_OK),
        _final("Ship", MEMO_OK),
    ])
    result = run_agent(build_context("exp_autopay_banner", df), llm=llm, reports_dir=None)
    assert not result.validation.passed
    assert any("SRM" in v for v in result.validation.rule_violations)


def test_evaluation_reports_full_accuracy_on_default_seed(dataset):
    summary = evaluate(reports_dir=None)["summary"]
    assert summary["decision_accuracy"] == 1.0
    assert summary["planted_issue_surfaced"] == 1.0
    assert summary["wrong_ship_decisions"] == 0
    assert summary["validator_pass_rate"] == 1.0
