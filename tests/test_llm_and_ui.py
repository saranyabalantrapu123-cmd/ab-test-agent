"""LLM mode selection, the LiteLLM adapter (with a faked completion), CSV upload and
a headless run of the Streamlit app."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from abagent import llm
from abagent.agent import run_agent
from abagent.upload import ColumnMapping, context_from_upload, infer_metric_type

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("model,env,offline", [
    ("mock", {"ANTHROPIC_API_KEY": "x"}, True),
    ("anthropic/claude-sonnet-4-5", {}, True),
    ("anthropic/claude-sonnet-4-5", {"ANTHROPIC_API_KEY": "x"}, False),
    ("openai/gpt-4o-mini", {"ANTHROPIC_API_KEY": "x"}, True),
    ("openai/gpt-4o-mini", {"OPENAI_API_KEY": "x"}, False),
])
def test_offline_detection(monkeypatch, model, env, offline):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "AZURE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_MODEL", model)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert llm.is_offline() is offline
    assert isinstance(llm.get_llm(), llm.MockLLM) is offline


def test_litellm_client_parses_tool_calls_and_tracks_usage(monkeypatch):
    fn = types.SimpleNamespace(name="check_srm", arguments="{}")
    message = types.SimpleNamespace(
        content="Plan: SRM first.",
        tool_calls=[types.SimpleNamespace(id="t1", function=fn)])
    response = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=message)],
        usage=types.SimpleNamespace(prompt_tokens=120, completion_tokens=30))
    fake = types.SimpleNamespace(
        completion=lambda **kw: response,
        completion_cost=lambda completion_response: 0.0021)
    monkeypatch.setitem(sys.modules, "litellm", fake)

    client = llm.LiteLLMClient(model="anthropic/claude-sonnet-4-5")
    out = client.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    assert out.tool_calls[0].name == "check_srm" and out.tool_calls[0].arguments == {}
    assert client.usage.to_dict() == {"calls": 1, "prompt_tokens": 120,
                                      "completion_tokens": 30, "estimated_cost_usd": 0.0021}
    assert out.as_message()["tool_calls"][0]["function"]["name"] == "check_srm"


def test_upload_with_column_mapping_runs_agent(dataset):
    df, truth = dataset
    raw = df[df["experiment_id"] == "exp_discount_nudge"].rename(columns={
        "variant": "arm", "converted": "bought", "revenue": "rev", "platform": "device"})
    raw["arm"] = raw["arm"].map({"control": "A", "treatment": "B"})
    ctx = context_from_upload(raw, ColumnMapping(
        variant="arm", control_label="A", treatment_label="B", primary_metric="bought",
        guardrails=["rev"], segments=["device"], date="assignment_date"))
    assert infer_metric_type(ctx.df["bought"]) == "proportion"
    assert ctx.config.metrics["rev"].type == "mean"
    result = run_agent(ctx, reports_dir=None)
    assert result.decision == truth["exp_discount_nudge"]["expected_decision"]
    assert result.validation.passed


def test_upload_rejects_unknown_columns(dataset):
    df, _ = dataset
    with pytest.raises(ValueError, match="not found"):
        context_from_upload(df, ColumnMapping("nope", "a", "b", "converted"))


def test_streamlit_app_runs_analysis_headless(dataset):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert any("offline demo mode" in w.value for w in at.warning)
    at.selectbox[0].set_value("exp_discount_nudge").run()
    at.button[0].click().run()
    assert not at.exception
    assert at.metric[0].value.endswith("Don't ship")
    assert any("PASSED" in s.value for s in at.success)
