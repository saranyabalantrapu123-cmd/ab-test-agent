"""The experiment-readout agent.

One loop serves both modes: the LLM (LiteLLM, or the rule-based MockLLM offline)
plans and calls statistical tools, then returns a decision + memo as JSON. The
harness enforces SRM-first, validates the memo (hallucinated numbers, rule
consistency, sections), gives a real model one chance to fix a failed memo, and
writes the memo and charts to disk.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import charts
from .config import DECISIONS, REPORTS_DIR, ExperimentContext
from .decision import Findings, decide
from .llm import LLMClient, LLMResponse, ToolCallRequest, get_llm, tool_message
from .memo import REQUIRED_SECTIONS, render_memo
from .tools import run_tool, tool_schemas
from .validator import ValidationResult, validate_memo

SYSTEM_PROMPT = f"""You are an experimentation analyst. Produce a ship decision for one A/B test.

Process:
1. Call get_experiment_metadata, then check_srm. SRM is ALWAYS the first statistical check.
2. If SRM is detected, the experiment is invalid: estimate the primary metric only for
   context, run check_duration, and decide Investigate.
3. Otherwise estimate the primary metric and every guardrail (use cuped=true for mean
   metrics), run power_analysis and check_duration, and run segment_breakdown for each
   registered segment.
4. Decide with this playbook:
   - SRM -> Investigate (never Ship).
   - Primary significantly worse -> Don't ship.
   - Primary significantly better: any guardrail significantly harmed -> Don't ship;
     read early and p >= sequential_alpha_now -> Inconclusive-extend; effect
     concentrated in a single segment level (heterogeneous, one corrected-significant
     winner) -> Ship with scope "<segment>=<level>"; otherwise Ship.
   - Primary not significant: adequately powered -> Don't ship; else Inconclusive-extend.

Before calling tools, state your plan in one short paragraph.

Final answer: reply with ONLY a JSON object:
{{"decision": one of {list(DECISIONS)}, "scope": "all" or "<segment>=<level>",
  "memo_markdown": "<memo>"}}
The memo must have these sections as markdown headings: {", ".join(REQUIRED_SECTIONS)}.
Every number in the memo must come from a tool output (you may round, and show
fractions as percentages). Do not compute new numbers. Put experiment ids in backticks.
"""

MAX_STEPS = 12


@dataclass
class AgentResult:
    experiment_id: str
    decision: str
    scope: str
    memo_markdown: str
    plan: str
    trace: list[dict[str, Any]]
    validation: ValidationResult
    mode: str
    model: str
    usage: dict[str, Any]
    findings: Findings = field(repr=False)
    charts: dict[str, str] = field(default_factory=dict)
    memo_path: str | None = None

    def summary(self) -> dict[str, Any]:
        return {"experiment_id": self.experiment_id, "decision": self.decision,
                "scope": self.scope, "validation": self.validation.summary(),
                "tool_calls": len(self.trace), "mode": self.mode, "usage": self.usage}


# ----------------------------------------------------------- MockLLM planner
def _executed(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map 'tool|args' -> result for every tool call already answered."""
    calls: dict[str, str] = {}
    results: dict[str, dict[str, Any]] = {}
    for m in messages:
        for tc in m.get("tool_calls", []) or []:
            calls[tc["id"]] = f"{tc['function']['name']}|{tc['function']['arguments']}"
        if m["role"] == "tool":
            results[calls[m["tool_call_id"]]] = json.loads(m["content"])
    return results


def _key(name: str, args: dict[str, Any]) -> str:
    return f"{name}|{json.dumps(args)}"


def mock_policy(messages: list[dict[str, Any]], _tools: list[dict[str, Any]] | None) -> LLMResponse:
    """Deterministic rule-based planner: decides the next batch of tool calls from
    what has already run, then writes the decision and templated memo."""
    done = _executed(messages)
    step = sum(1 for m in messages if m["role"] == "assistant")

    def calls(*specs: tuple[str, dict[str, Any]]) -> list[ToolCallRequest]:
        return [ToolCallRequest(f"call_{step}_{i}", n, a) for i, (n, a) in enumerate(specs)
                if _key(n, a) not in done]

    meta = done.get(_key("get_experiment_metadata", {}))
    if meta is None:
        return LLMResponse(None, calls(("get_experiment_metadata", {})))
    srm = done.get(_key("check_srm", {}))
    if srm is None:
        plan = ("Plan: check sample-ratio mismatch first; if allocation is valid, estimate "
                f"{meta['primary_metric']} and guardrails ({', '.join(meta['guardrail_metrics'])}),"
                " check power and test duration, then scan segments "
                f"({', '.join(meta['segments'])}) with Holm correction.")
        return LLMResponse(plan, calls(("check_srm", {})))

    primary = ("estimate_effect", {"metric": meta["primary_metric"]})
    if srm["srm_detected"]:
        batch = [primary, ("check_duration", {})]
    else:
        guardrails = [("estimate_effect", {"metric": g, "cuped": meta["metric_types"][g] == "mean"})
                      for g in meta["guardrail_metrics"]]
        batch = [primary, *guardrails, ("power_analysis", {}), ("check_duration", {})]
        pending = calls(*batch)
        if pending:
            return LLMResponse(None, pending)
        batch = [("segment_breakdown", {"segment": s}) for s in meta["segments"]]
    pending = calls(*batch)
    if pending:
        return LLMResponse(None, pending)

    trace = [{"tool": k.split("|", 1)[0], "arguments": json.loads(k.split("|", 1)[1]),
              "result": v} for k, v in done.items()]
    findings = Findings.from_trace(trace)
    decision = decide(findings)
    return LLMResponse(json.dumps({"decision": decision.decision, "scope": decision.scope,
                                   "memo_markdown": render_memo(findings, decision)}))


# --------------------------------------------------------------- harness
def parse_final(content: str | None) -> dict[str, Any]:
    """Extract the final JSON object from a model reply (tolerates code fences)."""
    if not content:
        raise ValueError("Empty final answer")
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
    blob = fenced.group(1) if fenced else content[content.find("{"): content.rfind("}") + 1]
    data = json.loads(blob)
    for key in ("decision", "memo_markdown"):
        if key not in data:
            raise ValueError(f"Final answer missing '{key}'")
    data.setdefault("scope", "all")
    return data


def run_agent(
    ctx: ExperimentContext,
    llm: LLMClient | None = None,
    reports_dir: Path | None = REPORTS_DIR,
    max_steps: int = MAX_STEPS,
) -> AgentResult:
    """Analyse one experiment end-to-end. Set reports_dir=None to skip writing files."""
    llm = llm or get_llm(mock_policy)
    exp_id = ctx.config.experiment_id
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyse experiment `{exp_id}` and write the decision memo."},
    ]
    trace: list[dict[str, Any]] = []
    plan = ""
    final: dict[str, Any] | None = None
    validation: ValidationResult | None = None
    retried = False

    for _ in range(max_steps):
        resp = llm.chat(messages, tool_schemas())
        if resp.tool_calls:
            if not plan and resp.content:
                plan = resp.content.strip()
            messages.append(resp.as_message())
            for tc in resp.tool_calls:
                srm_done = any(c["tool"] == "check_srm" for c in trace)
                if tc.name not in ("get_experiment_metadata", "check_srm") and not srm_done:
                    result = {"error": "check_srm must run before any other statistical check."}
                else:
                    result = run_tool(ctx, tc.name, tc.arguments)
                    trace.append({"tool": tc.name, "arguments": tc.arguments, "result": result})
                messages.append(tool_message(tc.id, result))
            continue
        try:
            final = parse_final(resp.content)
        except (ValueError, json.JSONDecodeError) as exc:
            messages.append(resp.as_message())
            messages.append({"role": "user", "content": f"Invalid final answer ({exc}). "
                             "Reply with only the JSON object."})
            continue
        validation = validate_memo(final["memo_markdown"], final["decision"], trace)
        if validation.passed or retried or llm.offline:
            break
        retried = True  # one repair attempt for a real model
        messages.append(resp.as_message())
        messages.append({"role": "user", "content": "Your memo failed validation: "
                         f"{validation.summary()} Fix it using only numbers from tool outputs "
                         "and the playbook rules, and reply with the JSON object again."})

    if final is None or validation is None:
        raise RuntimeError(f"Agent did not produce a final answer for {exp_id} in {max_steps} steps")

    findings = Findings.from_trace(trace)
    result = AgentResult(
        experiment_id=exp_id, decision=final["decision"], scope=final.get("scope", "all"),
        memo_markdown=final["memo_markdown"], plan=plan, trace=trace, validation=validation,
        mode="offline (rule-based MockLLM)" if llm.offline else "online",
        model=llm.model, usage=llm.usage.to_dict(), findings=findings,
    )
    if reports_dir is not None:
        write_outputs(result, Path(reports_dir))
    return result


def write_outputs(result: AgentResult, reports_dir: Path) -> None:
    """Write charts and the memo (plus a harness footer) to reports/."""
    label = result.decision + ("" if result.scope == "all" else f" ({result.scope})")
    chart_dir = reports_dir / "charts"
    paths = {
        "lift": charts.lift_chart(result.findings, label,
                                  chart_dir / f"{result.experiment_id}_lift.png"),
        "segments": charts.segment_chart(result.findings,
                                         chart_dir / f"{result.experiment_id}_segments.png"),
    }
    result.charts = {k: str(v) for k, v in paths.items() if v}
    images = "\n\n".join(f"![{k}](charts/{Path(v).name})" for k, v in result.charts.items())
    footer = (
        "\n## Charts\n\n" + (images or "_No charts (no effects estimated)._") + "\n\n---\n"
        f"_Generated by ab-test-agent in {result.mode} mode (model: `{result.model}`). "
        f"Validator: {result.validation.summary()} Tools called: "
        f"{', '.join(dict.fromkeys(c['tool'] for c in result.trace))}._\n"
    )
    path = reports_dir / f"{result.experiment_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.memo_markdown + footer)
    result.memo_path = str(path)
