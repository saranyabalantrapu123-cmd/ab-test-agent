"""Score the agent against the PLANTED ground truth.

Because the data is generated with known effects, we can measure whether the agent
reaches the right decision — the thing that matters to the business — not just
whether its prose sounds plausible.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .agent import AgentResult, mock_policy, run_agent
from .config import REPORTS_DIR, ROOT, build_context, load_data, load_ground_truth, load_registry
from .decision import decide
from .llm import LLMClient, get_llm, is_offline


def _llm_factory() -> LLMClient:
    return get_llm(mock_policy)


def score(result: AgentResult, truth: dict[str, Any]) -> dict[str, Any]:
    """Compare one agent run to its planted truth."""
    rules = decide(result.findings)  # rule-engine reading of the agent's own tool outputs
    flags = rules.flags
    expected_flags = truth["expected_flags"]
    flags_ok = all(flags.get(k) == v for k, v in expected_flags.items())
    decision_ok = result.decision == truth["expected_decision"]
    scope_ok = result.scope == truth["expected_scope"]
    return {
        "experiment_id": result.experiment_id,
        "scenario": truth["scenario"],
        "expected": truth["expected_decision"] + _scope(truth["expected_scope"]),
        "agent": result.decision + _scope(result.scope),
        "decision_correct": decision_ok and scope_ok,
        "planted_issue_surfaced": flags_ok,
        "wrong_ship": result.decision == "Ship" and truth["expected_decision"] != "Ship",
        "validator_passed": result.validation.passed,
        "tool_calls": len(result.trace),
        "rule_reading": rules.rule,
        "usage": result.usage,
    }


def _scope(scope: str) -> str:
    return "" if scope == "all" else f" ({scope})"


def evaluate(
    data_dir: Path | None = None,
    reports_dir: Path | None = REPORTS_DIR,
    llm_factory: Callable[[], LLMClient] = _llm_factory,
) -> dict[str, Any]:
    """Run the agent on every experiment and summarise accuracy."""
    df = load_data(data_dir)
    truth = load_ground_truth(data_dir)["experiments"]
    rows = []
    for exp_id in load_registry():
        llm = llm_factory()
        result = run_agent(build_context(exp_id, df), llm=llm, reports_dir=reports_dir)
        rows.append(score(result, truth[exp_id]))
    return {"mode": _mode_label(), "rows": rows, "summary": summarise(rows)}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    cost = sum(r["usage"].get("estimated_cost_usd", 0.0) for r in rows)
    tokens = sum(r["usage"].get("prompt_tokens", 0) + r["usage"].get("completion_tokens", 0)
                 for r in rows)
    return {
        "experiments": n,
        "decision_accuracy": sum(r["decision_correct"] for r in rows) / n,
        "planted_issue_surfaced": sum(r["planted_issue_surfaced"] for r in rows) / n,
        "wrong_ship_decisions": sum(r["wrong_ship"] for r in rows),
        "validator_pass_rate": sum(r["validator_passed"] for r in rows) / n,
        "total_tokens": tokens,
        "estimated_cost_usd": round(cost, 4),
    }


def seed_sweep(seeds: range, llm_factory: Callable[[], LLMClient] = _llm_factory) -> dict[str, Any]:
    """Regenerate the data with different seeds and re-score: separates agent errors
    from the statistical noise any fixed-alpha, fixed-power test design carries."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from generate_experiments import generate  # type: ignore[import-not-found]

    per_scenario: dict[str, list[bool]] = defaultdict(list)
    per_seed, misses = [], []
    for seed in seeds:
        with tempfile.TemporaryDirectory() as tmp:
            generate(seed, Path(tmp))
            res = evaluate(Path(tmp), reports_dir=None, llm_factory=llm_factory)
        for r in res["rows"]:
            per_scenario[r["scenario"]].append(r["decision_correct"])
            if not r["decision_correct"]:
                misses.append({"seed": seed, **{k: r[k] for k in
                               ("scenario", "expected", "agent", "rule_reading")}})
        per_seed.append({"seed": seed, **res["summary"]})
    return {
        "seeds": list(seeds),
        "mean_decision_accuracy": sum(s["decision_accuracy"] for s in per_seed) / len(per_seed),
        "total_wrong_ships": sum(s["wrong_ship_decisions"] for s in per_seed),
        "per_scenario_accuracy": {k: sum(v) / len(v) for k, v in per_scenario.items()},
        "misses": misses,
        "per_seed": per_seed,
    }


def _mode_label() -> str:
    if is_offline():
        return "rule-based offline mode (MockLLM, no API key)"
    from .llm import configured_model
    return f"online — {configured_model()}"


def render_markdown(result: dict[str, Any], sweep: dict[str, Any] | None = None) -> str:
    s = result["summary"]
    cols = ["experiment_id", "scenario", "expected", "agent", "decision_correct",
            "planted_issue_surfaced", "validator_passed", "tool_calls"]
    table = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in result["rows"]:
        cells = [("yes" if row[c] else "**no**") if isinstance(row[c], bool) else str(row[c])
                 for c in cols]
        table.append("| " + " | ".join(cells) + " |")
    lines = [
        "# Evaluation: agent decisions vs planted ground truth",
        "",
        f"**Mode:** {result['mode']}",
        "",
        *table,
        "",
        f"- Decision accuracy: **{s['decision_accuracy']:.0%}** "
        f"({s['experiments']} experiments)",
        f"- Planted issue surfaced: **{s['planted_issue_surfaced']:.0%}**",
        f"- Wrong ship decisions: **{s['wrong_ship_decisions']}**",
        f"- Validator pass rate: **{s['validator_pass_rate']:.0%}**",
        f"- Tokens: {s['total_tokens']:,}; estimated cost: ${s['estimated_cost_usd']:.4f}",
    ]
    if sweep:
        lines += [
            "",
            f"## Seed sweep ({len(sweep['seeds'])} regenerated datasets)",
            "",
            f"Mean decision accuracy: **{sweep['mean_decision_accuracy']:.1%}**; "
            f"wrong ship decisions across all runs: **{sweep['total_wrong_ships']}**.",
            "",
            "| Scenario | Decision accuracy |",
            "|---|---|",
            *[f"| {k} | {v:.0%} |" for k, v in sweep["per_scenario_accuracy"].items()],
        ]
        if sweep["misses"]:
            lines += [
                "",
                "Misses (each new seed is a fresh random draw, so a fixed-alpha, fixed-power "
                "design will occasionally show a chance guardrail regression, miss a real "
                "segment difference, or land just under the required sample):",
                "",
                "| Seed | Scenario | Expected | Agent | Why (rule engine on the tool outputs) |",
                "|---|---|---|---|---|",
                *[f"| {m['seed']} | {m['scenario']} | {m['expected']} | {m['agent']} | "
                  f"{m['rule_reading']} |" for m in sweep["misses"]],
            ]
    return "\n".join(lines) + "\n"


def write_report(result: dict[str, Any], sweep: dict[str, Any] | None,
                 reports_dir: Path = REPORTS_DIR) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "eval_results.json").write_text(
        json.dumps({"result": result, "seed_sweep": sweep}, indent=2, default=str))
    path = reports_dir / "eval_results.md"
    path.write_text(render_markdown(result, sweep))
    return path
