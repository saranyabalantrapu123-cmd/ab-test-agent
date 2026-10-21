"""Decision rules shared by the MockLLM planner and the consistency validator.

The rules encode an experimentation team's playbook:

1. Sample-ratio mismatch      -> Investigate (results are untrustworthy; never Ship)
2. Primary significantly worse -> Don't ship
3. Primary significantly better
     a. any guardrail significantly harmed      -> Don't ship
     b. read early and p above sequential alpha -> Inconclusive-extend
     c. effect concentrated in one segment       -> Ship, scoped to that segment
     d. otherwise                                -> Ship
4. Primary not significant
     a. adequately powered for the MDE -> Don't ship (a true flat result)
     b. underpowered                   -> Inconclusive-extend
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ToolCall = dict[str, Any]  # {"tool": str, "arguments": dict, "result": dict}


@dataclass
class Findings:
    """Structured view of the tool outputs produced during one agent run."""

    metadata: dict[str, Any] | None = None
    srm: dict[str, Any] | None = None
    primary: dict[str, Any] | None = None
    guardrails: dict[str, dict[str, Any]] = field(default_factory=dict)
    power: dict[str, Any] | None = None
    duration: dict[str, Any] | None = None
    segments: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_trace(cls, trace: list[ToolCall]) -> "Findings":
        f = cls()
        for call in trace:
            name, res = call["tool"], call["result"]
            if "error" in res:
                continue
            if name == "get_experiment_metadata":
                f.metadata = res
            elif name == "check_srm":
                f.srm = res
            elif name == "power_analysis" and (
                f.metadata is None or res.get("metric") == f.metadata["primary_metric"]
            ):
                f.power = res
            elif name == "check_duration":
                f.duration = res
            elif name == "segment_breakdown":
                f.segments.append(res)
            elif name == "estimate_effect" and f.metadata is not None:
                if res["metric"] == f.metadata["primary_metric"]:
                    f.primary = res
                elif res["metric"] in f.metadata["guardrail_metrics"]:
                    f.guardrails[res["metric"]] = res
        return f


def is_beneficial(result: dict[str, Any]) -> bool:
    return (result["abs_diff"] > 0) == bool(result["higher_is_better"])


def is_significant_win(result: dict[str, Any] | None) -> bool:
    return bool(result and result["significant"] and is_beneficial(result))


def is_significant_harm(result: dict[str, Any] | None) -> bool:
    return bool(result and result["significant"] and not is_beneficial(result))


def breached_guardrails(f: Findings) -> list[str]:
    return [m for m, r in f.guardrails.items() if is_significant_harm(r)]


def concentrated_segment(f: Findings) -> str | None:
    """'dim=level' when effects differ across levels (Cochran's Q) and exactly one
    level shows a significant beneficial effect after correction."""
    if not f.primary:
        return None
    for seg in f.segments:
        if seg["metric"] != f.primary["metric"] or not seg["heterogeneous"]:
            continue
        winners = [
            lv for lv in seg["levels"]
            if lv["significant_adjusted"]
            and (lv["abs_diff"] > 0) == bool(f.primary["higher_is_better"])
        ]
        if len(winners) == 1 and len(seg["levels"]) > 1:
            return f"{seg['segment']}={winners[0]['level']}"
    return None


@dataclass
class Decision:
    decision: str
    scope: str
    rule: str
    flags: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide(f: Findings) -> Decision:
    """Apply the playbook to the findings."""
    breaches = breached_guardrails(f)
    segment = concentrated_segment(f)
    powered = bool(f.power and f.power["adequately_powered"])
    flags = {
        "srm": bool(f.srm and f.srm["srm_detected"]),
        "guardrail_breach": bool(breaches),
        "breached_guardrails": breaches,
        "heterogeneous_segment": segment,
        "underpowered": bool(f.power) and not powered,
        "peeking_risk": bool(f.duration and f.duration["peeking_risk"]),
    }

    def out(decision: str, rule: str, scope: str = "all") -> Decision:
        return Decision(decision, scope, rule, flags)

    if f.srm is None:
        return out("Investigate", "SRM check was not run; results cannot be trusted.")
    if flags["srm"]:
        return out("Investigate", "Sample-ratio mismatch: allocation differs from design, "
                                  "so every metric is suspect.")
    if f.primary is None:
        return out("Investigate", "Primary metric could not be estimated.")
    if is_significant_harm(f.primary):
        return out("Don't ship", "Primary metric is significantly worse.")
    if is_significant_win(f.primary):
        if breaches:
            return out("Don't ship", "Primary metric improved but guardrail(s) "
                                     f"{', '.join(breaches)} significantly regressed.")
        if f.duration and f.duration["information_fraction"] < 1 and (
            f.primary["p_value"] >= f.duration["sequential_alpha_now"]
        ):
            return out("Inconclusive-extend", "Significant only at the nominal alpha on an "
                                              "early read; not past the sequential boundary.")
        if segment:
            return out("Ship", f"Primary metric improved, driven by {segment}; ship to that "
                               "segment only.", scope=segment)
        return out("Ship", "Primary metric significantly improved with no guardrail regressions.")
    if segment:
        return out("Investigate", f"No overall effect but a significant effect in {segment}; "
                                  "run a targeted follow-up.")
    if powered:
        return out("Don't ship", "No significant effect and the test was powered for the MDE.")
    return out("Inconclusive-extend", "No significant effect yet, but the test is "
                                      "underpowered for the MDE.")


def consistency_violations(decision: str, f: Findings) -> list[str]:
    """Hard rules any memo (LLM-written or not) must satisfy, given the tool outputs."""
    from .config import DECISIONS

    problems: list[str] = []
    if decision not in DECISIONS:
        problems.append(f"Decision '{decision}' is not one of {list(DECISIONS)}.")
        return problems
    if f.srm is None:
        problems.append("SRM check was never run.")
    if decision == "Ship":
        if f.srm is None or f.srm["srm_detected"]:
            problems.append("Ship is not allowed when SRM is detected or unchecked.")
        if not is_significant_win(f.primary):
            problems.append("Ship requires a significant improvement on the primary metric.")
        if breached_guardrails(f):
            problems.append("Ship is not allowed when a guardrail significantly regressed.")
    return problems
