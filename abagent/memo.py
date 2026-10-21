"""Templated decision memo (used by the MockLLM; real models write their own).

Every number printed here is formatted directly from a tool output, so the memo
passes the hallucination validator by construction.
"""

from __future__ import annotations

from typing import Any

from .decision import Decision, Findings, is_beneficial

CORRECTION_NAMES = {"holm": "Holm", "bh": "Benjamini-Hochberg"}
REQUIRED_SECTIONS = ("Decision", "Key numbers", "Guardrails", "Segment insights", "Risks",
                     "Next steps")


def pct(x: float | None, signed: bool = True, digits: int = 1) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"


def fmt_p(p: float | None) -> str:
    if p is None:
        return "n/a"
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def fmt_value(x: float, metric_type: str) -> str:
    return pct(x, signed=False, digits=2) if metric_type == "proportion" else f"${x:,.2f}"


def _arms(r: dict[str, Any]) -> tuple[float, float]:
    """(control, treatment) value for either a proportion or a mean result."""
    if r["metric_type"] == "proportion":
        return r["rate_control"], r["rate_treatment"]
    return r["mean_control"], r["mean_treatment"]


def _verdict(r: dict[str, Any]) -> str:
    if not r["significant"]:
        return "No significant change"
    return "Improved" if is_beneficial(r) else "**Regressed**"


def _effect_line(r: dict[str, Any]) -> str:
    level = int(round(r["ci_level"] * 100))
    return (f"{pct(r['relative_lift'])} relative ({level}% CI {pct(r['relative_ci_low'])} "
            f"to {pct(r['relative_ci_high'])}), p {fmt_p(r['p_value'])}")


def _key_numbers(f: Findings) -> list[str]:
    lines = []
    if f.srm:
        s = f.srm
        lines.append(
            f"- **Sample ratio:** {s['n_control']:,} control vs {s['n_treatment']:,} treatment "
            f"(treatment share {pct(s['observed_treatment_share'], signed=False)} vs designed "
            f"{pct(s['expected_split'][1], signed=False)}); SRM chi-square p {fmt_p(s['p_value'])}"
            f" -> {'**SRM detected**' if s['srm_detected'] else 'no SRM'}."
        )
    if f.primary:
        r = f.primary
        caveat = " *(not trustworthy: SRM)*" if f.srm and f.srm["srm_detected"] else ""
        vc, vt = _arms(r)
        lines.append(
            f"- **Primary — {r['metric']}:** {fmt_value(vc, r['metric_type'])} control vs "
            f"{fmt_value(vt, r['metric_type'])} treatment; {_effect_line(r)}{caveat}."
        )
    if f.power:
        p = f.power
        lines.append(
            f"- **Power:** {p['observed_n_per_arm']:,} users per arm vs {p['required_n_per_arm']:,} "
            f"required for a {pct(p['mde_relative'], signed=False, digits=0)} MDE; achieved power "
            f"{pct(p['achieved_power_at_mde'], signed=False, digits=0)}; smallest detectable lift now "
            f"{pct(p['detectable_lift_at_current_n'])}."
        )
    if f.duration:
        d = f.duration
        lines.append(
            f"- **Duration:** {d['days_observed']} of {d['planned_duration_days']} planned days "
            f"({d['first_day']} to {d['last_day']})."
        )
    return lines


def _guardrails(f: Findings) -> list[str]:
    if not f.guardrails:
        return ["- Not evaluated (results invalid until the SRM is resolved)."
                if f.srm and f.srm["srm_detected"] else "- No guardrail metrics evaluated."]
    lines = ["| Guardrail | Control | Treatment | Effect | Status |", "|---|---|---|---|---|"]
    for name, r in f.guardrails.items():
        vc, vt = _arms(r)
        lines.append(f"| {name} | {fmt_value(vc, r['metric_type'])} | "
                     f"{fmt_value(vt, r['metric_type'])} | {_effect_line(r)} | {_verdict(r)} |")
    cuped = [(n, r["cuped"]) for n, r in f.guardrails.items() if r.get("cuped")]
    for name, c in cuped:
        lines.append(f"\nThe {name} effect is CUPED-adjusted with its pre-period covariate, "
                     f"which cut variance by {pct(c['variance_reduction'], signed=False)}; "
                     "the control and treatment columns show raw arm means.")
    return lines


def _segments(f: Findings, decision: Decision) -> list[str]:
    if not f.segments:
        return ["- Not analysed (segment cuts on an invalid experiment would mislead)."
                if f.srm and f.srm["srm_detected"] else "- No segment breakdown run."]
    lines = []
    for seg in f.segments:
        het = (f"effects **differ** across levels (Cochran's Q p {fmt_p(seg['heterogeneity_p'])})"
               if seg["heterogeneous"] else
               f"no evidence effects differ (Cochran's Q p {fmt_p(seg['heterogeneity_p'])})")
        lines.append(f"- **{seg['segment']}** — {het}:")
        for lv in seg["levels"]:
            mark = " ✓" if lv["significant_adjusted"] else ""
            lines.append(
                f"  - {lv['level']}: {pct(lv['relative_lift'])} "
                f"(CI {pct(lv['relative_ci_low'])} to {pct(lv['relative_ci_high'])}), "
                f"{CORRECTION_NAMES.get(seg['correction'], seg['correction'])}-adjusted p {fmt_p(lv['p_adjusted'])}{mark}"
            )
    if decision.flags.get("heterogeneous_segment"):
        lines.append(f"\nThe overall lift is concentrated in **{decision.flags['heterogeneous_segment']}**; "
                     "other levels should not receive the change.")
    lines.append("\n✓ = significant after multiple-comparison correction.")
    return lines


def _risks(f: Findings, decision: Decision) -> list[str]:
    risks = []
    if f.srm and f.srm["srm_detected"]:
        risks.append("Sample-ratio mismatch means users were lost or mis-assigned non-randomly; "
                     "any lift may be an artefact of who is missing.")
    if decision.flags.get("breached_guardrails"):
        risks.append(f"Shipping would regress {', '.join(decision.flags['breached_guardrails'])}.")
    if f.duration:
        risks.extend(f.duration["warnings"])
    if f.power and not f.power["adequately_powered"] and not (f.srm and f.srm["srm_detected"]):
        risks.append("Underpowered: a real effect the size of the MDE could easily be missed.")
    if decision.decision == "Ship":
        risks.append("Novelty effects can fade; monitor the primary metric after rollout.")
    return [f"- {r}" for r in risks] or ["- No material risks identified."]


def _next_steps(f: Findings, decision: Decision) -> list[str]:
    d = decision.decision
    if d == "Investigate" and decision.flags.get("srm"):
        steps = ["Audit assignment and exposure logging by platform, brand and day to find "
                 "where treatment users are dropped.",
                 "Fix the bug, then restart the test with fresh assignment; do not reuse this data."]
    elif d == "Investigate":
        steps = ["Design a targeted follow-up test for the responsive segment."]
    elif d == "Don't ship" and decision.flags.get("guardrail_breach"):
        steps = ["Do not roll out. Explore a variant that keeps the primary gain without the "
                 "guardrail cost (e.g. a smaller incentive).",
                 "Re-test the revised variant with the same guardrails."]
    elif d == "Don't ship":
        steps = ["Close the test and keep control; the change did not move the primary metric.",
                 "Record the null result so the idea is not re-tested without a new hypothesis."]
    elif d == "Inconclusive-extend":
        p = f.power or {}
        more = ""
        if p.get("estimated_additional_days") is not None:
            more = (f" (about {p['additional_users_needed_per_arm']:,} more per arm, "
                    f"~{p['estimated_additional_days']} more days at the current pace)")
        steps = [f"Keep the test running until {p.get('required_n_per_arm', 0):,} users per arm{more}.",
                 "Do not act on interim reads; re-run this analysis at the planned end date."]
    else:
        scope = decision.scope
        target = "all users" if scope == "all" else f"{scope} only"
        steps = [f"Roll out to {target} behind a feature flag and monitor guardrails for two weeks.",
                 "Add this result to the experiment log with the measured lift."]
    return [f"{i}. {s}" for i, s in enumerate(steps, 1)]


def render_memo(f: Findings, decision: Decision) -> str:
    meta = f.metadata or {}
    scope = "" if decision.scope == "all" else f" — scope: **{decision.scope}**"
    parts = [
        f"# Decision memo: {meta.get('name', 'Experiment')} (`{meta.get('experiment_id', '')}`)",
        "",
        f"**Hypothesis:** {meta.get('hypothesis', 'n/a')}",
        "",
        "## Decision",
        f"**{decision.decision}**{scope}",
        "",
        f"Why: {decision.rule}",
        "",
        "## Key numbers",
        *_key_numbers(f),
        "",
        "## Guardrails",
        *_guardrails(f),
        "",
        "## Segment insights",
        *_segments(f, decision),
        "",
        "## Risks",
        *_risks(f, decision),
        "",
        "## Next steps",
        *_next_steps(f, decision),
    ]
    return "\n".join(parts) + "\n"
