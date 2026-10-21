"""Agent tools: thin wrappers that run stats functions on an experiment's data.

Each tool takes JSON-serialisable arguments and returns a JSON-serialisable dict, so
the same functions serve LiteLLM tool calling, the MockLLM planner and the UI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from . import stats
from .config import ExperimentContext

BOOTSTRAP_REPS = 1000


def _days_observed(ctx: ExperimentContext) -> int | None:
    col = ctx.config.date_column
    if not col or col not in ctx.df.columns:
        return None
    d = ctx.df[col].astype(str)
    return int((np.datetime64(d.max()) - np.datetime64(d.min())).astype(int) + 1)


def _metric(ctx: ExperimentContext, metric: str):
    if metric not in ctx.config.metrics:
        raise ValueError(f"Unknown metric '{metric}'. Known: {sorted(ctx.config.metrics)}")
    return ctx.config.metrics[metric]


def get_experiment_metadata(ctx: ExperimentContext) -> dict[str, Any]:
    """Pre-registered design: hypothesis, metrics, MDE, alpha, planned duration."""
    return ctx.config.to_public_dict()


def check_srm(ctx: ExperimentContext) -> dict[str, Any]:
    """Sample-ratio mismatch check. Must be the first statistical check."""
    res = stats.srm_check(len(ctx.control), len(ctx.treatment), ctx.config.expected_split)
    return res.to_dict()


def estimate_effect(ctx: ExperimentContext, metric: str, cuped: bool = False) -> dict[str, Any]:
    """Treatment effect on one metric (z-test for proportions, Welch+bootstrap for means)."""
    m = _metric(ctx, metric)
    c, t = ctx.control[m.column].to_numpy(), ctx.treatment[m.column].to_numpy()
    if m.type == "proportion":
        out = stats.two_proportion_ztest(
            int(c.sum()), c.size, int(t.sum()), t.size, ctx.config.alpha
        ).to_dict()
    else:
        cov = m.cuped_covariate if cuped and m.cuped_covariate in ctx.df.columns else None
        out = stats.welch_ttest(
            c, t, alpha=ctx.config.alpha, n_boot=BOOTSTRAP_REPS, seed=0,
            covariate_control=ctx.control[cov].to_numpy() if cov else None,
            covariate_treatment=ctx.treatment[cov].to_numpy() if cov else None,
        ).to_dict()
    return {"metric": metric, "metric_type": m.type, "higher_is_better": m.higher_is_better,
            **out}


def segment_breakdown(
    ctx: ExperimentContext, segment: str, metric: str | None = None, correction: str = "holm"
) -> dict[str, Any]:
    """Per-segment effects with multiple-comparison correction and a heterogeneity test.

    The heterogeneity threshold is Bonferroni-split across the registered segment
    dimensions, since the agent scans all of them.
    """
    metric = metric or ctx.config.primary_metric
    m = _metric(ctx, metric)
    if segment not in ctx.df.columns:
        raise ValueError(f"Unknown segment '{segment}'. Available: {ctx.config.segments}")
    levels = {}
    for level, rows in ctx.df.groupby(segment, sort=True):
        c = rows.loc[rows["variant"] == ctx.config.control_label, m.column].to_numpy()
        t = rows.loc[rows["variant"] == ctx.config.treatment_label, m.column].to_numpy()
        if c.size >= 2 and t.size >= 2:
            levels[str(level)] = (c, t)
    n_dims = max(1, len(ctx.config.segments))
    return stats.segment_breakdown(
        segment, levels, metric, m.type, alpha=ctx.config.alpha, correction=correction,
        heterogeneity_alpha=ctx.config.alpha / n_dims,
    ).to_dict()


def power_analysis(ctx: ExperimentContext, metric: str | None = None) -> dict[str, Any]:
    """Required sample size, achieved power and current MDE for the registered MDE."""
    metric = metric or ctx.config.primary_metric
    m = _metric(ctx, metric)
    c = ctx.control[m.column].to_numpy(float)
    n = int(min(len(ctx.control), len(ctx.treatment)))
    return stats.power_analysis(
        metric, m.type, baseline=float(c.mean()), observed_n_per_arm=n,
        mde_relative=ctx.config.mde_relative, alpha=ctx.config.alpha,
        power=ctx.config.power, sd=float(c.std(ddof=1)), days_observed=_days_observed(ctx),
    ).to_dict()


def check_duration(ctx: ExperimentContext) -> dict[str, Any]:
    """Peeking / duration check against the planned duration and required sample."""
    p = power_analysis(ctx)
    col = ctx.config.date_column
    dates = ctx.df[col].astype(str).tolist() if col and col in ctx.df.columns else None
    return stats.duration_check(
        dates, ctx.config.planned_duration_days, p["observed_n_per_arm"],
        p["required_n_per_arm"], ctx.config.alpha,
    ).to_dict()


@dataclass(frozen=True)
class Tool:
    name: str
    fn: Callable[..., dict[str, Any]]
    description: str
    parameters: dict[str, Any]

    def schema(self) -> dict[str, Any]:
        """OpenAI/LiteLLM function-calling schema."""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


_METRIC = {"type": "string", "description": "Metric name from the experiment metadata."}
TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("get_experiment_metadata", get_experiment_metadata,
         "Return the pre-registered design: hypothesis, primary and guardrail metrics, "
         "segments, MDE, alpha, power, planned duration.",
         {"type": "object", "properties": {}}),
    Tool("check_srm", check_srm,
         "Sample-ratio mismatch chi-square test. ALWAYS run this first; if srm_detected "
         "is true the experiment is invalid and must not be shipped.",
         {"type": "object", "properties": {}}),
    Tool("estimate_effect", estimate_effect,
         "Estimate the treatment effect on a metric: two-proportion z-test for "
         "proportions, Welch t-test with bootstrap CI for means. Set cuped=true for "
         "mean metrics to reduce variance with a pre-period covariate.",
         {"type": "object", "properties": {"metric": _METRIC,
          "cuped": {"type": "boolean", "default": False}}, "required": ["metric"]}),
    Tool("segment_breakdown", segment_breakdown,
         "Per-segment treatment effects with Holm or BH correction and Cochran's Q "
         "heterogeneity test.",
         {"type": "object", "properties": {
             "segment": {"type": "string", "description": "Segment column, e.g. platform."},
             "metric": _METRIC,
             "correction": {"type": "string", "enum": ["holm", "bh"], "default": "holm"}},
          "required": ["segment"]}),
    Tool("power_analysis", power_analysis,
         "Required users per arm for the registered MDE, achieved power, the smallest "
         "lift detectable at the current sample, and extra days needed.",
         {"type": "object", "properties": {"metric": _METRIC}}),
    Tool("check_duration", check_duration,
         "Duration / peeking check: days observed vs planned, information fraction and "
         "the O'Brien-Fleming sequential alpha valid right now.",
         {"type": "object", "properties": {}}),
]}


def tool_schemas() -> list[dict[str, Any]]:
    return [t.schema() for t in TOOLS.values()]


def run_tool(ctx: ExperimentContext, name: str, arguments: dict[str, Any] | str | None) -> dict[str, Any]:
    """Execute a tool by name; errors are returned as data so the agent can recover."""
    if isinstance(arguments, str):
        arguments = json.loads(arguments or "{}")
    arguments = arguments or {}
    if name not in TOOLS:
        return {"error": f"Unknown tool '{name}'. Available: {sorted(TOOLS)}"}
    try:
        return TOOLS[name].fn(ctx, **arguments)
    except (TypeError, ValueError, KeyError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
