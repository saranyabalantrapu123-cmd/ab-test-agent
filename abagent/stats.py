"""Statistical toolkit for A/B test readouts.

Every function is pure (no I/O, no globals) and returns a dataclass with a
``to_dict()`` so results can be passed to an LLM as JSON tool output, rendered in
the UI, and checked by the hallucination validator.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import stats as sps


def _clean(value: Any) -> Any:
    """Make numpy scalars / NaN JSON-friendly and round floats for readable output."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return float(f"{f:.6g}")
    return value


class _Result:
    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))  # type: ignore[call-overload]


# --------------------------------------------------------------------------- SRM
@dataclass
class SRMResult(_Result):
    n_control: int
    n_treatment: int
    expected_split: list[float]
    observed_treatment_share: float
    chi2: float
    p_value: float
    threshold: float
    srm_detected: bool


def srm_check(
    n_control: int,
    n_treatment: int,
    expected_split: Sequence[float] = (0.5, 0.5),
    threshold: float = 0.001,
) -> SRMResult:
    """Chi-square goodness-of-fit test of observed vs designed allocation.

    A strict threshold (p < 0.001) is industry standard: SRM indicates broken
    randomization or logging, which invalidates every downstream metric.
    """
    total = n_control + n_treatment
    weights = np.asarray(expected_split, dtype=float)
    expected = total * weights / weights.sum()
    chi2, p = sps.chisquare([n_control, n_treatment], f_exp=expected)
    return SRMResult(
        n_control=int(n_control),
        n_treatment=int(n_treatment),
        expected_split=[float(w) for w in weights / weights.sum()],
        observed_treatment_share=n_treatment / total,
        chi2=float(chi2),
        p_value=float(p),
        threshold=threshold,
        srm_detected=bool(p < threshold),
    )


# ------------------------------------------------------------ proportion z-test
@dataclass
class ProportionTestResult(_Result):
    n_control: int
    n_treatment: int
    conversions_control: int
    conversions_treatment: int
    rate_control: float
    rate_treatment: float
    abs_diff: float
    abs_ci_low: float
    abs_ci_high: float
    relative_lift: float
    relative_ci_low: float
    relative_ci_high: float
    z: float
    p_value: float
    ci_level: float
    significant: bool


def two_proportion_ztest(
    x_control: int, n_control: int, x_treatment: int, n_treatment: int, alpha: float = 0.05
) -> ProportionTestResult:
    """Pooled two-proportion z-test with Wald CI on the difference and a
    delta-method CI on the relative lift (log of the rate ratio)."""
    p_c, p_t = x_control / n_control, x_treatment / n_treatment
    pooled = (x_control + x_treatment) / (n_control + n_treatment)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n_control + 1 / n_treatment))
    diff = p_t - p_c
    z = diff / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * sps.norm.sf(abs(z))

    zcrit = sps.norm.ppf(1 - alpha / 2)
    se_unpooled = math.sqrt(p_c * (1 - p_c) / n_control + p_t * (1 - p_t) / n_treatment)
    abs_low, abs_high = diff - zcrit * se_unpooled, diff + zcrit * se_unpooled

    if p_c > 0 and p_t > 0:
        se_log = math.sqrt((1 - p_t) / x_treatment + (1 - p_c) / x_control)
        log_rr = math.log(p_t / p_c)
        rel_low = math.exp(log_rr - zcrit * se_log) - 1
        rel_high = math.exp(log_rr + zcrit * se_log) - 1
        rel = p_t / p_c - 1
    else:
        rel = rel_low = rel_high = float("nan")

    return ProportionTestResult(
        n_control=int(n_control), n_treatment=int(n_treatment),
        conversions_control=int(x_control), conversions_treatment=int(x_treatment),
        rate_control=p_c, rate_treatment=p_t, abs_diff=diff,
        abs_ci_low=abs_low, abs_ci_high=abs_high,
        relative_lift=rel, relative_ci_low=rel_low, relative_ci_high=rel_high,
        z=z, p_value=float(p_value), ci_level=1 - alpha, significant=bool(p_value < alpha),
    )


# ------------------------------------------------------------- means / revenue
def bootstrap_relative_lift_ci(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int = 0,
    chunk: int = 100,
) -> tuple[float, float]:
    """Percentile bootstrap CI for mean(treatment)/mean(control) - 1.

    Resamples in chunks so memory stays bounded for large arms.
    """
    rng = np.random.default_rng(seed)
    c, t = np.asarray(control, float), np.asarray(treatment, float)
    lifts = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        k = min(chunk, n_boot - start)
        mc = c[rng.integers(0, c.size, (k, c.size))].mean(axis=1)
        mt = t[rng.integers(0, t.size, (k, t.size))].mean(axis=1)
        lifts[start : start + k] = mt / mc - 1
    lo, hi = np.quantile(lifts, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


@dataclass
class CupedResult(_Result):
    theta: float
    covariate_mean: float
    correlation: float
    variance_reduction: float


def cuped_adjust(
    y: np.ndarray, x: np.ndarray
) -> tuple[np.ndarray, CupedResult]:
    """CUPED: y_adj = y - theta * (x - mean(x)), theta = cov(x, y) / var(x).

    theta is estimated on the pooled sample (both arms), which keeps the
    treatment-effect estimate unbiased because x is measured pre-experiment.
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    var_x = x.var(ddof=1)
    theta = float(np.cov(x, y, ddof=1)[0, 1] / var_x) if var_x > 0 else 0.0
    adjusted = y - theta * (x - x.mean())
    var_y = y.var(ddof=1)
    reduction = 1 - adjusted.var(ddof=1) / var_y if var_y > 0 else 0.0
    corr = float(np.corrcoef(x, y)[0, 1]) if var_x > 0 and var_y > 0 else 0.0
    return adjusted, CupedResult(theta, float(x.mean()), corr, float(reduction))


@dataclass
class MeanTestResult(_Result):
    n_control: int
    n_treatment: int
    mean_control: float
    mean_treatment: float
    abs_diff: float
    abs_ci_low: float
    abs_ci_high: float
    relative_lift: float
    relative_ci_low: float
    relative_ci_high: float
    t: float
    df: float
    p_value: float
    ci_level: float
    significant: bool
    ci_method: str
    cuped: CupedResult | None = None


def welch_ttest(
    control: Sequence[float],
    treatment: Sequence[float],
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int = 0,
    covariate_control: Sequence[float] | None = None,
    covariate_treatment: Sequence[float] | None = None,
) -> MeanTestResult:
    """Welch's unequal-variance t-test with a Welch CI on the absolute difference and
    a bootstrap percentile CI on the relative lift. If covariates are supplied, the
    test runs on CUPED-adjusted values (means reported are the raw arm means)."""
    c_raw, t_raw = np.asarray(control, float), np.asarray(treatment, float)
    c, t = c_raw, t_raw
    cuped: CupedResult | None = None
    if covariate_control is not None and covariate_treatment is not None:
        y = np.concatenate([c_raw, t_raw])
        x = np.concatenate([np.asarray(covariate_control, float),
                            np.asarray(covariate_treatment, float)])
        adjusted, cuped = cuped_adjust(y, x)
        c, t = adjusted[: c_raw.size], adjusted[c_raw.size :]

    res = sps.ttest_ind(t, c, equal_var=False)
    vc, vt = c.var(ddof=1) / c.size, t.var(ddof=1) / t.size
    se = math.sqrt(vc + vt)
    dof = (vc + vt) ** 2 / (vc**2 / (c.size - 1) + vt**2 / (t.size - 1))
    tcrit = sps.t.ppf(1 - alpha / 2, dof)
    diff = float(t.mean() - c.mean())
    mean_c = float(c_raw.mean())
    rel_low, rel_high = bootstrap_relative_lift_ci(
        # Shift adjusted values back to the raw control mean so the ratio is on
        # the original scale (CUPED-adjusted values are centred differently).
        c - c.mean() + mean_c, t - c.mean() + mean_c, alpha=alpha, n_boot=n_boot, seed=seed
    )
    return MeanTestResult(
        n_control=c.size, n_treatment=t.size,
        mean_control=mean_c, mean_treatment=float(t_raw.mean()),
        abs_diff=diff, abs_ci_low=diff - tcrit * se, abs_ci_high=diff + tcrit * se,
        relative_lift=diff / mean_c if mean_c else float("nan"),
        relative_ci_low=rel_low, relative_ci_high=rel_high,
        t=float(res.statistic), df=float(dof), p_value=float(res.pvalue),
        ci_level=1 - alpha, significant=bool(res.pvalue < alpha),
        ci_method=f"Welch (absolute), bootstrap percentile x{n_boot} (relative)",
        cuped=cuped,
    )


# ------------------------------------------------------ multiple comparisons
def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (controls family-wise error rate)."""
    p = np.asarray(p_values, float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def benjamini_hochberg_adjust(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (controls false discovery rate)."""
    p = np.asarray(p_values, float)
    m = p.size
    order = np.argsort(p)[::-1]
    adjusted = np.empty(m)
    running = 1.0
    for rank_from_top, idx in enumerate(order):
        rank = m - rank_from_top
        running = min(running, p[idx] * m / rank)
        adjusted[idx] = running
    return adjusted.tolist()


ADJUSTERS = {"holm": holm_adjust, "bh": benjamini_hochberg_adjust}


@dataclass
class SegmentLevelResult(_Result):
    level: str
    n_control: int
    n_treatment: int
    value_control: float
    value_treatment: float
    abs_diff: float
    relative_lift: float
    relative_ci_low: float
    relative_ci_high: float
    p_value: float
    p_adjusted: float
    significant_adjusted: bool


@dataclass
class SegmentBreakdownResult(_Result):
    segment: str
    metric: str
    correction: str
    alpha: float
    levels: list[SegmentLevelResult]
    heterogeneity_q: float
    heterogeneity_df: int
    heterogeneity_p: float
    heterogeneous: bool
    significant_levels: list[str] = field(default_factory=list)


def cochran_q(effects: Sequence[float], ses: Sequence[float]) -> tuple[float, int, float]:
    """Cochran's Q test that per-segment effects are equal (inverse-variance weights)."""
    e, s = np.asarray(effects, float), np.asarray(ses, float)
    mask = s > 0
    e, w = e[mask], 1 / s[mask] ** 2
    if e.size < 2:
        return 0.0, 0, 1.0
    pooled = (w * e).sum() / w.sum()
    q = float((w * (e - pooled) ** 2).sum())
    dof = e.size - 1
    return q, dof, float(sps.chi2.sf(q, dof))


def segment_breakdown(
    segment: str,
    levels: dict[str, tuple[np.ndarray, np.ndarray]],
    metric: str,
    metric_type: str,
    alpha: float = 0.05,
    correction: str = "holm",
    heterogeneity_alpha: float | None = None,
) -> SegmentBreakdownResult:
    """Per-level treatment effects for one segment dimension.

    ``levels`` maps level -> (control_values, treatment_values). Per-level p-values
    are corrected with Holm (FWER) or BH (FDR); Cochran's Q tests whether the
    absolute effect differs across levels.
    """
    if correction not in ADJUSTERS:
        raise ValueError(f"correction must be one of {sorted(ADJUSTERS)}")
    rows, effects, ses = [], [], []
    for level, (c, t) in levels.items():
        c, t = np.asarray(c, float), np.asarray(t, float)
        if metric_type == "proportion":
            r = two_proportion_ztest(int(c.sum()), c.size, int(t.sum()), t.size, alpha)
            vc, vt = r.rate_control, r.rate_treatment
            rel, lo, hi, p = r.relative_lift, r.relative_ci_low, r.relative_ci_high, r.p_value
            se = math.sqrt(vc * (1 - vc) / c.size + vt * (1 - vt) / t.size)
        else:
            res = sps.ttest_ind(t, c, equal_var=False)
            vc, vt, p = float(c.mean()), float(t.mean()), float(res.pvalue)
            se = math.sqrt(c.var(ddof=1) / c.size + t.var(ddof=1) / t.size)
            zc = sps.norm.ppf(1 - alpha / 2)
            rel = (vt - vc) / vc if vc else float("nan")
            lo = (vt - vc - zc * se) / vc if vc else float("nan")
            hi = (vt - vc + zc * se) / vc if vc else float("nan")
        rows.append([level, c.size, t.size, vc, vt, vt - vc, rel, lo, hi, p])
        effects.append(vt - vc)
        ses.append(se)

    adjusted = ADJUSTERS[correction]([r[-1] for r in rows])
    level_results = [
        SegmentLevelResult(*r, p_adjusted=pa, significant_adjusted=bool(pa < alpha))
        for r, pa in zip(rows, adjusted, strict=True)
    ]
    q, dof, q_p = cochran_q(effects, ses)
    het_alpha = alpha if heterogeneity_alpha is None else heterogeneity_alpha
    return SegmentBreakdownResult(
        segment=segment, metric=metric, correction=correction, alpha=alpha,
        levels=level_results, heterogeneity_q=q, heterogeneity_df=dof,
        heterogeneity_p=q_p, heterogeneous=bool(q_p < het_alpha),
        significant_levels=[lr.level for lr in level_results if lr.significant_adjusted],
    )


# ----------------------------------------------------------- power and MDE
def cohens_h(p1: float, p2: float) -> float:
    return 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))


def required_n_proportion(
    baseline: float, mde_relative: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Users per arm to detect a relative lift with a two-sided test (Cohen's h,
    identical to statsmodels NormalIndPower with ratio=1)."""
    h = abs(cohens_h(baseline, min(baseline * (1 + mde_relative), 1.0)))
    z = sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)
    return math.ceil(2 * (z / h) ** 2)


def power_proportion(
    baseline: float, mde_relative: float, n_per_arm: int, alpha: float = 0.05
) -> float:
    """Power of a two-sided test at n users per arm (normal approximation)."""
    h = abs(cohens_h(baseline, min(baseline * (1 + mde_relative), 1.0)))
    zcrit = sps.norm.ppf(1 - alpha / 2)
    shift = h * math.sqrt(n_per_arm / 2)
    return float(sps.norm.sf(zcrit - shift) + sps.norm.cdf(-zcrit - shift))


def mde_proportion(
    baseline: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Smallest relative lift detectable at n users per arm (inverts Cohen's h)."""
    z = sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)
    h = z * math.sqrt(2 / n_per_arm)
    phi = 2 * math.asin(math.sqrt(baseline)) + h
    p2 = math.sin(min(phi, math.pi) / 2) ** 2
    return p2 / baseline - 1


def required_n_mean(
    mean: float, sd: float, mde_relative: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Users per arm to detect a relative change in a mean (Cohen's d)."""
    d = abs(mean * mde_relative) / sd
    z = sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)
    return math.ceil(2 * (z / d) ** 2)


def power_mean(mean: float, sd: float, mde_relative: float, n_per_arm: int,
               alpha: float = 0.05) -> float:
    d = abs(mean * mde_relative) / sd
    zcrit = sps.norm.ppf(1 - alpha / 2)
    shift = d * math.sqrt(n_per_arm / 2)
    return float(sps.norm.sf(zcrit - shift) + sps.norm.cdf(-zcrit - shift))


def mde_mean(mean: float, sd: float, n_per_arm: int, alpha: float = 0.05,
             power: float = 0.8) -> float:
    z = sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)
    return z * sd * math.sqrt(2 / n_per_arm) / abs(mean)


@dataclass
class PowerResult(_Result):
    metric: str
    metric_type: str
    baseline: float
    mde_relative: float
    alpha: float
    target_power: float
    required_n_per_arm: int
    observed_n_per_arm: int
    achieved_power_at_mde: float
    detectable_lift_at_current_n: float
    additional_users_needed_per_arm: int
    adequately_powered: bool
    users_per_arm_per_day: float | None = None
    estimated_additional_days: int | None = None


def power_analysis(
    metric: str,
    metric_type: str,
    baseline: float,
    observed_n_per_arm: int,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.8,
    sd: float | None = None,
    days_observed: int | None = None,
) -> PowerResult:
    """Required sample, achieved power and current MDE for the pre-registered MDE."""
    if metric_type == "proportion":
        req = required_n_proportion(baseline, mde_relative, alpha, power)
        achieved = power_proportion(baseline, mde_relative, observed_n_per_arm, alpha)
        mde_now = mde_proportion(baseline, observed_n_per_arm, alpha, power)
    else:
        if sd is None:
            raise ValueError("sd is required for mean metrics")
        req = required_n_mean(baseline, sd, mde_relative, alpha, power)
        achieved = power_mean(baseline, sd, mde_relative, observed_n_per_arm, alpha)
        mde_now = mde_mean(baseline, sd, observed_n_per_arm, alpha, power)
    extra = max(0, req - observed_n_per_arm)
    per_day = observed_n_per_arm / days_observed if days_observed else None
    extra_days = math.ceil(extra / per_day) if per_day else None
    return PowerResult(
        metric=metric, metric_type=metric_type, baseline=baseline, mde_relative=mde_relative,
        alpha=alpha, target_power=power, required_n_per_arm=req,
        observed_n_per_arm=int(observed_n_per_arm), achieved_power_at_mde=achieved,
        detectable_lift_at_current_n=mde_now, additional_users_needed_per_arm=extra,
        adequately_powered=bool(observed_n_per_arm >= req),
        users_per_arm_per_day=per_day, estimated_additional_days=extra_days,
    )


# ------------------------------------------------------ duration / peeking
def obrien_fleming_alpha(information_fraction: float, alpha: float = 0.05) -> float:
    """Lan-DeMets O'Brien-Fleming alpha spent by information fraction t:
    alpha(t) = 2 - 2 * Phi(z_{alpha/2} / sqrt(t))."""
    t = min(max(information_fraction, 1e-9), 1.0)
    return float(2 - 2 * sps.norm.cdf(sps.norm.ppf(1 - alpha / 2) / math.sqrt(t)))


@dataclass
class DurationResult(_Result):
    first_day: str | None
    last_day: str | None
    days_observed: int
    planned_duration_days: int
    full_weeks_observed: int
    information_fraction: float
    nominal_alpha: float
    sequential_alpha_now: float
    peeking_risk: bool
    warnings: list[str]


def duration_check(
    dates: Sequence[str] | None,
    planned_duration_days: int,
    observed_n_per_arm: int,
    required_n_per_arm: int,
    alpha: float = 0.05,
) -> DurationResult:
    """Flag early reads ("peeking"): reading a test before its planned duration or
    sample inflates false positives. Reports the O'Brien-Fleming alpha that would be
    valid at the current information fraction."""
    first = last = None
    days = 0
    if dates is not None and len(dates):
        ordered = sorted(set(str(d) for d in dates))
        first, last = ordered[0], ordered[-1]
        days = (np.datetime64(last) - np.datetime64(first)).astype(int) + 1
    info = min(1.0, observed_n_per_arm / required_n_per_arm) if required_n_per_arm else 1.0
    warnings: list[str] = []
    if days and days < planned_duration_days:
        warnings.append(
            f"Read after {days} of {planned_duration_days} planned days; the decision "
            "should wait for the planned duration."
        )
    if days and days < 7:
        warnings.append("Less than one full weekly cycle observed; day-of-week effects are not averaged out.")
    if info < 1.0:
        warnings.append(
            f"Only {info:.0%} of the required sample collected; a nominal p-value read now "
            "overstates confidence (use the sequential alpha)."
        )
    return DurationResult(
        first_day=first, last_day=last, days_observed=int(days),
        planned_duration_days=planned_duration_days, full_weeks_observed=int(days // 7),
        information_fraction=info, nominal_alpha=alpha,
        sequential_alpha_now=obrien_fleming_alpha(info, alpha),
        peeking_risk=bool(warnings), warnings=warnings,
    )
