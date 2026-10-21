"""Each statistical function checked against scipy / statsmodels references or known values."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (confint_proportions_2indep, proportion_effectsize,
                                          proportions_ztest)
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from abagent import stats


def test_srm_matches_scipy_chisquare():
    res = stats.srm_check(20_000, 18_500)
    chi2, p = sps.chisquare([20_000, 18_500])
    assert res.chi2 == pytest.approx(chi2)
    assert res.p_value == pytest.approx(p)
    assert res.srm_detected


def test_srm_not_flagged_for_balanced_split_and_respects_custom_split():
    assert not stats.srm_check(20_000, 20_050).srm_detected
    # 30/70 design observed at 30/70 is fine; at 50/50 it is a mismatch
    assert not stats.srm_check(3_000, 7_000, (0.3, 0.7)).srm_detected
    assert stats.srm_check(5_000, 5_000, (0.3, 0.7)).srm_detected


def test_two_proportion_ztest_matches_statsmodels():
    r = stats.two_proportion_ztest(800, 10_000, 900, 10_000)
    z, p = proportions_ztest([900, 800], [10_000, 10_000])
    assert r.z == pytest.approx(z)
    assert r.p_value == pytest.approx(p)
    lo, hi = confint_proportions_2indep(900, 10_000, 800, 10_000, method="wald", compare="diff")
    assert (r.abs_ci_low, r.abs_ci_high) == pytest.approx((lo, hi))
    lo, hi = confint_proportions_2indep(900, 10_000, 800, 10_000, method="log", compare="ratio")
    assert (r.relative_ci_low + 1, r.relative_ci_high + 1) == pytest.approx((lo, hi))
    assert r.relative_lift == pytest.approx(0.125)
    assert r.significant


def test_welch_matches_scipy_and_statsmodels():
    rng = np.random.default_rng(1)
    c, t = rng.exponential(10, 3_000), rng.exponential(10.8, 2_500)
    r = stats.welch_ttest(c, t, n_boot=500)
    ref = sps.ttest_ind(t, c, equal_var=False)
    assert r.t == pytest.approx(ref.statistic)
    assert r.p_value == pytest.approx(ref.pvalue)
    lo, hi = CompareMeans(DescrStatsW(t), DescrStatsW(c)).tconfint_diff(usevar="unequal")
    assert (r.abs_ci_low, r.abs_ci_high) == pytest.approx((lo, hi))
    assert r.relative_ci_low < r.relative_lift < r.relative_ci_high


def test_bootstrap_is_seeded_and_agrees_with_delta_method():
    rng = np.random.default_rng(2)
    c, t = rng.normal(100, 20, 4_000), rng.normal(110, 20, 4_000)
    a = stats.bootstrap_relative_lift_ci(c, t, n_boot=2_000, seed=5)
    assert a == stats.bootstrap_relative_lift_ci(c, t, n_boot=2_000, seed=5)
    # delta-method CI on log(mean_t / mean_c) as an analytic reference
    se = np.sqrt(t.var(ddof=1) / (t.size * t.mean() ** 2) + c.var(ddof=1) / (c.size * c.mean() ** 2))
    ratio = np.log(t.mean() / c.mean())
    ref = np.exp(ratio + np.array([-1, 1]) * 1.959964 * se) - 1
    assert a == pytest.approx(tuple(ref), abs=0.003)


def test_cuped_theta_is_ols_slope_and_reduces_variance():
    rng = np.random.default_rng(3)
    x = rng.normal(50, 10, 10_000)
    y = 0.8 * x + rng.normal(0, 6, 10_000)
    adjusted, res = stats.cuped_adjust(y, x)
    slope = np.polyfit(x, y, 1)[0]
    assert res.theta == pytest.approx(slope)
    assert res.variance_reduction == pytest.approx(res.correlation**2, rel=1e-6)
    assert adjusted.mean() == pytest.approx(y.mean())


def test_cuped_welch_keeps_effect_and_tightens_ci():
    rng = np.random.default_rng(4)
    pre = rng.normal(50, 10, 8_000)
    y = pre + rng.normal(0, 5, 8_000)
    y[4_000:] += 1.0  # true effect
    plain = stats.welch_ttest(y[:4_000], y[4_000:], n_boot=200)
    cuped = stats.welch_ttest(y[:4_000], y[4_000:], n_boot=200,
                              covariate_control=pre[:4_000], covariate_treatment=pre[4_000:])
    assert cuped.cuped is not None and cuped.cuped.variance_reduction > 0.5
    assert (cuped.abs_ci_high - cuped.abs_ci_low) < (plain.abs_ci_high - plain.abs_ci_low)
    assert cuped.abs_ci_low < 1.0 < cuped.abs_ci_high


@pytest.mark.parametrize("method,ref", [("holm", "holm"), ("bh", "fdr_bh")])
def test_multiple_comparison_matches_statsmodels(method, ref):
    p = [0.001, 0.01, 0.02, 0.04, 0.3, 0.049]
    ours = stats.ADJUSTERS[method](p)
    assert ours == pytest.approx(multipletests(p, method=ref)[1].tolist())


def test_cochran_q_known_values():
    q, dof, p = stats.cochran_q([0.0, 0.2], [0.1, 0.1])
    # pooled = 0.1 -> Q = (0.1/0.1)^2 * 2 = 2
    assert (q, dof) == (pytest.approx(2.0), 1)
    assert p == pytest.approx(sps.chi2.sf(2.0, 1))


def test_segment_breakdown_detects_concentrated_effect():
    rng = np.random.default_rng(5)
    levels = {
        "app": (rng.random(20_000) < 0.08, rng.random(20_000) < 0.10),
        "web": (rng.random(20_000) < 0.08, rng.random(20_000) < 0.08),
    }
    res = stats.segment_breakdown("platform", levels, "conversion_rate", "proportion")
    assert res.heterogeneous
    assert res.significant_levels == ["app"]


def test_power_and_sample_size_match_statsmodels():
    h = proportion_effectsize(0.088, 0.08)
    expected_n = NormalIndPower().solve_power(effect_size=h, alpha=0.05, power=0.8)
    assert stats.required_n_proportion(0.08, 0.10) == int(np.ceil(expected_n))
    assert stats.power_proportion(0.08, 0.10, 10_000) == pytest.approx(
        NormalIndPower().power(effect_size=h, nobs1=10_000, alpha=0.05))
    # MDE at the required n should give back the target lift
    assert stats.mde_proportion(0.08, int(expected_n)) == pytest.approx(0.10, abs=1e-3)


def test_mean_power_matches_statsmodels():
    d = 0.05 * 20 / 25
    expected = NormalIndPower().solve_power(effect_size=d, alpha=0.05, power=0.8)
    assert stats.required_n_mean(20, 25, 0.05) == int(np.ceil(expected))
    assert stats.mde_mean(20, 25, int(expected)) == pytest.approx(0.05, rel=1e-3)


def test_power_analysis_extra_days():
    res = stats.power_analysis("conversion_rate", "proportion", 0.08, 1_500, 0.10,
                               days_observed=6)
    assert not res.adequately_powered
    assert res.users_per_arm_per_day == 250
    assert res.estimated_additional_days == int(np.ceil(res.additional_users_needed_per_arm / 250))


def test_obrien_fleming_alpha_spending():
    assert stats.obrien_fleming_alpha(1.0) == pytest.approx(0.05)
    assert stats.obrien_fleming_alpha(0.5) == pytest.approx(0.00557, abs=1e-4)
    assert stats.obrien_fleming_alpha(0.2) < stats.obrien_fleming_alpha(0.5)


def test_duration_check_warns_on_early_read():
    dates = [f"2026-05-{d:02d}" for d in range(18, 24)]
    res = stats.duration_check(dates, 14, 1_500, 18_000)
    assert res.days_observed == 6
    assert res.peeking_risk and len(res.warnings) == 3
    full = stats.duration_check([f"2026-03-{d:02d}" for d in range(2, 16)], 14, 20_000, 18_000)
    assert not full.peeking_risk and full.full_weeks_observed == 2
