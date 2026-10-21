"""Generate reproducible user-level A/B test data with PLANTED ground truth.

Each experiment in experiments.yaml gets a simulation scenario below. Because we know
the true effect we planted (a clear win, no effect, a sample-ratio bug, a guardrail
breach, an app-only effect, an underpowered test), we can score whether the agent
reaches the right ship decision — something you can never do with real data alone.

Usage:
    python scripts/generate_experiments.py [--seed 0] [--out data/]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abagent.config import DATA_FILE, GROUND_TRUTH_FILE, load_registry  # noqa: E402

BASE_CONVERSION = 0.08
BASE_SUPPORT_RATE = 0.03
MEDIAN_ORDER_VALUE = 55.0

# Planted truth per experiment. Keys:
#   n_per_arm            users randomized into each arm
#   run_days             days of data actually collected (vs planned_duration_days)
#   conv_lift            true relative lift in conversion for treatment users
#   segment_lift         optional {dimension: {level: relative lift}} overriding conv_lift
#   aov_effect           relative change in order value for converting treatment users
#   srm_drop             treatment users silently lost (biased toward low-intent users)
SCENARIOS: dict[str, dict[str, Any]] = {
    "exp_checkout_redesign": {
        "scenario": "clear_winner",
        "description": "Treatment lifts conversion ~15% with neutral guardrails.",
        "n_per_arm": 20_000, "run_days": 14, "conv_lift": 0.15, "aov_effect": 0.0,
        "expected_decision": "Ship", "expected_scope": "all",
        "expected_flags": {"srm": False, "guardrail_breach": False,
                           "heterogeneous_segment": None, "underpowered": False},
    },
    "exp_plan_page_copy": {
        "scenario": "flat",
        "description": "No true effect; test is adequately powered for the 10% MDE.",
        "n_per_arm": 20_000, "run_days": 14, "conv_lift": 0.0, "aov_effect": 0.0,
        "expected_decision": "Don't ship", "expected_scope": "all",
        "expected_flags": {"srm": False, "guardrail_breach": False,
                           "heterogeneous_segment": None, "underpowered": False},
    },
    "exp_autopay_banner": {
        "scenario": "sample_ratio_mismatch",
        "description": (
            "No true effect, but a client bug drops ~1,500 low-intent treatment users, "
            "creating a spurious lift. Result is invalid and must be investigated."
        ),
        "n_per_arm": 20_000, "run_days": 14, "conv_lift": 0.0, "aov_effect": 0.0,
        "srm_drop": 1_500,
        "expected_decision": "Investigate", "expected_scope": "all",
        # Once SRM is found the other metrics are untrustworthy, so only SRM is scored.
        "expected_flags": {"srm": True},
    },
    "exp_discount_nudge": {
        "scenario": "guardrail_breach",
        "description": (
            "Discount lifts conversion ~15% but cuts order value 30%, so revenue per "
            "user falls ~20%. Wins the primary metric, fails the revenue guardrail."
        ),
        "n_per_arm": 20_000, "run_days": 14, "conv_lift": 0.15, "aov_effect": -0.30,
        "expected_decision": "Don't ship", "expected_scope": "all",
        "expected_flags": {"srm": False, "guardrail_breach": True,
                           "heterogeneous_segment": None, "underpowered": False},
    },
    "exp_app_onboarding": {
        "scenario": "heterogeneous_effect",
        "description": "Lift of ~25% on app users only; web users see no effect.",
        "n_per_arm": 20_000, "run_days": 14, "conv_lift": 0.0, "aov_effect": 0.0,
        "segment_lift": {"platform": {"app": 0.25, "web": 0.0}},
        "expected_decision": "Ship", "expected_scope": "platform=app",
        "expected_flags": {"srm": False, "guardrail_breach": False,
                           "heterogeneous_segment": "platform=app", "underpowered": False},
    },
    "exp_family_bundle": {
        "scenario": "underpowered",
        "description": (
            "Small real lift (~5%) but only 1,500 users per arm after 6 of 14 planned "
            "days — far below the sample needed. Correct call is to keep running."
        ),
        "n_per_arm": 1_500, "run_days": 6, "conv_lift": 0.05, "aov_effect": 0.0,
        "brand_only": "Nimbus Prepaid",
        "expected_decision": "Inconclusive-extend", "expected_scope": "all",
        "expected_flags": {"srm": False, "guardrail_breach": False,
                           "heterogeneous_segment": None, "underpowered": True},
    },
}


def simulate_experiment(
    exp_id: str, spec: dict[str, Any], start_date: str, rng: np.random.Generator,
    first_user_id: int,
) -> pd.DataFrame:
    """Simulate user-level outcomes for one experiment."""
    n_per_arm = spec["n_per_arm"]
    n = 2 * n_per_arm
    variant = np.array(["control"] * n_per_arm + ["treatment"] * n_per_arm)
    rng.shuffle(variant)
    is_treat = variant == "treatment"

    platform = np.where(rng.random(n) < 0.55, "app", "web")
    if spec.get("brand_only"):
        brand = np.full(n, spec["brand_only"])
    else:
        brand = np.where(rng.random(n) < 0.65, "Nimbus Wireless", "Nimbus Prepaid")
    new_vs_returning = np.where(rng.random(n) < 0.35, "new", "returning")

    # Latent purchase intent drives conversion, order value, sessions and pre-period
    # behaviour, which is what makes CUPED covariates informative.
    intent = rng.normal(size=n)
    base = (
        BASE_CONVERSION
        * np.where(platform == "app", 1.10, 0.88)
        * np.where(new_vs_returning == "new", 0.85, 1.08)
        * np.where(brand == "Nimbus Prepaid", 0.92, 1.04)
        * np.exp(0.35 * intent - 0.35**2 / 2)
    )

    lift = np.full(n, spec["conv_lift"], dtype=float)
    for dim, levels in spec.get("segment_lift", {}).items():
        col = {"platform": platform, "brand": brand, "new_vs_returning": new_vs_returning}[dim]
        lift = np.array([levels.get(v, 0.0) for v in col])
    p_convert = np.clip(base * (1 + lift * is_treat), 0, 1)
    converted = rng.random(n) < p_convert

    order_value = rng.lognormal(np.log(MEDIAN_ORDER_VALUE) + 0.3 * intent, 0.6, n)
    order_value *= 1 + spec["aov_effect"] * is_treat
    # Returning customers also spend on add-ons (insurance, international passes) at a
    # stable personal rate; that rate drives both pre-period and in-test add-on revenue,
    # which is what makes pre_revenue a strong CUPED covariate.
    addon_rate = np.where(
        new_vs_returning == "returning",
        rng.lognormal(np.log(18) + 0.3 * intent, 0.6, n), 0.0,
    )
    addon_revenue = addon_rate * rng.lognormal(0, 0.2, n)
    pre_revenue = np.round(addon_rate * rng.lognormal(0, 0.2, n), 2)
    revenue = np.round(np.where(converted, order_value, 0.0) + addon_revenue, 2)
    activity = np.exp(1.0 + 0.4 * intent)
    pre_sessions = rng.poisson(activity)
    sessions = rng.poisson(activity * (1 + 0.5 * converted)) + 1
    support_contact = rng.random(n) < BASE_SUPPORT_RATE

    start = date.fromisoformat(start_date)
    day_offsets = rng.integers(0, spec["run_days"], n)
    assignment_date = [(start + timedelta(days=int(d))).isoformat() for d in day_offsets]

    df = pd.DataFrame({
        "user_id": [f"u{first_user_id + i:07d}" for i in range(n)],
        "experiment_id": exp_id,
        "variant": variant,
        "assignment_date": assignment_date,
        "platform": platform,
        "brand": brand,
        "new_vs_returning": new_vs_returning,
        "converted": converted.astype(int),
        "revenue": revenue,
        "sessions": sessions,
        "support_contact": support_contact.astype(int),
        "pre_revenue": pre_revenue,
        "pre_sessions": pre_sessions,
    })

    drop = spec.get("srm_drop", 0)
    if drop:
        # Bug: treatment users on slow, low-intent sessions never log an exposure.
        candidates = np.flatnonzero(is_treat & (intent < -1.0))
        dropped = rng.choice(candidates, size=drop, replace=False)
        df = df.drop(index=dropped)
    return df.sort_values(["assignment_date", "user_id"]).reset_index(drop=True)


def generate(seed: int = 0, out_dir: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate all experiments, write CSV + ground truth, return both."""
    out = Path(out_dir or ROOT / "data")
    out.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    frames, truth = [], {}
    next_user = 1
    for idx, (exp_id, cfg) in enumerate(registry.items()):
        spec = SCENARIOS[exp_id]
        rng = np.random.default_rng([seed, idx])  # per-experiment stream: stable if others change
        df = simulate_experiment(exp_id, spec, cfg.start_date or "2026-01-05", rng, next_user)
        next_user += 2 * spec["n_per_arm"]
        frames.append(df)
        truth[exp_id] = {
            "scenario": spec["scenario"],
            "description": spec["description"],
            "expected_decision": spec["expected_decision"],
            "expected_scope": spec["expected_scope"],
            "expected_flags": spec["expected_flags"],
            "planted": {
                "conversion_relative_lift": spec["conv_lift"],
                "segment_lift": spec.get("segment_lift"),
                "order_value_relative_effect": spec["aov_effect"],
                "srm_dropped_treatment_users": spec.get("srm_drop", 0),
                "users_per_arm_randomized": spec["n_per_arm"],
                "days_collected": spec["run_days"],
            },
        }
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(out / DATA_FILE, index=False)
    (out / GROUND_TRUTH_FILE).write_text(
        "# PLANTED ground truth — used only by the evaluator, never shown to the agent.\n"
        + yaml.safe_dump({"seed": seed, "experiments": truth}, sort_keys=False)
    )
    return data, truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    data, truth = generate(args.seed, args.out)
    print(f"Wrote {len(data):,} user rows for {len(truth)} experiments to {args.out}/")
    summary = data.groupby(["experiment_id", "variant"]).agg(
        users=("user_id", "size"), conv=("converted", "mean"), rpu=("revenue", "mean")
    )
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
