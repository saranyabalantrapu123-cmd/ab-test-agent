# Evaluation: agent decisions vs planted ground truth

**Mode:** rule-based offline mode (MockLLM, no API key)

| experiment_id | scenario | expected | agent | decision_correct | planted_issue_surfaced | validator_passed | tool_calls |
|---|---|---|---|---|---|---|---|
| exp_checkout_redesign | clear_winner | Ship | Ship | yes | yes | yes | 10 |
| exp_plan_page_copy | flat | Don't ship | Don't ship | yes | yes | yes | 10 |
| exp_autopay_banner | sample_ratio_mismatch | Investigate | Investigate | yes | yes | yes | 4 |
| exp_discount_nudge | guardrail_breach | Don't ship | Don't ship | yes | yes | yes | 10 |
| exp_app_onboarding | heterogeneous_effect | Ship (platform=app) | Ship (platform=app) | yes | yes | yes | 10 |
| exp_family_bundle | underpowered | Inconclusive-extend | Inconclusive-extend | yes | yes | yes | 10 |

- Decision accuracy: **100%** (6 experiments)
- Planted issue surfaced: **100%**
- Wrong ship decisions: **0**
- Validator pass rate: **100%**
- Tokens: 0; estimated cost: $0.0000

## Seed sweep (20 regenerated datasets)

Mean decision accuracy: **96.7%**; wrong ship decisions across all runs: **0**.

| Scenario | Decision accuracy |
|---|---|
| clear_winner | 90% |
| flat | 95% |
| sample_ratio_mismatch | 100% |
| guardrail_breach | 100% |
| heterogeneous_effect | 95% |
| underpowered | 100% |

Misses (each new seed is a fresh random draw, so a fixed-alpha, fixed-power design will occasionally show a chance guardrail regression, miss a real segment difference, or land just under the required sample):

| Seed | Scenario | Expected | Agent | Why (rule engine on the tool outputs) |
|---|---|---|---|---|
| 9 | clear_winner | Ship | Don't ship | Primary metric improved but guardrail(s) support_contact_rate significantly regressed. |
| 11 | clear_winner | Ship | Don't ship | Primary metric improved but guardrail(s) support_contact_rate significantly regressed. |
| 13 | flat | Don't ship | Inconclusive-extend | No significant effect yet, but the test is underpowered for the MDE. |
| 18 | heterogeneous_effect | Ship (platform=app) | Ship | Primary metric significantly improved with no guardrail regressions. |
