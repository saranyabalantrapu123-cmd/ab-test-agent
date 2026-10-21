"""Streamlit UI: pick an experiment (or upload a CSV), run the agent, review the readout.

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from abagent.agent import AgentResult, mock_policy, run_agent
from abagent.config import (DATA_DIR, DATA_FILE, REPORTS_DIR, ROOT, build_context, load_data,
                            load_registry)
from abagent.llm import OFFLINE_BANNER, configured_model, get_llm, is_offline
from abagent.upload import ColumnMapping, context_from_upload

DECISION_ICONS = {"Ship": "✅", "Don't ship": "⛔", "Investigate": "🔍",
                  "Inconclusive-extend": "⏳"}

st.set_page_config(page_title="A/B Test Agent", page_icon="🧪", layout="wide")
st.title("🧪 A/B Test Readout Agent")
st.caption("SRM check → effect estimates → guardrails → power & peeking → segments → "
           "validated decision memo.")
if is_offline():
    st.warning(OFFLINE_BANNER)
else:
    st.info(f"Using {configured_model()} via LiteLLM")


@st.cache_data(show_spinner="Loading experiment data…")
def _data() -> pd.DataFrame:
    return load_data()


def _ensure_demo_data() -> bool:
    if (DATA_DIR / DATA_FILE).exists():
        return True
    st.info("Demo data has not been generated yet.")
    if st.button("Generate demo data"):
        sys.path.insert(0, str(ROOT / "scripts"))
        from generate_experiments import generate

        with st.spinner("Generating 6 experiments…"):
            generate()
        st.rerun()
    return False


def _upload_context():
    file = st.sidebar.file_uploader("User-level CSV (one row per user)", type="csv")
    if file is None:
        st.info("Upload a CSV with one row per user: a variant column plus metric columns. "
                "Map the columns in the sidebar, then run the analysis.")
        return None
    df = pd.read_csv(file)
    cols = list(df.columns)
    sb = st.sidebar
    variant = sb.selectbox("Variant column", cols)
    labels = sorted(df[variant].astype(str).unique())
    control = sb.selectbox("Control label", labels)
    treatment = sb.selectbox("Treatment label", [x for x in labels if x != control] or labels)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and c != variant]
    primary = sb.selectbox("Primary metric column", numeric)
    guardrails = sb.multiselect("Guardrail metric columns", [c for c in numeric if c != primary])
    lower = sb.multiselect("Guardrails where lower is better", guardrails)
    segments = sb.multiselect("Segment columns",
                              [c for c in cols if c not in numeric and c != variant])
    date = sb.selectbox("Assignment date column (optional)", ["(none)", *cols])
    alpha = sb.number_input("Alpha", 0.001, 0.2, 0.05, 0.005)
    mde = sb.number_input("MDE (relative)", 0.01, 1.0, 0.10, 0.01)
    days = sb.number_input("Planned duration (days)", 1, 120, 14)
    st.dataframe(df.head(10), width="stretch")
    mapping = ColumnMapping(variant, control, treatment, primary, guardrails, lower, segments,
                            None if date == "(none)" else date)
    try:
        return context_from_upload(df, mapping, name=Path(file.name).stem, alpha=alpha,
                                   mde_relative=mde, planned_duration_days=int(days))
    except ValueError as exc:
        st.error(str(exc))
        return None


def _show(result: AgentResult) -> None:
    icon = DECISION_ICONS.get(result.decision, "")
    scope = "" if result.scope == "all" else f" — {result.scope}"
    c1, c2, c3 = st.columns(3)
    c1.metric("Decision", f"{icon} {result.decision}", scope.strip(" —") or None,
              delta_color="off")
    c2.metric("Tool calls", len(result.trace))
    c3.metric("Validator", "Passed" if result.validation.passed else "Failed")
    if result.validation.passed:
        st.success(result.validation.summary())
    else:
        st.error(result.validation.summary())
    if not is_offline():
        st.caption(f"Tokens: {result.usage['prompt_tokens'] + result.usage['completion_tokens']:,}"
                   f" · est. cost ${result.usage['estimated_cost_usd']:.4f}")

    memo_tab, charts_tab, checks_tab = st.tabs(["Memo", "Charts", "Checks run"])
    with memo_tab:
        if result.plan:
            st.caption(result.plan)
        st.markdown(result.memo_markdown)
        st.download_button("Download memo (.md)", result.memo_markdown,
                           file_name=f"{result.experiment_id}_memo.md")
    with charts_tab:
        for path in result.charts.values():
            st.image(path, width="stretch")
        if not result.charts:
            st.write("No charts for this run.")
    with checks_tab:
        for i, call in enumerate(result.trace, 1):
            args = ", ".join(f"{k}={v}" for k, v in call["arguments"].items())
            with st.expander(f"{i}. {call['tool']}({args})"):
                st.json(call["result"])


source = st.sidebar.radio("Data source", ["Demo experiments", "Upload CSV"])
ctx = None
if source == "Demo experiments":
    if _ensure_demo_data():
        registry = load_registry()
        exp_id = st.sidebar.selectbox(
            "Experiment", list(registry), format_func=lambda e: f"{registry[e].name} ({e})")
        cfg = registry[exp_id]
        st.subheader(cfg.name)
        st.write(f"**Hypothesis:** {cfg.hypothesis}")
        st.write(f"Primary: `{cfg.primary_metric}` · Guardrails: "
                 f"{', '.join(f'`{g}`' for g in cfg.guardrail_metrics)} · "
                 f"MDE {cfg.mde_relative:.0%} · alpha {cfg.alpha}")
        ctx = build_context(exp_id, _data())
else:
    ctx = _upload_context()

if ctx is not None and st.button("Run analysis", type="primary"):
    with st.spinner("Agent is running checks…"):
        out_dir = REPORTS_DIR if source == "Demo experiments" else REPORTS_DIR / "uploads"
        st.session_state["result"] = run_agent(ctx, llm=get_llm(mock_policy), reports_dir=out_dir)

result = st.session_state.get("result")
if result is not None and ctx is not None and result.experiment_id == ctx.config.experiment_id:
    _show(result)
