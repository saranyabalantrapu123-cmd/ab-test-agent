"""Build an ExperimentContext from an arbitrary user-level CSV plus a column mapping."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import ExperimentConfig, ExperimentContext, MetricDef


@dataclass
class ColumnMapping:
    variant: str
    control_label: str
    treatment_label: str
    primary_metric: str
    guardrails: list[str] = field(default_factory=list)
    lower_is_better: list[str] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)
    date: str | None = None


def infer_metric_type(series: pd.Series) -> str:
    """0/1 (or boolean) columns are proportions; anything else numeric is a mean."""
    values = set(pd.unique(series.dropna()))
    return "proportion" if values <= {0, 1} else "mean"


def context_from_upload(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    name: str = "Uploaded experiment",
    alpha: float = 0.05,
    mde_relative: float = 0.10,
    planned_duration_days: int = 14,
) -> ExperimentContext:
    """Normalise the uploaded table to the internal schema and describe it."""
    missing = [c for c in [mapping.variant, mapping.primary_metric, *mapping.guardrails,
                           *mapping.segments, *( [mapping.date] if mapping.date else [])]
               if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in upload: {missing}")
    data = df.copy()
    data["variant"] = data[mapping.variant].astype(str)
    labels = {mapping.control_label: "control", mapping.treatment_label: "treatment"}
    data = data[data["variant"].isin(labels)].copy()
    data["variant"] = data["variant"].map(labels)
    if data.empty:
        raise ValueError("No rows match the chosen control / treatment labels.")

    metrics: dict[str, MetricDef] = {}
    for col in dict.fromkeys([mapping.primary_metric, *mapping.guardrails]):
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
        metrics[col] = MetricDef(name=col, column=col, type=infer_metric_type(data[col]),
                                 higher_is_better=col not in mapping.lower_is_better)
    if mapping.date:
        data[mapping.date] = pd.to_datetime(data[mapping.date]).dt.strftime("%Y-%m-%d")

    config = ExperimentConfig(
        experiment_id="uploaded", name=name,
        hypothesis="User-uploaded experiment (no pre-registered hypothesis).",
        primary_metric=mapping.primary_metric,
        guardrail_metrics=[g for g in mapping.guardrails if g != mapping.primary_metric],
        metrics=metrics, segments=list(mapping.segments), mde_relative=mde_relative,
        alpha=alpha, planned_duration_days=planned_duration_days, date_column=mapping.date,
    )
    return ExperimentContext(config=config, df=data.reset_index(drop=True))
