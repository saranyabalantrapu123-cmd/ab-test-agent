"""Paths, experiment metadata loading and the in-memory experiment context."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("ABAGENT_DATA_DIR", ROOT / "data"))
REPORTS_DIR = Path(os.getenv("ABAGENT_REPORTS_DIR", ROOT / "reports"))
EXPERIMENTS_YAML = ROOT / "experiments.yaml"
DATA_FILE = "experiments.csv"
GROUND_TRUTH_FILE = "ground_truth.yaml"

DECISIONS = ("Ship", "Don't ship", "Investigate", "Inconclusive-extend")


@dataclass(frozen=True)
class MetricDef:
    """How a metric is computed from user-level rows."""

    name: str
    column: str
    type: str  # "proportion" | "mean"
    higher_is_better: bool = True
    cuped_covariate: str | None = None
    description: str = ""


@dataclass(frozen=True)
class ExperimentConfig:
    """Pre-registered metadata for one experiment."""

    experiment_id: str
    name: str
    hypothesis: str
    primary_metric: str
    guardrail_metrics: list[str]
    metrics: dict[str, MetricDef]
    segments: list[str]
    mde_relative: float = 0.10
    alpha: float = 0.05
    power: float = 0.80
    expected_split: tuple[float, float] = (0.5, 0.5)
    start_date: str | None = None
    planned_duration_days: int = 14
    brand_scope: str = ""
    control_label: str = "control"
    treatment_label: str = "treatment"
    date_column: str | None = "assignment_date"

    def to_public_dict(self) -> dict[str, Any]:
        """Metadata the agent is allowed to see (no ground truth)."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "hypothesis": self.hypothesis,
            "primary_metric": self.primary_metric,
            "guardrail_metrics": list(self.guardrail_metrics),
            "metric_types": {
                m: self.metrics[m].type for m in [self.primary_metric, *self.guardrail_metrics]
            },
            "segments": list(self.segments),
            "mde_relative": self.mde_relative,
            "alpha": self.alpha,
            "power": self.power,
            "expected_split": list(self.expected_split),
            "start_date": self.start_date,
            "planned_duration_days": self.planned_duration_days,
            "brand_scope": self.brand_scope,
        }


@dataclass
class ExperimentContext:
    """Everything the tools need: the config plus that experiment's user-level rows."""

    config: ExperimentConfig
    df: pd.DataFrame = field(repr=False)

    def arm(self, variant: str) -> pd.DataFrame:
        return self.df[self.df["variant"] == variant]

    @property
    def control(self) -> pd.DataFrame:
        return self.arm(self.config.control_label)

    @property
    def treatment(self) -> pd.DataFrame:
        return self.arm(self.config.treatment_label)


def _metric_defs(raw: dict[str, Any]) -> dict[str, MetricDef]:
    return {name: MetricDef(name=name, **spec) for name, spec in raw.items()}


def load_registry(path: Path = EXPERIMENTS_YAML) -> dict[str, ExperimentConfig]:
    """Load every experiment in experiments.yaml as an ExperimentConfig."""
    raw = yaml.safe_load(path.read_text())
    metrics = _metric_defs(raw["metrics"])
    segments = list(raw.get("segments", []))
    registry: dict[str, ExperimentConfig] = {}
    for exp_id, spec in raw["experiments"].items():
        spec = dict(spec)
        spec["expected_split"] = tuple(spec.get("expected_split", (0.5, 0.5)))
        registry[exp_id] = ExperimentConfig(
            experiment_id=exp_id, metrics=metrics, segments=segments, **spec
        )
    return registry


def load_data(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the generated user-level experiment table."""
    path = Path(data_dir or DATA_DIR) / DATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `make data` first.")
    return pd.read_csv(path)


def load_ground_truth(data_dir: Path | None = None) -> dict[str, Any]:
    path = Path(data_dir or DATA_DIR) / GROUND_TRUTH_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `make data` first.")
    return yaml.safe_load(path.read_text())


def build_context(
    experiment_id: str, df: pd.DataFrame | None = None, data_dir: Path | None = None
) -> ExperimentContext:
    """Context for a registered experiment, slicing its rows from the shared table."""
    registry = load_registry()
    if experiment_id not in registry:
        raise KeyError(f"Unknown experiment '{experiment_id}'. Known: {sorted(registry)}")
    data = df if df is not None else load_data(data_dir)
    rows = data[data["experiment_id"] == experiment_id].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"No rows for experiment '{experiment_id}' in the data file.")
    return ExperimentContext(config=registry[experiment_id], df=rows)
