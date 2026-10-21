"""Static matplotlib charts: lift with CI (primary + guardrails) and segment forest plot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .decision import Findings  # noqa: E402
from .memo import CORRECTION_NAMES  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
NEUTRAL = "#898781"


def _status(significant: bool, beneficial: bool) -> tuple[str, str]:
    if not significant:
        return NEUTRAL, "not significant"
    return (GOOD, "significant gain") if beneficial else (CRITICAL, "significant harm")


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle="--")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))


def _forest(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    """rows: label, lift, lo, hi (fractions), significant, beneficial, note."""
    y = list(range(len(rows)))[::-1]
    seen: set[str] = set()
    for yi, r in zip(y, rows, strict=True):
        color, status = _status(r["significant"], r["beneficial"])
        ax.plot([r["lo"] * 100, r["hi"] * 100], [yi, yi], color=color, linewidth=2.2,
                solid_capstyle="round")
        ax.scatter([r["lift"] * 100], [yi], s=60, color=color, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5,
                   label=status if status not in seen else None)
        seen.add(status)
        ax.annotate(r["note"], xy=(1.0, yi), xycoords=("axes fraction", "data"),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK_2)
    ax.set_yticks(y, [r["label"] for r in rows], fontsize=9.5, color=INK)
    ax.set_ylim(-0.7, len(rows) - 0.3)


def _finish(fig: Figure, ax: plt.Axes, title: str, subtitle: str, path: Path) -> Path:
    ax.set_xlabel("Relative lift vs control (95% CI)", color=INK_2, fontsize=9)
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 30),
                textcoords="offset points", fontsize=12.5, color=INK, fontweight="bold")
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 13),
                textcoords="offset points", fontsize=9.5, color=INK_2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16 if len(ax.get_yticks()) > 4 else -0.24),
              ncol=3, frameon=False, fontsize=8.5, labelcolor=INK_2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def _note(r: dict[str, Any], p_key: str = "p_value") -> str:
    p = r[p_key]
    ptxt = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    return f"{r['relative_lift'] * 100:+.1f}%  {ptxt}"


def lift_chart(f: Findings, decision_label: str, path: Path) -> Path | None:
    """Primary and guardrail lifts with CIs, coloured by significance and direction."""
    results = ([f.primary] if f.primary else []) + list(f.guardrails.values())
    if not results:
        return None
    rows = []
    for i, r in enumerate(results):
        beneficial = (r["abs_diff"] > 0) == bool(r["higher_is_better"])
        role = "primary" if i == 0 else "guardrail"
        rows.append({"label": f"{r['metric']}\n({role})", "lift": r["relative_lift"],
                     "lo": r["relative_ci_low"], "hi": r["relative_ci_high"],
                     "significant": r["significant"], "beneficial": beneficial,
                     "note": _note(r)})
    fig, ax = plt.subplots(figsize=(8.2, 1.1 + 0.85 * len(rows)), facecolor=SURFACE)
    _style(ax)
    _forest(ax, rows)
    name = (f.metadata or {}).get("name", "Experiment")
    return _finish(fig, ax, name, f"Decision: {decision_label}", path)


def segment_chart(f: Findings, path: Path) -> Path | None:
    """Per-segment lift on the primary metric with corrected significance."""
    if not f.segments or not f.primary:
        return None
    rows = []
    for seg in f.segments:
        for lv in seg["levels"]:
            beneficial = (lv["abs_diff"] > 0) == bool(f.primary["higher_is_better"])
            rows.append({"label": f"{seg['segment']}: {lv['level']}",
                         "lift": lv["relative_lift"], "lo": lv["relative_ci_low"],
                         "hi": lv["relative_ci_high"], "significant": lv["significant_adjusted"],
                         "beneficial": beneficial, "note": _note(lv, "p_adjusted")})
    fig, ax = plt.subplots(figsize=(8.2, 1.2 + 0.5 * len(rows)), facecolor=SURFACE)
    _style(ax)
    _forest(ax, rows)
    overall = f.primary["relative_lift"] * 100
    ax.axvline(overall, color=INK_2, linewidth=1, alpha=0.6)
    ax.annotate(f"overall {overall:+.1f}%", xy=(overall, len(rows) - 0.55), fontsize=8,
                color=INK_2, ha="left", va="bottom", xytext=(4, 0), textcoords="offset points",
                bbox={"boxstyle": "square,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"})
    name = (f.metadata or {}).get("name", "Experiment")
    correction = CORRECTION_NAMES.get(f.segments[0]["correction"], f.segments[0]["correction"])
    return _finish(fig, ax, f"{name}: {f.primary['metric']} by segment",
                   f"{correction}-adjusted p-values; colour = significant after correction", path)
