"""Detection benchmark for MetricGuard.

Runs the detector over many seeded synthetic series where the ground truth is
known (clean vs. gamed, and the exact gaming window), then reports:

* ROC-AUC of the risk score for separating gamed from clean series
* detection rate / false-positive rate at the HIGH-risk threshold
* window localization quality (IoU between flagged and true gaming window)

Everything is computed from scratch (no sklearn dependency) so the numbers in
the README can be reproduced with ``python -m metricguard benchmark``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .analysis import analyze_kpi_integrity
from .synthetic import SCENARIOS, gaming_window

HIGH_THRESHOLD = 0.50  # risk_score >= 0.50 -> HIGH or CRITICAL


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U), with ties counted as 0.5."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("roc_auc needs at least one positive and one negative example")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def window_iou(pred: tuple[int, int] | None, truth: tuple[int, int]) -> float:
    """Intersection-over-union of two half-open row ranges [start, end)."""
    if pred is None:
        return 0.0
    inter = max(0, min(pred[1], truth[1]) - max(pred[0], truth[0]))
    union = max(pred[1], truth[1]) - min(pred[0], truth[0])
    return inter / union if union > 0 else 0.0


@dataclass
class BenchmarkResult:
    rows: pd.DataFrame
    summary: pd.DataFrame
    intensity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_markdown(self) -> str:
        parts = ["## Detection summary", "", self.summary.to_markdown(index=False, floatfmt=".3f")]
        if not self.intensity_curve.empty:
            parts += [
                "",
                "## Sensitivity to gaming intensity",
                "",
                self.intensity_curve.to_markdown(index=False, floatfmt=".3f"),
            ]
        return "\n".join(parts) + "\n"


def _score_one(scenario: str, mode: str, seed: int, periods: int, intensity: float, window: int) -> dict:
    cfg = SCENARIOS[scenario]
    df = cfg["generate"](periods=periods, seed=seed, mode=mode, intensity=intensity)
    res = analyze_kpi_integrity(
        df,
        time_col=cfg["time_col"],
        kpi_col=cfg["kpi_col"],
        outcome_col=cfg["outcome_col"],
        extra_cols=cfg["extra_cols"],
        window=window,
        kpi_higher_better=cfg["kpi_higher_better"],
        outcome_higher_better=cfg["outcome_higher_better"],
    )
    row = {
        "scenario": scenario,
        "mode": mode,
        "seed": seed,
        "intensity": intensity,
        "label": int(mode == "gamed"),
        "risk_score": res.risk_score,
        "risk_level": res.risk_level,
        "window_iou": np.nan,
    }
    if mode == "gamed":
        row["window_iou"] = window_iou(res.suspicious_window, gaming_window(scenario, periods))
    return row


def _summarize(rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = []
    for key, g in rows.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        gamed, clean = g[g.label == 1], g[g.label == 0]
        rec = dict(zip(group_cols, key, strict=True))
        rec.update(
            {
                "n": len(g),
                "roc_auc": roc_auc(g.label, g.risk_score),
                "detection_rate": float((gamed.risk_score >= HIGH_THRESHOLD).mean()),
                "false_positive_rate": float((clean.risk_score >= HIGH_THRESHOLD).mean()),
                "mean_window_iou": float(gamed.window_iou.mean()),
            }
        )
        out.append(rec)
    return pd.DataFrame(out)


def run_benchmark(
    seeds: Iterable[int] = range(100),
    scenarios: Iterable[str] = ("safety", "productivity"),
    periods: int = 52,
    window: int = 8,
    intensities: Iterable[float] = (0.25, 0.5, 0.75, 1.0),
) -> BenchmarkResult:
    seeds = list(seeds)
    scenarios = list(scenarios)
    records = []
    for scenario in scenarios:
        for seed in seeds:
            # Clean series are independent of intensity: score them once.
            records.append(_score_one(scenario, "clean", seed, periods, 1.0, window))
            for intensity in intensities:
                records.append(_score_one(scenario, "gamed", seed, periods, intensity, window))
    rows = pd.DataFrame(records)

    clean = rows[rows["mode"] == "clean"]
    # Headline numbers use the default gaming intensity (1.0).
    default = pd.concat([clean, rows[(rows["mode"] == "gamed") & (rows.intensity == 1.0)]])
    summary = _summarize(default, ["scenario"])

    curve_parts = []
    for intensity in sorted(set(rows.intensity[rows["mode"] == "gamed"])):
        subset = pd.concat([clean, rows[(rows["mode"] == "gamed") & (rows.intensity == intensity)]])
        part = _summarize(subset, ["scenario"])
        part.insert(1, "intensity", intensity)
        curve_parts.append(part)
    curve = pd.concat(curve_parts, ignore_index=True).sort_values(["scenario", "intensity"])
    return BenchmarkResult(rows=rows, summary=summary, intensity_curve=curve.reset_index(drop=True))
