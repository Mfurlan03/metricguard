from pathlib import Path
import pandas as pd

from metricguard.synthetic import generate_safety_dataset
from metricguard.analysis import analyze_kpi_integrity


def test_analyze_runs():
    df = generate_safety_dataset(periods=40, seed=1, mode="gamed")
    res = analyze_kpi_integrity(
        df,
        time_col="week",
        kpi_col="safety_score",
        outcome_col="incidents",
        extra_cols=["near_misses", "overtime_hours"],
        window=8,
        kpi_higher_better=True,
        outcome_higher_better=False,
    )
    assert 0.0 <= res.risk_score <= 1.0
    assert res.risk_level in {"LOW", "MODERATE", "HIGH", "CRITICAL"}
