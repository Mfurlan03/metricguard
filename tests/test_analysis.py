import numpy as np
import pandas as pd
import pytest

from metricguard.analysis import (
    _excess_slope,
    _find_most_suspicious_window,
    _noise_scale,
    _risk_bucket,
    analyze_kpi_integrity,
)
from metricguard.synthetic import SCENARIOS, gaming_window


def _run(scenario: str, mode: str, seed: int = 0, **kw):
    cfg = SCENARIOS[scenario]
    df = cfg["generate"](seed=seed, mode=mode, **kw)
    return analyze_kpi_integrity(
        df,
        time_col=cfg["time_col"],
        kpi_col=cfg["kpi_col"],
        outcome_col=cfg["outcome_col"],
        extra_cols=cfg["extra_cols"],
        kpi_higher_better=cfg["kpi_higher_better"],
        outcome_higher_better=cfg["outcome_higher_better"],
    )


@pytest.mark.parametrize(
    "score,level",
    [(0.0, "LOW"), (0.249, "LOW"), (0.25, "MODERATE"), (0.5, "HIGH"), (0.75, "CRITICAL"), (1.0, "CRITICAL")],
)
def test_risk_bucket_boundaries(score, level):
    assert _risk_bucket(score) == level


def test_noise_scale_is_unit_invariant():
    rng = np.random.default_rng(0)
    x = pd.Series(np.cumsum(rng.normal(size=200)))
    assert _noise_scale(x * 100) == pytest.approx(100 * _noise_scale(x))


def test_noise_scale_handles_constant_series():
    assert _noise_scale(pd.Series([3.0] * 10)) == 1.0


def test_excess_slope_removes_linear_trend():
    x = np.arange(20, dtype=float)
    assert _excess_slope(x, 3, 11) == pytest.approx(0.0)


def test_suspicious_window_finds_injected_divergence():
    n = 40
    kpi = np.zeros(n)
    out = np.zeros(n)
    kpi[20:28] = np.linspace(0, 5, 8)
    kpi[28:] = 5
    out[20:28] = np.linspace(0, -5, 8)
    out[28:] = -5
    win, score = _find_most_suspicious_window(pd.Series(kpi), pd.Series(out), window=8)
    assert win == (20, 28)
    assert score > 0


def test_suspicious_window_none_for_short_series():
    win, score = _find_most_suspicious_window(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), window=8)
    assert win is None and score == 0.0


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_gamed_scores_higher_than_clean(scenario):
    gamed = np.mean([_run(scenario, "gamed", seed=s).risk_score for s in range(10)])
    clean = np.mean([_run(scenario, "clean", seed=s).risk_score for s in range(10)])
    assert gamed > clean + 0.2


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_clean_data_is_not_flagged_high(scenario):
    # Regression test: v0.2 flagged every clean productivity series as HIGH risk.
    levels = [_run(scenario, "clean", seed=s).risk_level for s in range(10)]
    assert all(lvl in {"LOW", "MODERATE"} for lvl in levels)


def test_scale_invariance_of_risk_score():
    cfg = SCENARIOS["productivity"]
    df = cfg["generate"](seed=4, mode="gamed")
    scaled = df.copy()
    scaled[cfg["kpi_col"]] *= 1000
    kw = dict(time_col="week", kpi_col=cfg["kpi_col"], outcome_col=cfg["outcome_col"], extra_cols=cfg["extra_cols"])
    assert analyze_kpi_integrity(df, **kw).risk_score == pytest.approx(
        analyze_kpi_integrity(scaled, **kw).risk_score, abs=0.02
    )


def test_flagged_window_overlaps_ground_truth():
    res = _run("safety", "gamed", seed=0)
    assert res.suspicious_window is not None
    gs, ge = gaming_window("safety", 52)
    i, j = res.suspicious_window
    assert i < ge and j > gs


def test_unsorted_input_is_handled():
    cfg = SCENARIOS["safety"]
    df = cfg["generate"](seed=1, mode="gamed")
    shuffled = df.sample(frac=1.0, random_state=0)
    kw = dict(time_col="week", kpi_col="safety_score", outcome_col="incidents", extra_cols=cfg["extra_cols"])
    assert analyze_kpi_integrity(df, **kw).risk_score == pytest.approx(analyze_kpi_integrity(shuffled, **kw).risk_score)
