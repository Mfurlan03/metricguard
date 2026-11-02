import subprocess
import sys

import pandas as pd
import pytest

from metricguard.benchmark import roc_auc, run_benchmark, window_iou
from metricguard.synthetic import gaming_window, generate_safety_dataset


def test_roc_auc_perfect_and_random():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5


def test_roc_auc_requires_both_classes():
    with pytest.raises(ValueError):
        roc_auc([1, 1], [0.2, 0.3])


@pytest.mark.parametrize(
    "pred,truth,expected",
    [((0, 10), (0, 10), 1.0), ((0, 5), (5, 10), 0.0), ((0, 10), (5, 15), 5 / 15), (None, (0, 5), 0.0)],
)
def test_window_iou(pred, truth, expected):
    assert window_iou(pred, truth) == pytest.approx(expected)


def test_intensity_zero_matches_clean():
    clean = generate_safety_dataset(seed=3, mode="clean")
    zero = generate_safety_dataset(seed=3, mode="gamed", intensity=0.0)
    gs, _ = gaming_window("safety", 52)
    # Outside the gaming window nothing changes; KPI and extras are untouched at intensity 0.
    pd.testing.assert_series_equal(clean["safety_score"], zero["safety_score"])
    pd.testing.assert_frame_equal(clean.iloc[:gs], zero.iloc[:gs])


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        generate_safety_dataset(mode="bogus")


def test_benchmark_small_run():
    res = run_benchmark(seeds=range(5), intensities=(1.0,))
    assert set(res.summary.scenario) == {"safety", "productivity"}
    assert (res.summary.roc_auc >= 0.8).all()
    assert "Detection summary" in res.to_markdown()


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "metricguard", *args], capture_output=True, text=True, check=True)


def test_cli_generate_and_run(tmp_path):
    csv = tmp_path / "d.csv"
    report = tmp_path / "r.md"
    _cli("generate", "--scenario", "safety", "--mode", "gamed", "--out", str(csv))
    _cli(
        "run",
        "--csv",
        str(csv),
        "--time",
        "week",
        "--kpi",
        "safety_score",
        "--outcome",
        "incidents",
        "--extras",
        "near_misses",
        "overtime_hours",
        "--out",
        str(report),
        "--plot",
    )
    assert report.exists() and report.with_suffix(".png").exists()
    assert "Risk" in report.read_text(encoding="utf-8")


def test_cli_missing_column_errors(tmp_path):
    csv = tmp_path / "d.csv"
    _cli("generate", "--out", str(csv))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "metricguard",
            "run",
            "--csv",
            str(csv),
            "--time",
            "week",
            "--kpi",
            "nope",
            "--outcome",
            "incidents",
            "--out",
            str(tmp_path / "r.md"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0 and "Missing required columns" in proc.stderr
