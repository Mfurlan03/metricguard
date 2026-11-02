from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analysis import analyze_kpi_integrity
from .plotting import save_basic_plot
from .report import build_markdown_report, write_report


def run_metricguard(
    csv_path: Path,
    time_col: str,
    kpi_col: str,
    outcome_col: str,
    extra_cols: list[str],
    report_path: Path,
    window: int = 8,
    save_plot: bool = False,
    kpi_higher_better: bool = True,
    outcome_higher_better: bool = False,
) -> None:
    df = pd.read_csv(csv_path)

    missing = [c for c in [time_col, kpi_col, outcome_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    cols = [time_col, kpi_col, outcome_col] + [c for c in extra_cols if c in df.columns]
    d = df[cols].copy().sort_values(time_col).reset_index(drop=True)

    result = analyze_kpi_integrity(
        d,
        time_col=time_col,
        kpi_col=kpi_col,
        outcome_col=outcome_col,
        extra_cols=extra_cols,
        window=window,
        kpi_higher_better=kpi_higher_better,
        outcome_higher_better=outcome_higher_better,
    )

    report = build_markdown_report(
        d,
        time_col=time_col,
        kpi_col=kpi_col,
        outcome_col=outcome_col,
        extra_cols=extra_cols,
        result=result,
        kpi_higher_better=kpi_higher_better,
        outcome_higher_better=outcome_higher_better,
    )
    write_report(report_path, report)

    if save_plot:
        plot_path = report_path.with_suffix(".png")
        save_basic_plot(
            d,
            time_col=time_col,
            kpi_col=kpi_col,
            outcome_col=outcome_col,
            corr_series=result.corr_series,
            out_path=plot_path,
            suspicious_window=result.suspicious_window,
        )
