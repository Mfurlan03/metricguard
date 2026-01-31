from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def save_basic_plot(
    df: pd.DataFrame,
    time_col: str,
    kpi_col: str,
    outcome_col: str,
    corr_series: pd.Series,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = df[time_col]

    fig = plt.figure()
    plt.plot(t, df[kpi_col], label=kpi_col)
    plt.plot(t, df[outcome_col], label=outcome_col)
    plt.legend()
    plt.title("KPI vs Outcome over Time")
    plt.xlabel(time_col)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    fig2 = plt.figure()
    plt.plot(t, corr_series, label=f"rolling corr({kpi_col}, {outcome_col})")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.legend()
    plt.title("Rolling KPI–Outcome Correlation")
    plt.xlabel(time_col)
    plt.tight_layout()
    fig2.savefig(out_path.with_name(out_path.stem + "_corr.png"), dpi=160)
    plt.close(fig2)
