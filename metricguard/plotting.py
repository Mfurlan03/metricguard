from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

KPI_COLOR = "#2a6fdb"
OUTCOME_COLOR = "#d1495b"
WINDOW_COLOR = "#f4a261"


def _shade_window(ax, t: pd.Series, window: tuple[int, int] | None) -> None:
    if window is None:
        return
    i, j = window
    ax.axvspan(t.iloc[i], t.iloc[j - 1], color=WINDOW_COLOR, alpha=0.25, label="flagged window")


def save_basic_plot(
    df: pd.DataFrame,
    time_col: str,
    kpi_col: str,
    outcome_col: str,
    corr_series: pd.Series,
    out_path: Path,
    suspicious_window: tuple[int, int] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = df[time_col]

    # KPI and outcome usually live on very different scales, so give each its own axis.
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    _shade_window(ax1, t, suspicious_window)
    ax1.plot(t, df[kpi_col], color=KPI_COLOR, linewidth=2, label=f"{kpi_col} (KPI)")
    ax1.set_xlabel(time_col)
    ax1.set_ylabel(kpi_col, color=KPI_COLOR)
    ax1.tick_params(axis="y", labelcolor=KPI_COLOR)

    ax2 = ax1.twinx()
    ax2.plot(t, df[outcome_col], color=OUTCOME_COLOR, linewidth=2, label=f"{outcome_col} (outcome)")
    ax2.set_ylabel(outcome_col, color=OUTCOME_COLOR)
    ax2.tick_params(axis="y", labelcolor=OUTCOME_COLOR)

    handles = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax1.legend(handles=handles, loc="upper left", frameon=False)
    ax1.set_title("KPI vs. outcome over time")
    for side in ("top",):
        ax1.spines[side].set_visible(False)
        ax2.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(9, 3.5))
    _shade_window(ax, t, suspicious_window)
    ax.plot(t, corr_series, color="#444444", linewidth=2, label=f"rolling corr({kpi_col}, {outcome_col})")
    ax.axhline(0, linestyle="--", linewidth=1, color="#888888")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(time_col)
    ax.set_ylabel("correlation (good-direction)")
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("Rolling KPI–outcome correlation")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(out_path.with_name(out_path.stem + "_corr.png"), dpi=160)
    plt.close(fig2)
