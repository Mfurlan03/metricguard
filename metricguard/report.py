from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from .analysis import MetricGuardResult


def build_markdown_report(
    df: pd.DataFrame,
    time_col: str,
    kpi_col: str,
    outcome_col: str,
    extra_cols: list[str],
    result: MetricGuardResult,
    kpi_higher_better: bool,
    outcome_higher_better: bool,
) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    def dir_str(hb: bool) -> str:
        return "higher is better" if hb else "lower is better"

    lines: list[str] = []
    lines.append("# MetricGuard Report\n")
    lines.append(f"- Generated: **{now}**")
    lines.append(f"- Time: `{time_col}`")
    lines.append(f"- KPI: `{kpi_col}` ({dir_str(kpi_higher_better)})")
    lines.append(f"- Outcome: `{outcome_col}` ({dir_str(outcome_higher_better)})")
    if extra_cols:
        lines.append(f"- Extras: {', '.join(f'`{c}`' for c in extra_cols)}")
    lines.append("")

    lines.append("## KPI Integrity Verdict")
    lines.append(f"- **Risk Level:** `{result.risk_level}`")
    lines.append(f"- **Risk Score:** `{result.risk_score:.2f}` (0 = low concern, 1 = high concern)")
    lines.append("")

    lines.append("## Key Findings")
    for f in result.key_findings:
        lines.append(f"- {f}")
    lines.append("")

    if result.suspicious_window is not None:
        i, j = result.suspicious_window
        t0 = df.iloc[i][time_col]
        t1 = df.iloc[j - 1][time_col]
        lines.append("## Most Suspicious Window")
        lines.append(f"- Window: `{t0}` → `{t1}`")
        lines.append("- Preview:")
        lines.append("")
        preview_cols = [time_col, kpi_col, outcome_col] + [c for c in extra_cols if c in df.columns]
        lines.append(df.iloc[i:j][preview_cols].to_markdown(index=False))
        lines.append("")

    lines.append("## Diagnostic Signals")
    for k, v in result.signals.items():
        lines.append(f"- `{k}`: {v:.3f}")
    lines.append("")

    lines.append("## Recommendations")
    if result.risk_level in {"HIGH", "CRITICAL"}:
        lines.append("- Review incentives: what behavior improves KPI without improving reality?")
        lines.append("- Validate data definitions: did reporting/reclassification change?")
        if extra_cols:
            lines.append("- Investigate contradictions: why are outcome-adjacent signals worsening?")
        lines.append("- Consider replacing KPI with a composite that is harder to game.")
    else:
        lines.append("- Continue monitoring; add more outcome-adjacent signals to strengthen detection.")
    lines.append("")

    lines.append("## Appendix: First 10 rows")
    lines.append("")
    lines.append(df.head(10).to_markdown(index=False))

    return "\n".join(lines)


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
