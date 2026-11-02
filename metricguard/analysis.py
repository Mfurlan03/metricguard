from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MetricGuardResult:
    risk_level: str
    risk_score: float
    key_findings: list[str]
    signals: dict[str, float]
    corr_series: pd.Series
    suspicious_window: tuple[int, int] | None  # row indices (start, end)


def _risk_bucket(score: float) -> str:
    if score < 0.25:
        return "LOW"
    if score < 0.50:
        return "MODERATE"
    if score < 0.75:
        return "HIGH"
    return "CRITICAL"


def _robust_zscore(a: pd.Series) -> pd.Series:
    x = pd.to_numeric(a, errors="coerce").astype(float)
    med = np.nanmedian(x.values)
    mad = np.nanmedian(np.abs(x.values - med))
    if mad == 0 or np.isnan(mad):
        return (x - med) * 0.0
    return 0.6745 * (x - med) / mad


def _pct_change_safe(a: pd.Series) -> pd.Series:
    x = pd.to_numeric(a, errors="coerce").astype(float)
    return x.pct_change().replace([np.inf, -np.inf], np.nan)


def _rolling_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window).corr(y)


def _direction_adjust(series: pd.Series, higher_better: bool) -> pd.Series:
    # convert “good direction” to higher=better always for scoring
    # if higher is NOT better, flip sign so “higher adjusted” means “better”
    x = pd.to_numeric(series, errors="coerce").astype(float)
    return x if higher_better else -x


def _noise_scale(x: pd.Series) -> float:
    """Robust estimate of step-to-step noise (MAD of first differences, in sigma units).

    Dividing a series by this makes window slopes comparable across KPIs measured in
    very different units (a 0-1 score vs. hundreds of tickets).
    """
    d = np.diff(pd.to_numeric(x, errors="coerce").to_numpy(dtype=float))
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return 1.0
    mad = 1.4826 * np.median(np.abs(d - np.median(d)))
    if not np.isfinite(mad) or mad == 0:
        sd = float(np.std(d))
        return sd if sd > 0 else 1.0
    return float(mad)


def _standardize(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    return (x - np.nanmedian(x.to_numpy())) / _noise_scale(x)


def _excess_slope(x: np.ndarray, i: int, j: int) -> float:
    """Change over rows [i, j) minus the change expected from the series' overall trend."""
    span = j - 1 - i
    global_step = (x[-1] - x[0]) / max(1, len(x) - 1)
    return float((x[j - 1] - x[i]) - global_step * span)


def _find_most_suspicious_window(
    kpi_good: pd.Series,
    out_good: pd.Series,
    window: int,
) -> tuple[tuple[int, int] | None, float]:
    """
    Find the window where the KPI (good-direction) improves faster than its usual trend
    while the outcome (good-direction) deteriorates relative to its usual trend.

    Both series are expected in noise-scaled units. Returns ((start, end_exclusive), score),
    or (None, 0.0) if no window shows that pattern.
    """
    if len(kpi_good) < window + 2:
        return None, 0.0

    kg = kpi_good.to_numpy(dtype=float)
    og = out_good.to_numpy(dtype=float)

    best: tuple[int, int] | None = None
    best_score = 0.0
    for i in range(0, len(kg) - window + 1):
        j = i + window
        if np.any(np.isnan(kg[i:j])) or np.any(np.isnan(og[i:j])):
            continue
        k_excess = _excess_slope(kg, i, j)
        o_excess = _excess_slope(og, i, j)
        # suspicious when KPI improves (positive) but outcome worsens (negative)
        score = max(0.0, k_excess) * max(0.0, -o_excess)
        if score > best_score:
            best_score = score
            best = (i, j)
    return best, best_score


# Calibration constants. Tuned on held-out calibration seeds (1000-1099) with
# `python -m metricguard benchmark --seeds 1000:1100`, never on the reported seeds.
WINDOW_SCORE_SCALE = 15.0  # window score (noise units^2) that maps to full strength
CONTRADICTION_Z = 1.0  # excess deterioration (noise units) needed to count as a contradiction


def analyze_kpi_integrity(
    df: pd.DataFrame,
    time_col: str,
    kpi_col: str,
    outcome_col: str,
    extra_cols: list[str] | None = None,
    window: int = 8,
    kpi_higher_better: bool = True,
    outcome_higher_better: bool = False,
) -> MetricGuardResult:
    extra_cols = extra_cols or []
    d = df.copy().sort_values(time_col).reset_index(drop=True)

    # Normalize “good direction”
    kpi_good = _direction_adjust(d[kpi_col], kpi_higher_better)
    out_good = _direction_adjust(d[outcome_col], outcome_higher_better)

    # Rolling correlation uses raw series (but direction affects interpretation)
    # Using good-direction for both makes correlation “positive is good alignment”
    corr = _rolling_corr(kpi_good, out_good, window=window)
    corr_valid = corr.dropna()

    # --- Signal 1: correlation breakdown ---
    corr_decay = 0.0
    corr_instability = 0.0
    if len(corr_valid) >= max(10, window + 2):
        third = len(corr_valid) // 3
        first = float(corr_valid.iloc[:third].mean())
        last = float(corr_valid.iloc[-third:].mean())
        corr_decay = max(0.0, abs(first) - abs(last))

        # sign flips / instability
        flips = int(((corr_valid.shift(1) * corr_valid) < 0).sum())
        corr_instability = min(1.0, flips / max(5, third))

    # --- Signal 2: “KPI moves, outcome doesn’t” via change volatility gap ---
    kpi_chg = _robust_zscore(_pct_change_safe(kpi_good))
    out_chg = _robust_zscore(_pct_change_safe(out_good))
    kpi_vol = float(np.nanmedian(np.abs(kpi_chg.values)))
    out_vol = float(np.nanmedian(np.abs(out_chg.values)))
    drift_gap = max(0.0, kpi_vol - out_vol)

    # --- Signal 3: suspicious window (KPI up, outcome down, beyond normal trend) ---
    kpi_z = _standardize(kpi_good)
    out_z = _standardize(out_good)
    suspicious, window_score = _find_most_suspicious_window(kpi_z, out_z, window=window)
    suspicious_strength = min(1.0, window_score / WINDOW_SCORE_SCALE)

    # --- Signal 4: contradictions with extras inside the suspicious window ---
    # Extras are assumed "higher is worse" (overtime, reopens, near misses), so their
    # good direction is the negated series. A contradiction is an extra that worsens
    # noticeably faster than its own long-run trend while the KPI is being inflated.
    contradiction_hits = 0.0
    contradiction_notes: list[str] = []
    present_extras = [c for c in extra_cols if c in d.columns]
    if suspicious is not None:
        i, j = suspicious
        for col in present_extras:
            ex_z = _standardize(_direction_adjust(d[col], higher_better=False)).to_numpy(dtype=float)
            if np.any(np.isnan(ex_z[i:j])):
                continue
            if _excess_slope(ex_z, i, j) < -CONTRADICTION_Z:
                contradiction_hits += 1.0
                contradiction_notes.append(f"KPI improves while `{col}` worsens faster than its normal trend.")

    contradictions_norm = contradiction_hits / len(present_extras) if present_extras else 0.0

    # --- Normalize components into 0..1 ---
    corr_component = min(1.0, corr_decay / 0.35)
    drift_component = min(1.0, drift_gap / 0.60)
    contradiction_component = min(1.0, contradictions_norm)
    window_component = min(1.0, suspicious_strength)
    instability_component = min(1.0, corr_instability)

    risk_score = (
        0.20 * corr_component
        + 0.25 * instability_component
        + 0.15 * drift_component
        + 0.25 * contradiction_component
        + 0.15 * window_component
    )
    # Escalation rule:
    # If we have BOTH a suspicious window AND contradictions,
    # we treat this as high concern even if global decay is weak.
    if contradiction_component > 0.4 and window_component > 0.4:
        risk_score = max(risk_score, 0.65)

    risk_score = float(np.clip(risk_score, 0.0, 1.0))
    risk_level = _risk_bucket(risk_score)

    findings: list[str] = []
    if corr_component > 0.35:
        findings.append(f"KPI–outcome alignment weakened (corr decay ~{corr_decay:.2f}).")
    if instability_component > 0.25:
        findings.append("KPI–outcome relationship is unstable (frequent correlation sign flips).")
    if drift_component > 0.35:
        findings.append("KPI changes are much more volatile than outcome changes (possible metric inflation).")
    if contradiction_component > 0.0:
        findings.append("Contradictions detected with related signals:")
        findings.extend(contradiction_notes)
    if window_component > 0.25 and suspicious is not None:
        i, j = suspicious
        findings.append(f"Most suspicious window: rows {i}–{j - 1} (KPI improves while outcome worsens).")

    if not findings:
        findings.append("No strong KPI integrity issues detected with current thresholds.")

    signals = {
        "corr_decay": float(corr_decay),
        "corr_instability": float(corr_instability),
        "drift_gap": float(drift_gap),
        "contradictions_norm": float(contradictions_norm),
        "suspicious_strength": float(suspicious_strength),
    }

    return MetricGuardResult(
        risk_level=risk_level,
        risk_score=risk_score,
        key_findings=findings,
        signals=signals,
        corr_series=corr,
        suspicious_window=suspicious,
    )
