from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class MetricGuardResult:
    risk_level: str
    risk_score: float
    key_findings: List[str]
    signals: Dict[str, float]
    corr_series: pd.Series
    suspicious_window: Tuple[int, int] | None  # row indices (start, end)


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


def _find_most_suspicious_window(
    kpi_good: pd.Series,
    out_good: pd.Series,
    window: int,
) -> Tuple[int, int] | None:
    """
    Find window where KPI (good-direction) increases but outcome (good-direction) decreases/stalls.
    Returns (start_idx, end_idx) row indices, end exclusive.
    """
    if len(kpi_good) < window + 2:
        return None

    # Window score: KPI slope positive AND outcome slope negative => suspicious
    best = None
    best_score = 0.0

    kg = kpi_good.to_numpy(dtype=float)
    og = out_good.to_numpy(dtype=float)

    for i in range(0, len(kg) - window + 1):
        a = kg[i:i + window]
        b = og[i:i + window]
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            continue

        k_slope = float(a[-1] - a[0])
        o_slope = float(b[-1] - b[0])

        # suspicious when KPI improves (positive) but outcome worsens (negative)
        score = max(0.0, k_slope) * max(0.0, -o_slope)
        if score > best_score:
            best_score = score
            best = (i, i + window)

    return best


def analyze_kpi_integrity(
    df: pd.DataFrame,
    time_col: str,
    kpi_col: str,
    outcome_col: str,
    extra_cols: Optional[List[str]] = None,
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

    # --- Signal 3: contradictions with extras ---
    # Convert extras to "good direction" assumption: most extras are “higher is worse”
    # so we flip them (good direction = -extra)
    contradiction_hits = 0.0
    contradiction_notes: List[str] = []
    k_trend = float(np.nanmean(np.diff(kpi_good.astype(float).values)))

    for col in extra_cols:
        if col not in d.columns:
            continue
        ex_good = _direction_adjust(d[col], higher_better=False)  # treat higher as worse by default
        e_trend = float(np.nanmean(np.diff(ex_good.astype(float).values)))

        # If KPI trend is improving (k_trend positive), but extra “good direction” trend is negative, that's bad:
        # (because extra is worsening)
        if k_trend > 0 and e_trend < 0:
            contradiction_hits += 1.0
            contradiction_notes.append(f"- KPI improves while `{col}` worsens (contradiction).")

    contradictions_norm = contradiction_hits / max(1.0, len([c for c in extra_cols if c in d.columns])) if extra_cols else 0.0

    # --- Signal 4: suspicious window existence ---
    suspicious = _find_most_suspicious_window(kpi_good, out_good, window=window)
    suspicious_strength = 0.0
    if suspicious is not None:
        i, j = suspicious
        k_slope = float(kpi_good.iloc[j - 1] - kpi_good.iloc[i])
        o_slope = float(out_good.iloc[j - 1] - out_good.iloc[i])
        suspicious_strength = min(1.0, max(0.0, k_slope) * max(0.0, -o_slope) / 0.5)

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

    findings: List[str] = []
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
        findings.append(f"Most suspicious window: rows {i}–{j-1} (KPI improves while outcome worsens).")

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
