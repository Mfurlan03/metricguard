from __future__ import annotations

import numpy as np
import pandas as pd

# Fraction of the series where gaming starts / stops, per scenario.
# Exposed so evaluation code can score window localization against ground truth.
GAMING_WINDOWS = {
    "safety": (0.45, 0.70),
    "productivity": (0.45, 0.75),
}


def gaming_window(scenario: str, periods: int) -> tuple[int, int]:
    """Ground-truth gaming window (start, end-exclusive) row indices for a scenario."""
    start_frac, end_frac = GAMING_WINDOWS[scenario]
    return int(periods * start_frac), int(periods * end_frac)


def generate_safety_dataset(
    periods: int = 52, seed: int = 7, mode: str = "gamed", intensity: float = 1.0
) -> pd.DataFrame:
    """
    Safety scenario:
    KPI: safety_score (higher better)
    Outcome: incidents (lower better)
    Extras: near_misses (higher worse), overtime_hours (higher worse)

    mode="clean": KPI improves and incidents fall.
    mode="gamed": KPI inflates while incidents plateau/worsen; contradictions rise.
    intensity: scales the size of the gaming effect (0 = no gaming, 1 = default).
    """
    _check_mode(mode)
    rng = np.random.default_rng(seed)
    t = np.arange(1, periods + 1)

    # underlying true safety improves gradually
    true_safety = np.clip(0.35 + 0.004 * t, 0, 0.70)

    # KPI tracks true safety, plus noise
    safety_score = np.clip(true_safety + rng.normal(0, 0.03, size=periods), 0, 1)

    # incidents decrease with true safety
    incidents = 7.5 - 6.5 * true_safety + rng.normal(0, 0.5, size=periods)

    near_misses = 4.0 + 3.0 * (1 - true_safety) + rng.normal(0, 0.5, size=periods)
    overtime = 10 + 12 * (t / periods) + rng.normal(0, 2.0, size=periods)

    if mode == "gamed":
        # KPI inflation window
        gs, ge = gaming_window("safety", periods)

        inflation = np.zeros_like(t, dtype=float)
        inflation[gs:ge] = np.linspace(0.00, 0.25 * intensity, ge - gs)
        inflation[ge:] = inflation[ge - 1]
        safety_score = np.clip(safety_score + inflation, 0, 1)

        # outcome decouples and worsens during inflation (strong signal)
        plateau = float(np.mean(incidents[gs - 6 : gs]))
        incidents[gs:ge] = plateau + np.linspace(0.0, 2.0 * intensity, ge - gs) + rng.normal(0, 0.45, size=(ge - gs))

        # contradictions rise
        near_misses[gs:ge] += np.linspace(0.0, 2.5 * intensity, ge - gs)
        overtime[gs:ge] += np.linspace(0.0, 10.0 * intensity, ge - gs)

    df = pd.DataFrame(
        {
            "week": t,
            "safety_score": np.round(safety_score, 4),
            "incidents": np.round(incidents, 3),
            "near_misses": np.round(near_misses, 3),
            "overtime_hours": np.round(overtime, 2),
        }
    )
    return df


def generate_productivity_dataset(
    periods: int = 52, seed: int = 11, mode: str = "gamed", intensity: float = 1.0
) -> pd.DataFrame:
    """
    Productivity scenario:
    KPI: tickets_closed (higher better)
    Outcome: defects (lower better)
    Extras: reopens (higher worse), overtime_hours (higher worse)

    mode="clean": tickets_closed rises while defects fall.
    mode="gamed": tickets_closed rises but defects/reopens rise (rush work).
    intensity: scales the size of the gaming effect (0 = no gaming, 1 = default).
    """
    _check_mode(mode)
    rng = np.random.default_rng(seed)
    t = np.arange(1, periods + 1)

    # baseline output increases slightly over time
    base_output = 40 + 0.6 * t + rng.normal(0, 2.0, size=periods)
    tickets_closed = np.clip(base_output, 0, None)

    # quality improves slowly in clean mode
    defects = 22 - 0.15 * t + rng.normal(0, 1.2, size=periods)
    reopens = 14 - 0.10 * t + rng.normal(0, 1.0, size=periods)

    overtime = 6 + 0.2 * t + rng.normal(0, 1.5, size=periods)

    if mode == "gamed":
        gs, ge = gaming_window("productivity", periods)

        # KPI spikes (teams optimize throughput)
        tickets_closed[gs:ge] += np.linspace(0.0, 35.0 * intensity, ge - gs)

        # quality degrades
        defects[gs:ge] = defects[gs:ge] + np.linspace(0.0, 10.0 * intensity, ge - gs) + rng.normal(0, 1.0, ge - gs)
        reopens[gs:ge] = reopens[gs:ge] + np.linspace(0.0, 8.0 * intensity, ge - gs) + rng.normal(0, 0.8, ge - gs)
        overtime[gs:ge] += np.linspace(0.0, 12.0 * intensity, ge - gs)

    defects = np.clip(defects, 0, None)
    reopens = np.clip(reopens, 0, None)
    overtime = np.clip(overtime, 0, None)

    df = pd.DataFrame(
        {
            "week": t,
            "tickets_closed": np.round(tickets_closed, 2),
            "defects": np.round(defects, 2),
            "reopens": np.round(reopens, 2),
            "overtime_hours": np.round(overtime, 2),
        }
    )
    return df


def _check_mode(mode: str) -> None:
    if mode not in {"clean", "gamed"}:
        raise ValueError(f"mode must be 'clean' or 'gamed', got {mode!r}")


SCENARIOS = {
    "safety": {
        "generate": generate_safety_dataset,
        "time_col": "week",
        "kpi_col": "safety_score",
        "outcome_col": "incidents",
        "extra_cols": ["near_misses", "overtime_hours"],
        "kpi_higher_better": True,
        "outcome_higher_better": False,
    },
    "productivity": {
        "generate": generate_productivity_dataset,
        "time_col": "week",
        "kpi_col": "tickets_closed",
        "outcome_col": "defects",
        "extra_cols": ["reopens", "overtime_hours"],
        "kpi_higher_better": True,
        "outcome_higher_better": False,
    },
}
