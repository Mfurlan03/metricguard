from __future__ import annotations

import numpy as np
import pandas as pd


def generate_safety_dataset(periods: int = 52, seed: int = 7, mode: str = "gamed") -> pd.DataFrame:
    """
    Safety scenario:
    KPI: safety_score (higher better)
    Outcome: incidents (lower better)
    Extras: near_misses (higher worse), overtime_hours (higher worse)

    mode="clean": KPI improves and incidents fall.
    mode="gamed": KPI inflates while incidents plateau/worsen; contradictions rise.
    """
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
        gs = int(periods * 0.45)
        ge = int(periods * 0.70)

        inflation = np.zeros_like(t, dtype=float)
        inflation[gs:ge] = np.linspace(0.00, 0.25, ge - gs)
        inflation[ge:] = inflation[ge - 1]
        safety_score = np.clip(safety_score + inflation, 0, 1)

        # outcome decouples and worsens during inflation (strong signal)
        plateau = float(np.mean(incidents[gs - 6:gs]))
        incidents[gs:ge] = (
            plateau
            + np.linspace(0.0, 2.0, ge - gs)
            + rng.normal(0, 0.45, size=(ge - gs))
        )

        # contradictions rise
        near_misses[gs:ge] += np.linspace(0.0, 2.5, ge - gs)
        overtime[gs:ge] += np.linspace(0.0, 10.0, ge - gs)

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


def generate_productivity_dataset(periods: int = 52, seed: int = 11, mode: str = "gamed") -> pd.DataFrame:
    """
    Productivity scenario:
    KPI: tickets_closed (higher better)
    Outcome: defects (lower better)
    Extras: reopens (higher worse), overtime_hours (higher worse)

    mode="clean": tickets_closed rises while defects fall.
    mode="gamed": tickets_closed rises but defects/reopens rise (rush work).
    """
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
        gs = int(periods * 0.45)
        ge = int(periods * 0.75)

        # KPI spikes (teams optimize throughput)
        tickets_closed[gs:ge] += np.linspace(0.0, 35.0, ge - gs)

        # quality degrades
        defects[gs:ge] = defects[gs:ge] + np.linspace(0.0, 10.0, ge - gs) + rng.normal(0, 1.0, ge - gs)
        reopens[gs:ge] = reopens[gs:ge] + np.linspace(0.0, 8.0, ge - gs) + rng.normal(0, 0.8, ge - gs)
        overtime[gs:ge] += np.linspace(0.0, 12.0, ge - gs)

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
