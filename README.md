# MetricGuard

**MetricGuard** detects when KPIs stop representing reality.

It identifies **KPI gaming, metric drift, and KPI–outcome decoupling** using
interpretable, behavior-aware analytics.

> When a measure becomes a target, it ceases to be a good measure.  
> — *Goodhart’s Law*

---

## Why MetricGuard Exists

Organizations rely on KPIs to manage performance, safety, and quality.

Over time, these metrics often fail silently:
- teams optimize the metric, not the outcome
- reporting practices change
- incentives create shortcuts
- dashboards look “green” while reality worsens

**MetricGuard detects these failures early.**

---

## Quickstart

```bash
pip install -e .

python -m metricguard generate --scenario safety --mode gamed --out data/safety_gamed.csv

python -m metricguard run   --csv data/safety_gamed.csv   --time week   --kpi safety_score   --outcome incidents   --extras near_misses overtime_hours   --out reports/safety_gamed.md   --plot   --kpi-higher-better true   --outcome-higher-better false
```

---
## Example

## Example: KPI Gaming in Safety Metrics

In this synthetic safety scenario:

- The safety KPI improves steadily
- Incident counts worsen during a localized period
- Overtime increases as teams optimize the metric

MetricGuard flags this as **HIGH KPI integrity risk** and identifies the exact failure window.

### Rolling KPI–Outcome Relationship
This plot shows instability and sign flips in the KPI–outcome correlation,
indicating a breakdown in metric validity.

<p align="center">
  <img src="docs/safety_corr.png" width="700">
</p>

### Identified KPI Gaming Window
MetricGuard pinpoints the exact period where the KPI improves while outcomes worsen.

<p align="center">
  <img src="docs/safety_gamed.png" width="700">
</p>

