"""MetricGuard: detect KPI gaming, metric drift, and KPI-outcome decoupling."""

from .analysis import MetricGuardResult, analyze_kpi_integrity
from .benchmark import run_benchmark

__all__ = ["MetricGuardResult", "analyze_kpi_integrity", "run_benchmark"]
__version__ = "0.3.0"
