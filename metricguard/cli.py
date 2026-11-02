from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_metricguard
from .synthetic import generate_productivity_dataset, generate_safety_dataset


def main() -> None:
    parser = argparse.ArgumentParser(prog="metricguard", description="MetricGuard CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- generate ---
    g = sub.add_parser("generate", help="Generate a synthetic dataset")
    g.add_argument("--scenario", choices=["safety", "productivity"], default="safety")
    g.add_argument("--out", type=str, required=True, help="Output CSV path")
    g.add_argument("--periods", type=int, default=52, help="Number of rows/time periods")
    g.add_argument("--seed", type=int, default=7, help="Random seed")
    g.add_argument(
        "--mode",
        choices=["clean", "gamed"],
        default="gamed",
        help="clean=healthy metric, gamed=decoupling & contradictions",
    )

    # --- run ---
    r = sub.add_parser("run", help="Run MetricGuard analysis on a CSV")
    r.add_argument("--csv", type=str, required=True, help="Input CSV path")
    r.add_argument("--time", type=str, required=True, help="Time column name (sorted)")
    r.add_argument("--kpi", type=str, required=True, help="KPI column name")
    r.add_argument("--outcome", type=str, required=True, help="Outcome column name")

    r.add_argument(
        "--extras", type=str, nargs="*", default=[], help="Optional extra columns for contradictions (space-separated)"
    )

    r.add_argument("--out", type=str, required=True, help="Output report (.md) path")
    r.add_argument("--window", type=int, default=8, help="Rolling window length")
    r.add_argument("--plot", action="store_true", help="Save plots alongside report")

    # Direction flags (this is a big “real product” feature)
    r.add_argument("--kpi-higher-better", type=str, default="true", help="true/false: is a higher KPI value better?")
    r.add_argument(
        "--outcome-higher-better",
        type=str,
        default="false",
        help="true/false: is a higher outcome value better? (often false, e.g. incidents)",
    )

    # --- benchmark ---
    b = sub.add_parser("benchmark", help="Measure detection accuracy on labeled synthetic data")
    b.add_argument("--seeds", type=str, default="0:100", help="Seed range as start:stop (default 0:100)")
    b.add_argument("--periods", type=int, default=52)
    b.add_argument("--window", type=int, default=8)
    b.add_argument("--out", type=str, default=None, help="Optional markdown output path")

    args = parser.parse_args()

    if args.cmd == "benchmark":
        from .benchmark import run_benchmark

        start, stop = (int(v) for v in args.seeds.split(":"))
        result = run_benchmark(seeds=range(start, stop), periods=args.periods, window=args.window)
        md = result.to_markdown()
        print(md)
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(md, encoding="utf-8")
        return

    if args.cmd == "generate":
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if args.scenario == "safety":
            df = generate_safety_dataset(periods=args.periods, seed=args.seed, mode=args.mode)
        else:
            df = generate_productivity_dataset(periods=args.periods, seed=args.seed, mode=args.mode)

        df.to_csv(out_path, index=False)
        print(f"✅ Wrote {args.scenario} dataset ({args.mode}) to: {out_path}")
        return

    # run
    def to_bool(s: str) -> bool:
        s = s.strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
        raise ValueError("Expected true/false")

    kpi_hb = to_bool(args.kpi_higher_better)
    out_hb = to_bool(args.outcome_higher_better)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_metricguard(
        csv_path=Path(args.csv),
        time_col=args.time,
        kpi_col=args.kpi,
        outcome_col=args.outcome,
        extra_cols=list(args.extras),
        report_path=out_path,
        window=args.window,
        save_plot=args.plot,
        kpi_higher_better=kpi_hb,
        outcome_higher_better=out_hb,
    )
    print(f"✅ Wrote report to: {out_path}")
