from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import pandas as pd

from .backtest import BacktestReport
from .selection import PairResearchReport


def _maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None
    return plt


def make_run_id(prefix: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def save_research_report(
    report: PairResearchReport,
    artifacts_root: str | Path,
) -> PairResearchReport:
    run_id = report.run_id or make_run_id("research")
    run_dir = Path(artifacts_root) / run_id
    diagnostics_dir = run_dir / "pair_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    rankings_path = run_dir / "ranked_pairs.csv"
    summary_path = run_dir / "summary.json"
    report.rankings.to_csv(rankings_path, index=False)
    _write_json(summary_path, report.summary)

    diagnostic_paths: dict[str, str] = {}
    for pair_id, frame in report.diagnostics.items():
        path = diagnostics_dir / f"{pair_id}.csv"
        frame.to_csv(path, index=True)
        diagnostic_paths[pair_id] = str(path)

    artifacts = {
        "run_dir": str(run_dir),
        "rankings": str(rankings_path),
        "summary": str(summary_path),
        "pair_diagnostics": str(diagnostics_dir),
    }

    plt = _maybe_import_matplotlib()
    if plt is not None and not report.rankings.empty:
        figure, axis = plt.subplots(figsize=(10, 5))
        top = report.rankings.head(10).copy()
        axis.bar(top["pair_id"], top["ranking_score"], color="#1f77b4")
        axis.set_title("Top Ranked Pairs")
        axis.set_ylabel("Ranking Score")
        axis.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        plot_path = run_dir / "top_ranked_pairs.png"
        figure.savefig(plot_path, dpi=200)
        plt.close(figure)
        artifacts["top_ranked_pairs_plot"] = str(plot_path)

    report.run_id = run_id
    report.artifact_paths = {**artifacts, **diagnostic_paths}
    return report


def save_backtest_report(
    report: BacktestReport,
    artifacts_root: str | Path,
) -> BacktestReport:
    run_id = report.run_id or make_run_id("backtest")
    run_dir = Path(artifacts_root) / run_id
    diagnostics_dir = run_dir / "pair_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    equity_path = run_dir / "equity_curve.csv"
    trades_path = run_dir / "trade_log.csv"
    _write_json(summary_path, report.summary)
    report.equity_curve.to_csv(equity_path, index=True)
    report.trade_log.to_csv(trades_path, index=False)

    artifacts = {
        "run_dir": str(run_dir),
        "summary": str(summary_path),
        "equity_curve": str(equity_path),
        "trade_log": str(trades_path),
        "pair_diagnostics": str(diagnostics_dir),
    }

    for pair_id, frame in report.pair_diagnostics.items():
        path = diagnostics_dir / f"{pair_id}.csv"
        frame.to_csv(path, index=True)

    plt = _maybe_import_matplotlib()
    if plt is not None and not report.equity_curve.empty:
        figure, axis = plt.subplots(figsize=(10, 5))
        report.equity_curve["equity"].plot(ax=axis, color="#0f766e", lw=1.8)
        axis.set_title("Portfolio Equity Curve")
        axis.set_ylabel("Equity")
        axis.grid(True, linestyle="--", alpha=0.4)
        figure.tight_layout()
        plot_path = run_dir / "equity_curve.png"
        figure.savefig(plot_path, dpi=200)
        plt.close(figure)
        artifacts["equity_plot"] = str(plot_path)

    report.run_id = run_id
    report.artifact_paths = artifacts
    return report


def render_run_summary(run_dir: str | Path) -> str:
    path = Path(run_dir) / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"No summary.json found in {run_dir}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    lines = [f"Run summary for {run_dir}"]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)

