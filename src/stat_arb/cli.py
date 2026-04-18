from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from .backtest import run_portfolio_backtest
from .config import ProjectConfig, load_project_config
from .data import MarketDataStore
from .reporting import render_run_summary, save_backtest_report, save_research_report
from .selection import PairResearchReport, research_pairs
from .universe import build_universe, load_universe_metadata


def _load_or_seed_metadata(store: MarketDataStore, config: ProjectConfig) -> pd.DataFrame:
    try:
        return store.load_metadata()
    except FileNotFoundError:
        metadata = load_universe_metadata(config.metadata_file, config.etf_metadata_file)
        store.save_metadata(metadata)
        return metadata


def _ensure_price_data(
    store: MarketDataStore,
    config: ProjectConfig,
    dataset_name: str,
) -> None:
    try:
        store.load_raw_prices(dataset_name)
    except FileNotFoundError:
        if config.legacy_prices_file and Path(config.legacy_prices_file).exists():
            store.import_legacy_wide_prices(config.legacy_prices_file, dataset_name=dataset_name)
            return
        raise


def _resolve_universe(
    store: MarketDataStore,
    config: ProjectConfig,
    dataset_name: str = "market_prices",
) -> pd.DataFrame:
    try:
        return store.load_universe(config.universe.name)
    except FileNotFoundError:
        metadata = _load_or_seed_metadata(store, config)
        _ensure_price_data(store, config, dataset_name)
        universe = build_universe(
            store,
            config.universe,
            config.research.start_date,
            config.research.end_date,
            metadata=metadata,
            dataset_name=dataset_name,
        )
        store.save_universe(universe, config.universe.name)
        return universe


def cmd_download(args: argparse.Namespace) -> int:
    config = load_project_config(args.config)
    if args.metadata_file:
        config.metadata_file = args.metadata_file
    if args.etf_metadata_file:
        config.etf_metadata_file = args.etf_metadata_file
    store = MarketDataStore(config.data_root)
    metadata = load_universe_metadata(config.metadata_file, config.etf_metadata_file)
    store.save_metadata(metadata)
    result = store.download_prices(
        metadata["symbol"].tolist(),
        start=args.start_date or config.research.start_date or "2012-01-01",
        end=args.end_date or config.research.end_date or "2024-01-01",
        batch_size=args.batch_size,
        retries=args.retries,
        dataset_name=args.dataset_name,
    )
    print(f"Saved metadata for {len(metadata)} symbols.")
    print(f"Downloaded {result.downloaded_symbols} symbols to {result.storage_path}.")
    if result.skipped_symbols:
        print(f"Skipped {len(result.skipped_symbols)} previously cached symbols.")
    return 0


def cmd_build_universe(args: argparse.Namespace) -> int:
    config = load_project_config(args.config)
    store = MarketDataStore(config.data_root)
    _ensure_price_data(store, config, args.dataset_name)
    metadata = _load_or_seed_metadata(store, config)
    universe = build_universe(
        store,
        config.universe,
        args.start_date or config.research.start_date,
        args.end_date or config.research.end_date,
        metadata=metadata,
        dataset_name=args.dataset_name,
    )
    path = store.save_universe(universe, config.universe.name)
    print(f"Built universe '{config.universe.name}' with {len(universe)} symbols.")
    print(path)
    return 0


def _run_research(config: ProjectConfig, dataset_name: str) -> PairResearchReport:
    store = MarketDataStore(config.data_root)
    _ensure_price_data(store, config, dataset_name)
    _load_or_seed_metadata(store, config)
    universe = _resolve_universe(store, config, dataset_name)
    prices = store.load_prices(
        universe,
        config.research.start_date,
        config.research.end_date,
        ("close", "volume"),
        dataset_name=dataset_name,
    )
    report = research_pairs(
        prices["close"],
        prices["volume"],
        universe,
        config.research,
        config.strategy,
    )
    return save_research_report(report, config.artifacts_root)


def cmd_research(args: argparse.Namespace) -> int:
    config = load_project_config(args.config)
    report = _run_research(config, args.dataset_name)
    print(f"Research artifacts written to {report.artifact_paths['run_dir']}")
    if not report.rankings.empty:
        print(report.top_pairs(args.top_n).to_string(index=False))
    else:
        print(report.summary.get("message", "No research results generated."))
    return 0


def _load_rankings_from_run_dir(run_dir: str | Path) -> pd.DataFrame:
    rankings_path = Path(run_dir) / "ranked_pairs.csv"
    if not rankings_path.exists():
        raise FileNotFoundError(f"No ranked_pairs.csv found in {run_dir}")
    return pd.read_csv(rankings_path)


def cmd_backtest(args: argparse.Namespace) -> int:
    config = load_project_config(args.config)
    store = MarketDataStore(config.data_root)
    _ensure_price_data(store, config, args.dataset_name)
    universe = _resolve_universe(store, config, args.dataset_name)
    prices = store.load_prices(
        universe,
        config.research.start_date,
        config.research.end_date,
        ("close",),
        dataset_name=args.dataset_name,
    )

    if args.research_run:
        rankings = _load_rankings_from_run_dir(args.research_run)
    else:
        rankings = _run_research(config, args.dataset_name).rankings

    report = run_portfolio_backtest(
        prices["close"],
        rankings,
        config.strategy,
        config.research,
        initial_capital=config.initial_capital,
        top_n=args.top_n,
    )
    report = save_backtest_report(report, config.artifacts_root)
    print(f"Backtest artifacts written to {report.artifact_paths['run_dir']}")
    print(pd.Series(report.summary).to_string())
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    print(render_run_summary(args.run_dir))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stat-arb",
        description="Research-grade statistical arbitrage lab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    download.add_argument("--config", default=None)
    download.add_argument("--metadata-file", default=None)
    download.add_argument("--etf-metadata-file", default=None)
    download.add_argument("--start-date", default=None)
    download.add_argument("--end-date", default=None)
    download.add_argument("--batch-size", type=int, default=50)
    download.add_argument("--retries", type=int, default=3)
    download.add_argument("--dataset-name", default="market_prices")
    download.set_defaults(func=cmd_download)

    build_cmd = subparsers.add_parser("build-universe", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    build_cmd.add_argument("--config", default=None)
    build_cmd.add_argument("--start-date", default=None)
    build_cmd.add_argument("--end-date", default=None)
    build_cmd.add_argument("--dataset-name", default="market_prices")
    build_cmd.set_defaults(func=cmd_build_universe)

    research = subparsers.add_parser("research", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    research.add_argument("--config", default=None)
    research.add_argument("--dataset-name", default="market_prices")
    research.add_argument("--top-n", type=int, default=10)
    research.set_defaults(func=cmd_research)

    backtest = subparsers.add_parser("backtest", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    backtest.add_argument("--config", default=None)
    backtest.add_argument("--dataset-name", default="market_prices")
    backtest.add_argument("--research-run", default=None, help="Existing research artifact directory.")
    backtest.add_argument("--top-n", type=int, default=None)
    backtest.set_defaults(func=cmd_backtest)

    report = subparsers.add_parser("report", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    report.add_argument("--config", default=None)
    report.add_argument("run_dir", help="Artifact directory to summarize.")
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
