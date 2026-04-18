"""Research-grade statistical arbitrage toolkit."""

from .backtest import BacktestReport, run_portfolio_backtest
from .config import (
    ProjectConfig,
    ResearchConfig,
    StrategyConfig,
    UniverseSpec,
    load_project_config,
)
from .data import MarketDataStore
from .selection import PairResearchReport, research_pairs
from .universe import build_universe

__all__ = [
    "BacktestReport",
    "MarketDataStore",
    "PairResearchReport",
    "ProjectConfig",
    "ResearchConfig",
    "StrategyConfig",
    "UniverseSpec",
    "build_universe",
    "load_project_config",
    "research_pairs",
    "run_portfolio_backtest",
]

