from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from stat_arb.backtest import run_portfolio_backtest  # noqa: E402
from stat_arb.config import ResearchConfig, StrategyConfig  # noqa: E402
from stat_arb.selection import research_pairs  # noqa: E402
from tests.helpers import make_synthetic_market  # noqa: E402


class BacktestTests(unittest.TestCase):
    def test_backtest_accounts_for_costs_and_generates_trades(self) -> None:
        close, volume, _prices, metadata = make_synthetic_market(num_days=720, include_etf=False)
        research = ResearchConfig(
            train_size=260,
            validation_size=120,
            test_size=120,
            max_candidate_pairs=10,
            min_return_correlation=0.05,
            fdr_alpha=1.0,
        )
        zero_cost_strategy = StrategyConfig(
            entry_z=1.2,
            exit_z=0.3,
            stop_z=3.0,
            max_holding_days=18,
            commission_bps=0.0,
            slippage_bps=0.0,
            max_active_pairs=1,
        )
        rankings = research_pairs(close, volume, metadata, research, zero_cost_strategy).rankings
        report_low_cost = run_portfolio_backtest(
            close,
            rankings,
            zero_cost_strategy,
            research,
            initial_capital=100_000,
            top_n=1,
        )

        high_cost_strategy = StrategyConfig(
            entry_z=1.2,
            exit_z=0.3,
            stop_z=3.0,
            max_holding_days=18,
            commission_bps=20.0,
            slippage_bps=20.0,
            max_active_pairs=1,
        )
        report_high_cost = run_portfolio_backtest(
            close,
            rankings,
            high_cost_strategy,
            research,
            initial_capital=100_000,
            top_n=1,
        )

        self.assertGreater(report_low_cost.summary["trade_count"], 0)
        self.assertGreater(
            report_low_cost.summary["final_equity"],
            report_high_cost.summary["final_equity"],
        )
        pair_id = rankings.iloc[0]["pair_id"]
        self.assertIn("target_notional_a", report_low_cost.pair_diagnostics[pair_id].columns)
        self.assertIn("pair_pnl", report_low_cost.pair_diagnostics[pair_id].columns)


if __name__ == "__main__":
    unittest.main()

