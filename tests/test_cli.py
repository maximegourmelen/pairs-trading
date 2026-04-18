from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from stat_arb.cli import main  # noqa: E402
from stat_arb.data import MarketDataStore  # noqa: E402
from tests.helpers import make_synthetic_market, write_json_yaml  # noqa: E402


class CliTests(unittest.TestCase):
    def test_cli_builds_universe_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            close, _volume, _prices, metadata = make_synthetic_market(num_days=720, include_etf=True)

            equities = metadata[metadata["asset_type"] == "equity"].rename(
                columns={
                    "symbol": "Symbol",
                    "name": "Security",
                    "sector": "GICS Sector",
                    "industry": "GICS Sub-Industry",
                }
            )
            etfs = metadata[metadata["asset_type"] == "etf"].rename(
                columns={"symbol": "Symbol", "name": "Name"}
            )

            equity_path = temp_path / "equities.csv"
            etf_path = temp_path / "etfs.csv"
            legacy_path = temp_path / "legacy_prices.csv"
            config_path = temp_path / "config.yaml"
            data_root = temp_path / "data"
            artifacts_root = temp_path / "artifacts"

            equities.to_csv(equity_path, index=False)
            etfs.to_csv(etf_path, index=False)
            close.to_csv(legacy_path)

            write_json_yaml(
                config_path,
                {
                    "data_root": str(data_root),
                    "artifacts_root": str(artifacts_root),
                    "initial_capital": 100000,
                    "metadata_file": str(equity_path),
                    "legacy_prices_file": str(legacy_path),
                    "etf_metadata_file": str(etf_path),
                    "universe": {
                        "name": "test_universe",
                        "universe_type": "equities_plus_etfs",
                        "min_history_days": 500,
                        "min_average_dollar_volume": 0,
                        "min_price": 5.0,
                        "allowed_sectors": [],
                        "excluded_symbols": [],
                        "included_symbols": [],
                        "etf_symbols": ["SPY"],
                    },
                    "research": {
                        "start_date": "2018-01-01",
                        "end_date": "2021-12-31",
                        "train_size": 250,
                        "validation_size": 100,
                        "test_size": 100,
                        "require_same_sector": False,
                        "require_same_industry": False,
                        "min_return_correlation": 0.05,
                        "max_candidate_pairs": 10,
                        "beta_window": 90,
                        "beta_mode": "rolling",
                        "fdr_alpha": 1.0,
                    },
                    "strategy": {
                        "beta_model": "rolling_ols",
                        "zscore_lookback": 45,
                        "entry_z": 1.2,
                        "exit_z": 0.3,
                        "stop_z": 3.0,
                        "max_holding_days": 20,
                        "pair_allocation_pct": 0.1,
                        "max_gross_exposure_pct": 0.6,
                        "max_net_exposure_pct": 0.1,
                        "max_active_pairs": 2,
                        "commission_bps": 1.0,
                        "slippage_bps": 1.0,
                        "borrow_bps_annual": 20.0,
                        "max_pair_gross_pct": 0.15,
                    },
                },
            )

            self.assertEqual(main(["build-universe", "--config", str(config_path)]), 0)
            store = MarketDataStore(data_root)
            universe = store.load_universe("test_universe")
            self.assertIn("SPY", universe["symbol"].tolist())

            self.assertEqual(main(["research", "--config", str(config_path), "--top-n", "3"]), 0)
            research_runs = sorted(artifacts_root.glob("research_*"))
            self.assertTrue(research_runs)
            self.assertTrue((research_runs[-1] / "ranked_pairs.csv").exists())

            self.assertEqual(main(["backtest", "--config", str(config_path), "--top-n", "1"]), 0)
            backtest_runs = sorted(artifacts_root.glob("backtest_*"))
            self.assertTrue(backtest_runs)
            self.assertTrue((backtest_runs[-1] / "summary.json").exists())
            self.assertEqual(main(["report", "--config", str(config_path), str(backtest_runs[-1])]), 0)


if __name__ == "__main__":
    unittest.main()
