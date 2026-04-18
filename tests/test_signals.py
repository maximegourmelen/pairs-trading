from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from stat_arb.config import StrategyConfig  # noqa: E402
from stat_arb.signals import RollingOLSModel, generate_stateful_signals, rolling_zscore  # noqa: E402
from tests.helpers import make_synthetic_market  # noqa: E402


class SignalTests(unittest.TestCase):
    def test_rolling_ols_beta_tracks_synthetic_relationship(self) -> None:
        close, _volume, _prices, _metadata = make_synthetic_market(num_days=400, include_etf=False)
        beta = RollingOLSModel(lookback=120).estimate_beta(close["AAA"], close["BBB"])
        self.assertAlmostEqual(float(beta.iloc[-30:].mean()), 1.35, delta=0.20)

    def test_rolling_zscore_matches_manual_computation(self) -> None:
        series = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
        zscore = rolling_zscore(series, lookback=3, min_periods=3)
        manual = (14.0 - np.mean([12.0, 13.0, 14.0])) / np.std([12.0, 13.0, 14.0], ddof=0)
        self.assertAlmostEqual(float(zscore.iloc[-1]), float(manual), places=8)

    def test_stateful_signals_persist_until_exit(self) -> None:
        zscore = pd.Series(
            [-2.1, -1.7, -1.2, -0.4, 0.2, 2.2, 2.1, 0.3],
            index=pd.bdate_range("2024-01-01", periods=8),
        )
        strategy = StrategyConfig(entry_z=2.0, exit_z=0.5, stop_z=3.5, max_holding_days=10)
        signals = generate_stateful_signals(zscore, strategy)
        self.assertListEqual(signals["side"].tolist(), [1, 1, 1, 0, 0, -1, -1, 0])
        self.assertEqual(signals.iloc[3]["exit_reason"], "mean_reversion_exit")
        self.assertEqual(signals.iloc[-1]["exit_reason"], "mean_reversion_exit")


if __name__ == "__main__":
    unittest.main()

