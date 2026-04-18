from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from stat_arb.config import ResearchConfig, StrategyConfig  # noqa: E402
from stat_arb.selection import research_pairs  # noqa: E402
from tests.helpers import make_synthetic_market  # noqa: E402


class SelectionTests(unittest.TestCase):
    def test_research_pipeline_is_deterministic_and_out_of_sample(self) -> None:
        close, volume, _prices, metadata = make_synthetic_market(num_days=720, include_etf=False)
        research = ResearchConfig(
            train_size=260,
            validation_size=120,
            test_size=120,
            max_candidate_pairs=10,
            min_return_correlation=0.05,
            fdr_alpha=1.0,
        )
        strategy = StrategyConfig(entry_z=1.25, exit_z=0.35, stop_z=3.5, max_holding_days=20)

        report_one = research_pairs(close, volume, metadata, research, strategy)
        report_two = research_pairs(close, volume, metadata, research, strategy)

        self.assertFalse(report_one.rankings.empty)
        self.assertEqual(report_one.rankings.iloc[0]["pair_id"], "AAA__BBB")
        pd.testing.assert_frame_equal(
            report_one.rankings.head(3).reset_index(drop=True),
            report_two.rankings.head(3).reset_index(drop=True),
        )
        self.assertIn("train_start", report_one.summary)
        self.assertIn("test_end", report_one.summary)
        self.assertIn("validation_sharpe", report_one.rankings.columns)
        self.assertIn("test_sharpe", report_one.rankings.columns)


if __name__ == "__main__":
    unittest.main()

