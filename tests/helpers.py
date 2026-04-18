from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from stat_arb.data import MarketDataStore  # noqa: E402


def make_synthetic_market(
    num_days: int = 720,
    *,
    include_etf: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(17)
    dates = pd.bdate_range("2018-01-01", periods=num_days)

    asset_b = 55 + np.cumsum(rng.normal(0, 0.7, num_days))
    mean_reverting_noise = np.zeros(num_days)
    for idx in range(1, num_days):
        mean_reverting_noise[idx] = 0.86 * mean_reverting_noise[idx - 1] + rng.normal(0, 0.28)
    asset_a = 1.35 * asset_b + mean_reverting_noise + 8
    asset_c = 35 + np.cumsum(rng.normal(0, 1.4, num_days))
    spy = 95 + np.cumsum(rng.normal(0, 0.6, num_days))

    close = pd.DataFrame(
        {
            "AAA": asset_a,
            "BBB": asset_b,
            "CCC": asset_c,
        },
        index=dates,
    )
    if include_etf:
        close["SPY"] = spy

    volume = pd.DataFrame(
        {
            "AAA": 1_500_000 + rng.integers(0, 50_000, num_days),
            "BBB": 1_700_000 + rng.integers(0, 50_000, num_days),
            "CCC": 900_000 + rng.integers(0, 60_000, num_days),
        },
        index=dates,
    )
    if include_etf:
        volume["SPY"] = 2_500_000 + rng.integers(0, 80_000, num_days)

    long_prices = (
        close.stack()
        .rename("close")
        .to_frame()
        .join(volume.reindex(columns=close.columns, fill_value=0.0).stack().rename("volume"))
        .reset_index()
    )
    long_prices.columns = ["date", "symbol", "close", "volume"]
    long_prices["symbol"] = long_prices["symbol"].astype(str)
    price_store = long_prices.set_index(["date", "symbol"]).sort_index()

    metadata_rows = [
        {"symbol": "AAA", "name": "Alpha Corp", "sector": "Industrials", "industry": "Machinery", "asset_type": "equity"},
        {"symbol": "BBB", "name": "Beta Corp", "sector": "Industrials", "industry": "Machinery", "asset_type": "equity"},
        {"symbol": "CCC", "name": "Gamma Corp", "sector": "Utilities", "industry": "Electric Utilities", "asset_type": "equity"},
    ]
    if include_etf:
        metadata_rows.append(
            {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "sector": "ETF", "industry": "ETF", "asset_type": "etf"}
        )
    metadata = pd.DataFrame(metadata_rows)
    return close, volume, price_store, metadata


def seed_store(base_path: str | Path, *, include_etf: bool = True) -> tuple[MarketDataStore, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    close, volume, prices, metadata = make_synthetic_market(include_etf=include_etf)
    store = MarketDataStore(base_path)
    store.save_metadata(metadata)
    store.save_prices(prices)
    return store, close, volume, metadata


def write_json_yaml(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

