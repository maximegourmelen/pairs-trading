from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import UniverseSpec
from .data import MarketDataStore


EQUITY_COLUMN_MAP = {
    "Symbol": "symbol",
    "Security": "name",
    "GICS Sector": "sector",
    "GICS Sub-Industry": "industry",
}


def load_universe_metadata(
    equities_path: str | Path,
    etf_path: str | Path | None = None,
) -> pd.DataFrame:
    equities = pd.read_csv(equities_path).rename(columns=EQUITY_COLUMN_MAP)
    equities = equities.assign(asset_type="equity")
    keep_columns = ["symbol", "name", "sector", "industry", "asset_type"]
    equities = equities[[column for column in keep_columns if column in equities.columns]]

    frames = [equities]
    if etf_path:
        etfs = pd.read_csv(etf_path)
        rename_map = {}
        if "Symbol" in etfs.columns:
            rename_map["Symbol"] = "symbol"
        if "Name" in etfs.columns:
            rename_map["Name"] = "name"
        etfs = etfs.rename(columns=rename_map)
        if "symbol" not in etfs.columns:
            raise ValueError("ETF metadata must include a Symbol or symbol column.")
        if "name" not in etfs.columns:
            etfs["name"] = etfs["symbol"]
        if "sector" not in etfs.columns:
            etfs["sector"] = "ETF"
        if "industry" not in etfs.columns:
            etfs["industry"] = "ETF"
        etfs["asset_type"] = "etf"
        frames.append(etfs[keep_columns])

    metadata = pd.concat(frames, ignore_index=True)
    metadata["symbol"] = metadata["symbol"].astype(str).str.upper()
    metadata = metadata.drop_duplicates(subset=["symbol"], keep="last")
    return metadata.sort_values("symbol").reset_index(drop=True)


def build_universe(
    store: MarketDataStore,
    spec: UniverseSpec,
    start: str | None,
    end: str | None,
    *,
    metadata: Optional[pd.DataFrame] = None,
    dataset_name: str = "market_prices",
) -> pd.DataFrame:
    metadata_frame = metadata.copy() if metadata is not None else store.load_metadata()
    metadata_frame["symbol"] = metadata_frame["symbol"].astype(str).str.upper()

    prices = store.load_prices(metadata_frame["symbol"].tolist(), start, end, ("close", "volume"), dataset_name)
    close = prices["close"]
    volume = prices["volume"].reindex_like(close).fillna(0.0)

    history_days = close.notna().sum()
    avg_dollar_volume = (close * volume).replace([float("inf"), float("-inf")], pd.NA).mean(skipna=True)
    last_price = close.ffill().iloc[-1]

    stats = pd.DataFrame(
        {
            "symbol": close.columns,
            "history_days": history_days.reindex(close.columns).fillna(0).astype(int).values,
            "avg_dollar_volume": avg_dollar_volume.reindex(close.columns).fillna(0.0).values,
            "last_price": last_price.reindex(close.columns).fillna(0.0).values,
        }
    )
    universe = metadata_frame.merge(stats, on="symbol", how="inner")

    if spec.universe_type == "equities":
        universe = universe[universe["asset_type"] == "equity"]
    elif spec.universe_type == "equities_plus_etfs":
        allowed_etfs = {symbol.upper() for symbol in spec.etf_symbols}
        if allowed_etfs:
            universe = universe[
                (universe["asset_type"] == "equity")
                | (universe["symbol"].isin(allowed_etfs))
            ]
    else:
        raise ValueError(f"Unsupported universe_type: {spec.universe_type}")

    universe = universe[
        (universe["history_days"] >= spec.min_history_days)
        & (universe["avg_dollar_volume"] >= spec.min_average_dollar_volume)
        & (universe["last_price"] >= spec.min_price)
    ]

    if spec.allowed_sectors:
        universe = universe[universe["sector"].isin(spec.allowed_sectors)]
    if spec.excluded_symbols:
        universe = universe[~universe["symbol"].isin([symbol.upper() for symbol in spec.excluded_symbols])]
    if spec.included_symbols:
        include = {symbol.upper() for symbol in spec.included_symbols}
        universe = universe[universe["symbol"].isin(include)]

    universe = universe.sort_values(
        by=["asset_type", "avg_dollar_volume", "symbol"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return universe

