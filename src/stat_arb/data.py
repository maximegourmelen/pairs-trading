from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence
import time

import pandas as pd


def _require_yfinance():
    try:
        import yfinance as yf  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "yfinance is required for downloads. Install it in the active environment."
        ) from exc
    return yf


@dataclass
class DownloadResult:
    requested_symbols: int
    downloaded_symbols: int
    skipped_symbols: list[str]
    storage_path: str


class MarketDataStore:
    """Persistent market data cache with Parquet-first storage."""

    def __init__(self, base_path: str | Path = "data") -> None:
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def _resolve_dataset_path(self, stem: str) -> Optional[Path]:
        for suffix in (".parquet", ".pkl", ".csv"):
            candidate = self.raw_path / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
            processed_candidate = self.processed_path / f"{stem}{suffix}"
            if processed_candidate.exists():
                return processed_candidate
        return None

    def _write_frame(self, frame: pd.DataFrame, path: Path) -> Path:
        parquet_path = path.with_suffix(".parquet")
        try:
            frame.to_parquet(parquet_path)
            return parquet_path
        except Exception:
            pickle_path = path.with_suffix(".pkl")
            frame.to_pickle(pickle_path)
            return pickle_path

    def _read_frame(self, path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".pkl":
            return pd.read_pickle(path)
        return pd.read_csv(path)

    def save_metadata(self, metadata: pd.DataFrame, name: str = "universe_metadata") -> Path:
        frame = metadata.copy()
        return self._write_frame(frame.reset_index(drop=True), self.raw_path / name)

    def load_metadata(self, name: str = "universe_metadata") -> pd.DataFrame:
        path = self._resolve_dataset_path(name)
        if path is None:
            raise FileNotFoundError(f"No metadata dataset found for '{name}'.")
        return self._read_frame(path)

    def save_universe(self, universe: pd.DataFrame, name: str) -> Path:
        frame = universe.copy()
        return self._write_frame(frame.reset_index(drop=True), self.processed_path / name)

    def load_universe(self, name: str) -> pd.DataFrame:
        path = self._resolve_dataset_path(name)
        if path is None:
            raise FileNotFoundError(f"No universe dataset found for '{name}'.")
        return self._read_frame(path)

    def save_prices(self, prices: pd.DataFrame, name: str = "market_prices") -> Path:
        frame = prices.copy().sort_index()
        if not isinstance(frame.index, pd.MultiIndex):
            raise ValueError("Prices must use a MultiIndex of [date, symbol].")
        frame = frame.reset_index()
        return self._write_frame(frame, self.raw_path / name)

    def load_raw_prices(self, name: str = "market_prices") -> pd.DataFrame:
        path = self._resolve_dataset_path(name)
        if path is None:
            raise FileNotFoundError(f"No price dataset found for '{name}'.")
        frame = self._read_frame(path)
        if "date" not in frame.columns or "symbol" not in frame.columns:
            raise ValueError("Stored price data must include 'date' and 'symbol' columns.")
        frame["date"] = pd.to_datetime(frame["date"])
        return frame.set_index(["date", "symbol"]).sort_index()

    def import_legacy_wide_prices(
        self,
        csv_path: str | Path,
        *,
        dataset_name: str = "market_prices",
    ) -> Path:
        frame = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        try:
            stacked = frame.stack(future_stack=True).dropna()
        except TypeError:
            stacked = frame.stack(dropna=True)
        long_close = stacked.rename("close").reset_index()
        long_close.columns = ["date", "symbol", "close"]
        long_close["volume"] = 0.0
        long_close["symbol"] = long_close["symbol"].astype(str).str.upper()
        prices = long_close.set_index(["date", "symbol"]).sort_index()
        return self.save_prices(prices, dataset_name)

    def load_prices(
        self,
        universe: pd.DataFrame | Sequence[str],
        start: str | None,
        end: str | None,
        fields: Iterable[str] = ("close", "volume"),
        dataset_name: str = "market_prices",
    ) -> dict[str, pd.DataFrame]:
        frame = self.load_raw_prices(dataset_name)
        symbols = (
            list(universe["symbol"].astype(str))
            if isinstance(universe, pd.DataFrame)
            else [str(symbol) for symbol in universe]
        )
        if start is not None:
            frame = frame[frame.index.get_level_values("date") >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame.index.get_level_values("date") <= pd.Timestamp(end)]
        frame = frame[frame.index.get_level_values("symbol").isin(symbols)]

        aligned: dict[str, pd.DataFrame] = {}
        for field in fields:
            if field not in frame.columns:
                raise KeyError(f"Field '{field}' is not available in the price store.")
            wide = frame[field].unstack("symbol").sort_index()
            ordered = [symbol for symbol in symbols if symbol in wide.columns]
            aligned[field] = wide.loc[:, ordered]
        return aligned

    def download_prices(
        self,
        tickers: Sequence[str],
        start: str,
        end: str,
        *,
        batch_size: int = 50,
        retries: int = 3,
        pause_seconds: float = 0.0,
        resume: bool = True,
        dataset_name: str = "market_prices",
    ) -> DownloadResult:
        yf = _require_yfinance()
        tickers = [str(symbol).upper() for symbol in tickers]
        skipped: list[str] = []
        existing: Optional[pd.DataFrame] = None

        if resume:
            try:
                existing = self.load_raw_prices(dataset_name)
                seen = set(existing.index.get_level_values("symbol"))
                skipped = sorted(symbol for symbol in tickers if symbol in seen)
                tickers = [symbol for symbol in tickers if symbol not in seen]
            except FileNotFoundError:
                existing = None

        all_rows: list[pd.DataFrame] = []
        for batch_start in range(0, len(tickers), batch_size):
            batch = tickers[batch_start : batch_start + batch_size]
            attempt = 0
            last_error: Optional[Exception] = None
            while attempt < retries:
                attempt += 1
                try:
                    raw = yf.download(
                        batch,
                        start=start,
                        end=end,
                        progress=False,
                        auto_adjust=False,
                        group_by="column",
                        threads=True,
                    )
                    if raw.empty:
                        break
                    all_rows.extend(self._normalise_download(raw, batch))
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    time.sleep(min(attempt, 5))
            if last_error is not None:
                raise RuntimeError(f"Failed to download batch {batch}: {last_error}") from last_error
            if pause_seconds:
                time.sleep(pause_seconds)

        combined = existing.copy() if existing is not None else pd.DataFrame()
        downloaded_symbols = 0
        if all_rows:
            fresh = pd.concat(all_rows, ignore_index=True)
            fresh["date"] = pd.to_datetime(fresh["date"])
            fresh = fresh.set_index(["date", "symbol"]).sort_index()
            downloaded_symbols = int(fresh.index.get_level_values("symbol").nunique())
            combined = pd.concat([combined, fresh]).reset_index()
            combined = combined.drop_duplicates(subset=["date", "symbol"], keep="last")
            combined = combined.set_index(["date", "symbol"]).sort_index()
        storage_path = self.save_prices(
            combined if not combined.empty else pd.DataFrame(columns=["close", "volume"]).set_index(
                pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])
            ),
            dataset_name,
        )
        return DownloadResult(
            requested_symbols=len(tickers) + len(skipped),
            downloaded_symbols=downloaded_symbols,
            skipped_symbols=skipped,
            storage_path=str(storage_path),
        )

    def _normalise_download(self, raw: pd.DataFrame, symbols: Sequence[str]) -> list[pd.DataFrame]:
        rows: list[pd.DataFrame] = []
        if isinstance(raw.columns, pd.MultiIndex):
            for symbol in symbols:
                if ("Close", symbol) in raw.columns:
                    close = raw[("Adj Close", symbol)] if ("Adj Close", symbol) in raw.columns else raw[("Close", symbol)]
                    volume = raw[("Volume", symbol)] if ("Volume", symbol) in raw.columns else pd.Series(index=close.index, data=0.0)
                    frame = pd.DataFrame(
                        {
                            "date": close.index,
                            "symbol": symbol,
                            "close": close.astype(float),
                            "volume": volume.astype(float),
                        }
                    ).dropna(subset=["close"])
                    if not frame.empty:
                        rows.append(frame)
        else:
            symbol = symbols[0]
            close = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
            volume = raw["Volume"] if "Volume" in raw.columns else pd.Series(index=close.index, data=0.0)
            frame = pd.DataFrame(
                {
                    "date": close.index,
                    "symbol": symbol,
                    "close": close.astype(float),
                    "volume": volume.astype(float),
                }
            ).dropna(subset=["close"])
            if not frame.empty:
                rows.append(frame)
        return rows
