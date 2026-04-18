from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import ResearchConfig, StrategyConfig
from .signals import build_pair_signal_frame


def _annualized_return(returns: pd.Series) -> float:
    cleaned = returns.dropna()
    if cleaned.empty:
        return 0.0
    equity = (1.0 + cleaned).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    periods = len(cleaned)
    return float((1.0 + total_return) ** (252.0 / periods) - 1.0) if periods else 0.0


def _annualized_volatility(returns: pd.Series) -> float:
    cleaned = returns.dropna()
    return float(cleaned.std(ddof=0) * np.sqrt(252)) if not cleaned.empty else 0.0


def _sharpe_ratio(returns: pd.Series) -> float:
    vol = _annualized_volatility(returns)
    if vol == 0:
        return 0.0
    return _annualized_return(returns) / vol


def _max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    drawdown = equity_curve / peak - 1.0
    return float(drawdown.min())


@dataclass
class BacktestReport:
    run_id: str
    summary: dict[str, object]
    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    pair_diagnostics: dict[str, pd.DataFrame]
    artifact_paths: dict[str, str] = field(default_factory=dict)


def _coerce_float(value: object, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    return float(value)


def _prepare_pair_frame(
    close: pd.DataFrame,
    symbol_a: str,
    symbol_b: str,
    strategy: StrategyConfig,
    research: ResearchConfig,
) -> pd.DataFrame:
    pair_close = close[[symbol_a, symbol_b]].dropna()
    frame = build_pair_signal_frame(pair_close[symbol_a], pair_close[symbol_b], strategy, research.beta_window)
    frame["ret_a"] = frame["price_a"].pct_change().fillna(0.0)
    frame["ret_b"] = frame["price_b"].pct_change().fillna(0.0)
    raw_weight_a = frame["side"]
    raw_weight_b = -frame["side"] * frame["beta"]
    gross = (raw_weight_a.abs() + raw_weight_b.abs()).replace(0.0, np.nan)
    frame["weight_a"] = (raw_weight_a / gross).fillna(0.0)
    frame["weight_b"] = (raw_weight_b / gross).fillna(0.0)
    frame["pair_id"] = f"{symbol_a}__{symbol_b}"
    return frame


def _extract_trade_log(pair_diagnostics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    trades: list[dict[str, object]] = []
    for pair_id, frame in pair_diagnostics.items():
        active_trade: dict[str, object] | None = None
        running_pnl = 0.0
        previous_side = 0
        for date, row in frame.iterrows():
            side = int(row["side"])
            if active_trade is not None:
                running_pnl += float(row.get("pair_pnl", 0.0))
            if previous_side == 0 and side != 0:
                active_trade = {
                    "pair_id": pair_id,
                    "symbol_a": pair_id.split("__")[0],
                    "symbol_b": pair_id.split("__")[1],
                    "entry_date": str(date.date()),
                    "side": side,
                    "entry_z": float(row.get("entry_z", 0.0) or 0.0),
                }
                running_pnl = 0.0
            elif active_trade is not None and previous_side != 0 and side == 0:
                active_trade["exit_date"] = str(date.date())
                active_trade["exit_reason"] = row.get("exit_reason", "")
                active_trade["pnl"] = running_pnl
                trades.append(active_trade)
                active_trade = None
                running_pnl = 0.0
            previous_side = side
        if active_trade is not None:
            active_trade["exit_date"] = str(frame.index[-1].date())
            active_trade["exit_reason"] = "open_at_end"
            active_trade["pnl"] = running_pnl
            trades.append(active_trade)
    return pd.DataFrame(trades)


def run_portfolio_backtest(
    close: pd.DataFrame,
    rankings: pd.DataFrame,
    strategy: StrategyConfig,
    research: ResearchConfig,
    *,
    initial_capital: float,
    top_n: int | None = None,
) -> BacktestReport:
    if rankings.empty:
        empty_curve = pd.DataFrame(columns=["equity", "daily_return", "gross_exposure", "net_exposure"])
        return BacktestReport(
            run_id="",
            summary={"message": "No ranked pairs were supplied for backtesting."},
            equity_curve=empty_curve,
            trade_log=pd.DataFrame(),
            pair_diagnostics={},
        )

    selected = rankings.head(top_n or strategy.max_active_pairs)
    pair_frames: dict[str, pd.DataFrame] = {}
    for row in selected.itertuples(index=False):
        pair_frames[row.pair_id] = _prepare_pair_frame(
            close,
            row.symbol_a,
            row.symbol_b,
            strategy,
            research,
        )

    common_index = pd.Index(sorted(set().union(*(frame.index for frame in pair_frames.values()))))
    commission_rate = (strategy.commission_bps + strategy.slippage_bps) / 10_000.0
    borrow_rate_daily = strategy.borrow_bps_annual / 10_000.0 / 252.0

    holdings = {pair_id: {"a": 0.0, "b": 0.0} for pair_id in pair_frames}
    pair_histories = {
        pair_id: frame.reindex(common_index).copy()
        for pair_id, frame in pair_frames.items()
    }
    for pair_id, frame in pair_histories.items():
        frame["side"] = frame["side"].fillna(0.0)
        frame["weight_a"] = frame["weight_a"].fillna(0.0)
        frame["weight_b"] = frame["weight_b"].fillna(0.0)
        frame["ret_a"] = frame["ret_a"].fillna(0.0)
        frame["ret_b"] = frame["ret_b"].fillna(0.0)
        frame["entry_z"] = frame["entry_z"].astype(float)
        frame["exit_reason"] = frame["exit_reason"].fillna("")
        pair_histories[pair_id] = frame
    equity_records: list[dict[str, float | str]] = []
    equity = initial_capital

    for date in common_index:
        target_book: dict[str, tuple[float, float]] = {}
        pair_budget = min(equity * strategy.pair_allocation_pct, equity * strategy.max_pair_gross_pct)
        for pair_id, frame in pair_histories.items():
            row = frame.loc[date]
            weight_a = _coerce_float(row.get("weight_a", 0.0))
            weight_b = _coerce_float(row.get("weight_b", 0.0))
            target_book[pair_id] = (weight_a * pair_budget, weight_b * pair_budget)

        gross = sum(abs(a) + abs(b) for a, b in target_book.values())
        net = sum(a + b for a, b in target_book.values())
        scale = 1.0
        if equity > 0 and gross > equity * strategy.max_gross_exposure_pct:
            scale = min(scale, (equity * strategy.max_gross_exposure_pct) / gross)
        if equity > 0 and abs(net) > equity * strategy.max_net_exposure_pct and abs(net) > 0:
            scale = min(scale, (equity * strategy.max_net_exposure_pct) / abs(net))
        if scale < 1.0:
            target_book = {
                pair_id: (targets[0] * scale, targets[1] * scale)
                for pair_id, targets in target_book.items()
            }
            gross = sum(abs(a) + abs(b) for a, b in target_book.values())
            net = sum(a + b for a, b in target_book.values())

        total_pnl = 0.0
        for pair_id, frame in pair_histories.items():
            row = frame.loc[date]
            previous_a = holdings[pair_id]["a"]
            previous_b = holdings[pair_id]["b"]
            target_a, target_b = target_book[pair_id]
            pnl = previous_a * _coerce_float(row.get("ret_a", 0.0)) + previous_b * _coerce_float(row.get("ret_b", 0.0))
            traded_notional = abs(target_a - previous_a) + abs(target_b - previous_b)
            transaction_cost = traded_notional * commission_rate
            borrow_cost = (
                abs(min(previous_a, 0.0)) + abs(min(previous_b, 0.0))
            ) * borrow_rate_daily
            pair_pnl = pnl - transaction_cost - borrow_cost
            total_pnl += pair_pnl

            holdings[pair_id]["a"] = target_a
            holdings[pair_id]["b"] = target_b
            pair_histories[pair_id].loc[date, "target_notional_a"] = target_a
            pair_histories[pair_id].loc[date, "target_notional_b"] = target_b
            pair_histories[pair_id].loc[date, "pair_pnl"] = pair_pnl
            pair_histories[pair_id].loc[date, "pair_gross_exposure"] = abs(target_a) + abs(target_b)

        previous_equity = equity
        equity = max(equity + total_pnl, 0.0)
        daily_return = total_pnl / previous_equity if previous_equity else 0.0
        equity_records.append(
            {
                "date": str(date.date()),
                "equity": equity,
                "daily_return": daily_return,
                "gross_exposure": gross,
                "net_exposure": net,
            }
        )

    equity_curve = pd.DataFrame(equity_records)
    if not equity_curve.empty:
        equity_curve["date"] = pd.to_datetime(equity_curve["date"])
        equity_curve = equity_curve.set_index("date")

    returns = equity_curve["daily_return"] if not equity_curve.empty else pd.Series(dtype=float)
    trade_log = _extract_trade_log(pair_histories)
    summary = {
        "initial_capital": initial_capital,
        "final_equity": float(equity_curve["equity"].iloc[-1]) if not equity_curve.empty else initial_capital,
        "total_return": float(equity_curve["equity"].iloc[-1] / initial_capital - 1.0) if not equity_curve.empty else 0.0,
        "annualized_return": _annualized_return(returns),
        "annualized_volatility": _annualized_volatility(returns),
        "sharpe_ratio": _sharpe_ratio(returns),
        "max_drawdown": _max_drawdown(equity_curve["equity"]) if not equity_curve.empty else 0.0,
        "average_gross_exposure": float(equity_curve["gross_exposure"].mean()) if not equity_curve.empty else 0.0,
        "average_net_exposure": float(equity_curve["net_exposure"].mean()) if not equity_curve.empty else 0.0,
        "trade_count": int(len(trade_log)),
        "pairs_traded": int(len(pair_histories)),
    }
    return BacktestReport(
        run_id="",
        summary=summary,
        equity_curve=equity_curve,
        trade_log=trade_log,
        pair_diagnostics=pair_histories,
    )
