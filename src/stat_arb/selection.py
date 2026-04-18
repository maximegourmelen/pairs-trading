from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from .config import ResearchConfig, StrategyConfig
from .signals import build_pair_signal_frame, generate_stateful_signals, rolling_zscore


def _cointegration_pvalue(asset_a: pd.Series, asset_b: pd.Series) -> float:
    try:
        from statsmodels.tsa.stattools import coint  # type: ignore

        _score, pvalue, _ = coint(asset_a, asset_b)
        return float(pvalue)
    except Exception:
        return 1.0


def _annualized_sharpe(returns: pd.Series) -> float:
    cleaned = returns.dropna()
    if cleaned.empty or cleaned.std(ddof=0) == 0:
        return 0.0
    return float((cleaned.mean() / cleaned.std(ddof=0)) * np.sqrt(252))


def _max_drawdown(returns: pd.Series) -> float:
    cleaned = returns.fillna(0.0)
    equity_curve = (1.0 + cleaned).cumprod()
    peak = equity_curve.cummax()
    drawdown = equity_curve / peak - 1.0
    return float(abs(drawdown.min()))


def _half_life(spread: pd.Series) -> float:
    aligned = pd.concat([spread.shift(1).rename("lag"), spread.diff().rename("delta")], axis=1).dropna()
    if aligned.empty:
        return float("inf")
    x = aligned["lag"].to_numpy()
    y = aligned["delta"].to_numpy()
    beta, _intercept = np.polyfit(x, y, 1)
    if beta >= 0:
        return float("inf")
    return float(np.log(2) / abs(beta))


def _pair_returns(frame: pd.DataFrame, strategy: StrategyConfig) -> pd.Series:
    side = frame["side"].shift(1).fillna(0.0)
    beta = frame["beta"].shift(1).ffill().fillna(1.0)
    gross = (1.0 + beta.abs()).replace(0.0, np.nan)
    weight_a = side / gross
    weight_b = (-side * beta) / gross
    returns = (
        weight_a * frame["price_a"].pct_change().fillna(0.0)
        + weight_b * frame["price_b"].pct_change().fillna(0.0)
    )
    turnover = weight_a.diff().abs().fillna(weight_a.abs()) + weight_b.diff().abs().fillna(weight_b.abs())
    cost_rate = (strategy.commission_bps + strategy.slippage_bps) / 10_000.0
    borrow_rate = strategy.borrow_bps_annual / 10_000.0 / 252.0
    borrow_cost = weight_a.clip(upper=0).abs() * borrow_rate + weight_b.clip(upper=0).abs() * borrow_rate
    return (returns - turnover * cost_rate - borrow_cost).fillna(0.0)


def _split_windows(index: pd.Index, research: ResearchConfig) -> tuple[pd.Index, pd.Index, pd.Index]:
    total = research.train_size + research.validation_size + research.test_size
    if len(index) < total:
        raise ValueError(
            f"Need at least {total} observations for the configured train/validation/test split."
        )
    sample = index[-total:]
    train = sample[: research.train_size]
    validation = sample[research.train_size : research.train_size + research.validation_size]
    test = sample[research.train_size + research.validation_size :]
    return train, validation, test


def _expanding_beta(asset_a: pd.Series, asset_b: pd.Series, min_periods: int) -> pd.Series:
    mean_a = asset_a.expanding(min_periods=min_periods).mean()
    mean_b = asset_b.expanding(min_periods=min_periods).mean()
    covariance = ((asset_a - mean_a) * (asset_b - mean_b)).expanding(min_periods=min_periods).mean()
    variance = ((asset_b - mean_b) ** 2).expanding(min_periods=min_periods).mean()
    beta = covariance / variance.replace(0.0, np.nan)
    return beta.ffill().fillna(1.0)


def _build_research_frame(
    asset_a: pd.Series,
    asset_b: pd.Series,
    strategy: StrategyConfig,
    research: ResearchConfig,
) -> pd.DataFrame:
    if strategy.beta_model == "rolling_ols" and research.beta_mode == "expanding":
        beta = _expanding_beta(asset_a, asset_b, min_periods=max(20, research.beta_window // 4))
        spread = asset_a - beta * asset_b
        zscore = rolling_zscore(spread, strategy.zscore_lookback)
        signals = generate_stateful_signals(zscore, strategy)
        frame = pd.concat(
            [
                asset_a.rename("price_a"),
                asset_b.rename("price_b"),
                beta.rename("beta"),
                spread.rename("spread"),
                zscore.rename("zscore"),
                signals,
            ],
            axis=1,
        )
        frame["model"] = "expanding_ols"
        return frame
    return build_pair_signal_frame(asset_a, asset_b, strategy, research.beta_window)


def _benjamini_hochberg(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    ordered = values.sort_values()
    adjusted = pd.Series(index=ordered.index, dtype=float)
    total = len(ordered)
    running = 1.0
    for rank, (idx, value) in enumerate(reversed(list(ordered.items())), start=1):
        denominator = total - rank + 1
        candidate = min(running, float(value) * total / denominator)
        adjusted.loc[idx] = candidate
        running = candidate
    return adjusted.reindex(values.index).fillna(1.0)


def _rank_metric(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    cleaned = series.replace([np.inf, -np.inf], np.nan)
    if cleaned.dropna().nunique() <= 1:
        return pd.Series(0.5, index=series.index)
    return cleaned.rank(pct=True, ascending=higher_is_better).fillna(0.0)


def prescreen_pairs(
    close: pd.DataFrame,
    metadata: pd.DataFrame,
    research: ResearchConfig,
) -> pd.DataFrame:
    returns = close.pct_change()
    symbol_meta = metadata.set_index("symbol")
    candidates = []
    for symbol_a, symbol_b in combinations(close.columns, 2):
        if symbol_a not in symbol_meta.index or symbol_b not in symbol_meta.index:
            continue
        meta_a = symbol_meta.loc[symbol_a]
        meta_b = symbol_meta.loc[symbol_b]
        if research.require_same_sector and meta_a.get("sector") != meta_b.get("sector"):
            continue
        if research.require_same_industry and meta_a.get("industry") != meta_b.get("industry"):
            continue
        correlation = returns[symbol_a].corr(returns[symbol_b])
        if pd.isna(correlation) or correlation < research.min_return_correlation:
            continue
        liquidity_values = [
            meta_a.get("avg_dollar_volume", np.nan),
            meta_b.get("avg_dollar_volume", np.nan),
        ]
        finite_liquidity = [float(value) for value in liquidity_values if pd.notna(value)]
        liquidity_score = float(np.mean(finite_liquidity)) if finite_liquidity else 0.0
        candidates.append(
            {
                "symbol_a": symbol_a,
                "symbol_b": symbol_b,
                "sector": meta_a.get("sector"),
                "industry_a": meta_a.get("industry"),
                "industry_b": meta_b.get("industry"),
                "return_correlation": float(correlation),
                "liquidity_score": liquidity_score,
            }
        )
    frame = pd.DataFrame(candidates)
    if frame.empty:
        return frame
    frame = frame.sort_values(
        by=["return_correlation", "liquidity_score", "symbol_a", "symbol_b"],
        ascending=[False, False, True, True],
    )
    return frame.head(research.max_candidate_pairs).reset_index(drop=True)


@dataclass
class PairResearchReport:
    run_id: str
    rankings: pd.DataFrame
    diagnostics: dict[str, pd.DataFrame]
    summary: dict[str, object]
    artifact_paths: dict[str, str] = field(default_factory=dict)

    def top_pairs(self, count: int = 10) -> pd.DataFrame:
        return self.rankings.head(count).copy()


def research_pairs(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    metadata: pd.DataFrame,
    research: ResearchConfig,
    strategy: StrategyConfig,
) -> PairResearchReport:
    del volume
    candidates = prescreen_pairs(close, metadata, research)
    results: list[dict[str, object]] = []
    diagnostics: dict[str, pd.DataFrame] = {}
    window_summary: Optional[dict[str, str]] = None

    for row in candidates.itertuples(index=False):
        pair_close = close[[row.symbol_a, row.symbol_b]].dropna()
        try:
            train_idx, validation_idx, test_idx = _split_windows(pair_close.index, research)
        except ValueError:
            continue

        frame = _build_research_frame(pair_close[row.symbol_a], pair_close[row.symbol_b], strategy, research)
        frame["pair_return"] = _pair_returns(frame, strategy)
        frame["window"] = "ignored"
        frame.loc[train_idx, "window"] = "train"
        frame.loc[validation_idx, "window"] = "validation"
        frame.loc[test_idx, "window"] = "test"
        frame["pair_id"] = f"{row.symbol_a}__{row.symbol_b}"

        train_close = pair_close.loc[train_idx]
        train_pvalue = _cointegration_pvalue(train_close[row.symbol_a], train_close[row.symbol_b])
        validation_returns = frame.loc[validation_idx, "pair_return"]
        test_returns = frame.loc[test_idx, "pair_return"]
        half_life = _half_life(frame.loc[train_idx, "spread"])
        turnover = float(frame["side"].diff().abs().loc[test_idx].fillna(0.0).mean())
        stability = float(1.0 / (1.0 + frame.loc[test_idx, "beta"].diff().abs().mean(skipna=True)))
        max_drawdown = _max_drawdown(test_returns)
        result = {
            "pair_id": f"{row.symbol_a}__{row.symbol_b}",
            "symbol_a": row.symbol_a,
            "symbol_b": row.symbol_b,
            "sector": row.sector,
            "return_correlation": row.return_correlation,
            "train_pvalue": train_pvalue,
            "half_life": half_life,
            "validation_sharpe": _annualized_sharpe(validation_returns),
            "test_sharpe": _annualized_sharpe(test_returns),
            "stability": stability,
            "drawdown": max_drawdown,
            "turnover": turnover,
            "validation_total_return": float((1.0 + validation_returns).prod() - 1.0),
            "test_total_return": float((1.0 + test_returns).prod() - 1.0),
            "model": frame["model"].iloc[-1],
        }
        results.append(result)
        diagnostics[result["pair_id"]] = frame

        if window_summary is None:
            window_summary = {
                "train_start": str(train_idx[0].date()),
                "train_end": str(train_idx[-1].date()),
                "validation_start": str(validation_idx[0].date()),
                "validation_end": str(validation_idx[-1].date()),
                "test_start": str(test_idx[0].date()),
                "test_end": str(test_idx[-1].date()),
            }

    rankings = pd.DataFrame(results)
    if rankings.empty:
        summary = {
            "candidates_considered": int(len(candidates)),
            "pairs_after_fdr": 0,
            "message": "No valid pairs survived the configured research pipeline.",
        }
        if window_summary:
            summary.update(window_summary)
        return PairResearchReport(run_id="", rankings=rankings, diagnostics=diagnostics, summary=summary)

    rankings["fdr_pvalue"] = _benjamini_hochberg(rankings["train_pvalue"])
    rankings = rankings[rankings["fdr_pvalue"] <= research.fdr_alpha].copy()
    if rankings.empty:
        summary = {
            "candidates_considered": int(len(candidates)),
            "pairs_after_fdr": 0,
            "message": "All candidate pairs failed the multiple-testing threshold.",
        }
        if window_summary:
            summary.update(window_summary)
        return PairResearchReport(run_id="", rankings=rankings, diagnostics=diagnostics, summary=summary)

    weights = research.ranking_weights
    score = pd.Series(0.0, index=rankings.index)
    for metric, weight in weights.items():
        if metric not in rankings.columns:
            continue
        higher_is_better = metric not in {"half_life", "drawdown", "turnover"}
        score += _rank_metric(rankings[metric], higher_is_better=higher_is_better) * float(weight)
    rankings["ranking_score"] = score
    rankings = rankings.sort_values(
        by=["ranking_score", "test_sharpe", "validation_sharpe", "fdr_pvalue"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    summary = {
        "candidates_considered": int(len(candidates)),
        "pairs_evaluated": int(len(results)),
        "pairs_after_fdr": int(len(rankings)),
        "fdr_alpha": research.fdr_alpha,
        "beta_mode": research.beta_mode,
    }
    if window_summary:
        summary.update(window_summary)

    top_pair_ids = rankings["pair_id"].head(10).tolist()
    top_diagnostics = {pair_id: diagnostics[pair_id] for pair_id in top_pair_ids if pair_id in diagnostics}
    return PairResearchReport(
        run_id="",
        rankings=rankings,
        diagnostics=top_diagnostics,
        summary=summary,
    )
