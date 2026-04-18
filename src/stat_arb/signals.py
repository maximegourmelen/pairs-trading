from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .config import StrategyConfig


class SpreadModel(Protocol):
    name: str

    def estimate_beta(self, asset_a: pd.Series, asset_b: pd.Series) -> pd.Series:
        ...


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0.0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


@dataclass
class RollingOLSModel:
    lookback: int
    name: str = "rolling_ols"

    def estimate_beta(self, asset_a: pd.Series, asset_b: pd.Series) -> pd.Series:
        mean_a = asset_a.rolling(self.lookback, min_periods=max(20, self.lookback // 4)).mean()
        mean_b = asset_b.rolling(self.lookback, min_periods=max(20, self.lookback // 4)).mean()
        covariance = ((asset_a - mean_a) * (asset_b - mean_b)).rolling(
            self.lookback,
            min_periods=max(20, self.lookback // 4),
        ).mean()
        variance = ((asset_b - mean_b) ** 2).rolling(
            self.lookback,
            min_periods=max(20, self.lookback // 4),
        ).mean()
        beta = _safe_divide(covariance, variance)
        return beta.ffill().fillna(1.0)


@dataclass
class KalmanBetaModel:
    delta: float = 0.0001
    observation_variance: float = 0.001
    name: str = "kalman_beta"

    def estimate_beta(self, asset_a: pd.Series, asset_b: pd.Series) -> pd.Series:
        beta = np.zeros(len(asset_a))
        covariance = np.ones(len(asset_a))
        beta[0] = 1.0
        covariance[0] = 1.0
        process_variance = self.delta / (1.0 - self.delta)

        for idx in range(1, len(asset_a)):
            predicted_beta = beta[idx - 1]
            predicted_cov = covariance[idx - 1] + process_variance
            observation = asset_b.iloc[idx]
            innovation_variance = (observation ** 2) * predicted_cov + self.observation_variance
            if innovation_variance <= 0:
                beta[idx] = predicted_beta
                covariance[idx] = predicted_cov
                continue
            innovation = asset_a.iloc[idx] - predicted_beta * observation
            kalman_gain = predicted_cov * observation / innovation_variance
            beta[idx] = predicted_beta + kalman_gain * innovation
            covariance[idx] = max((1.0 - kalman_gain * observation) * predicted_cov, 1e-6)

        return pd.Series(beta, index=asset_a.index, name="beta").ffill().fillna(1.0)


def build_spread_model(strategy: StrategyConfig, lookback: int) -> SpreadModel:
    if strategy.beta_model == "kalman_beta":
        return KalmanBetaModel(
            delta=strategy.kalman_delta,
            observation_variance=strategy.kalman_observation_variance,
        )
    return RollingOLSModel(lookback=lookback)


def rolling_zscore(series: pd.Series, lookback: int, min_periods: int | None = None) -> pd.Series:
    window = lookback
    minimum = min_periods or max(10, lookback // 4)
    rolling_mean = series.rolling(window, min_periods=minimum).mean()
    rolling_std = series.rolling(window, min_periods=minimum).std(ddof=0)
    zscore = (series - rolling_mean) / rolling_std.replace(0.0, np.nan)
    return zscore.replace([np.inf, -np.inf], np.nan)


def build_pair_signal_frame(
    asset_a: pd.Series,
    asset_b: pd.Series,
    strategy: StrategyConfig,
    beta_lookback: int,
) -> pd.DataFrame:
    model = build_spread_model(strategy, beta_lookback)
    beta = model.estimate_beta(asset_a, asset_b)
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
    frame["model"] = model.name
    return frame


def generate_stateful_signals(zscore: pd.Series, strategy: StrategyConfig) -> pd.DataFrame:
    columns = {
        "side": [],
        "entry_z": [],
        "holding_days": [],
        "exit_reason": [],
    }
    active_side = 0
    holding_days = 0
    entry_z = np.nan

    for value in zscore.fillna(0.0):
        exit_reason = ""
        if active_side == 0:
            if value <= -strategy.entry_z:
                active_side = 1
                holding_days = 1
                entry_z = value
            elif value >= strategy.entry_z:
                active_side = -1
                holding_days = 1
                entry_z = value
        else:
            holding_days += 1
            should_exit = abs(value) <= strategy.exit_z
            should_stop = (
                active_side == 1 and value <= -strategy.stop_z
            ) or (
                active_side == -1 and value >= strategy.stop_z
            )
            should_time_stop = holding_days > strategy.max_holding_days
            if should_exit or should_stop or should_time_stop:
                if should_exit:
                    exit_reason = "mean_reversion_exit"
                elif should_stop:
                    exit_reason = "zscore_stop"
                else:
                    exit_reason = "time_stop"
                active_side = 0
                holding_days = 0
                entry_z = np.nan

        columns["side"].append(active_side)
        columns["entry_z"].append(entry_z)
        columns["holding_days"].append(holding_days)
        columns["exit_reason"].append(exit_reason)

    frame = pd.DataFrame(columns, index=zscore.index)
    frame["position_a"] = frame["side"]
    frame["position_b"] = -frame["side"]
    return frame

