from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import json


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _coerce_date(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def _coerce_float_mapping(values: Mapping[str, Any]) -> Dict[str, float]:
    return {str(key): float(value) for key, value in values.items()}


def load_mapping_file(path: str | Path) -> Dict[str, Any]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded or {}
    except ModuleNotFoundError:
        return json.loads(text)


@dataclass
class UniverseSpec:
    name: str = "core_universe"
    universe_type: str = "equities"
    min_history_days: int = 756
    min_average_dollar_volume: float = 0.0
    min_price: float = 5.0
    allowed_sectors: list[str] = field(default_factory=list)
    excluded_symbols: list[str] = field(default_factory=list)
    included_symbols: list[str] = field(default_factory=list)
    etf_symbols: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "UniverseSpec":
        defaults = cls()
        payload = dict(data or {})
        return cls(
            name=str(payload.get("name", defaults.name)),
            universe_type=str(payload.get("universe_type", defaults.universe_type)),
            min_history_days=int(payload.get("min_history_days", defaults.min_history_days)),
            min_average_dollar_volume=float(
                payload.get(
                    "min_average_dollar_volume",
                    defaults.min_average_dollar_volume,
                )
            ),
            min_price=float(payload.get("min_price", defaults.min_price)),
            allowed_sectors=[str(item) for item in payload.get("allowed_sectors", [])],
            excluded_symbols=[str(item) for item in payload.get("excluded_symbols", [])],
            included_symbols=[str(item) for item in payload.get("included_symbols", [])],
            etf_symbols=[str(item) for item in payload.get("etf_symbols", [])],
        )


@dataclass
class ResearchConfig:
    start_date: Optional[str] = "2012-01-01"
    end_date: Optional[str] = "2024-01-01"
    train_size: int = 504
    validation_size: int = 252
    test_size: int = 252
    require_same_sector: bool = True
    require_same_industry: bool = False
    min_return_correlation: float = 0.25
    max_candidate_pairs: int = 200
    beta_window: int = 126
    beta_mode: str = "rolling"
    fdr_alpha: float = 0.05
    ranking_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "validation_sharpe": 0.30,
            "test_sharpe": 0.35,
            "half_life": 0.15,
            "stability": 0.10,
            "drawdown": 0.05,
            "turnover": 0.05,
        }
    )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ResearchConfig":
        defaults = cls()
        payload = dict(data or {})
        ranking_weights = payload.get("ranking_weights")
        return cls(
            start_date=_coerce_date(payload.get("start_date", defaults.start_date)),
            end_date=_coerce_date(payload.get("end_date", defaults.end_date)),
            train_size=int(payload.get("train_size", defaults.train_size)),
            validation_size=int(payload.get("validation_size", defaults.validation_size)),
            test_size=int(payload.get("test_size", defaults.test_size)),
            require_same_sector=bool(
                payload.get("require_same_sector", defaults.require_same_sector)
            ),
            require_same_industry=bool(
                payload.get("require_same_industry", defaults.require_same_industry)
            ),
            min_return_correlation=float(
                payload.get("min_return_correlation", defaults.min_return_correlation)
            ),
            max_candidate_pairs=int(
                payload.get("max_candidate_pairs", defaults.max_candidate_pairs)
            ),
            beta_window=int(payload.get("beta_window", defaults.beta_window)),
            beta_mode=str(payload.get("beta_mode", defaults.beta_mode)),
            fdr_alpha=float(payload.get("fdr_alpha", defaults.fdr_alpha)),
            ranking_weights=_coerce_float_mapping(ranking_weights or defaults.ranking_weights),
        )


@dataclass
class StrategyConfig:
    beta_model: str = "rolling_ols"
    zscore_lookback: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5
    max_holding_days: int = 30
    pair_allocation_pct: float = 0.10
    max_gross_exposure_pct: float = 0.75
    max_net_exposure_pct: float = 0.10
    max_active_pairs: int = 5
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    borrow_bps_annual: float = 30.0
    max_pair_gross_pct: float = 0.20
    kalman_delta: float = 0.0001
    kalman_observation_variance: float = 0.001

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "StrategyConfig":
        defaults = cls()
        payload = dict(data or {})
        return cls(
            beta_model=str(payload.get("beta_model", defaults.beta_model)),
            zscore_lookback=int(payload.get("zscore_lookback", defaults.zscore_lookback)),
            entry_z=float(payload.get("entry_z", defaults.entry_z)),
            exit_z=float(payload.get("exit_z", defaults.exit_z)),
            stop_z=float(payload.get("stop_z", defaults.stop_z)),
            max_holding_days=int(
                payload.get("max_holding_days", defaults.max_holding_days)
            ),
            pair_allocation_pct=float(
                payload.get("pair_allocation_pct", defaults.pair_allocation_pct)
            ),
            max_gross_exposure_pct=float(
                payload.get("max_gross_exposure_pct", defaults.max_gross_exposure_pct)
            ),
            max_net_exposure_pct=float(
                payload.get("max_net_exposure_pct", defaults.max_net_exposure_pct)
            ),
            max_active_pairs=int(
                payload.get("max_active_pairs", defaults.max_active_pairs)
            ),
            commission_bps=float(payload.get("commission_bps", defaults.commission_bps)),
            slippage_bps=float(payload.get("slippage_bps", defaults.slippage_bps)),
            borrow_bps_annual=float(
                payload.get("borrow_bps_annual", defaults.borrow_bps_annual)
            ),
            max_pair_gross_pct=float(
                payload.get("max_pair_gross_pct", defaults.max_pair_gross_pct)
            ),
            kalman_delta=float(payload.get("kalman_delta", defaults.kalman_delta)),
            kalman_observation_variance=float(
                payload.get(
                    "kalman_observation_variance",
                    defaults.kalman_observation_variance,
                )
            ),
        )


@dataclass
class ProjectConfig:
    data_root: str = "data"
    artifacts_root: str = "artifacts"
    initial_capital: float = 1_000_000.0
    metadata_file: str = "flat-ui__data-Fri Jul 04 2025.csv"
    legacy_prices_file: Optional[str] = "all data from 2000-01-01 to 2024-01-01.csv"
    etf_metadata_file: Optional[str] = None
    universe: UniverseSpec = field(default_factory=UniverseSpec)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ProjectConfig":
        payload = dict(data or {})
        defaults = asdict(cls())
        merged = _deep_merge(defaults, payload)
        return cls(
            data_root=str(merged["data_root"]),
            artifacts_root=str(merged["artifacts_root"]),
            initial_capital=float(merged["initial_capital"]),
            metadata_file=str(merged["metadata_file"]),
            legacy_prices_file=merged.get("legacy_prices_file"),
            etf_metadata_file=merged.get("etf_metadata_file"),
            universe=UniverseSpec.from_mapping(merged.get("universe")),
            research=ResearchConfig.from_mapping(merged.get("research")),
            strategy=StrategyConfig.from_mapping(merged.get("strategy")),
        )


def load_project_config(path: str | Path | None) -> ProjectConfig:
    if path is None:
        return ProjectConfig()
    return ProjectConfig.from_mapping(load_mapping_file(path))
