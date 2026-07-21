from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from twelvedata import TDClient

from config.settings import settings
from data import database
from models.schemas import OHLCVPoint, Ticker


DEFAULT_INTERVAL = "15min"
DEFAULT_LOOKBACK_DAYS = 10
DEFAULT_TIMEZONE = "UTC"


class TwelveDataError(RuntimeError):
    """Raised when Twelve Data prices cannot be fetched or parsed."""


class MissingTwelveDataConfigError(TwelveDataError):
    """Raised when TWELVE_DATA_API_KEY is missing."""


class TimeSeriesRequest(Protocol):
    def as_json(self) -> tuple[dict[str, Any], ...]: ...


class TimeSeriesClient(Protocol):
    def time_series(self, **kwargs: Any) -> TimeSeriesRequest: ...


def get_twelve_data_api_key() -> str:
    if settings.twelve_data_api_key is None:
        raise MissingTwelveDataConfigError(
            "TWELVE_DATA_API_KEY must be set before fetching prices."
        )
    return settings.twelve_data_api_key.get_secret_value()


def _ticker_symbol(ticker: str) -> str:
    return ticker.upper()


def _create_client() -> TDClient:
    return TDClient(apikey=get_twelve_data_api_key())


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise TwelveDataError("Twelve Data bar field 'datetime' must be a string.")

    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise TwelveDataError(f"Twelve Data returned an invalid datetime: {value}") from exc

    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _bar_to_point(ticker: str, bar: dict[str, Any]) -> OHLCVPoint:
    required_fields = ("datetime", "open", "high", "low", "close", "volume")
    missing = next((field for field in required_fields if field not in bar), None)
    if missing is not None:
        raise TwelveDataError(f"Twelve Data bar is missing expected field: {missing}")

    try:
        return OHLCVPoint(
            ticker=_ticker_symbol(ticker),
            timestamp=_parse_timestamp(bar["datetime"]),
            open=float(bar["open"]),
            high=float(bar["high"]),
            low=float(bar["low"]),
            close=float(bar["close"]),
            volume=float(bar["volume"]),
        )
    except (TypeError, ValueError) as exc:
        raise TwelveDataError("Twelve Data returned a non-numeric OHLCV value.") from exc


def parse_ohlcv_points(
    ticker: str,
    bars: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    limit: int,
) -> list[OHLCVPoint]:
    if not isinstance(bars, (tuple, list)):
        raise TwelveDataError("Twelve Data response bars must be a sequence.")

    points = [_bar_to_point(ticker, bar) for bar in bars]
    points.sort(key=lambda point: point.timestamp)
    return points[-limit:]


def fetch_intraday_ohlcv(
    ticker: str,
    limit: int = settings.max_ticker_datapoints,
    *,
    to_date: datetime | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    client: TimeSeriesClient | None = None,
) -> list[OHLCVPoint]:
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be greater than zero")

    end = (to_date or datetime.now(timezone.utc)).astimezone(timezone.utc)
    start = end - timedelta(days=lookback_days)

    try:
        request = (client or _create_client()).time_series(
            symbol=_ticker_symbol(ticker),
            interval=DEFAULT_INTERVAL,
            outputsize=limit,
            timezone=DEFAULT_TIMEZONE,
            order="DESC",
            start_date=start.strftime("%Y-%m-%d %H:%M:%S"),
            end_date=end.strftime("%Y-%m-%d %H:%M:%S"),
        )
        bars = request.as_json()
    except MissingTwelveDataConfigError:
        raise
    except Exception as exc:
        raise TwelveDataError(f"Twelve Data request failed: {exc}") from exc

    return parse_ohlcv_points(ticker, bars, limit)


def refresh_ticker_data(
    ticker: str,
    limit: int = settings.max_ticker_datapoints,
    *,
    next_fetch_minutes: int = settings.price_fetch_interval_minutes,
) -> list[OHLCVPoint]:
    points = fetch_intraday_ohlcv(ticker, limit=limit)
    if not points:
        return []

    now = datetime.now(timezone.utc)
    latest = points[-1]
    database.upsert_ticker(
        Ticker(
            ticker=_ticker_symbol(ticker),
            last_fetched=now,
            next_fetch_time=now + timedelta(minutes=next_fetch_minutes),
            current_price=latest.close,
        )
    )

    database.upsert_ticker_data(points)
    database.delete_old_ticker_data(_ticker_symbol(ticker), keep=limit)
    return points
