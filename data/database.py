from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional, TypeVar

from pydantic import BaseModel
from supabase import Client, create_client

from config.settings import settings
from models.schemas import (
    AgentOutput,
    Alert,
    OHLCVPoint,
    Subscription,
    Ticker,
    User,
)


class DatabaseError(RuntimeError):
    """Raised when a database operation fails or returns unexpected data."""


class MissingSupabaseConfigError(DatabaseError):
    """Raised when Supabase credentials are missing from settings."""


ModelT = TypeVar("ModelT", bound=BaseModel)


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    if settings.supabase_url is None or settings.supabase_key is None:
        raise MissingSupabaseConfigError(
            "SUPABASE_URL and SUPABASE_KEY must be set before using the database."
        )

    return create_client(
        str(settings.supabase_url).rstrip("/"),
        settings.supabase_key.get_secret_value(),
    )


def _client(client: Optional[Client] = None) -> Client:
    return client or get_supabase_client()


def _model_payload(model: BaseModel, *, exclude_none: bool = True) -> dict[str, Any]:
    return model.model_dump(mode="json", exclude_none=exclude_none)


def _single_row(data: Any) -> Optional[dict[str, Any]]:
    if data is None:
        return None
    if isinstance(data, list):
        if not data:
            return None
        if len(data) > 1:
            raise DatabaseError(f"Expected one row, received {len(data)} rows.")
        return data[0]
    if isinstance(data, dict):
        return data
    raise DatabaseError(f"Unexpected Supabase response data type: {type(data)!r}")


def _require_single_row(data: Any, operation: str) -> dict[str, Any]:
    row = _single_row(data)
    if row is None:
        raise DatabaseError(f"{operation} did not return a row.")
    return row


def _parse_model(model_type: type[ModelT], row: dict[str, Any]) -> ModelT:
    return model_type.model_validate(row)


def _parse_model_list(model_type: type[ModelT], rows: Any) -> list[ModelT]:
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise DatabaseError(f"Expected a list of rows, received {type(rows)!r}.")
    return [_parse_model(model_type, row) for row in rows]


def _execute(builder: Any) -> Any:
    return builder.execute()


def _ticker_symbol(ticker: str) -> str:
    return ticker.upper()


def create_user(user: User, *, client: Optional[Client] = None) -> User:
    response = _execute(_client(client).table("users").insert(_model_payload(user)))
    row = _require_single_row(response.data, "create_user")
    return _parse_model(User, row)


def upsert_user(user: User, *, client: Optional[Client] = None) -> User:
    response = _execute(
        _client(client)
        .table("users")
        .upsert(_model_payload(user), on_conflict="user_id")
    )
    row = _require_single_row(response.data, "upsert_user")
    return _parse_model(User, row)


def get_user(user_id: str, *, client: Optional[Client] = None) -> Optional[User]:
    response = _execute(
        _client(client)
        .table("users")
        .select("*")
        .eq("user_id", user_id)
        .limit(1)
    )
    row = _single_row(response.data)
    return _parse_model(User, row) if row else None


def upsert_ticker(ticker: Ticker, *, client: Optional[Client] = None) -> Ticker:
    payload = _model_payload(ticker)
    payload["ticker"] = _ticker_symbol(payload["ticker"])

    response = _execute(
        _client(client)
        .table("tickers")
        .upsert(payload, on_conflict="ticker")
    )
    row = _require_single_row(response.data, "upsert_ticker")
    return _parse_model(Ticker, row)


def get_ticker(ticker: str, *, client: Optional[Client] = None) -> Optional[Ticker]:
    response = _execute(
        _client(client)
        .table("tickers")
        .select("*")
        .eq("ticker", _ticker_symbol(ticker))
        .limit(1)
    )
    row = _single_row(response.data)
    return _parse_model(Ticker, row) if row else None


def list_tickers(*, client: Optional[Client] = None) -> list[Ticker]:
    response = _execute(
        _client(client)
        .table("tickers")
        .select("*")
        .order("ticker")
    )
    return _parse_model_list(Ticker, response.data)


def upsert_subscription(
    subscription: Subscription, *, client: Optional[Client] = None
) -> Subscription:
    payload = _model_payload(subscription)
    payload["ticker"] = _ticker_symbol(payload["ticker"])

    response = _execute(
        _client(client)
        .table("subscriptions")
        .upsert(payload, on_conflict="user_id,ticker")
    )
    row = _require_single_row(response.data, "upsert_subscription")
    return _parse_model(Subscription, row)


def get_subscription(
    user_id: str, ticker: str, *, client: Optional[Client] = None
) -> Optional[Subscription]:
    response = _execute(
        _client(client)
        .table("subscriptions")
        .select("*")
        .eq("user_id", user_id)
        .eq("ticker", _ticker_symbol(ticker))
        .limit(1)
    )
    row = _single_row(response.data)
    return _parse_model(Subscription, row) if row else None


def list_subscriptions(*, client: Optional[Client] = None) -> list[Subscription]:
    response = _execute(
        _client(client)
        .table("subscriptions")
        .select("*")
        .order("ticker")
    )
    return _parse_model_list(Subscription, response.data)


def list_subscriptions_for_user(
    user_id: str, *, client: Optional[Client] = None
) -> list[Subscription]:
    response = _execute(
        _client(client)
        .table("subscriptions")
        .select("*")
        .eq("user_id", user_id)
        .order("ticker")
    )
    return _parse_model_list(Subscription, response.data)


def list_subscriptions_for_ticker(
    ticker: str, *, client: Optional[Client] = None
) -> list[Subscription]:
    response = _execute(
        _client(client)
        .table("subscriptions")
        .select("*")
        .eq("ticker", _ticker_symbol(ticker))
        .order("user_id")
    )
    return _parse_model_list(Subscription, response.data)


def delete_subscription(
    user_id: str, ticker: str, *, client: Optional[Client] = None
) -> None:
    _execute(
        _client(client)
        .table("subscriptions")
        .delete()
        .eq("user_id", user_id)
        .eq("ticker", _ticker_symbol(ticker))
    )


def insert_ticker_data(point: OHLCVPoint, *, client: Optional[Client] = None) -> OHLCVPoint:
    payload = _model_payload(point)
    payload["ticker"] = _ticker_symbol(payload["ticker"])

    response = _execute(_client(client).table("ticker_data").upsert(payload))
    row = _require_single_row(response.data, "insert_ticker_data")
    return _parse_model(OHLCVPoint, row)


def upsert_ticker_data(
    points: list[OHLCVPoint], *, client: Optional[Client] = None
) -> list[OHLCVPoint]:
    if not points:
        return []

    payloads = []
    for point in points:
        payload = _model_payload(point)
        payload["ticker"] = _ticker_symbol(payload["ticker"])
        payloads.append(payload)

    response = _execute(
        _client(client)
        .table("ticker_data")
        .upsert(payloads, on_conflict="ticker,timestamp")
    )
    return _parse_model_list(OHLCVPoint, response.data)


def get_latest_ticker_data(
    ticker: str, limit: int = settings.max_ticker_datapoints, *, client: Optional[Client] = None
) -> list[OHLCVPoint]:
    response = _execute(
        _client(client)
        .table("ticker_data")
        .select("*")
        .eq("ticker", _ticker_symbol(ticker))
        .order("timestamp", desc=True)
        .limit(limit)
    )
    points = _parse_model_list(OHLCVPoint, response.data)
    return list(reversed(points))


def delete_old_ticker_data(
    ticker: str, keep: int = settings.max_ticker_datapoints, *, client: Optional[Client] = None
) -> None:
    response = _execute(
        _client(client)
        .table("ticker_data")
        .select("timestamp")
        .eq("ticker", _ticker_symbol(ticker))
        .order("timestamp", desc=True)
        .range(keep, keep)
    )
    cutoff_row = _single_row(response.data)
    if cutoff_row is None:
        return

    _execute(
        _client(client)
        .table("ticker_data")
        .delete()
        .eq("ticker", _ticker_symbol(ticker))
        .lte("timestamp", cutoff_row["timestamp"])
    )


def _agent_output_payload(output: AgentOutput) -> dict[str, Any]:
    payload = _model_payload(output)
    allowed_columns = {
        "ticker",
        "user_id",
        "event_type",
        "summary",
        "recommendation",
        "confidence",
        "timestamp",
        "price_at_update",
        "searched_web",
    }
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in allowed_columns and value is not None
    }
    row = {key: value for key, value in payload.items() if key in allowed_columns}
    row["ticker"] = _ticker_symbol(row["ticker"])
    if metadata:
        row["metadata"] = metadata
    return row


def insert_agent_output(
    output: AgentOutput, *, client: Optional[Client] = None
) -> AgentOutput:
    response = _execute(
        _client(client)
        .table("updates")
        .insert(_agent_output_payload(output))
    )
    row = _require_single_row(response.data, "insert_agent_output")
    return _parse_model(AgentOutput, row)


def list_updates_for_user(
    user_id: str, limit: int = 50, *, client: Optional[Client] = None
) -> list[AgentOutput]:
    response = _execute(
        _client(client)
        .table("updates")
        .select("*")
        .eq("user_id", user_id)
        .order("timestamp", desc=True)
        .limit(limit)
    )
    return _parse_model_list(AgentOutput, response.data)


def list_updates_for_user_ticker(
    user_id: str, ticker: str, limit: int = 20, *, client: Optional[Client] = None
) -> list[AgentOutput]:
    response = _execute(
        _client(client)
        .table("updates")
        .select("*")
        .eq("user_id", user_id)
        .eq("ticker", _ticker_symbol(ticker))
        .order("timestamp", desc=True)
        .limit(limit)
    )
    return _parse_model_list(AgentOutput, response.data)


def insert_alert(alert: Alert, *, client: Optional[Client] = None) -> Alert:
    payload = _model_payload(alert)
    payload["ticker"] = _ticker_symbol(payload["ticker"])

    response = _execute(_client(client).table("alerts").insert(payload))
    row = _require_single_row(response.data, "insert_alert")
    return _parse_model(Alert, row)


def list_alerts_for_user(
    user_id: str, limit: int = 50, *, client: Optional[Client] = None
) -> list[Alert]:
    response = _execute(
        _client(client)
        .table("alerts")
        .select("*")
        .eq("user_id", user_id)
        .order("timestamp", desc=True)
        .limit(limit)
    )
    return _parse_model_list(Alert, response.data)


def list_alerts_for_user_ticker(
    user_id: str, ticker: str, limit: int = 20, *, client: Optional[Client] = None
) -> list[Alert]:
    response = _execute(
        _client(client)
        .table("alerts")
        .select("*")
        .eq("user_id", user_id)
        .eq("ticker", _ticker_symbol(ticker))
        .order("timestamp", desc=True)
        .limit(limit)
    )
    return _parse_model_list(Alert, response.data)
