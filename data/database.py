from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Optional, TypeAlias, TypeVar

from pydantic import BaseModel
from pydantic import SecretStr
from supabase import Client, create_client

from config.settings import settings
from models.schemas import (
    AgentOutput,
    AgentType,
    Alert,
    AlertType,
    CrossPortfolioOutput,
    EventType,
    HypothesisOutput,
    OHLCVPoint,
    Subscription,
    Ticker,
    User,
)
from security.credentials import (
    CredentialCipher,
    UserApiKeyMetadata,
    get_credential_cipher,
    normalize_provider,
)


class DatabaseError(RuntimeError):
    """Raised when a database operation fails or returns unexpected data."""


class MissingSupabaseConfigError(DatabaseError):
    """Raised when Supabase credentials are missing from settings."""


ModelT = TypeVar("ModelT", bound=BaseModel)
StoredAgentOutput: TypeAlias = AgentOutput | HypothesisOutput | CrossPortfolioOutput

MAX_HISTORY_QUERY_LIMIT = 100
MAX_PORTFOLIO_UPDATE_SCAN = 500
PORTFOLIO_UPDATE_SCAN_PER_TICKER = 20

_AGENT_TYPE_BY_EVENT = {
    EventType.SCHEDULED_UPDATE: AgentType.SCHEDULED_REVIEW,
    EventType.SHARP_MOVE: AgentType.SHARP_MOVE,
    EventType.MOTIVE_CHECK: AgentType.MOTIVE,
}


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


@lru_cache(maxsize=1)
def get_supabase_admin_client() -> Client:
    if settings.supabase_url is None or settings.supabase_secret_key is None:
        raise MissingSupabaseConfigError(
            "SUPABASE_URL and SUPABASE_SECRET_KEY must be set before accessing credentials."
        )

    return create_client(
        str(settings.supabase_url).rstrip("/"),
        settings.supabase_secret_key.get_secret_value(),
    )


def _client(client: Optional[Client] = None) -> Client:
    return client or get_supabase_client()


def _credential_client(client: Optional[Client] = None) -> Client:
    return client or get_supabase_admin_client()


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


def _history_limit(limit: int, *, maximum: int = MAX_HISTORY_QUERY_LIMIT) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= maximum:
        raise ValueError(f"limit must be an integer between 1 and {maximum}.")
    return limit


def _since_timestamp(since: datetime | None) -> str | None:
    if since is None:
        return None
    if since.tzinfo is None or since.utcoffset() is None:
        raise ValueError("since must be timezone-aware.")
    return since.astimezone(timezone.utc).isoformat()


def _parse_datetime(value: Any, field_name: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise DatabaseError(f"Invalid {field_name} timestamp returned by database.") from exc
    raise DatabaseError(f"Missing or invalid {field_name} timestamp returned by database.")


def _credential_metadata(row: dict[str, Any]) -> UserApiKeyMetadata:
    validated = row.get("last_validated_at")
    return UserApiKeyMetadata(
        user_id=row["user_id"],
        provider=normalize_provider(row["provider"]),
        created_at=_parse_datetime(row.get("created_at"), "created_at"),
        updated_at=_parse_datetime(row.get("updated_at"), "updated_at"),
        last_validated_at=(
            _parse_datetime(validated, "last_validated_at") if validated is not None else None
        ),
    )


def store_user_api_key(
    user_id: str,
    provider: str,
    api_key: SecretStr | str,
    *,
    client: Optional[Client] = None,
    cipher: CredentialCipher | None = None,
) -> UserApiKeyMetadata:
    normalized_provider = normalize_provider(provider)
    protector = cipher or get_credential_cipher()
    encrypted_api_key = protector.encrypt(
        api_key,
        user_id=user_id,
        provider=normalized_provider,
    )
    payload = {
        "user_id": user_id,
        "provider": normalized_provider,
        "encrypted_api_key": encrypted_api_key,
        "encryption_version": 1,
        "last_validated_at": None,
    }
    response = _execute(
        _credential_client(client)
        .table("user_api_keys")
        .upsert(payload, on_conflict="user_id,provider")
    )
    row = _require_single_row(response.data, "store_user_api_key")
    return _credential_metadata(row)


def resolve_user_api_key(
    user_id: str,
    provider: str,
    *,
    client: Optional[Client] = None,
    cipher: CredentialCipher | None = None,
) -> SecretStr | None:
    normalized_provider = normalize_provider(provider)
    response = _execute(
        _credential_client(client)
        .table("user_api_keys")
        .select("user_id,provider,encrypted_api_key,encryption_version")
        .eq("user_id", user_id)
        .eq("provider", normalized_provider)
        .limit(1)
    )
    row = _single_row(response.data)
    if row is None:
        return None
    if row.get("encryption_version") != 1:
        raise DatabaseError("Stored credential uses an unsupported encryption version.")
    protector = cipher or get_credential_cipher()
    return protector.decrypt(
        row["encrypted_api_key"],
        user_id=user_id,
        provider=normalized_provider,
    )


def list_user_api_key_metadata(
    user_id: str,
    *,
    client: Optional[Client] = None,
) -> list[UserApiKeyMetadata]:
    response = _execute(
        _credential_client(client)
        .table("user_api_keys")
        .select("user_id,provider,created_at,updated_at,last_validated_at")
        .eq("user_id", user_id)
        .order("provider")
    )
    if response.data is None:
        return []
    if not isinstance(response.data, list):
        raise DatabaseError("Expected a list of credential metadata rows.")
    return [_credential_metadata(row) for row in response.data]


def mark_user_api_key_validated(
    user_id: str,
    provider: str,
    *,
    validated_at: datetime | None = None,
    client: Optional[Client] = None,
) -> UserApiKeyMetadata:
    normalized_provider = normalize_provider(provider)
    timestamp = validated_at or datetime.now(timezone.utc)
    response = _execute(
        _credential_client(client)
        .table("user_api_keys")
        .update({"last_validated_at": timestamp.isoformat()})
        .eq("user_id", user_id)
        .eq("provider", normalized_provider)
    )
    row = _require_single_row(response.data, "mark_user_api_key_validated")
    return _credential_metadata(row)


def delete_user_api_key(
    user_id: str,
    provider: str,
    *,
    client: Optional[Client] = None,
) -> None:
    _execute(
        _credential_client(client)
        .table("user_api_keys")
        .delete()
        .eq("user_id", user_id)
        .eq("provider", normalize_provider(provider))
    )


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


def delete_user(user_id: str, *, client: Optional[Client] = None) -> None:
    _execute(
        _client(client)
        .table("users")
        .delete()
        .eq("user_id", user_id)
    )


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


def _agent_output_payload(output: StoredAgentOutput) -> dict[str, Any]:
    payload = _model_payload(output)
    if isinstance(output, CrossPortfolioOutput):
        return {
            "user_id": payload["user_id"],
            "ticker": None,
            "timestamp": payload["timestamp"],
            "agent_type": AgentType.CROSS_PORTFOLIO.value,
            "event_type": None,
            "summary": payload["summary"],
            "recommendation": None,
            "confidence": None,
            "price_at_update": None,
            "searched_web": False,
            "metadata": {
                "correlations_flagged": payload["correlations_flagged"],
                "tickers_analyzed": [
                    _ticker_symbol(ticker) for ticker in payload["tickers_analyzed"]
                ],
            },
        }

    if isinstance(output, HypothesisOutput):
        agent_type = AgentType.HYPOTHESIS
    else:
        if output.event_type == EventType.HYPOTHESIS_SCAN:
            raise DatabaseError(
                "Hypothesis outputs must use the HypothesisOutput schema."
            )
        try:
            agent_type = _AGENT_TYPE_BY_EVENT[output.event_type]
        except KeyError as exc:
            raise DatabaseError(
                f"No agent type maps to event type {output.event_type.value}."
            ) from exc

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
    if not metadata.get("sources"):
        metadata.pop("sources", None)
    if metadata.get("web_search_requests") == 0:
        metadata.pop("web_search_requests", None)
    row = {key: value for key, value in payload.items() if key in allowed_columns}
    row["ticker"] = _ticker_symbol(row["ticker"])
    row["agent_type"] = agent_type.value
    row["metadata"] = metadata
    return row


def _parse_agent_output_row(row: dict[str, Any]) -> StoredAgentOutput:
    try:
        agent_type = AgentType(row["agent_type"])
    except (KeyError, ValueError, TypeError) as exc:
        raise DatabaseError("Stored update has an unknown or missing agent_type.") from exc

    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise DatabaseError("Stored update metadata must be an object.")

    if agent_type == AgentType.CROSS_PORTFOLIO:
        payload = {
            "user_id": row.get("user_id"),
            "timestamp": row.get("timestamp"),
            "summary": row.get("summary"),
            "correlations_flagged": metadata.get("correlations_flagged", []),
            "tickers_analyzed": metadata.get("tickers_analyzed", []),
        }
        return _parse_model(CrossPortfolioOutput, payload)

    payload = {
        key: row.get(key)
        for key in (
            "ticker",
            "user_id",
            "event_type",
            "summary",
            "recommendation",
            "confidence",
            "timestamp",
            "price_at_update",
            "searched_web",
        )
    }
    payload["sources"] = metadata.get("sources", [])
    payload["web_search_requests"] = metadata.get("web_search_requests", 0)
    if agent_type == AgentType.HYPOTHESIS:
        payload.update(
            flagged=metadata.get("flagged", False),
            recommended_next_scan_days=metadata.get("recommended_next_scan_days"),
        )
        return _parse_model(HypothesisOutput, payload)

    expected_agent_type = _AGENT_TYPE_BY_EVENT.get(payload["event_type"])
    if expected_agent_type != agent_type:
        raise DatabaseError(
            "Stored update agent_type does not match its event_type."
        )
    return _parse_model(AgentOutput, payload)


def _parse_agent_output_list(rows: Any) -> list[StoredAgentOutput]:
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise DatabaseError(f"Expected a list of rows, received {type(rows)!r}.")
    return [_parse_agent_output_row(row) for row in rows]


def insert_agent_output(
    output: StoredAgentOutput, *, client: Optional[Client] = None
) -> StoredAgentOutput:
    response = _execute(
        _client(client)
        .table("updates")
        .insert(_agent_output_payload(output))
    )
    row = _require_single_row(response.data, "insert_agent_output")
    return _parse_agent_output_row(row)


def get_latest_update(
    user_id: str,
    agent_type: AgentType | str,
    ticker: str | None = None,
    *,
    client: Optional[Client] = None,
) -> StoredAgentOutput | None:
    normalized_agent_type = AgentType(agent_type)
    query = (
        _client(client)
        .table("updates")
        .select("*")
        .eq("user_id", user_id)
        .eq("agent_type", normalized_agent_type.value)
    )
    if ticker is not None:
        query = query.eq("ticker", _ticker_symbol(ticker))

    response = _execute(
        query.order("timestamp", desc=True).order("id", desc=True).limit(1)
    )
    row = _single_row(response.data)
    return _parse_agent_output_row(row) if row else None


def list_recent_updates(
    user_id: str,
    *,
    ticker: str | None = None,
    agent_type: AgentType | str | None = None,
    limit: int = 10,
    since: datetime | None = None,
    client: Optional[Client] = None,
) -> list[StoredAgentOutput]:
    validated_limit = _history_limit(limit)
    normalized_since = _since_timestamp(since)
    query = _client(client).table("updates").select("*").eq("user_id", user_id)
    if ticker is not None:
        query = query.eq("ticker", _ticker_symbol(ticker))
    if agent_type is not None:
        query = query.eq("agent_type", AgentType(agent_type).value)
    if normalized_since is not None:
        query = query.gte("timestamp", normalized_since)

    response = _execute(
        query.order("timestamp", desc=True)
        .order("id", desc=True)
        .limit(validated_limit)
    )
    return _parse_agent_output_list(response.data)


def list_latest_portfolio_updates(
    user_id: str, *, client: Optional[Client] = None
) -> list[StoredAgentOutput]:
    database_client = _client(client)
    subscriptions = list_subscriptions_for_user(user_id, client=database_client)
    subscribed_tickers = {subscription.ticker for subscription in subscriptions}
    if not subscribed_tickers:
        return []

    scan_limit = min(
        MAX_PORTFOLIO_UPDATE_SCAN,
        max(len(subscribed_tickers), len(subscribed_tickers) * PORTFOLIO_UPDATE_SCAN_PER_TICKER),
    )
    response = _execute(
        database_client.table("updates")
        .select("*")
        .eq("user_id", user_id)
        .in_("ticker", sorted(subscribed_tickers))
        .order("timestamp", desc=True)
        .order("id", desc=True)
        .limit(scan_limit)
    )
    recent_outputs = _parse_agent_output_list(response.data)

    latest_by_ticker: dict[str, StoredAgentOutput] = {}
    for output in recent_outputs:
        ticker = getattr(output, "ticker", None)
        if ticker is not None and ticker not in latest_by_ticker:
            latest_by_ticker[ticker] = output
    return list(latest_by_ticker.values())


def list_updates_for_user(
    user_id: str, limit: int = 50, *, client: Optional[Client] = None
) -> list[StoredAgentOutput]:
    return list_recent_updates(user_id, limit=limit, client=client)


def list_updates_for_user_ticker(
    user_id: str, ticker: str, limit: int = 20, *, client: Optional[Client] = None
) -> list[StoredAgentOutput]:
    return list_recent_updates(
        user_id,
        ticker=ticker,
        limit=limit,
        client=client,
    )


def insert_alert(alert: Alert, *, client: Optional[Client] = None) -> Alert:
    payload = _model_payload(alert)
    payload["ticker"] = _ticker_symbol(payload["ticker"])

    response = _execute(_client(client).table("alerts").insert(payload))
    row = _require_single_row(response.data, "insert_alert")
    return _parse_model(Alert, row)


def list_alerts_for_user(
    user_id: str, limit: int = 50, *, client: Optional[Client] = None
) -> list[Alert]:
    return list_recent_alerts(user_id, limit=limit, client=client)


def list_alerts_for_user_ticker(
    user_id: str, ticker: str, limit: int = 20, *, client: Optional[Client] = None
) -> list[Alert]:
    return list_recent_alerts(
        user_id,
        ticker=ticker,
        limit=limit,
        client=client,
    )


def list_recent_alerts(
    user_id: str,
    *,
    ticker: str | None = None,
    alert_type: AlertType | str | None = None,
    limit: int = 10,
    since: datetime | None = None,
    client: Optional[Client] = None,
) -> list[Alert]:
    validated_limit = _history_limit(limit)
    normalized_since = _since_timestamp(since)
    query = _client(client).table("alerts").select("*").eq("user_id", user_id)
    if ticker is not None:
        query = query.eq("ticker", _ticker_symbol(ticker))
    if alert_type is not None:
        query = query.eq("alert_type", AlertType(alert_type).value)
    if normalized_since is not None:
        query = query.gte("timestamp", normalized_since)

    response = _execute(
        query.order("timestamp", desc=True)
        .order("id", desc=True)
        .limit(validated_limit)
    )
    return _parse_model_list(Alert, response.data)
