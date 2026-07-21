from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime

from config.settings import settings
from data import database
from models.schemas import (
    AgentContext,
    EventType,
    OHLCVPoint,
    Subscription,
    Ticker,
    TickerRegistry,
)


EventHandler = Callable[[AgentContext], None]
TickerLoader = Callable[[], list[Ticker]]
SubscriptionLoader = Callable[[], list[Subscription]]
DatapointLoader = Callable[[str, int], list[OHLCVPoint]]


class EventBusError(RuntimeError):
    """Raised when an event cannot be routed."""


class UnknownTickerError(EventBusError):
    """Raised when an event references a ticker outside the registry."""


class HandlerNotRegisteredError(EventBusError):
    """Raised when an event type has no registered handler."""


class MissingTickerDataError(EventBusError):
    """Raised when a ticker event has no price history to dispatch."""


@dataclass(frozen=True)
class DispatchFailure:
    user_id: str
    error_type: str
    message: str


@dataclass(frozen=True)
class DispatchResult:
    ticker: str
    event_type: EventType
    attempted: int
    succeeded: int
    failures: list[DispatchFailure] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return len(self.failures)


class EventBus:
    def __init__(
        self,
        *,
        ticker_loader: TickerLoader | None = None,
        subscription_loader: SubscriptionLoader | None = None,
        datapoint_loader: DatapointLoader | None = None,
        max_datapoints: int = settings.max_ticker_datapoints,
    ) -> None:
        if max_datapoints <= 0:
            raise ValueError("max_datapoints must be greater than zero")

        self.registry: dict[str, TickerRegistry] = {}
        self._handlers: dict[EventType, EventHandler] = {}
        self._ticker_loader = ticker_loader or database.list_tickers
        self._subscription_loader = subscription_loader or database.list_subscriptions
        self._datapoint_loader = datapoint_loader or database.get_latest_ticker_data
        self._max_datapoints = max_datapoints

    @staticmethod
    def _ticker_symbol(ticker: str) -> str:
        return ticker.upper()

    def load_registry(self) -> dict[str, TickerRegistry]:
        tickers = {
            self._ticker_symbol(ticker.ticker): ticker
            for ticker in self._ticker_loader()
        }
        registry: dict[str, TickerRegistry] = {}

        for subscription in self._subscription_loader():
            symbol = self._ticker_symbol(subscription.ticker)
            ticker = tickers.get(symbol)
            ticker_registry = registry.setdefault(
                symbol,
                TickerRegistry(
                    ticker=symbol,
                    current_price=ticker.current_price if ticker else None,
                    last_fetched=ticker.last_fetched if ticker else None,
                ),
            )
            ticker_registry.add_subscriber(
                subscription.model_copy(update={"ticker": symbol})
            )

        self.registry = registry
        return self.registry

    def get_registry(self, ticker: str) -> TickerRegistry:
        symbol = self._ticker_symbol(ticker)
        try:
            return self.registry[symbol]
        except KeyError as exc:
            raise UnknownTickerError(f"Ticker {symbol} is not active in the registry.") from exc

    def refresh_subscription(self, subscription: Subscription) -> TickerRegistry:
        symbol = self._ticker_symbol(subscription.ticker)
        ticker_registry = self.registry.setdefault(
            symbol,
            TickerRegistry(ticker=symbol),
        )
        ticker_registry.add_subscriber(
            subscription.model_copy(update={"ticker": symbol})
        )
        return ticker_registry

    def remove_subscription(self, user_id: str, ticker: str) -> bool:
        symbol = self._ticker_symbol(ticker)
        ticker_registry = self.registry.get(symbol)
        if ticker_registry is None:
            return False

        existed = ticker_registry.get_subscriber(user_id) is not None
        ticker_registry.remove_subscriber(user_id)
        if not ticker_registry.is_active:
            del self.registry[symbol]
        return existed

    def update_ticker_state(
        self,
        ticker: str,
        *,
        current_price: float,
        last_fetched: datetime | None = None,
    ) -> TickerRegistry:
        if current_price < 0:
            raise ValueError("current_price must be non-negative")

        ticker_registry = self.get_registry(ticker)
        ticker_registry.current_price = current_price
        if last_fetched is not None:
            ticker_registry.last_fetched = last_fetched
        return ticker_registry

    def register_handler(self, event_type: EventType, handler: EventHandler) -> None:
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._handlers[event_type] = handler

    def unregister_handler(self, event_type: EventType) -> bool:
        return self._handlers.pop(event_type, None) is not None

    def _select_subscribers(
        self,
        ticker_registry: TickerRegistry,
        target_user_ids: Iterable[str] | None,
    ) -> list[Subscription]:
        if target_user_ids is None:
            return list(ticker_registry.subscribers)

        targets = set(target_user_ids)
        return [
            subscription
            for subscription in ticker_registry.subscribers
            if subscription.user_id in targets
        ]

    @staticmethod
    def _build_agent_context(
        subscription: Subscription,
        datapoints: list[OHLCVPoint],
        event_type: EventType,
    ) -> AgentContext:
        current_price = datapoints[-1].close
        price_change = current_price - subscription.avg_price
        unrealized_pnl = price_change * subscription.shares
        unrealized_pnl_pct = price_change / subscription.avg_price

        return AgentContext(
            ticker=subscription.ticker,
            datapoints=datapoints,
            subscription=subscription,
            event_type=event_type,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
        )

    def emit(
        self,
        event_type: EventType,
        ticker: str,
        *,
        target_user_ids: Iterable[str] | None = None,
    ) -> DispatchResult:
        symbol = self._ticker_symbol(ticker)
        ticker_registry = self.get_registry(symbol)
        handler = self._handlers.get(event_type)
        if handler is None:
            raise HandlerNotRegisteredError(
                f"No handler is registered for event type {event_type.value}."
            )

        subscribers = self._select_subscribers(ticker_registry, target_user_ids)
        if not subscribers:
            return DispatchResult(
                ticker=symbol,
                event_type=event_type,
                attempted=0,
                succeeded=0,
            )

        datapoints = self._datapoint_loader(symbol, self._max_datapoints)
        if not datapoints:
            raise MissingTickerDataError(f"Ticker {symbol} has no price history.")

        datapoints = sorted(datapoints, key=lambda point: point.timestamp)[
            -self._max_datapoints:
        ]
        ticker_registry.current_price = datapoints[-1].close

        succeeded = 0
        failures: list[DispatchFailure] = []
        for subscription in subscribers:
            context = self._build_agent_context(subscription, datapoints, event_type)
            try:
                handler(context)
                succeeded += 1
            except Exception as exc:
                failures.append(
                    DispatchFailure(
                        user_id=subscription.user_id,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )

        return DispatchResult(
            ticker=symbol,
            event_type=event_type,
            attempted=len(subscribers),
            succeeded=succeeded,
            failures=failures,
        )
