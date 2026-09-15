from __future__ import annotations

from dataclasses import dataclass

from supabase import Client

from data.database import (
    StoredAgentOutput,
    delete_subscription,
    get_latest_ticker_data,
    get_ticker,
    get_user,
    list_alerts_for_user,
    list_latest_portfolio_updates,
    list_subscriptions_for_user,
    list_updates_for_user,
    list_users,
    upsert_subscription,
    upsert_ticker,
)
from models.schemas import (
    Alert,
    Motive,
    OHLCVPoint,
    Subscription,
    Ticker,
    UpdateInterval,
    User,
)


@dataclass(frozen=True)
class DashboardPosition:
    subscription: Subscription
    current_price: float | None
    datapoints: list[OHLCVPoint]
    latest_output: StoredAgentOutput | None

    @property
    def current_value(self) -> float | None:
        if self.current_price is None:
            return None
        return self.current_price * self.subscription.shares

    @property
    def cost_value(self) -> float:
        return self.subscription.avg_price * self.subscription.shares

    @property
    def unrealized_pnl(self) -> float | None:
        if self.current_value is None:
            return None
        return self.current_value - self.cost_value

    @property
    def unrealized_pnl_pct(self) -> float | None:
        if self.current_price is None:
            return None
        return (
            self.current_price - self.subscription.avg_price
        ) / self.subscription.avg_price


@dataclass(frozen=True)
class DashboardData:
    user: User
    positions: list[DashboardPosition]
    updates: list[StoredAgentOutput]
    alerts: list[Alert]


def resolve_mvp_user_id(
    configured_user_id: str | None = None,
    *,
    client: Client | None = None,
) -> str:
    """Use the configured user, or infer the sole user in an MVP database."""
    configured = (configured_user_id or "").strip()
    if configured:
        return configured

    users = list_users(limit=2, client=client)
    if not users:
        raise ValueError("Create a user before opening the dashboard.")
    if len(users) > 1:
        raise ValueError("Set MVP_USER_ID when the database contains multiple users.")
    return users[0].user_id


def load_dashboard_data(
    user_id: str,
    *,
    client: Client | None = None,
) -> DashboardData:
    user = get_user(user_id, client=client)
    if user is None:
        raise ValueError("The configured MVP user does not exist.")

    subscriptions = list_subscriptions_for_user(user_id, client=client)
    latest_outputs = list_latest_portfolio_updates(user_id, client=client)
    latest_by_ticker = {
        output.ticker: output
        for output in latest_outputs
        if getattr(output, "ticker", None) is not None
    }
    positions = []
    for subscription in subscriptions:
        datapoints = get_latest_ticker_data(
            subscription.ticker,
            limit=150,
            client=client,
        )
        current_price = datapoints[-1].close if datapoints else None
        positions.append(
            DashboardPosition(
                subscription=subscription,
                current_price=current_price,
                datapoints=datapoints,
                latest_output=latest_by_ticker.get(subscription.ticker),
            )
        )

    return DashboardData(
        user=user,
        positions=positions,
        updates=list_updates_for_user(user_id, limit=100, client=client),
        alerts=list_alerts_for_user(user_id, limit=100, client=client),
    )


def save_subscription(
    user_id: str,
    ticker: str,
    *,
    avg_price: float,
    shares: float,
    motive: Motive,
    update_interval: UpdateInterval,
    sharp_move_threshold: float,
    client: Client | None = None,
) -> Subscription:
    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker is required.")
    if get_ticker(symbol, client=client) is None:
        upsert_ticker(Ticker(ticker=symbol), client=client)
    return upsert_subscription(
        Subscription(
            user_id=user_id,
            ticker=symbol,
            avg_price=avg_price,
            shares=shares,
            motive=motive,
            update_interval=update_interval,
            sharp_move_threshold=sharp_move_threshold,
        ),
        client=client,
    )


def remove_subscription(
    user_id: str,
    ticker: str,
    *,
    client: Client | None = None,
) -> None:
    delete_subscription(user_id, ticker, client=client)
