from __future__ import annotations

from datetime import timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config.settings import settings
from config.startup import StartupConfigurationError, validate_dashboard_settings
from dashboard.services import (
    DashboardData,
    DashboardPosition,
    load_dashboard_data,
    remove_subscription,
    resolve_mvp_user_id,
    save_subscription,
)
from models.schemas import Motive, UpdateInterval


ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = ROOT / "public" / "Sapient_Logo.png"

st.set_page_config(
    page_title="Sapient",
    page_icon=str(LOGO_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --sapient-ink: #17212b;
        --sapient-muted: #66727f;
        --sapient-line: #dfe5e8;
        --sapient-green: #167d68;
        --sapient-coral: #c7524a;
        --sapient-blue: #356b9a;
        --sapient-bg: #f7f9f8;
    }
    .stApp { background: var(--sapient-bg); color: var(--sapient-ink); }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--sapient-line); }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--sapient-line);
        border-radius: 6px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] { color: var(--sapient-muted); }
    [data-testid="stMetricValue"] { color: var(--sapient-ink); }
    [data-testid="stDataFrame"] { border: 1px solid var(--sapient-line); border-radius: 6px; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-color: var(--sapient-line);
        border-radius: 6px;
    }
    .sapient-header {
        border-bottom: 1px solid var(--sapient-line);
        margin-bottom: 1.2rem;
        padding-bottom: 0.7rem;
    }
    .sapient-header h1 { font-size: 1.7rem; margin: 0; letter-spacing: 0; }
    .sapient-header p { color: var(--sapient-muted); margin: 0.2rem 0 0; }
    h1, h2, h3 { letter-spacing: 0 !important; }
    .stButton button, .stFormSubmitButton button { border-radius: 6px; }
    @media (max-width: 700px) {
        [data-testid="stMetric"] { padding: 10px 12px; }
        .sapient-header h1 { font-size: 1.4rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=45, show_spinner=False)
def cached_dashboard(user_id: str) -> DashboardData:
    return load_dashboard_data(user_id)


def money(value: float | None) -> str:
    return "--" if value is None else f"${value:,.2f}"


def percentage(value: float | None) -> str:
    return "--" if value is None else f"{value * 100:+.2f}%"


def position_rows(positions: list[DashboardPosition]) -> pd.DataFrame:
    rows = []
    for position in positions:
        subscription = position.subscription
        latest = position.latest_output
        rows.append(
            {
                "Ticker": subscription.ticker,
                "Motive": subscription.motive.value.title(),
                "Price": position.current_price,
                "Average cost": subscription.avg_price,
                "Shares": subscription.shares,
                "Market value": position.current_value,
                "P&L": position.unrealized_pnl,
                "P&L %": (
                    position.unrealized_pnl_pct * 100
                    if position.unrealized_pnl_pct is not None
                    else None
                ),
                "Last analysis": (
                    latest.timestamp.astimezone(timezone.utc)
                    if latest is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def price_chart(position: DashboardPosition) -> go.Figure:
    points = position.datapoints
    timestamps = [point.timestamp for point in points]
    increasing = "#167d68"
    decreasing = "#c7524a"
    volume_colors = [
        increasing if point.close >= point.open else decreasing for point in points
    ]
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.74, 0.26],
    )
    figure.add_trace(
        go.Candlestick(
            x=timestamps,
            open=[point.open for point in points],
            high=[point.high for point in points],
            low=[point.low for point in points],
            close=[point.close for point in points],
            increasing_line_color=increasing,
            decreasing_line_color=decreasing,
            name=position.subscription.ticker,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=timestamps,
            y=[point.volume for point in points],
            marker_color=volume_colors,
            name="Volume",
        ),
        row=2,
        col=1,
    )
    figure.update_layout(
        height=540,
        margin=dict(l=8, r=8, t=20, b=8),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#17212b", size=12),
        showlegend=False,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    figure.update_xaxes(gridcolor="#edf0f1", showline=False)
    figure.update_yaxes(gridcolor="#edf0f1", showline=False)
    return figure


def render_overview(data: DashboardData) -> None:
    positions = data.positions
    total_value = sum(position.current_value or 0.0 for position in positions)
    total_cost = sum(position.cost_value for position in positions)
    total_pnl = total_value - total_cost if positions else 0.0
    total_pnl_pct = total_pnl / total_cost if total_cost > 0 else 0.0

    metric_columns = st.columns(4)
    metric_columns[0].metric("Tracked value", money(total_value))
    metric_columns[1].metric(
        "Unrealized P&L",
        money(total_pnl),
        percentage(total_pnl_pct),
    )
    metric_columns[2].metric("Tracked tickers", str(len(positions)))
    metric_columns[3].metric("Alerts", str(len(data.alerts)))

    st.subheader("Positions")
    if not positions:
        st.info("No tracked positions.")
        return
    st.dataframe(
        position_rows(positions),
        width="stretch",
        hide_index=True,
        column_config={
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Average cost": st.column_config.NumberColumn(format="$%.2f"),
            "Shares": st.column_config.NumberColumn(format="%.4f"),
            "Market value": st.column_config.NumberColumn(format="$%.2f"),
            "P&L": st.column_config.NumberColumn(format="$%.2f"),
            "P&L %": st.column_config.NumberColumn(format="%+.2f%%"),
            "Last analysis": st.column_config.DatetimeColumn(format="MMM D, h:mm a"),
        },
    )

    st.subheader("Latest Analysis")
    columns = st.columns(2)
    for index, position in enumerate(positions):
        latest = position.latest_output
        with columns[index % 2].container(border=True):
            st.markdown(f"**{position.subscription.ticker}**")
            if latest is None:
                st.caption("No analysis available")
                continue
            st.write(latest.summary or "No material hypothesis was confirmed.")
            recommendation = getattr(latest, "recommendation", None)
            if recommendation:
                st.caption(f"Next: {recommendation}")


def render_ticker(data: DashboardData) -> None:
    if not data.positions:
        st.info("No tracked positions.")
        return
    tickers = [position.subscription.ticker for position in data.positions]
    selected = st.selectbox("Ticker", tickers)
    position = next(
        item for item in data.positions if item.subscription.ticker == selected
    )

    columns = st.columns(4)
    columns[0].metric("Current price", money(position.current_price))
    columns[1].metric("Average cost", money(position.subscription.avg_price))
    columns[2].metric("Unrealized P&L", money(position.unrealized_pnl))
    columns[3].metric("Return", percentage(position.unrealized_pnl_pct))

    if position.datapoints:
        st.plotly_chart(price_chart(position), width="stretch", config={"displaylogo": False})
    else:
        st.warning("Price history unavailable.")

    latest = position.latest_output
    if latest is not None:
        st.subheader("Latest Analysis")
        st.write(latest.summary or "No material hypothesis was confirmed.")
        recommendation = getattr(latest, "recommendation", None)
        if recommendation:
            st.info(recommendation)
        sources = getattr(latest, "sources", [])
        for source in sources:
            st.markdown(f"[{source.title}]({source.url})")


def render_activity(data: DashboardData) -> None:
    mode = st.segmented_control(
        "Activity",
        options=["Alerts", "Agent updates"],
        default="Alerts",
    )
    if mode == "Alerts":
        rows = [
            {
                "Time": alert.timestamp,
                "Ticker": alert.ticker or "Portfolio",
                "Type": alert.alert_type.value.replace("_", " ").title(),
                "Message": alert.message,
            }
            for alert in data.alerts
        ]
    else:
        rows = [
            {
                "Time": output.timestamp,
                "Ticker": getattr(output, "ticker", None) or "Portfolio",
                "Type": (
                    getattr(output, "event_type", None).value.replace("_", " ").title()
                    if getattr(output, "event_type", None) is not None
                    else "Cross Portfolio"
                ),
                "Summary": output.summary or "No material hypothesis confirmed.",
                "Recommendation": getattr(output, "recommendation", None),
            }
            for output in data.updates
        ]
    if not rows:
        st.info("No activity recorded.")
        return
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "Time": st.column_config.DatetimeColumn(format="MMM D, YYYY, h:mm a"),
        },
    )


def render_settings(user_id: str, data: DashboardData) -> None:
    st.subheader("Add or Update Position")
    with st.form("subscription-form", border=True):
        first_row = st.columns(3)
        ticker = first_row[0].text_input("Ticker", max_chars=20).strip().upper()
        average_price = first_row[1].number_input(
            "Average cost",
            min_value=0.01,
            value=100.0,
            step=0.01,
        )
        shares = first_row[2].number_input(
            "Shares",
            min_value=0.0001,
            value=1.0,
            step=0.1,
            format="%.4f",
        )
        second_row = st.columns(3)
        motive = second_row[0].selectbox(
            "Motive",
            options=list(Motive),
            format_func=lambda value: value.value.title(),
        )
        interval = second_row[1].selectbox(
            "Review interval",
            options=list(UpdateInterval),
            format_func=lambda value: value.value.title(),
        )
        threshold_pct = second_row[2].number_input(
            "Sharp move threshold (%)",
            min_value=0.1,
            max_value=50.0,
            value=1.0,
            step=0.1,
        )
        submitted = st.form_submit_button(
            "Save position",
            icon=":material/save:",
            type="primary",
        )
        if submitted:
            try:
                save_subscription(
                    user_id,
                    ticker,
                    avg_price=average_price,
                    shares=shares,
                    motive=motive,
                    update_interval=interval,
                    sharp_move_threshold=threshold_pct / 100,
                )
            except Exception:
                st.error("Position could not be saved.")
            else:
                cached_dashboard.clear()
                st.success(f"{ticker} saved.")
                st.rerun()

    st.subheader("Remove Position")
    if not data.positions:
        st.caption("No tracked positions")
        return
    remove_columns = st.columns([3, 1])
    selected = remove_columns[0].selectbox(
        "Tracked ticker",
        [position.subscription.ticker for position in data.positions],
    )
    if remove_columns[1].button(
        "Remove",
        icon=":material/delete:",
        type="secondary",
        width="stretch",
    ):
        try:
            remove_subscription(user_id, selected)
        except Exception:
            st.error("Position could not be removed.")
        else:
            cached_dashboard.clear()
            st.rerun()


def render_header(data: DashboardData) -> None:
    logo_column, title_column = st.columns([1, 8], vertical_alignment="center")
    logo_column.image(str(LOGO_PATH), width=58)
    display_name = data.user.email or data.user.whatsapp_number
    title_column.markdown(
        f'<div class="sapient-header"><h1>Sapient</h1><p>{display_name}</p></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    try:
        validate_dashboard_settings(settings)
    except StartupConfigurationError as exc:
        st.error(str(exc))
        st.stop()

    try:
        user_id = resolve_mvp_user_id(settings.mvp_user_id)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except Exception:
        st.error("The dashboard user could not be resolved.")
        st.stop()
    try:
        data = cached_dashboard(user_id)
    except Exception:
        st.error("Dashboard data could not be loaded.")
        st.stop()

    render_header(data)
    with st.sidebar:
        st.image(str(LOGO_PATH), width=44)
        page = st.radio(
            "Navigation",
            ["Overview", "Ticker detail", "Activity", "Positions"],
            label_visibility="collapsed",
        )
        if st.button("Refresh", icon=":material/refresh:", width="stretch"):
            cached_dashboard.clear()
            st.rerun()

    renderers: dict[str, Any] = {
        "Overview": lambda: render_overview(data),
        "Ticker detail": lambda: render_ticker(data),
        "Activity": lambda: render_activity(data),
        "Positions": lambda: render_settings(user_id, data),
    }
    renderers[page]()


if __name__ == "__main__":
    main()
