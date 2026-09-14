-- Sentient database schema for Supabase/Postgres.

create table if not exists public.users (
    user_id text primary key,
    whatsapp_number text not null unique,
    email text unique,
    created_at timestamptz not null default now(),
    preferences jsonb not null default '{}'::jsonb
);

create table if not exists public.tickers (
    ticker text primary key,
    last_fetched timestamptz,
    next_fetch_time timestamptz,
    current_price numeric(18, 6),
    constraint tickers_current_price_non_negative
        check (current_price is null or current_price >= 0),
    constraint tickers_ticker_uppercase
        check (ticker = upper(ticker))
);

create table if not exists public.ticker_data (
    ticker text not null references public.tickers(ticker) on delete cascade,
    "timestamp" timestamptz not null,
    open numeric(18, 6) not null,
    high numeric(18, 6) not null,
    low numeric(18, 6) not null,
    close numeric(18, 6) not null,
    volume numeric(20, 2) not null,
    primary key (ticker, "timestamp"),
    constraint ticker_data_open_non_negative check (open >= 0),
    constraint ticker_data_high_non_negative check (high >= 0),
    constraint ticker_data_low_non_negative check (low >= 0),
    constraint ticker_data_close_non_negative check (close >= 0),
    constraint ticker_data_volume_non_negative check (volume >= 0),
    constraint ticker_data_high_is_highest check (high >= open and high >= low and high >= close),
    constraint ticker_data_low_is_lowest check (low <= open and low <= high and low <= close)
);

create table if not exists public.subscriptions (
    user_id text not null references public.users(user_id) on delete cascade,
    ticker text not null references public.tickers(ticker) on delete cascade,
    avg_price numeric(18, 6) not null,
    shares numeric(18, 6) not null,
    motive text not null,
    update_interval text not null,
    sharp_move_threshold numeric(8, 6) not null default 0.01,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    primary key (user_id, ticker),
    constraint subscriptions_avg_price_positive check (avg_price > 0),
    constraint subscriptions_shares_positive check (shares > 0),
    constraint subscriptions_threshold_valid check (
        sharp_move_threshold >= 0.001
        and sharp_move_threshold <= 0.5
    ),
    constraint subscriptions_motive_valid check (
        motive in ('holding', 'short-term', 'watching')
    ),
    constraint subscriptions_update_interval_valid check (
        update_interval in ('daily', 'weekly')
    )
);

alter table public.subscriptions
    alter column sharp_move_threshold set default 0.01;

create table if not exists public.updates (
    id bigint generated always as identity primary key,
    user_id text not null references public.users(user_id) on delete cascade,
    ticker text references public.tickers(ticker) on delete set null,
    "timestamp" timestamptz not null default now(),
    agent_type text not null,
    event_type text,
    summary text,
    recommendation text,
    confidence text,
    price_at_update numeric(18, 6),
    searched_web boolean not null default false,
    metadata jsonb not null default '{}'::jsonb,
    constraint updates_price_at_update_non_negative
        check (price_at_update is null or price_at_update >= 0),
    constraint updates_agent_type_valid check (
        agent_type in (
            'scheduled_review',
            'sharp_move',
            'motive',
            'hypothesis',
            'cross_portfolio'
        )
    ),
    constraint updates_event_type_valid check (
        event_type in (
            'scheduled_update',
            'sharp_move',
            'motive_check',
            'hypothesis_scan'
        )
    ),
    constraint updates_confidence_valid check (
        confidence in ('high', 'medium', 'low')
    ),
    constraint updates_output_shape_valid check (
        (
            agent_type = 'cross_portfolio'
            and ticker is null
            and event_type is null
            and summary is not null
        )
        or
        (
            agent_type <> 'cross_portfolio'
            and ticker is not null
            and event_type is not null
            and recommendation is not null
            and confidence is not null
            and (summary is not null or agent_type = 'hypothesis')
        )
    )
);

alter table public.updates
    add column if not exists agent_type text;

update public.updates
set agent_type = case event_type
    when 'scheduled_update' then 'scheduled_review'
    when 'sharp_move' then 'sharp_move'
    when 'motive_check' then 'motive'
    when 'hypothesis_scan' then 'hypothesis'
end
where agent_type is null;

alter table public.updates
    alter column agent_type set not null,
    alter column event_type drop not null,
    alter column summary drop not null,
    alter column recommendation drop not null,
    alter column confidence drop not null;

alter table public.updates
    drop constraint if exists updates_agent_type_valid,
    drop constraint if exists updates_output_shape_valid;

alter table public.updates
    add constraint updates_agent_type_valid check (
        agent_type in (
            'scheduled_review',
            'sharp_move',
            'motive',
            'hypothesis',
            'cross_portfolio'
        )
    ),
    add constraint updates_output_shape_valid check (
        (
            agent_type = 'cross_portfolio'
            and ticker is null
            and event_type is null
            and summary is not null
        )
        or
        (
            agent_type <> 'cross_portfolio'
            and ticker is not null
            and event_type is not null
            and recommendation is not null
            and confidence is not null
            and (summary is not null or agent_type = 'hypothesis')
        )
    );

create table if not exists public.alerts (
    id bigint generated always as identity primary key,
    user_id text not null references public.users(user_id) on delete cascade,
    ticker text references public.tickers(ticker) on delete set null,
    "timestamp" timestamptz not null default now(),
    alert_type text not null,
    message text not null,
    trigger_details jsonb not null default '{}'::jsonb,
    constraint alerts_alert_type_valid check (
        alert_type in (
            'sharp_move',
            'motive_flag',
            'hypothesis',
            'cross_portfolio',
            'scheduled'
        )
    )
);

create index if not exists idx_subscriptions_ticker
    on public.subscriptions(ticker);

create index if not exists idx_subscriptions_user_id
    on public.subscriptions(user_id);

create index if not exists idx_ticker_data_ticker_timestamp_desc
    on public.ticker_data(ticker, "timestamp" desc);

create index if not exists idx_tickers_next_fetch_time
    on public.tickers(next_fetch_time)
    where next_fetch_time is not null;

create index if not exists idx_updates_user_timestamp_desc
    on public.updates(user_id, "timestamp" desc);

create index if not exists idx_updates_user_ticker_timestamp_desc
    on public.updates(user_id, ticker, "timestamp" desc);

create index if not exists idx_updates_user_agent_timestamp_desc
    on public.updates(user_id, agent_type, "timestamp" desc);

create index if not exists idx_alerts_user_timestamp_desc
    on public.alerts(user_id, "timestamp" desc);

create index if not exists idx_alerts_user_ticker_timestamp_desc
    on public.alerts(user_id, ticker, "timestamp" desc);

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists set_subscriptions_updated_at on public.subscriptions;

create trigger set_subscriptions_updated_at
before update on public.subscriptions
for each row
execute function public.set_updated_at();
