# MVP deployment

Sapient runs as two persistent processes built from the same Docker image:

- **Runtime worker:** market refresh, monitoring, scheduling, agents, and optional
  WhatsApp delivery. It exposes only health endpoints on port `8080`.
- **Dashboard:** the single-user Streamlit interface on port `8501`.

Keeping these separate prevents an interactive dashboard restart from stopping the
market scheduler and lets each process receive only the environment variables it
needs.

## Local container run

Validate configuration first:

```bash
uv run python -m config.preflight runtime
uv run python -m config.preflight dashboard
```

Then build and start both services:

```bash
docker compose up --build -d
docker compose ps
```

Open `http://localhost:8501`. Worker health is available at
`http://localhost:8080/health/ready`.

Stop both services with:

```bash
docker compose down
```

## Railway

Create two persistent Railway services pointing to this same repository. Railway
will detect the root `Dockerfile`.

Configure these service-specific start commands:

```text
runtime:   uv run python main.py
dashboard: uv run streamlit run dashboard/app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true --browser.gatherUsageStats=false
```

For the runtime service, set the health-check path to `/health/ready`. For the
dashboard service, use `/_stcore/health`. Generate a public domain only for the
dashboard. The runtime does not require a public domain.

Use Railway shared variables for common non-secret configuration, then scope
secrets to the service that needs them. Do not set `HEALTH_PORT` on Railway; the
runtime automatically accepts Railway's injected `PORT` value.

Railway currently recommends separate services and service-specific start commands
for web and worker processes sharing a repository. See the official
[services documentation](https://docs.railway.com/services) and
[multi-service guidance](https://docs.railway.com/guides/saas-backend).

## Required variables

Runtime:

```text
SUPABASE_URL
SUPABASE_KEY
SUPABASE_SECRET_KEY
CREDENTIAL_ENCRYPTION_KEY
CREDENTIAL_ENCRYPTION_KEY_ID
TWELVE_DATA_API_KEY
```

Dashboard:

```text
SUPABASE_URL
SUPABASE_KEY
MVP_USER_ID (only if the database has multiple users)
```

Add the three Twilio variables and set `WHATSAPP_ENABLED=true` only when WhatsApp
is ready. Add LangSmith variables only when tracing is desired.

## External handoff checklist

These are the only MVP steps that require account access:

1. Create the two hosting services and enter their environment variables.
2. Join the Twilio WhatsApp Sandbox or approve a production sender.
3. Enter the Twilio credentials and enable WhatsApp.
4. Send one real alert and confirm its corresponding `alerts` row.
5. Confirm the dashboard domain loads the expected single-user portfolio.

Do not expose the dashboard publicly beyond personal testing until authentication
and Supabase row-level security are implemented.
