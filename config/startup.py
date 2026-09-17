from __future__ import annotations

from collections.abc import Iterable

from config.settings import Settings
from security.credentials import (
    CredentialConfigurationError,
    create_credential_cipher,
)


class StartupConfigurationError(RuntimeError):
    """Raised when a service cannot safely start with its current settings."""


def _missing(setting_names: Iterable[str], config: Settings) -> list[str]:
    return [
        name.upper()
        for name in setting_names
        if getattr(config, name) is None
    ]


def validate_dashboard_settings(config: Settings) -> None:
    missing = _missing(("supabase_url", "supabase_key"), config)
    if missing:
        raise StartupConfigurationError(
            f"Missing dashboard configuration: {', '.join(missing)}."
        )


def validate_runtime_settings(config: Settings) -> None:
    required = (
        "supabase_url",
        "supabase_key",
        "supabase_secret_key",
        "credential_encryption_key",
        "twelve_data_api_key",
    )
    missing = _missing(required, config)
    if config.whatsapp_enabled:
        missing.extend(
            _missing(
                (
                    "twilio_account_sid",
                    "twilio_auth_token",
                    "twilio_whatsapp_from",
                ),
                config,
            )
        )
    if missing:
        unique = ", ".join(dict.fromkeys(missing))
        raise StartupConfigurationError(
            f"Missing runtime configuration: {unique}."
        )

    try:
        create_credential_cipher(config)
    except CredentialConfigurationError:
        raise StartupConfigurationError(
            "Credential encryption configuration is invalid."
        ) from None
