import base64
import os
import unittest
from unittest.mock import patch

from config.settings import Settings
from config.startup import (
    StartupConfigurationError,
    validate_dashboard_settings,
    validate_runtime_settings,
)


def settings_for(**overrides) -> Settings:
    values = {
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_KEY": "anon-key",
        "SUPABASE_SECRET_KEY": "secret-key",
        "CREDENTIAL_ENCRYPTION_KEY": base64.urlsafe_b64encode(b"k" * 32).decode(),
        "TWELVE_DATA_API_KEY": "twelve-key",
    }
    values.update(overrides)
    with patch.dict(os.environ, values, clear=True):
        return Settings(_env_file=None)


class StartupValidationTests(unittest.TestCase):
    def test_runtime_accepts_core_config_without_twilio(self):
        validate_runtime_settings(settings_for())

    def test_runtime_requires_twilio_only_when_enabled(self):
        with self.assertRaisesRegex(
            StartupConfigurationError,
            "TWILIO_ACCOUNT_SID",
        ):
            validate_runtime_settings(settings_for(WHATSAPP_ENABLED="true"))

        validate_runtime_settings(
            settings_for(
                WHATSAPP_ENABLED="true",
                TWILIO_ACCOUNT_SID="account",
                TWILIO_AUTH_TOKEN="token",
                TWILIO_WHATSAPP_FROM="whatsapp:+14155238886",
            )
        )

    def test_runtime_lists_missing_config_without_values(self):
        with patch.dict(os.environ, {}, clear=True):
            config = Settings(_env_file=None)

        with self.assertRaises(StartupConfigurationError) as raised:
            validate_runtime_settings(config)

        message = str(raised.exception)
        self.assertIn("SUPABASE_URL", message)
        self.assertIn("TWELVE_DATA_API_KEY", message)
        self.assertNotIn("anon-key", message)

    def test_runtime_rejects_invalid_encryption_key_without_echoing_it(self):
        config = settings_for(CREDENTIAL_ENCRYPTION_KEY="not-a-valid-key")

        with self.assertRaises(StartupConfigurationError) as raised:
            validate_runtime_settings(config)

        self.assertEqual(
            str(raised.exception),
            "Credential encryption configuration is invalid.",
        )
        self.assertNotIn("not-a-valid-key", str(raised.exception))

    def test_dashboard_needs_only_public_database_config(self):
        validate_dashboard_settings(settings_for())

        with patch.dict(os.environ, {}, clear=True):
            config = Settings(_env_file=None)
        with self.assertRaisesRegex(StartupConfigurationError, "SUPABASE_URL"):
            validate_dashboard_settings(config)


if __name__ == "__main__":
    unittest.main()
