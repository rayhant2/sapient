import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from config.settings import Environment, LogLevel, Settings


def load_settings(env: dict[str, str] | None = None) -> Settings:
    with patch.dict(os.environ, env or {}, clear=True):
        return Settings(_env_file=None)


class SettingsTests(unittest.TestCase):
    def test_defaults_load_without_real_secrets(self):
        settings = load_settings()

        self.assertEqual(settings.environment, Environment.LOCAL)
        self.assertEqual(settings.log_level, LogLevel.INFO)
        self.assertEqual(settings.max_ticker_datapoints, 150)
        self.assertEqual(settings.price_fetch_interval_minutes, 15)
        self.assertEqual(settings.twelve_data_requests_per_minute, 8)
        self.assertEqual(settings.twelve_data_rate_limit_buffer_seconds, 1.0)
        self.assertEqual(settings.twelve_data_max_retries, 2)
        self.assertEqual(settings.twelve_data_retry_base_delay_seconds, 5.0)
        self.assertEqual(settings.sharp_move_check_interval_minutes, 15)
        self.assertEqual(settings.default_sharp_move_threshold, 0.025)
        self.assertEqual(settings.default_hypothesis_scan_days, 3)
        self.assertFalse(settings.langsmith_tracing)
        self.assertIsNone(settings.supabase_url)
        self.assertIsNone(settings.twelve_data_api_key)

    def test_env_vars_override_defaults(self):
        settings = load_settings(
            {
                "ENVIRONMENT": "test",
                "LOG_LEVEL": "debug",
                "LANGSMITH_TRACING": "true",
                "LANGSMITH_PROJECT": "sentient-tests",
                "MAX_TICKER_DATAPOINTS": "200",
                "PRICE_FETCH_INTERVAL_MINUTES": "30",
                "TWELVE_DATA_REQUESTS_PER_MINUTE": "15",
                "TWELVE_DATA_RATE_LIMIT_BUFFER_SECONDS": "5",
                "TWELVE_DATA_MAX_RETRIES": "3",
                "TWELVE_DATA_RETRY_BASE_DELAY_SECONDS": "7.5",
                "DEFAULT_SHARP_MOVE_THRESHOLD": "0.04",
                "TWILIO_WHATSAPP_FROM": "whatsapp:+15551234567",
            }
        )

        self.assertEqual(settings.environment, Environment.TEST)
        self.assertEqual(settings.log_level, LogLevel.DEBUG)
        self.assertTrue(settings.langsmith_tracing)
        self.assertEqual(settings.langsmith_project, "sentient-tests")
        self.assertEqual(settings.max_ticker_datapoints, 200)
        self.assertEqual(settings.price_fetch_interval_minutes, 30)
        self.assertEqual(settings.twelve_data_requests_per_minute, 15)
        self.assertEqual(settings.twelve_data_rate_limit_buffer_seconds, 5.0)
        self.assertEqual(settings.twelve_data_max_retries, 3)
        self.assertEqual(settings.twelve_data_retry_base_delay_seconds, 7.5)
        self.assertEqual(settings.default_sharp_move_threshold, 0.04)
        self.assertEqual(settings.twilio_whatsapp_from, "whatsapp:+15551234567")

    def test_secret_fields_are_loaded_as_secret_values(self):
        settings = load_settings(
            {
                "SUPABASE_KEY": "supabase-secret",
                "TWELVE_DATA_API_KEY": "twelve-data-secret",
                "ANTHROPIC_API_KEY": "anthropic-secret",
            }
        )

        self.assertEqual(settings.supabase_key.get_secret_value(), "supabase-secret")
        self.assertEqual(
            settings.twelve_data_api_key.get_secret_value(), "twelve-data-secret"
        )
        self.assertEqual(settings.anthropic_api_key.get_secret_value(), "anthropic-secret")

    def test_empty_env_values_are_ignored(self):
        settings = load_settings(
            {
                "SUPABASE_URL": "",
                "TWELVE_DATA_API_KEY": "",
            }
        )

        self.assertIsNone(settings.supabase_url)
        self.assertIsNone(settings.twelve_data_api_key)

    def test_environment_rejects_unknown_value(self):
        with self.assertRaises(ValidationError):
            load_settings({"ENVIRONMENT": "banana"})

    def test_threshold_rejects_values_outside_bounds(self):
        with self.assertRaises(ValidationError):
            load_settings({"DEFAULT_SHARP_MOVE_THRESHOLD": "0.0005"})

        with self.assertRaises(ValidationError):
            load_settings({"DEFAULT_SHARP_MOVE_THRESHOLD": "0.51"})

    def test_intervals_and_counts_must_be_positive(self):
        invalid_values = [
            {"MAX_TICKER_DATAPOINTS": "0"},
            {"PRICE_FETCH_INTERVAL_MINUTES": "0"},
            {"TWELVE_DATA_REQUESTS_PER_MINUTE": "0"},
            {"SHARP_MOVE_CHECK_INTERVAL_MINUTES": "-1"},
            {"DEFAULT_HYPOTHESIS_SCAN_DAYS": "0"},
        ]

        for env in invalid_values:
            with self.subTest(env=env):
                with self.assertRaises(ValidationError):
                    load_settings(env)

        scheduler_values = [
            {"TWELVE_DATA_MAX_RETRIES": "-1"},
            {"TWELVE_DATA_RATE_LIMIT_BUFFER_SECONDS": "-1"},
            {"TWELVE_DATA_RETRY_BASE_DELAY_SECONDS": "-1"},
        ]
        for env in scheduler_values:
            with self.subTest(env=env):
                with self.assertRaises(ValidationError):
                    load_settings(env)

    def test_twilio_whatsapp_sender_must_use_whatsapp_prefix(self):
        with self.assertRaises(ValidationError):
            load_settings({"TWILIO_WHATSAPP_FROM": "+15551234567"})

    def test_environment_helper_properties(self):
        local = load_settings({"ENVIRONMENT": "local"})
        test = load_settings({"ENVIRONMENT": "test"})
        production = load_settings({"ENVIRONMENT": "production"})

        self.assertTrue(local.is_local)
        self.assertTrue(test.is_test)
        self.assertTrue(production.is_production)


if __name__ == "__main__":
    unittest.main()
