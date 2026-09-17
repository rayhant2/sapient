from enum import Enum
from typing import Optional

from pydantic import AliasChoices, AnyUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    LOCAL = "local"
    TEST = "test"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    TEXT = "text"
    JSON = "json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
    )

    environment: Environment = Environment.LOCAL
    log_level: LogLevel = LogLevel.INFO
    log_format: LogFormat = LogFormat.TEXT

    supabase_url: Optional[AnyUrl] = None
    supabase_key: Optional[SecretStr] = None
    supabase_secret_key: Optional[SecretStr] = None

    credential_encryption_key: Optional[SecretStr] = None
    credential_encryption_key_id: str = "primary-v1"
    credential_previous_encryption_keys: Optional[SecretStr] = None

    twelve_data_api_key: Optional[SecretStr] = None
    anthropic_api_key: Optional[SecretStr] = None
    anthropic_model: str = "claude-sonnet-4-5"
    agent_model_max_tokens: int = 2048
    agent_model_timeout_seconds: float = 60.0
    agent_model_max_retries: int = 2
    agent_web_search_max_uses: int = 2
    agent_web_search_max_continuations: int = 1
    agent_graph_recursion_limit: int = 20

    langsmith_tracing: bool = False
    langsmith_api_key: Optional[SecretStr] = None
    langsmith_project: str = "sentient"

    twilio_account_sid: Optional[SecretStr] = None
    twilio_auth_token: Optional[SecretStr] = None
    twilio_whatsapp_from: Optional[str] = None
    whatsapp_enabled: bool = False
    mvp_user_id: Optional[str] = None

    health_host: str = "0.0.0.0"
    health_port: int = Field(
        default=8080,
        validation_alias=AliasChoices("health_port", "PORT"),
    )

    max_ticker_datapoints: int = 150
    price_fetch_interval_minutes: int = 15
    twelve_data_requests_per_minute: int = 8
    twelve_data_rate_limit_buffer_seconds: float = 1.0
    twelve_data_max_retries: int = 2
    twelve_data_retry_base_delay_seconds: float = 5.0
    sharp_move_check_interval_minutes: int = 15
    default_sharp_move_threshold: float = 0.01
    default_hypothesis_scan_days: int = 3

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: str | LogLevel) -> str | LogLevel:
        if isinstance(value, str):
            return value.upper()
        return value

    @field_validator(
        "max_ticker_datapoints",
        "price_fetch_interval_minutes",
        "twelve_data_requests_per_minute",
        "sharp_move_check_interval_minutes",
        "default_hypothesis_scan_days",
        "agent_model_max_tokens",
        "agent_web_search_max_uses",
        "agent_graph_recursion_limit",
        "health_port",
    )
    @classmethod
    def must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Setting must be greater than zero")
        return value

    @field_validator("twelve_data_max_retries")
    @classmethod
    def retries_must_be_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("twelve_data_max_retries must be non-negative")
        return value

    @field_validator("agent_model_max_retries")
    @classmethod
    def agent_retries_must_be_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("agent_model_max_retries must be non-negative")
        return value

    @field_validator("agent_web_search_max_uses")
    @classmethod
    def web_search_uses_must_be_bounded(cls, value: int) -> int:
        if not 1 <= value <= 5:
            raise ValueError("agent_web_search_max_uses must be between 1 and 5")
        return value

    @field_validator("agent_web_search_max_continuations")
    @classmethod
    def web_search_continuations_must_be_bounded(cls, value: int) -> int:
        if not 0 <= value <= 2:
            raise ValueError(
                "agent_web_search_max_continuations must be between 0 and 2"
            )
        return value

    @field_validator("agent_graph_recursion_limit")
    @classmethod
    def graph_recursion_limit_must_be_bounded(cls, value: int) -> int:
        if not 1 <= value <= 100:
            raise ValueError("agent_graph_recursion_limit must be between 1 and 100")
        return value

    @field_validator(
        "twelve_data_rate_limit_buffer_seconds",
        "twelve_data_retry_base_delay_seconds",
    )
    @classmethod
    def scheduler_delays_must_be_valid(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Scheduler delays must be non-negative")
        return value

    @field_validator("agent_model_timeout_seconds")
    @classmethod
    def agent_timeout_must_be_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("agent_model_timeout_seconds must be greater than zero")
        return value

    @field_validator("anthropic_model")
    @classmethod
    def anthropic_model_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("anthropic_model must not be blank")
        return normalized

    @field_validator("health_host")
    @classmethod
    def health_host_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("health_host must not be blank")
        return normalized

    @field_validator("default_sharp_move_threshold")
    @classmethod
    def threshold_must_be_valid(cls, value: float) -> float:
        if not (0.001 <= value <= 0.5):
            raise ValueError("default_sharp_move_threshold must be between 0.1% and 50%")
        return value

    @field_validator("langsmith_project")
    @classmethod
    def langsmith_project_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("langsmith_project must not be blank")
        return value

    @field_validator("credential_encryption_key_id")
    @classmethod
    def credential_key_id_must_be_valid(cls, value: str) -> str:
        if not value or len(value) > 64:
            raise ValueError("credential_encryption_key_id must be 1-64 characters")
        if not all(character.isalnum() or character in "._-" for character in value):
            raise ValueError("credential_encryption_key_id contains invalid characters")
        return value

    @field_validator("twilio_whatsapp_from")
    @classmethod
    def twilio_whatsapp_from_must_be_valid(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.startswith("whatsapp:+"):
            raise ValueError("twilio_whatsapp_from must start with 'whatsapp:+'")
        return value

    @property
    def is_local(self) -> bool:
        return self.environment == Environment.LOCAL

    @property
    def is_test(self) -> bool:
        return self.environment == Environment.TEST

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION


settings = Settings()
