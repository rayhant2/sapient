from __future__ import annotations

import base64
import binascii
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Mapping

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import SecretStr

from config.settings import Settings, settings


_ENVELOPE_VERSION = 1
_KEY_BYTES = 32
_NONCE_BYTES = 12
_PROVIDER_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")
_KEY_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class CredentialSecurityError(RuntimeError):
    """Base exception for credential protection failures."""


class CredentialConfigurationError(CredentialSecurityError):
    """Raised when encryption configuration is missing or invalid."""


class CredentialEncryptionError(CredentialSecurityError):
    """Raised when a credential cannot be encrypted."""


class CredentialDecryptionError(CredentialSecurityError):
    """Raised when a stored credential cannot be safely decrypted."""


@dataclass(frozen=True)
class UserApiKeyMetadata:
    user_id: str
    provider: str
    created_at: datetime
    updated_at: datetime
    last_validated_at: datetime | None = None


def normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if not _PROVIDER_PATTERN.fullmatch(normalized):
        raise ValueError("Provider must be a lowercase identifier between 2 and 32 characters.")
    return normalized


def _secret_value(value: SecretStr | str) -> str:
    return value.get_secret_value() if isinstance(value, SecretStr) else value


def _decode_base64(value: str) -> bytes:
    if not isinstance(value, str):
        raise CredentialDecryptionError("Stored credential could not be decrypted.")
    try:
        return base64.b64decode(value.encode("ascii"), altchars=b"-_", validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise CredentialDecryptionError("Stored credential could not be decrypted.") from exc


def _decode_key(value: SecretStr | str) -> bytes:
    try:
        key = base64.b64decode(
            _secret_value(value).encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise CredentialConfigurationError(
            "Credential encryption keys must be URL-safe base64 values."
        ) from exc
    if len(key) != _KEY_BYTES:
        raise CredentialConfigurationError(
            "Credential encryption keys must decode to exactly 32 bytes."
        )
    return key


def _validate_key_id(key_id: str) -> str:
    if not _KEY_ID_PATTERN.fullmatch(key_id):
        raise CredentialConfigurationError("Credential encryption key ID is invalid.")
    return key_id


def _associated_data(*, user_id: str, provider: str, version: int) -> bytes:
    if not user_id.strip():
        raise ValueError("user_id must not be blank.")
    payload = {
        "context": "sentient:user-api-key",
        "provider": normalize_provider(provider),
        "user_id": user_id,
        "version": version,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


class CredentialCipher:
    """Owns the complete versioned AES-256-GCM credential envelope."""

    def __init__(
        self,
        *,
        active_key: SecretStr | str,
        active_key_id: str,
        previous_keys: Mapping[str, SecretStr | str] | None = None,
    ) -> None:
        self._active_key_id = _validate_key_id(active_key_id)
        keys = {
            _validate_key_id(key_id): _decode_key(key)
            for key_id, key in (previous_keys or {}).items()
        }
        keys[self._active_key_id] = _decode_key(active_key)
        self._keys = keys

    @property
    def active_key_id(self) -> str:
        return self._active_key_id

    def encrypt(self, api_key: SecretStr | str, *, user_id: str, provider: str) -> str:
        plaintext = _secret_value(api_key)
        if not plaintext or plaintext != plaintext.strip():
            raise CredentialEncryptionError(
                "API key must not be blank or contain surrounding whitespace."
            )

        nonce = os.urandom(_NONCE_BYTES)
        aad = _associated_data(
            user_id=user_id,
            provider=provider,
            version=_ENVELOPE_VERSION,
        )
        ciphertext = AESGCM(self._keys[self._active_key_id]).encrypt(
            nonce,
            plaintext.encode("utf-8"),
            aad,
        )
        envelope = {
            "ct": base64.urlsafe_b64encode(ciphertext).decode("ascii"),
            "kid": self._active_key_id,
            "n": base64.urlsafe_b64encode(nonce).decode("ascii"),
            "v": _ENVELOPE_VERSION,
        }
        return json.dumps(envelope, sort_keys=True, separators=(",", ":"))

    def decrypt(self, envelope: str, *, user_id: str, provider: str) -> SecretStr:
        try:
            payload = json.loads(envelope)
            if not isinstance(payload, dict) or set(payload) != {"ct", "kid", "n", "v"}:
                raise ValueError
            version = payload["v"]
            key_id = payload["kid"]
            if version != _ENVELOPE_VERSION or not isinstance(key_id, str):
                raise ValueError
            key = self._keys.get(key_id)
            if key is None:
                raise ValueError
            nonce = _decode_base64(payload["n"])
            ciphertext = _decode_base64(payload["ct"])
            if len(nonce) != _NONCE_BYTES or len(ciphertext) < 16:
                raise ValueError
            aad = _associated_data(user_id=user_id, provider=provider, version=version)
            plaintext = AESGCM(key).decrypt(nonce, ciphertext, aad).decode("utf-8")
            if not plaintext:
                raise ValueError
            return SecretStr(plaintext)
        except (CredentialDecryptionError, InvalidTag, UnicodeDecodeError, ValueError, TypeError):
            raise CredentialDecryptionError(
                "Stored credential could not be decrypted."
            ) from None

    def rotate(self, envelope: str, *, user_id: str, provider: str) -> str:
        plaintext = self.decrypt(envelope, user_id=user_id, provider=provider)
        return self.encrypt(plaintext, user_id=user_id, provider=provider)


def _previous_keys_from_settings(
    config: Settings = settings,
) -> dict[str, SecretStr]:
    configured = config.credential_previous_encryption_keys
    if configured is None:
        return {}
    try:
        values = json.loads(configured.get_secret_value())
    except json.JSONDecodeError as exc:
        raise CredentialConfigurationError(
            "CREDENTIAL_PREVIOUS_ENCRYPTION_KEYS must be a JSON object."
        ) from exc
    if not isinstance(values, dict) or not all(
        isinstance(key_id, str) and isinstance(key, str)
        for key_id, key in values.items()
    ):
        raise CredentialConfigurationError(
            "CREDENTIAL_PREVIOUS_ENCRYPTION_KEYS must map key IDs to base64 keys."
        )
    return {key_id: SecretStr(key) for key_id, key in values.items()}


def create_credential_cipher(config: Settings) -> CredentialCipher:
    if config.credential_encryption_key is None:
        raise CredentialConfigurationError(
            "CREDENTIAL_ENCRYPTION_KEY must be configured before storing user API keys."
        )
    return CredentialCipher(
        active_key=config.credential_encryption_key,
        active_key_id=config.credential_encryption_key_id,
        previous_keys=_previous_keys_from_settings(config),
    )


@lru_cache(maxsize=1)
def get_credential_cipher() -> CredentialCipher:
    return create_credential_cipher(settings)
