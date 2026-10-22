"""Manual Kite Connect login URL and request-token exchange foundation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from multi_agent_investment_system.core.configuration import KiteConnectCredentials
from multi_agent_investment_system.core.exceptions import AuthenticationError

_LOGIN_URL = "https://kite.zerodha.com/connect/login"
_SESSION_TOKEN_URL = "https://api.kite.trade/session/token"
_EXCHANGE_ERROR_MESSAGE = "Kite request-token exchange failed"


def _decode_payload(body: bytes) -> dict[str, object] | None:
    """Decode a provider payload without exposing its contents on failure."""

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _error_category(
    http_status: int | None,
    payload: dict[str, object] | None,
    *,
    malformed_response: bool = False,
) -> str:
    """Map only explicit transport or structured provider evidence."""

    if http_status == 429:
        return "rate_limit"

    error_type = payload.get("error_type") if payload else None
    message = payload.get("message") if payload else None
    error_type_text = error_type.casefold() if isinstance(error_type, str) else ""
    message_text = message.casefold() if isinstance(message, str) else ""

    if "checksum" in message_text:
        return "checksum_or_credentials"
    if "api_key" in message_text or "api key" in message_text:
        return "checksum_or_credentials"
    if "request_token" in message_text or "request token" in message_text:
        if any(word in message_text for word in ("invalid", "expired", "used")):
            return "invalid_or_expired_request_token"
    if error_type_text == "permissionexception" or any(
        phrase in message_text
        for phrase in ("permission denied", "not permitted", "access denied")
    ):
        return "app_access_or_permission"
    if http_status in (401, 403):
        return "app_access_or_permission"
    if malformed_response:
        return "malformed_provider_response"
    return "unknown_authentication_failure"


@dataclass(frozen=True, slots=True)
class KiteAccessToken:
    """An in-memory access token omitted from generated representations."""

    value: str = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value.strip():
            raise AuthenticationError("Kite authentication returned no access token")


class KiteConnectAuthenticator:
    """Build the interactive login URL and exchange its request token.

    The caller must perform the user login in a browser and supply the
    short-lived request token returned to the registered redirect URL. This
    component does not automate login, persist tokens, or perform trading.
    """

    def __init__(self, credentials: KiteConnectCredentials) -> None:
        self._credentials = credentials

    def login_url(self) -> str:
        """Return the official v3 login URL for the configured API key."""

        query = urlencode({"api_key": self._credentials.api_key, "v": "3"})
        return f"{_LOGIN_URL}?{query}"

    def exchange_request_token(self, request_token: str) -> KiteAccessToken:
        """Exchange a user-provided request token for an access token.

        Sensitive values are sent only in the HTTPS form body. Failures use a
        generic message and a fixed machine-readable category; provider details
        and exception causes are never exposed.
        """

        if not isinstance(request_token, str) or not request_token.strip():
            raise AuthenticationError("Kite request token must not be empty")

        checksum = hashlib.sha256(
            (
                self._credentials.api_key
                + request_token
                + self._credentials.api_secret
            ).encode("utf-8")
        ).hexdigest()
        body = urlencode(
            {
                "api_key": self._credentials.api_key,
                "request_token": request_token,
                "checksum": checksum,
            }
        ).encode("ascii")
        request = Request(
            _SESSION_TOKEN_URL,
            data=body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Kite-Version": "3",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=10) as response:
                http_status = getattr(response, "status", None)
                if http_status is None:
                    http_status = response.getcode()
                payload = _decode_payload(response.read())

            if not isinstance(http_status, int) or not 200 <= http_status < 300:
                category = _error_category(http_status, payload)
                raise AuthenticationError(_EXCHANGE_ERROR_MESSAGE, category=category)
            if payload is None:
                raise AuthenticationError(
                    _EXCHANGE_ERROR_MESSAGE,
                    category="malformed_provider_response",
                )
            if payload.get("status") != "success":
                category = _error_category(http_status, payload)
                raise AuthenticationError(_EXCHANGE_ERROR_MESSAGE, category=category)

            data = payload.get("data")
            token_value = data.get("access_token") if isinstance(data, dict) else None
            if not isinstance(token_value, str) or not token_value.strip():
                raise AuthenticationError(
                    _EXCHANGE_ERROR_MESSAGE,
                    category="malformed_provider_response",
                )
            return KiteAccessToken(token_value)
        except AuthenticationError:
            raise
        except HTTPError as error:
            try:
                payload = _decode_payload(error.read())
            except Exception:
                payload = None
            category = _error_category(
                error.code, payload, malformed_response=payload is None
            )
            raise AuthenticationError(
                _EXCHANGE_ERROR_MESSAGE, category=category
            ) from None
        except (URLError, TimeoutError):
            raise AuthenticationError(
                _EXCHANGE_ERROR_MESSAGE,
                category="network_or_timeout",
            ) from None
        except Exception:
            raise AuthenticationError(
                _EXCHANGE_ERROR_MESSAGE,
                category="unknown_authentication_failure",
            ) from None
