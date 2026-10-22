"""Offline tests for manual Kite Connect authentication primitives."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest

from multi_agent_investment_system.core.configuration import KiteConnectCredentials
from multi_agent_investment_system.core.exceptions import AuthenticationError
from multi_agent_investment_system.tools import kite_authentication
from multi_agent_investment_system.tools.kite_authentication import (
    KiteConnectAuthenticator,
)

_TEST_API_KEY = "dummy-api-key"
_TEST_API_SECRET = "dummy-api-secret"
_TEST_REQUEST_TOKEN = "dummy-request-token"
_TEST_ACCESS_TOKEN = "dummy-access-token"


def _authenticator() -> KiteConnectAuthenticator:
    return KiteConnectAuthenticator(
        KiteConnectCredentials(
            api_key=_TEST_API_KEY,
            api_secret=_TEST_API_SECRET,
        )
    )


def test_login_url_uses_api_key_and_kite_v3_login_route() -> None:
    parsed_url = urlparse(_authenticator().login_url())

    assert parsed_url.scheme == "https"
    assert parsed_url.netloc == "kite.zerodha.com"
    assert parsed_url.path == "/connect/login"
    assert parse_qs(parsed_url.query) == {"api_key": [_TEST_API_KEY], "v": ["3"]}


def test_exchange_posts_official_checksum_and_hides_access_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []

    class FakeResponse(BytesIO):
        status = 200

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse(
            json.dumps(
                {"status": "success", "data": {"access_token": _TEST_ACCESS_TOKEN}}
            ).encode("utf-8")
        )

    monkeypatch.setattr(kite_authentication, "urlopen", fake_urlopen)

    access_token = _authenticator().exchange_request_token(_TEST_REQUEST_TOKEN)

    request, timeout = requests[0]
    form_data = parse_qs(request.data.decode("ascii"))
    expected_checksum = hashlib.sha256(
        (_TEST_API_KEY + _TEST_REQUEST_TOKEN + _TEST_API_SECRET).encode("utf-8")
    ).hexdigest()
    assert request.full_url == "https://api.kite.trade/session/token"
    assert request.get_method() == "POST"
    assert request.get_header("X-kite-version") == "3"
    assert form_data == {
        "api_key": [_TEST_API_KEY],
        "request_token": [_TEST_REQUEST_TOKEN],
        "checksum": [expected_checksum],
    }
    assert _TEST_API_SECRET not in request.data.decode("ascii")
    assert timeout == 10
    assert access_token.value == _TEST_ACCESS_TOKEN
    assert _TEST_ACCESS_TOKEN not in repr(access_token)


def test_exchange_sanitizes_transport_errors_and_suppresses_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_with_sensitive_values(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(
            f"{_TEST_REQUEST_TOKEN} {_TEST_API_SECRET} {_TEST_ACCESS_TOKEN}"
        )

    monkeypatch.setattr(kite_authentication, "urlopen", fail_with_sensitive_values)

    with pytest.raises(AuthenticationError) as error:
        _authenticator().exchange_request_token(_TEST_REQUEST_TOKEN)

    message = str(error.value)
    assert message == "Kite request-token exchange failed"
    assert _TEST_REQUEST_TOKEN not in message
    assert _TEST_API_SECRET not in message
    assert _TEST_ACCESS_TOKEN not in message
    assert error.value.__cause__ is None
    assert error.value.category == "unknown_authentication_failure"


def test_exchange_rejects_empty_request_token_without_network_call() -> None:
    with pytest.raises(AuthenticationError, match="must not be empty"):
        _authenticator().exchange_request_token("   ")


@pytest.mark.parametrize(
    "payload",
    [b"not-json", b'{"status":"success","data":{}}'],
)
def test_malformed_exchange_responses_are_categorized(
    monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> None:
    class FakeResponse(BytesIO):
        status = 200

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

    monkeypatch.setattr(
        kite_authentication, "urlopen", lambda *_args, **_kwargs: FakeResponse(payload)
    )

    with pytest.raises(AuthenticationError) as error:
        _authenticator().exchange_request_token(_TEST_REQUEST_TOKEN)

    assert str(error.value) == "Kite request-token exchange failed"
    assert error.value.category == "malformed_provider_response"
    assert error.value.__cause__ is None


def test_http_rate_limit_response_is_categorized_without_exposing_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_body = json.dumps(
        {
            "status": "error",
            "message": f"denied {_TEST_REQUEST_TOKEN} {_TEST_API_SECRET}",
            "error_type": "GeneralException",
        }
    ).encode()

    def fail_with_http_error(*_args: object, **_kwargs: object) -> None:
        raise HTTPError(
            "https://api.kite.trade/session/token",
            429,
            "too many requests",
            {},
            BytesIO(sensitive_body),
        )

    monkeypatch.setattr(kite_authentication, "urlopen", fail_with_http_error)

    with pytest.raises(AuthenticationError) as error:
        _authenticator().exchange_request_token(_TEST_REQUEST_TOKEN)

    assert error.value.category == "rate_limit"
    assert str(error.value) == "Kite request-token exchange failed"
    assert _TEST_REQUEST_TOKEN not in str(error.value)
    assert _TEST_API_SECRET not in str(error.value)
    assert error.value.__cause__ is None


def test_structured_checksum_error_is_categorized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse(BytesIO):
        status = 200

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

    monkeypatch.setattr(
        kite_authentication,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(
            json.dumps(
                {
                    "status": "error",
                    "message": f"Invalid checksum {_TEST_API_SECRET}",
                    "error_type": "InputException",
                }
            ).encode(),
        ),
    )

    with pytest.raises(AuthenticationError) as error:
        _authenticator().exchange_request_token(_TEST_REQUEST_TOKEN)

    assert error.value.category == "checksum_or_credentials"
    assert _TEST_API_SECRET not in str(error.value)


@pytest.mark.parametrize("status", [400, 403])
def test_non_success_http_status_is_checked(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    class FakeResponse(BytesIO):
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

    response = FakeResponse(b'{"status":"success","data":{}}')
    response.status = status
    monkeypatch.setattr(
        kite_authentication, "urlopen", lambda *_args, **_kwargs: response
    )

    with pytest.raises(AuthenticationError) as error:
        _authenticator().exchange_request_token(_TEST_REQUEST_TOKEN)

    assert error.value.category == (
        "app_access_or_permission"
        if status == 403
        else "unknown_authentication_failure"
    )


def test_network_failure_has_sanitized_network_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise TimeoutError(f"timeout with {_TEST_REQUEST_TOKEN} {_TEST_API_SECRET}")

    monkeypatch.setattr(kite_authentication, "urlopen", fail)

    with pytest.raises(AuthenticationError) as error:
        _authenticator().exchange_request_token(_TEST_REQUEST_TOKEN)

    assert error.value.category == "network_or_timeout"
    assert str(error.value) == "Kite request-token exchange failed"
    assert _TEST_REQUEST_TOKEN not in str(error.value)
    assert _TEST_API_SECRET not in str(error.value)
    assert error.value.__cause__ is None
