"""Offline tests for the optional OpenAI Responses API adapter."""

from __future__ import annotations

import asyncio
import builtins
from types import SimpleNamespace

import pytest

from multi_agent_investment_system.core.exceptions import (
    ConfigurationError,
    DataValidationError,
)
from multi_agent_investment_system.llm.openai_provider import (
    DEFAULT_OPENAI_MODEL,
    OpenAIModelProvider,
    OpenAIModelProviderError,
)
from multi_agent_investment_system.llm.provider import (
    ChatMessage,
    ModelRequest,
    ModelResponse,
)

_SCHEMA = (
    '{"type":"object","additionalProperties":false,'
    '"required":["action","task_id"],"properties":{'
    '"action":{"const":"execute_ingestion"},'
    '"task_id":{"type":"string"}}}'
)


class _FakeResponses:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeClient:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.responses = _FakeResponses(response, error)


def _response(
    text: str = '{"action":"complete"}',
    *,
    status: str = "completed",
    model: str = "gpt-6-luna-2026-10-01",
    output=(),
):
    return SimpleNamespace(
        status=status,
        output_text=text,
        model=model,
        output=output,
    )


def _request(schema: str | None = None) -> ModelRequest:
    return ModelRequest(
        messages=(
            ChatMessage("system", "Return a constrained action."),
            ChatMessage("user", "Continue task-1."),
        ),
        logical_role="coordinator",
        output_schema=schema,
    )


def test_maps_model_messages_to_responses_input_and_returns_identifiers(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-6-luna-configured")
    response = _response()
    client = _FakeClient(response)
    provider = OpenAIModelProvider(client=client)

    result = asyncio.run(provider.generate(_request()))

    assert client.responses.calls == [
        {
            "model": "gpt-6-luna-configured",
            "input": [
                {"role": "system", "content": "Return a constrained action."},
                {"role": "user", "content": "Continue task-1."},
            ],
        }
    ]
    assert result == ModelResponse(
        '{"action":"complete"}',
        provider_id="openai",
        model_id="gpt-6-luna-2026-10-01",
    )


def test_schema_is_validated_and_sent_as_strict_structured_output() -> None:
    client = _FakeClient(_response())
    provider = OpenAIModelProvider(client=client, model="gpt-6-luna")

    asyncio.run(provider.generate(_request(_SCHEMA)))

    assert client.responses.calls[0]["text"] == {
        "format": {
            "type": "json_schema",
            "name": "agent_response",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["action", "task_id"],
                "properties": {
                    "action": {"const": "execute_ingestion"},
                    "task_id": {"type": "string"},
                },
            },
        }
    }


@pytest.mark.parametrize(
    "schema",
    [
        "not-json",
        '{"type":"string"}',
        '{"type":"object","properties":{"x":{"type":"string"}},'
        '"required":[],"additionalProperties":false}',
        '{"type":"object","properties":{},"required":[]}',
        '{"type":"object","type":"string"}',
    ],
)
def test_invalid_or_unsupported_schema_fails_before_api_call(schema: str) -> None:
    client = _FakeClient(_response())
    provider = OpenAIModelProvider(client=client)

    with pytest.raises(DataValidationError):
        asyncio.run(provider.generate(_request(schema)))

    assert client.responses.calls == []


def test_missing_api_key_is_actionable(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        OpenAIModelProvider()


def test_missing_optional_sdk_is_actionable_without_network(monkeypatch) -> None:
    original_import = builtins.__import__

    def without_openai(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("test simulates optional SDK absence")
        return original_import(name, *args, **kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setattr(builtins, "__import__", without_openai)

    with pytest.raises(ConfigurationError, match="optional 'openai' dependency"):
        OpenAIModelProvider()


@pytest.mark.parametrize(
    ("response", "category"),
    [
        (_response(""), "empty_output"),
        (
            _response(
                "",
                output=(
                    SimpleNamespace(
                        type="message",
                        content=(SimpleNamespace(type="refusal", refusal="private"),),
                    ),
                ),
            ),
            "refusal",
        ),
        (_response("partial", status="incomplete"), "incomplete"),
        (_response("", status="failed"), "response_error"),
    ],
)
def test_empty_refused_and_unsuccessful_responses_are_errors(
    response, category: str
) -> None:
    provider = OpenAIModelProvider(client=_FakeClient(response))

    with pytest.raises(OpenAIModelProviderError) as error:
        asyncio.run(provider.generate(_request()))

    assert error.value.category == category


def test_timeout_is_bounded_and_reported_without_sdk_details() -> None:
    async def delayed_response(**kwargs):
        await asyncio.sleep(1)

    client = SimpleNamespace(responses=SimpleNamespace(create=delayed_response))
    provider = OpenAIModelProvider(client=client, timeout_seconds=0.001)

    with pytest.raises(OpenAIModelProviderError) as error:
        asyncio.run(provider.generate(_request()))

    assert error.value.category == "timeout"
    assert str(error.value) == "OpenAI request timed out"


def test_sdk_failures_are_sanitized() -> None:
    class FakeSDKError(Exception):
        pass

    client = _FakeClient(error=FakeSDKError("secret header and prompt details"))
    provider = OpenAIModelProvider(client=client)
    provider._sdk_error_type = FakeSDKError

    with pytest.raises(OpenAIModelProviderError) as error:
        asyncio.run(provider.generate(_request()))

    assert error.value.category == "sdk_error"
    assert str(error.value) == "OpenAI request failed"
    assert "secret header" not in str(error.value)


def test_injected_client_keeps_tests_offline_and_default_model_is_configurable(
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    client = _FakeClient(_response(model=""))
    provider = OpenAIModelProvider(client=client)

    result = asyncio.run(provider.generate(_request()))

    assert DEFAULT_OPENAI_MODEL == "gpt-6-luna"
    assert client.responses.calls[0]["model"] == "gpt-6-luna"
    assert result.model_id == "gpt-6-luna"
    assert len(client.responses.calls) == 1
