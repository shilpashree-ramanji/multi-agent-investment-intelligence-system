"""Optional OpenAI Responses API adapter for the provider-neutral model contract."""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import os
from typing import Protocol, cast

from multi_agent_investment_system.core.exceptions import (
    ConfigurationError,
    DataValidationError,
    StockInvestmentError,
)
from multi_agent_investment_system.llm.provider import (
    ChatMessage,
    ModelRequest,
    ModelResponse,
)

DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_OPENAI_TIMEOUT_SECONDS = 30.0
_SUPPORTED_SCHEMA_TYPES = frozenset(
    {"string", "number", "integer", "boolean", "object", "array", "null"}
)
_SUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "anyOf",
        "description",
        "title",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "pattern",
        "format",
        "minItems",
        "maxItems",
    }
)


class OpenAIModelProviderError(StockInvestmentError):
    """Sanitized error from an OpenAI request or response."""

    def __init__(self, message: str, *, category: str) -> None:
        super().__init__(message)
        self.category = category


class _ResponsesEndpoint(Protocol):
    async def create(self, **kwargs: object) -> object: ...


class _AsyncClient(Protocol):
    responses: _ResponsesEndpoint


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _validate_schema_node(node: object, *, root: bool = False, depth: int = 0) -> None:
    if depth > 10 or not isinstance(node, dict):
        raise DataValidationError("OpenAI output schema has an invalid structure")
    if any(not isinstance(key, str) for key in node):
        raise DataValidationError("OpenAI output schema has an invalid key")
    if set(node) - _SUPPORTED_SCHEMA_KEYS:
        raise DataValidationError(
            "OpenAI output schema uses an unsupported schema keyword"
        )

    schema_type = node.get("type")
    if root and schema_type != "object":
        raise DataValidationError("OpenAI output schema root must be an object")
    if schema_type is not None:
        types = schema_type if isinstance(schema_type, list) else [schema_type]
        if (
            not types
            or any(
                not isinstance(value, str) or value not in _SUPPORTED_SCHEMA_TYPES
                for value in types
            )
            or len(set(types)) != len(types)
        ):
            raise DataValidationError("OpenAI output schema has an invalid type")

    if "enum" in node and (not isinstance(node["enum"], list) or not node["enum"]):
        raise DataValidationError("OpenAI output schema has an invalid enum")
    if "description" in node and not isinstance(node["description"], str):
        raise DataValidationError("OpenAI output schema description must be text")
    if "title" in node and not isinstance(node["title"], str):
        raise DataValidationError("OpenAI output schema title must be text")

    if schema_type == "object" or "properties" in node:
        properties = node.get("properties")
        required = node.get("required")
        if (
            schema_type not in ("object", ["object", "null"], ["null", "object"])
            or not isinstance(properties, dict)
            or not isinstance(required, list)
            or any(not isinstance(key, str) for key in properties)
            or any(not isinstance(key, str) for key in required)
            or len(set(required)) != len(required)
            or set(required) != set(properties)
            or node.get("additionalProperties") is not False
        ):
            raise DataValidationError(
                "OpenAI strict object schemas require all properties and "
                "additionalProperties=false"
            )
        for child in properties.values():
            _validate_schema_node(child, depth=depth + 1)
    elif "required" in node or "additionalProperties" in node:
        raise DataValidationError(
            "OpenAI required/additionalProperties keywords need an object schema"
        )

    if schema_type == "array":
        if "items" not in node:
            raise DataValidationError("OpenAI array schema must define items")
        _validate_schema_node(node["items"], depth=depth + 1)
    elif "items" in node:
        raise DataValidationError("OpenAI items keyword needs an array schema")

    if "anyOf" in node:
        alternatives = node["anyOf"]
        if root or not isinstance(alternatives, list) or not alternatives:
            raise DataValidationError("OpenAI output schema has an invalid anyOf")
        for child in alternatives:
            _validate_schema_node(child, depth=depth + 1)


def _parse_output_schema(raw_schema: str) -> dict[str, object]:
    try:
        schema = json.loads(
            raw_schema,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        raise DataValidationError("OpenAI output schema is not valid JSON") from None
    _validate_schema_node(schema, root=True)
    return cast(dict[str, object], schema)


def _contains_refusal(response: object) -> bool:
    output = getattr(response, "output", ())
    if not isinstance(output, (list, tuple)):
        return False
    for item in output:
        content = getattr(item, "content", ())
        if getattr(item, "type", None) == "refusal":
            return True
        if isinstance(content, (list, tuple)) and any(
            getattr(part, "type", None) == "refusal" for part in content
        ):
            return True
    return False


class OpenAIModelProvider:
    """Async OpenAI Responses adapter; importing it does not require the SDK.

    ``client`` may be injected for offline tests. Without an injected client,
    the optional SDK is loaded only when this class is constructed and the key
    is read from ``OPENAI_API_KEY``.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        client: _AsyncClient | None = None,
        timeout_seconds: float = DEFAULT_OPENAI_TIMEOUT_SECONDS,
    ) -> None:
        configured_model = (
            model
            if model is not None
            else os.environ.get("OPENAI_MODEL", "").strip() or DEFAULT_OPENAI_MODEL
        )
        if not isinstance(configured_model, str) or not configured_model.strip():
            raise ConfigurationError("OPENAI_MODEL must not be empty")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ConfigurationError("OpenAI timeout must be a positive finite number")

        self._model = configured_model.strip()
        self._timeout_seconds = float(timeout_seconds)
        self._sdk_error_type: type[BaseException] | None = None
        self._sdk_timeout_type: type[BaseException] | None = None
        self._client = client if client is not None else self._create_client()
        create = getattr(getattr(self._client, "responses", None), "create", None)
        if not callable(create):
            raise ConfigurationError("OpenAI client does not expose Responses create")

    def _create_client(self) -> _AsyncClient:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY must be configured")
        try:
            sdk = importlib.import_module("openai")
        except ImportError:
            raise ConfigurationError(
                "OpenAI SDK is unavailable; install the optional 'openai' dependency"
            ) from None

        client_type = getattr(sdk, "AsyncOpenAI", None)
        if not callable(client_type):
            raise ConfigurationError("Installed OpenAI SDK lacks AsyncOpenAI")
        self._sdk_error_type = getattr(sdk, "OpenAIError", None)
        self._sdk_timeout_type = getattr(sdk, "APITimeoutError", None)
        try:
            client = client_type(
                api_key=api_key,
                timeout=self._timeout_seconds,
                max_retries=0,
            )
        except Exception:
            raise ConfigurationError("Could not initialize the OpenAI client") from None
        return cast(_AsyncClient, client)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate text and reject refusals, incomplete, or empty responses."""

        self._validate_request(request)
        payload: dict[str, object] = {
            "model": self._model,
            "input": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
        }
        if request.output_schema is not None:
            schema = _parse_output_schema(request.output_schema)
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "agent_response",
                    "strict": True,
                    "schema": schema,
                }
            }

        try:
            response = await asyncio.wait_for(
                self._client.responses.create(**payload),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            raise OpenAIModelProviderError(
                "OpenAI request timed out", category="timeout"
            ) from None
        except Exception as error:
            if self._is_sdk_timeout(error):
                raise OpenAIModelProviderError(
                    "OpenAI request timed out", category="timeout"
                ) from None
            category = "sdk_error" if self._is_sdk_error(error) else "request_error"
            raise OpenAIModelProviderError(
                "OpenAI request failed", category=category
            ) from None

        status = getattr(response, "status", None)
        if status != "completed":
            if status == "incomplete":
                raise OpenAIModelProviderError(
                    "OpenAI returned an incomplete response", category="incomplete"
                )
            raise OpenAIModelProviderError(
                "OpenAI returned an unsuccessful response", category="response_error"
            )
        if _contains_refusal(response):
            raise OpenAIModelProviderError(
                "OpenAI refused the request", category="refusal"
            )

        content = getattr(response, "output_text", None)
        if not isinstance(content, str) or not content.strip():
            raise OpenAIModelProviderError(
                "OpenAI returned empty output", category="empty_output"
            )
        response_model = getattr(response, "model", None)
        model_id = (
            response_model.strip()
            if isinstance(response_model, str) and response_model.strip()
            else self._model
        )
        return ModelResponse(
            content=content,
            provider_id="openai",
            model_id=model_id,
        )

    @staticmethod
    def _validate_request(request: ModelRequest) -> None:
        if not isinstance(request, ModelRequest):
            raise DataValidationError("OpenAI model request is malformed")
        if (
            not isinstance(request.logical_role, str)
            or not request.logical_role.strip()
            or not isinstance(request.messages, tuple)
            or not request.messages
            or any(
                not isinstance(message, ChatMessage)
                or not isinstance(message.role, str)
                or message.role not in {"system", "user", "assistant"}
                or not isinstance(message.content, str)
                for message in request.messages
            )
        ):
            raise DataValidationError("OpenAI model request is malformed")
        if request.output_schema is not None and not isinstance(
            request.output_schema, str
        ):
            raise DataValidationError("OpenAI output schema must be text")

    def _is_sdk_error(self, error: BaseException) -> bool:
        return isinstance(self._sdk_error_type, type) and isinstance(
            error, self._sdk_error_type
        )

    def _is_sdk_timeout(self, error: BaseException) -> bool:
        return isinstance(self._sdk_timeout_type, type) and isinstance(
            error, self._sdk_timeout_type
        )
