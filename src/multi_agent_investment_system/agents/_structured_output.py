"""Strict parsing for small model action envelopes."""

from __future__ import annotations

import json

from multi_agent_investment_system.llm.provider import ModelResponse

MAX_MODEL_RESPONSE_BYTES = 4096


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Model response contains duplicate JSON keys")
        result[key] = value
    return result


def parse_action(
    response: ModelResponse,
    *,
    allowed_actions: frozenset[str],
    task_id: str | None = None,
    maximum_response_bytes: int = MAX_MODEL_RESPONSE_BYTES,
) -> str:
    """Accept only an exact JSON object containing an allow-listed action."""

    if not isinstance(response, ModelResponse) or not isinstance(response.content, str):
        raise ValueError("Model response is not a valid action response")
    if (
        isinstance(maximum_response_bytes, bool)
        or not isinstance(maximum_response_bytes, int)
        or maximum_response_bytes < 1
    ):
        raise ValueError("Model response size limit is invalid")
    try:
        response_size = len(response.content.encode("utf-8"))
    except UnicodeEncodeError:
        raise ValueError("Model response is not valid UTF-8 text") from None
    if response_size > maximum_response_bytes:
        raise ValueError("Model response exceeds the configured size limit")
    try:
        payload = json.loads(
            response.content,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        raise ValueError("Model response is not valid JSON") from None
    expected_keys = {"action"} if task_id is None else {"action", "task_id"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("Model response has an unexpected shape")
    action = payload.get("action")
    if not isinstance(action, str) or action not in allowed_actions:
        raise ValueError("Model response contains a disallowed action")
    if task_id is not None and payload.get("task_id") != task_id:
        raise ValueError("Model response task does not match the assigned task")
    return action
