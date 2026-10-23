"""Strict parsing for small model action envelopes."""

from __future__ import annotations

import json

from multi_agent_investment_system.llm.provider import (
    ModelOutputRejectionCategory,
    ModelResponse,
)

MAX_MODEL_RESPONSE_BYTES = 4096


class CollectionPlanValidationError(ValueError):
    """Safe parser failure carrying only a fixed rejection category."""

    def __init__(self, category: ModelOutputRejectionCategory) -> None:
        self.category = category
        super().__init__("Model collection plan failed validation")


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


def parse_collection_plan(
    response: ModelResponse,
    *,
    allowed_actions: frozenset[str],
    task_id: str,
    maximum_response_bytes: int = MAX_MODEL_RESPONSE_BYTES,
) -> tuple[str, str, str]:
    """Parse an action plus bounded rationale from the collection Coordinator."""

    if not isinstance(response, ModelResponse) or not isinstance(response.content, str):
        raise CollectionPlanValidationError(
            ModelOutputRejectionCategory.INVALID_RESPONSE
        )
    try:
        size = len(response.content.encode("utf-8"))
    except UnicodeEncodeError:
        raise CollectionPlanValidationError(
            ModelOutputRejectionCategory.INVALID_UTF8
        ) from None
    if size > maximum_response_bytes:
        raise CollectionPlanValidationError(
            ModelOutputRejectionCategory.RESPONSE_TOO_LARGE
        )
    try:
        payload = json.loads(
            response.content,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        raise CollectionPlanValidationError(
            ModelOutputRejectionCategory.INVALID_JSON
        ) from None
    if not isinstance(payload, dict) or set(payload) != {
        "action",
        "task_id",
        "reason",
        "expected_outcome",
        "completion_condition",
    }:
        raise CollectionPlanValidationError(
            ModelOutputRejectionCategory.UNEXPECTED_SHAPE
        )
    action = payload["action"]
    reason = payload["reason"]
    expected_outcome = payload["expected_outcome"]
    if not isinstance(action, str) or action not in allowed_actions:
        raise CollectionPlanValidationError(
            ModelOutputRejectionCategory.DISALLOWED_ACTION
        )
    if payload["task_id"] != task_id:
        raise CollectionPlanValidationError(
            ModelOutputRejectionCategory.TASK_ID_MISMATCH
        )
    for value in (reason, expected_outcome):
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 300
            or any(
                ord(character) < 32 and character not in "\t\n\r" for character in value
            )
        ):
            raise CollectionPlanValidationError(
                ModelOutputRejectionCategory.INVALID_RATIONALE
            )
    expected_condition = (
        "quality_clean_and_persisted"
        if action in {"collect_range", "retry_collection"}
        else "stop_without_coverage_claim"
    )
    if payload["completion_condition"] != expected_condition:
        raise CollectionPlanValidationError(
            ModelOutputRejectionCategory.COMPLETION_CONDITION_MISMATCH
        )
    return action, reason.strip(), expected_outcome.strip()
