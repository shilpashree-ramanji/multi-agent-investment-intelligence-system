"""Offline behavior tests for the Coordinator and Kite ingestion worker."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from multi_agent_investment_system.agents._structured_output import (
    CollectionPlanValidationError,
    parse_collection_plan,
)
from multi_agent_investment_system.agents.coordinator import CoordinatorAgent
from multi_agent_investment_system.agents.zerodha_ingestion_worker import (
    ZerodhaIngestionWorkerAgent,
)
from multi_agent_investment_system.core.exceptions import (
    DataValidationError,
    MarketDataErrorCategory,
    MarketDataProviderError,
)
from multi_agent_investment_system.data.ingestion import (
    DailyMarketDataIngestionService,
)
from multi_agent_investment_system.data.service import (
    DailyMarketDataExtractionService,
)
from multi_agent_investment_system.data.storage import SQLiteDailyCandleStore
from multi_agent_investment_system.llm.openai_provider import OpenAIModelProvider
from multi_agent_investment_system.llm.provider import (
    ModelOutputRejectionCategory,
    ModelProviderDiagnostic,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
)
from multi_agent_investment_system.orchestration.kite_ingestion import (
    KiteIngestionAssignment,
    KiteIngestionPolicy,
    KiteIngestionRequest,
    KiteWorkflowErrorCode,
    KiteWorkflowStatus,
    WorkerTaskStatus,
)
from multi_agent_investment_system.tools.market_data import MarketDataProvider


def _request(**overrides: object) -> KiteIngestionRequest:
    values: dict[str, object] = {
        "run_id": "run-1",
        "task_id": "task-1",
        "symbol": "RELIANCE",
        "start": date(2026, 10, 20),
        "end": date(2026, 10, 21),
    }
    values.update(overrides)
    return KiteIngestionRequest(**values)  # type: ignore[arg-type]


def _candle(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "symbol": "RELIANCE",
        "observed_at": datetime(2026, 10, 20, 10, tzinfo=UTC),
        "open": Decimal("100.00"),
        "high": Decimal("105.00"),
        "low": Decimal("99.00"),
        "close": Decimal("103.25"),
        "volume": 1000,
        "source": "fake-kite",
        "retrieved_at": datetime(2026, 10, 21, 8, tzinfo=UTC),
        "instrument_token": 12345,
    }
    record.update(overrides)
    return record


def _plan(action: str, task_id: str = "task-1") -> str:
    condition = (
        "quality_clean_and_persisted"
        if action in {"collect_range", "retry_collection"}
        else "stop_without_coverage_claim"
    )
    return json.dumps(
        {
            "action": action,
            "task_id": task_id,
            "reason": f"Observed state supports {action}",
            "expected_outcome": "A bounded, inspectable result",
            "completion_condition": condition,
        }
    )


@pytest.mark.parametrize(
    ("response", "allowed_actions", "task_id", "maximum_response_bytes", "category"),
    [
        (
            object(),
            frozenset({"escalate"}),
            "task-1",
            4096,
            ModelOutputRejectionCategory.INVALID_RESPONSE,
        ),
        (
            ModelResponse("malformed fake_api_key_and_prompt"),
            frozenset({"escalate"}),
            "task-1",
            4096,
            ModelOutputRejectionCategory.INVALID_JSON,
        ),
        (
            ModelResponse('{"action":"escalate"}'),
            frozenset({"escalate"}),
            "task-1",
            4096,
            ModelOutputRejectionCategory.UNEXPECTED_SHAPE,
        ),
        (
            ModelResponse(_plan("unauthorized_action")),
            frozenset({"escalate"}),
            "task-1",
            4096,
            ModelOutputRejectionCategory.DISALLOWED_ACTION,
        ),
        (
            ModelResponse(_plan("escalate", task_id="other-task")),
            frozenset({"escalate"}),
            "task-1",
            4096,
            ModelOutputRejectionCategory.TASK_ID_MISMATCH,
        ),
        (
            ModelResponse(
                json.dumps(
                    {
                        "action": "escalate",
                        "task_id": "task-1",
                        "reason": " ",
                        "expected_outcome": "No collection occurs",
                        "completion_condition": "stop_without_coverage_claim",
                    }
                )
            ),
            frozenset({"escalate"}),
            "task-1",
            4096,
            ModelOutputRejectionCategory.INVALID_RATIONALE,
        ),
        (
            ModelResponse(
                json.dumps(
                    {
                        "action": "escalate",
                        "task_id": "task-1",
                        "reason": "Review is required",
                        "expected_outcome": "No collection occurs",
                        "completion_condition": "quality_clean_and_persisted",
                    }
                )
            ),
            frozenset({"escalate"}),
            "task-1",
            4096,
            ModelOutputRejectionCategory.COMPLETION_CONDITION_MISMATCH,
        ),
        (
            ModelResponse("{}"),
            frozenset({"escalate"}),
            "task-1",
            1,
            ModelOutputRejectionCategory.RESPONSE_TOO_LARGE,
        ),
        (
            ModelResponse("\ud800"),
            frozenset({"escalate"}),
            "task-1",
            4096,
            ModelOutputRejectionCategory.INVALID_UTF8,
        ),
    ],
)
def test_collection_plan_parser_exposes_only_fixed_rejection_categories(
    response,
    allowed_actions,
    task_id,
    maximum_response_bytes,
    category,
) -> None:
    with pytest.raises(CollectionPlanValidationError) as error:
        parse_collection_plan(
            response,
            allowed_actions=allowed_actions,
            task_id=task_id,
            maximum_response_bytes=maximum_response_bytes,
        )

    assert error.value.category is category
    assert "fake_api_key_and_prompt" not in str(error.value)


@pytest.mark.parametrize(
    ("action", "condition"),
    [
        ("collect_range", "quality_clean_and_persisted"),
        ("retry_collection", "quality_clean_and_persisted"),
        ("complete", "stop_without_coverage_claim"),
        ("escalate", "stop_without_coverage_claim"),
    ],
)
def test_collection_plan_parser_accepts_matching_action_condition_pairs(
    action, condition
) -> None:
    payload = json.loads(_plan(action))
    payload["completion_condition"] = condition

    parsed_action, _, _ = parse_collection_plan(
        ModelResponse(json.dumps(payload)),
        allowed_actions=frozenset({action}),
        task_id="task-1",
    )

    assert parsed_action == action


@pytest.mark.parametrize(
    ("action", "condition"),
    [
        ("collect_range", "stop_without_coverage_claim"),
        ("retry_collection", "stop_without_coverage_claim"),
        ("complete", "quality_clean_and_persisted"),
        ("escalate", "quality_clean_and_persisted"),
    ],
)
def test_collection_plan_parser_rejects_inconsistent_action_condition_pairs(
    action, condition
) -> None:
    payload = json.loads(_plan(action))
    payload["completion_condition"] = condition

    with pytest.raises(CollectionPlanValidationError) as error:
        parse_collection_plan(
            ModelResponse(json.dumps(payload)),
            allowed_actions=frozenset({action}),
            task_id="task-1",
        )

    assert (
        error.value.category
        is ModelOutputRejectionCategory.COMPLETION_CONDITION_MISMATCH
    )
class FakeModelProvider:
    def __init__(self, *contents: str, error: Exception | None = None) -> None:
        self.contents = list(contents)
        self.error = error
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        if not self.contents:
            raise AssertionError("Unexpected model request")
        return ModelResponse(
            self.contents.pop(0), provider_id="offline-fake", model_id="test-model"
        )


class CapturingResponsesAPI:
    def __init__(
        self, output_text: str, *, error: Exception | None = None
    ) -> None:
        self.output_text = output_text
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            status="completed",
            output_text=self.output_text,
            model="offline-test-model",
            output=(),
        )


class FakeMarketDataProvider(MarketDataProvider):
    def __init__(
        self,
        records: Sequence[Mapping[str, object]] = (),
        error: Exception | None = None,
    ) -> None:
        self.records = records
        self.error = error
        self.calls: list[tuple[str, date, date]] = []

    async def get_daily_candles(
        self, symbol: str, start: date, end: date
    ) -> Sequence[Mapping[str, object]]:
        self.calls.append((symbol, start, end))
        if self.error is not None:
            raise self.error
        return self.records


def _agents(tmp_path, records=(), *, market_error=None, coordinator_outputs=None):
    market_provider = FakeMarketDataProvider(records, market_error)
    store = SQLiteDailyCandleStore(tmp_path / "market.sqlite3")
    service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(market_provider), store
    )
    policy = KiteIngestionPolicy(maximum_calendar_days=5)
    worker_model = FakeModelProvider(
        json.dumps({"action": "execute_ingestion", "task_id": "task-1"})
    )
    worker = ZerodhaIngestionWorkerAgent(worker_model, service, policy)
    outputs = coordinator_outputs or (
        _plan("collect_range"),
        _plan("complete"),
    )
    coordinator_model = FakeModelProvider(*outputs)
    coordinator = CoordinatorAgent(coordinator_model, worker, policy)
    return coordinator, coordinator_model, worker_model, market_provider, store


def test_coordinator_delegates_to_model_backed_worker_and_ingests_offline(
    tmp_path,
) -> None:
    coordinator, coordinator_model, worker_model, market, store = _agents(
        tmp_path,
        [_candle()],
    )

    state = asyncio.run(coordinator.run(_request()))

    assert state.status is KiteWorkflowStatus.COMPLETED
    assert state.transitions == (
        KiteWorkflowStatus.PENDING,
        KiteWorkflowStatus.COORDINATING,
        KiteWorkflowStatus.ASSIGNED,
        KiteWorkflowStatus.WORKER_RUNNING,
        KiteWorkflowStatus.REVIEWING,
        KiteWorkflowStatus.COMPLETED,
    )
    assert len(state.assignments) == 1
    assert state.worker_result is not None
    assert state.worker_result.status is WorkerTaskStatus.SUCCEEDED
    assert state.worker_result.counts is not None
    assert state.worker_result.counts.stored == 1
    assert state.worker_result.evidence[0].source == "fake-kite"
    assert [item.logical_role for item in coordinator_model.requests] == [
        "coordinator",
        "coordinator",
    ]
    assert [item.logical_role for item in worker_model.requests] == [
        "zerodha_ingestion_worker"
    ]
    from multi_agent_investment_system.llm.openai_provider import (
        _parse_output_schema,
    )

    _parse_output_schema(coordinator_model.requests[0].output_schema or "")
    assert market.calls == [("RELIANCE", date(2026, 10, 20), date(2026, 10, 21))]
    assert (
        store.fetch_daily_candles("RELIANCE", date(2026, 10, 20), date(2026, 10, 21))[
            0
        ]["source"]
        == "fake-kite"
    )


@pytest.mark.parametrize(
    ("plan", "expected_status", "expected_error"),
    [
        (_plan("escalate"), KiteWorkflowStatus.ESCALATED, None),
        (
            _plan("escalate", task_id="different-task"),
            KiteWorkflowStatus.FAILED,
            KiteWorkflowErrorCode.INVALID_MODEL_OUTPUT,
        ),
        (
            json.dumps(
                {
                    "action": "escalate",
                    "task_id": "task-1",
                    "reason": "Review is required",
                    "expected_outcome": "No collection is performed",
                    "completion_condition": "quality_clean_and_persisted",
                }
            ),
            KiteWorkflowStatus.FAILED,
            KiteWorkflowErrorCode.INVALID_MODEL_OUTPUT,
        ),
    ],
)
def test_coordinator_schema_is_sent_to_responses_and_plan_is_validated(
    tmp_path, plan, expected_status, expected_error
) -> None:
    market = FakeMarketDataProvider()
    store = SQLiteDailyCandleStore(tmp_path / "market.sqlite3")
    service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(market), store
    )
    policy = KiteIngestionPolicy(maximum_calendar_days=5)
    worker = ZerodhaIngestionWorkerAgent(
        FakeModelProvider(
            json.dumps({"action": "execute_ingestion", "task_id": "task-1"})
        ),
        service,
        policy,
    )
    responses = CapturingResponsesAPI(plan)
    coordinator = CoordinatorAgent(
        OpenAIModelProvider(
            client=SimpleNamespace(responses=responses), model="offline-test-model"
        ),
        worker,
        policy,
    )

    state = asyncio.run(coordinator.run(_request()))

    assert state.status is expected_status
    assert state.error_code is expected_error
    assert market.calls == []
    assert len(responses.calls) == 1
    model_input = responses.calls[0]["input"]
    assert isinstance(model_input, list)
    system_instruction = model_input[0]["content"]
    assert isinstance(system_instruction, str)
    assert "collect_range and retry_collection require" in system_instruction
    assert "complete and escalate require" in system_instruction
    assert "never claim full historical or session coverage" in system_instruction
    text = responses.calls[0]["text"]
    assert isinstance(text, dict)
    output_format = text["format"]
    assert isinstance(output_format, dict)
    assert output_format["type"] == "json_schema"
    assert output_format["strict"] is True
    schema = output_format["schema"]
    assert isinstance(schema, dict)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "action",
        "task_id",
        "reason",
        "expected_outcome",
        "completion_condition",
    ]
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert properties["action"] == {"enum": ["collect_range", "escalate"]}
    assert properties["task_id"] == {"enum": ["task-1"]}
    assert properties["completion_condition"] == {
        "enum": ["quality_clean_and_persisted", "stop_without_coverage_claim"]
    }


def test_coordinator_reports_parser_category_without_retaining_model_output(
    tmp_path,
) -> None:
    raw_model_output = 'malformed output fake_api_key secret prompt contents "'
    coordinator, _, _, market, _ = _agents(
        tmp_path,
        coordinator_outputs=(raw_model_output,),
    )

    state = asyncio.run(coordinator.run(_request()))

    assert state.status is KiteWorkflowStatus.FAILED
    assert state.error_code is KiteWorkflowErrorCode.INVALID_MODEL_OUTPUT
    rejected_trace = state.action_trace[-1]
    assert (
        rejected_trace.rejection_category
        is ModelOutputRejectionCategory.INVALID_JSON
    )
    assert rejected_trace.action.value == "rejected_model_output"
    assert not rejected_trace.authorized
    assert not rejected_trace.executed
    assert raw_model_output not in repr(state)
    assert market.calls == []


def test_coordinator_rejects_mismatched_condition_before_worker_invocation(
    tmp_path,
) -> None:
    invalid_plan = json.loads(_plan("collect_range"))
    invalid_plan["completion_condition"] = "stop_without_coverage_claim"
    coordinator, _, worker_model, market, _ = _agents(
        tmp_path,
        [_candle()],
        coordinator_outputs=(json.dumps(invalid_plan),),
    )

    state = asyncio.run(coordinator.run(_request()))

    assert state.error_code is KiteWorkflowErrorCode.INVALID_MODEL_OUTPUT
    assert (
        state.action_trace[-1].rejection_category
        is ModelOutputRejectionCategory.COMPLETION_CONDITION_MISMATCH
    )
    assert worker_model.requests == []
    assert market.calls == []


def test_coordinator_rejects_invalid_delegation_without_invoking_worker(
    tmp_path,
) -> None:
    coordinator, _, worker_model, market, _ = _agents(
        tmp_path,
        [_candle()],
        coordinator_outputs=(
            json.dumps({"action": "delegate_ingestion", "task_id": "other-task"}),
        ),
    )

    state = asyncio.run(coordinator.run(_request()))

    assert state.status is KiteWorkflowStatus.FAILED
    assert state.error_code is KiteWorkflowErrorCode.INVALID_MODEL_OUTPUT
    assert worker_model.requests == []
    assert market.calls == []


def test_worker_rejects_invalid_model_output_before_ingestion(tmp_path) -> None:
    market = FakeMarketDataProvider([_candle()])
    store = SQLiteDailyCandleStore(tmp_path / "market.sqlite3")
    service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(market), store
    )
    worker = ZerodhaIngestionWorkerAgent(
        FakeModelProvider(
            '{"action":"execute_ingestion","task_id":"task-1","start":"2020-01-01"}'
        ),
        service,
        KiteIngestionPolicy(maximum_calendar_days=5),
    )

    result = asyncio.run(
        worker.execute(
            _request(),
            KiteIngestionAssignment.for_request(_request()),
        )
    )

    assert result.status is WorkerTaskStatus.FAILED
    assert result.error_code is KiteWorkflowErrorCode.INVALID_MODEL_OUTPUT
    assert market.calls == []
    assert (
        store.fetch_daily_candles("RELIANCE", date(2026, 10, 20), date(2026, 10, 21))
        == ()
    )


def test_model_provider_failure_is_reported_without_market_data_call(tmp_path) -> None:
    market = FakeMarketDataProvider([_candle()])
    service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(market),
        SQLiteDailyCandleStore(tmp_path / "market.sqlite3"),
    )
    responses = CapturingResponsesAPI(
        "", error=RuntimeError("private model provider detail")
    )
    worker = ZerodhaIngestionWorkerAgent(
        OpenAIModelProvider(
            client=SimpleNamespace(responses=responses), model="offline-test-model"
        ),
        service,
        KiteIngestionPolicy(maximum_calendar_days=5),
    )

    result = asyncio.run(
        worker.execute(
            _request(),
            KiteIngestionAssignment.for_request(_request()),
        )
    )

    assert result.status is WorkerTaskStatus.FAILED
    assert result.error_code is KiteWorkflowErrorCode.MODEL_PROVIDER_ERROR
    assert result.model_provider_diagnostic.to_dict() == {
        "exception_class": "RuntimeError"
    }
    assert "private model provider detail" not in repr(result)
    assert len(responses.calls) == 1
    assert market.calls == []


def test_worker_sends_strict_single_action_schema_and_ingests_valid_response(
    tmp_path,
) -> None:
    market = FakeMarketDataProvider([_candle()])
    service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(market),
        SQLiteDailyCandleStore(tmp_path / "market.sqlite3"),
    )
    responses = CapturingResponsesAPI(
        '{"action":"execute_ingestion","task_id":"task-1"}'
    )
    worker = ZerodhaIngestionWorkerAgent(
        OpenAIModelProvider(
            client=SimpleNamespace(responses=responses), model="offline-test-model"
        ),
        service,
        KiteIngestionPolicy(maximum_calendar_days=5),
    )

    result = asyncio.run(
        worker.execute(
            _request(),
            KiteIngestionAssignment.for_request(_request()),
        )
    )

    assert result.status is WorkerTaskStatus.SUCCEEDED
    assert result.error_code is None
    assert market.calls == [("RELIANCE", date(2026, 10, 20), date(2026, 10, 21))]
    assert len(responses.calls) == 1
    request = responses.calls[0]
    model_input = request["input"]
    assert isinstance(model_input, list)
    assert model_input[0]["role"] == "system"
    assert model_input[1]["role"] == "user"
    assert json.loads(model_input[1]["content"])["task_id"] == "task-1"
    output_format = request["text"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["strict"] is True
    schema = output_format["schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["action", "task_id"]
    assert schema["properties"] == {
        "action": {"enum": ["execute_ingestion"]},
        "task_id": {"type": "string"},
    }


def test_worker_rejects_unassigned_task_id_before_ingestion(tmp_path) -> None:
    market = FakeMarketDataProvider([_candle()])
    service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(market),
        SQLiteDailyCandleStore(tmp_path / "market.sqlite3"),
    )
    responses = CapturingResponsesAPI(
        '{"action":"execute_ingestion","task_id":"different-task"}'
    )
    worker = ZerodhaIngestionWorkerAgent(
        OpenAIModelProvider(
            client=SimpleNamespace(responses=responses), model="offline-test-model"
        ),
        service,
        KiteIngestionPolicy(maximum_calendar_days=5),
    )

    result = asyncio.run(
        worker.execute(
            _request(),
            KiteIngestionAssignment.for_request(_request()),
        )
    )

    assert result.status is WorkerTaskStatus.FAILED
    assert result.error_code is KiteWorkflowErrorCode.INVALID_MODEL_OUTPUT
    assert len(responses.calls) == 1
    assert market.calls == []


def test_coordinator_preserves_safe_provider_diagnostic_without_raw_error(
    tmp_path,
) -> None:
    coordinator, _, _, market, _ = _agents(tmp_path)
    diagnostic = ModelProviderDiagnostic(
        exception_class="FakeAPIStatusError",
        http_status=400,
        provider_error_code="invalid_request_error",
        provider_error_type="invalid_request_error",
    )
    secret_content = "fake-key response-body prompt-secret"
    coordinator._model_provider = FakeModelProvider(
        error=ModelProviderError(
            secret_content,
            category="sdk_error",
            diagnostic=diagnostic,
        )
    )

    state = asyncio.run(coordinator.run(_request()))

    assert state.error_code is KiteWorkflowErrorCode.MODEL_PROVIDER_ERROR
    assert state.model_provider_diagnostic == diagnostic
    assert secret_content not in repr(state)
    assert market.calls == []


def test_coordinator_action_trace_discards_hostile_model_prose(tmp_path) -> None:
    hostile_values = (
        "fake_api_key_value",
        "fake_access_token_value",
        "Authorization: Bearer fake_header_value",
        "prompt and response body sentinel",
    )
    hostile_plan = json.dumps(
        {
            "action": "collect_range",
            "task_id": "task-1",
            "reason": " ".join(hostile_values),
            "expected_outcome": "leak " + " ".join(hostile_values),
            "completion_condition": "quality_clean_and_persisted",
        }
    )
    coordinator, _, _, _, _ = _agents(
        tmp_path,
        [_candle()],
        coordinator_outputs=(hostile_plan, _plan("complete")),
    )

    state = asyncio.run(coordinator.run(_request()))

    assert state.status is KiteWorkflowStatus.COMPLETED
    trace_text = repr(state.action_trace)
    assert all(value not in trace_text for value in hostile_values)


def test_invalid_and_incomplete_candles_are_not_persisted_and_are_escalated(
    tmp_path,
) -> None:
    incomplete = _candle(close=None)
    invalid = _candle(observed_at=datetime(2026, 10, 21, 10, tzinfo=UTC), volume=-1)
    coordinator, _, _, _, store = _agents(
        tmp_path,
        [incomplete, invalid],
        coordinator_outputs=(
            _plan("collect_range"),
            _plan("escalate"),
        ),
    )

    state = asyncio.run(coordinator.run(_request()))

    assert state.status is KiteWorkflowStatus.ESCALATED
    assert state.error_code is KiteWorkflowErrorCode.DATA_QUALITY_REVIEW
    assert state.worker_result is not None
    assert state.worker_result.counts is not None
    assert state.worker_result.counts.incomplete == 1
    assert state.worker_result.counts.invalid == 1
    assert state.worker_result.counts.stored == 0
    assert len(state.worker_result.evidence) == 2
    assert all(item.validation_issue_codes for item in state.worker_result.evidence)
    assert (
        store.fetch_daily_candles("RELIANCE", date(2026, 10, 20), date(2026, 10, 21))
        == ()
    )


def test_coordinator_cannot_complete_when_service_reports_invalid_candles(
    tmp_path,
) -> None:
    coordinator, _, _, _, store = _agents(
        tmp_path,
        [_candle(close=None)],
        coordinator_outputs=(
            _plan("collect_range"),
            _plan("complete"),
        ),
    )

    state = asyncio.run(coordinator.run(_request()))

    assert state.status is KiteWorkflowStatus.FAILED
    assert state.error_code is KiteWorkflowErrorCode.INVALID_MODEL_OUTPUT
    assert state.worker_result is not None
    assert state.worker_result.counts is not None
    assert state.worker_result.counts.stored == 0
    assert (
        store.fetch_daily_candles("RELIANCE", date(2026, 10, 20), date(2026, 10, 21))
        == ()
    )


def test_failed_market_request_is_sanitized_and_forces_escalation(tmp_path) -> None:
    coordinator, _, _, _, _ = _agents(
        tmp_path,
        market_error=MarketDataProviderError(
            "sensitive response body credential",
            retryable=True,
            category=MarketDataErrorCategory.HTTP_ERROR,
            http_status=503,
        ),
        coordinator_outputs=(
            _plan("collect_range"),
            _plan("escalate"),
        ),
    )

    state = asyncio.run(coordinator.run(_request()))

    assert state.status is KiteWorkflowStatus.ESCALATED
    assert state.error_code is KiteWorkflowErrorCode.MARKET_DATA_ERROR
    assert state.worker_result is not None
    assert state.worker_result.error_code is KiteWorkflowErrorCode.MARKET_DATA_ERROR
    assert state.worker_result.retryable is True
    assert state.worker_result.market_data_diagnostic is not None
    assert state.worker_result.market_data_diagnostic.to_dict() == {
        "category": "http_error",
        "http_status": 503,
    }
    assert "sensitive response body credential" not in repr(state)


def test_unexpected_extraction_failure_has_fixed_sanitized_category(tmp_path) -> None:
    coordinator, _, _, _, _ = _agents(
        tmp_path,
        market_error=RuntimeError("sensitive response body credential"),
        coordinator_outputs=(_plan("collect_range"), _plan("escalate")),
    )

    state = asyncio.run(coordinator.run(_request()))

    assert state.error_code is KiteWorkflowErrorCode.MARKET_DATA_ERROR
    assert state.worker_result is not None
    assert state.worker_result.retryable is False
    assert state.worker_result.market_data_diagnostic is not None
    assert state.worker_result.market_data_diagnostic.to_dict() == {
        "category": "unexpected_extraction_error"
    }
    assert "sensitive response body credential" not in repr(state)


def test_range_policy_rejects_oversized_task_before_any_model_or_market_call(
    tmp_path,
) -> None:
    coordinator, coordinator_model, worker_model, market, _ = _agents(
        tmp_path,
        [_candle()],
    )

    state = asyncio.run(
        coordinator.run(_request(start=date(2026, 1, 1), end=date(2026, 1, 10)))
    )

    assert state.status is KiteWorkflowStatus.FAILED
    assert state.error_code is KiteWorkflowErrorCode.VALIDATION_ERROR
    assert coordinator_model.requests == []
    assert worker_model.requests == []
    assert market.calls == []


def test_successful_workflow_values_are_read_back_from_sqlite(tmp_path) -> None:
    coordinator, _, _, _, store = _agents(tmp_path, [_candle()])
    asyncio.run(coordinator.run(_request()))

    with sqlite3.connect(store.database_path) as connection:
        row = connection.execute(
            "SELECT close, volume FROM daily_ohlcv WHERE symbol = ?",
            ("RELIANCE",),
        ).fetchone()
    assert row == ("103.25", "1000")


def test_request_and_state_reject_invalid_ranges_and_transitions() -> None:
    with pytest.raises(DataValidationError):
        _request(start=date(2026, 10, 22), end=date(2026, 10, 21))

    from multi_agent_investment_system.orchestration.kite_ingestion import (
        KiteIngestionWorkflowState,
    )

    state = KiteIngestionWorkflowState(request=_request())
    with pytest.raises(DataValidationError, match="transition"):
        state.transition(KiteWorkflowStatus.COMPLETED)


def test_observed_existing_coverage_changes_the_coordinators_next_action(
    tmp_path,
) -> None:
    class ObservationModel:
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            observed = json.loads(request.messages[1].content)
            if observed.get("last_execution"):
                action = "complete"
            elif observed["stored_dates"]:
                action = "escalate"
            else:
                action = "collect_range"
            return ModelResponse(_plan(action))

    async def run_case(database_path, seed: bool):
        market = FakeMarketDataProvider([_candle()])
        store = SQLiteDailyCandleStore(database_path)
        if seed:
            from multi_agent_investment_system.data.extraction import (
                normalize_and_validate_daily_candles,
            )

            store.upsert_validated(
                normalize_and_validate_daily_candles([_candle()]).valid_records
            )
        service = DailyMarketDataIngestionService(
            DailyMarketDataExtractionService(market), store
        )
        policy = KiteIngestionPolicy(maximum_calendar_days=5)
        worker = ZerodhaIngestionWorkerAgent(
            FakeModelProvider(
                json.dumps({"action": "execute_ingestion", "task_id": "task-1"})
            ),
            service,
            policy,
        )
        model = ObservationModel()
        state = await CoordinatorAgent(model, worker, policy).run(_request())
        return state, model, market

    empty_state, empty_model, empty_market = asyncio.run(
        run_case(tmp_path / "empty-observation.sqlite3", seed=False)
    )
    existing_state, existing_model, existing_market = asyncio.run(
        run_case(tmp_path / "existing-observation.sqlite3", seed=True)
    )

    assert [
        item.action.value
        for item in empty_state.action_trace
        if item.action.value != "inspect_coverage"
    ][0] == "collect_range"
    assert empty_state.status is KiteWorkflowStatus.COMPLETED
    assert [
        item.action.value
        for item in existing_state.action_trace
        if item.action.value != "inspect_coverage"
    ][0] == "escalate"
    assert existing_state.status is KiteWorkflowStatus.ESCALATED
    assert empty_market.calls
    assert existing_market.calls == []
    assert json.loads(empty_model.requests[0].messages[1].content)["stored_dates"] == []
    assert json.loads(existing_model.requests[0].messages[1].content)["stored_dates"]


def test_unsupported_model_action_is_rejected_before_any_provider_call(
    tmp_path,
) -> None:
    coordinator, model, _, market, _ = _agents(
        tmp_path,
        [_candle()],
        coordinator_outputs=(_plan("retry_collection"),),
    )

    state = asyncio.run(coordinator.run(_request()))

    assert state.status is KiteWorkflowStatus.FAILED
    assert state.error_code is KiteWorkflowErrorCode.INVALID_MODEL_OUTPUT
    assert market.calls == []
    assert state.action_trace[-1].authorized is False
    assert state.action_trace[-1].executed is False
    assert len(model.requests) == 1


def test_retryable_provider_error_is_retried_once_and_then_replanned(tmp_path) -> None:
    class FlakyProvider(FakeMarketDataProvider):
        async def get_daily_candles(self, symbol, start, end):
            self.calls.append((symbol, start, end))
            if len(self.calls) == 1:
                raise MarketDataProviderError(
                    "temporary provider failure", retryable=True
                )
            return self.records

    market = FlakyProvider([_candle()])
    service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(market),
        SQLiteDailyCandleStore(tmp_path / "retry.sqlite3"),
    )
    policy = KiteIngestionPolicy(maximum_calendar_days=5)
    worker = ZerodhaIngestionWorkerAgent(
        FakeModelProvider(
            json.dumps({"action": "execute_ingestion", "task_id": "task-1"}),
            json.dumps({"action": "execute_ingestion", "task_id": "task-1"}),
        ),
        service,
        policy,
    )
    coordinator = CoordinatorAgent(
        FakeModelProvider(
            _plan("collect_range"), _plan("retry_collection"), _plan("complete")
        ),
        worker,
        policy,
    )

    state = asyncio.run(coordinator.run(_request()))

    assert state.status is KiteWorkflowStatus.COMPLETED
    assert len(market.calls) == 2
    assert [
        item.action.value
        for item in state.action_trace
        if item.action.value != "inspect_coverage"
    ] == [
        "collect_range",
        "retry_collection",
        "complete",
    ]


def test_collection_attempt_limit_and_iteration_limit_force_escalation(
    tmp_path,
) -> None:
    market = FakeMarketDataProvider([_candle(close=None)])
    service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(market),
        SQLiteDailyCandleStore(tmp_path / "budget.sqlite3"),
    )
    policy = KiteIngestionPolicy(
        maximum_calendar_days=5,
        maximum_iterations=1,
        maximum_collection_attempts=1,
    )
    worker = ZerodhaIngestionWorkerAgent(
        FakeModelProvider(
            json.dumps({"action": "execute_ingestion", "task_id": "task-1"})
        ),
        service,
        policy,
    )
    coordinator = CoordinatorAgent(
        FakeModelProvider(_plan("collect_range")), worker, policy
    )

    state = asyncio.run(coordinator.run(_request()))

    assert state.status is KiteWorkflowStatus.ESCALATED
    assert state.error_code is KiteWorkflowErrorCode.ITERATION_BUDGET_EXHAUSTED
    assert len(market.calls) == 1
    assert state.worker_result is not None
    assert state.worker_result.counts is not None
    assert state.worker_result.counts.incomplete == 1
    assert state.worker_result.counts.stored == 0


def test_collection_attempt_budget_rejects_an_additional_retry(tmp_path) -> None:
    market = FakeMarketDataProvider([_candle(close=None)])
    service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(market),
        SQLiteDailyCandleStore(tmp_path / "attempt-budget.sqlite3"),
    )
    policy = KiteIngestionPolicy(
        maximum_calendar_days=5,
        maximum_iterations=4,
        maximum_collection_attempts=1,
    )
    worker = ZerodhaIngestionWorkerAgent(
        FakeModelProvider(
            json.dumps({"action": "execute_ingestion", "task_id": "task-1"})
        ),
        service,
        policy,
    )
    coordinator = CoordinatorAgent(
        FakeModelProvider(_plan("collect_range"), _plan("retry_collection")),
        worker,
        policy,
    )

    state = asyncio.run(coordinator.run(_request()))

    assert state.status is KiteWorkflowStatus.FAILED
    assert state.error_code is KiteWorkflowErrorCode.INVALID_MODEL_OUTPUT
    assert len(market.calls) == 1
    assert state.action_trace[-1].authorized is False
    assert state.action_trace[-1].executed is False


def test_retryable_flag_is_exposed_only_for_retryable_provider_failures() -> None:
    error = MarketDataProviderError("temporary failure", retryable=True)
    assert error.retryable is True
    assert MarketDataProviderError("not found").retryable is False
