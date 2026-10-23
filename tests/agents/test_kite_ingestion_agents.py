"""Offline behavior tests for the Coordinator and Kite ingestion worker."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from multi_agent_investment_system.agents.coordinator import CoordinatorAgent
from multi_agent_investment_system.agents.zerodha_ingestion_worker import (
    ZerodhaIngestionWorkerAgent,
)
from multi_agent_investment_system.core.exceptions import (
    DataValidationError,
    MarketDataProviderError,
)
from multi_agent_investment_system.data.ingestion import (
    DailyMarketDataIngestionService,
)
from multi_agent_investment_system.data.service import (
    DailyMarketDataExtractionService,
)
from multi_agent_investment_system.data.storage import SQLiteDailyCandleStore
from multi_agent_investment_system.llm.provider import ModelRequest, ModelResponse
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
        json.dumps({"action": "delegate_ingestion", "task_id": "task-1"}),
        json.dumps({"action": "complete"}),
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
    assert market.calls == [("RELIANCE", date(2026, 10, 20), date(2026, 10, 21))]
    assert (
        store.fetch_daily_candles("RELIANCE", date(2026, 10, 20), date(2026, 10, 21))[
            0
        ]["source"]
        == "fake-kite"
    )


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
    worker = ZerodhaIngestionWorkerAgent(
        FakeModelProvider(error=RuntimeError("private model provider detail")),
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
    assert "private model provider detail" not in repr(result)
    assert market.calls == []


def test_invalid_and_incomplete_candles_are_not_persisted_and_are_escalated(
    tmp_path,
) -> None:
    incomplete = _candle(close=None)
    invalid = _candle(observed_at=datetime(2026, 10, 21, 10, tzinfo=UTC), volume=-1)
    coordinator, _, _, _, store = _agents(
        tmp_path,
        [incomplete, invalid],
        coordinator_outputs=(
            json.dumps({"action": "delegate_ingestion", "task_id": "task-1"}),
            json.dumps({"action": "escalate"}),
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
            json.dumps({"action": "delegate_ingestion", "task_id": "task-1"}),
            json.dumps({"action": "complete"}),
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
        market_error=MarketDataProviderError("secret provider detail"),
        coordinator_outputs=(
            json.dumps({"action": "delegate_ingestion", "task_id": "task-1"}),
            json.dumps({"action": "escalate"}),
        ),
    )

    state = asyncio.run(coordinator.run(_request()))

    assert state.status is KiteWorkflowStatus.ESCALATED
    assert state.error_code is KiteWorkflowErrorCode.MARKET_DATA_ERROR
    assert state.worker_result is not None
    assert state.worker_result.error_code is KiteWorkflowErrorCode.MARKET_DATA_ERROR
    assert "secret provider detail" not in repr(state)


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
