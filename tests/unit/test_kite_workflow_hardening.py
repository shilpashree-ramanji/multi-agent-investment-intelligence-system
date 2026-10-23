"""Offline tests for strict Kite workflow boundaries and runtime wiring."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from multi_agent_investment_system.agents._structured_output import (
    MAX_MODEL_RESPONSE_BYTES,
    parse_action,
)
from multi_agent_investment_system.agents.zerodha_ingestion_worker import (
    ZerodhaIngestionWorkerAgent,
)
from multi_agent_investment_system.core.configuration import KiteConnectCredentials
from multi_agent_investment_system.core.exceptions import (
    ConfigurationError,
    MarketDataProviderError,
)
from multi_agent_investment_system.core.kite_runtime_settings import (
    KiteWorkflowRuntimeSettings,
)
from multi_agent_investment_system.data.ingestion import (
    DailyMarketDataIngestionService,
)
from multi_agent_investment_system.data.service import (
    DailyMarketDataExtractionService,
)
from multi_agent_investment_system.data.storage import SQLiteDailyCandleStore
from multi_agent_investment_system.llm.provider import ModelResponse
from multi_agent_investment_system.orchestration.kite_ingestion import (
    KiteCoverageReason,
    KiteCoverageStatus,
    KiteIngestionAssignment,
    KiteIngestionPolicy,
    KiteIngestionRequest,
    WorkerTaskStatus,
)
from multi_agent_investment_system.orchestration.kite_runtime import (
    build_kite_coordinator,
)
from multi_agent_investment_system.tools.kite_authentication import KiteAccessToken
from multi_agent_investment_system.tools.kite_symbol_guard import (
    RequestedSymbolGuard,
)
from multi_agent_investment_system.tools.market_data import MarketDataProvider


def _coordinator_plan(action: str, task_id: str = "task-a") -> str:
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


def _request(**overrides: object) -> KiteIngestionRequest:
    values: dict[str, object] = {
        "run_id": "run-a",
        "task_id": "task-a",
        "symbol": "RELIANCE",
        "start": date(2025, 1, 1),
        "end": date(2025, 1, 3),
    }
    values.update(overrides)
    return KiteIngestionRequest(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "content",
    [
        '{"action":"execute_ingestion","action":"other","task_id":"task-a"}',
        '{"action":"execute_ingestion","task_id":"task-a","task_id":"other"}',
    ],
)
def test_action_parser_rejects_duplicate_action_or_task_keys(content: str) -> None:
    with pytest.raises(ValueError):
        parse_action(
            ModelResponse(content),
            allowed_actions=frozenset({"execute_ingestion"}),
            task_id="task-a",
        )


def test_action_parser_rejects_response_over_explicit_byte_limit() -> None:
    response = ModelResponse(" " * (MAX_MODEL_RESPONSE_BYTES + 1))

    with pytest.raises(ValueError, match="size limit"):
        parse_action(response, allowed_actions=frozenset({"ok"}))


def test_assignment_fingerprint_binds_run_task_symbol_and_requested_dates() -> None:
    request = _request()
    assignment = KiteIngestionAssignment.for_request(request)
    assert assignment.request_fingerprint == request.fingerprint

    for altered in (
        _request(run_id="run-b"),
        _request(task_id="task-b"),
        _request(symbol="TCS"),
        _request(start=date(2025, 1, 2)),
        _request(end=date(2025, 1, 4)),
    ):
        assert altered.fingerprint != assignment.request_fingerprint


class _MarketProvider(MarketDataProvider):
    def __init__(self, records: Sequence[Mapping[str, object]]) -> None:
        self.records = records
        self.calls = 0

    async def get_daily_candles(self, symbol, start, end):
        self.calls += 1
        return self.records


@pytest.mark.parametrize("record", [{"symbol": "TCS"}, {"close": 10}])
def test_kite_symbol_guard_rejects_mismatched_or_missing_symbol(record) -> None:
    provider = _MarketProvider([record])

    with pytest.raises(MarketDataProviderError, match="mismatched symbol"):
        asyncio.run(
            RequestedSymbolGuard(provider).get_daily_candles(
                "RELIANCE", date(2025, 1, 1), date(2025, 1, 3)
            )
        )


def test_kite_symbol_mismatch_fails_before_sqlite_persistence(tmp_path) -> None:
    from multi_agent_investment_system.agents.coordinator import CoordinatorAgent

    class _Model:
        def __init__(self, *responses: str) -> None:
            self.responses = iter(responses)

        async def generate(self, request):
            return ModelResponse(next(self.responses))

    # A real Kite candle shape carrying a different identity must be rejected
    # by the Kite boundary before the general ingestion service stores it.
    record = {
        "symbol": "TCS",
        "observed_at": datetime(2025, 1, 2, 10, tzinfo=UTC),
        "open": 10,
        "high": 11,
        "low": 9,
        "close": 10,
        "volume": 100,
        "source": "fake-kite",
    }
    market = _MarketProvider([record])
    guarded = RequestedSymbolGuard(market)
    store = SQLiteDailyCandleStore(tmp_path / "symbol-guard.sqlite3")
    service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(guarded), store
    )
    policy = KiteIngestionPolicy(maximum_calendar_days=5)
    worker = ZerodhaIngestionWorkerAgent(
        _Model('{"action":"execute_ingestion","task_id":"task-a"}'),
        service,
        policy,
    )
    coordinator = CoordinatorAgent(
        _Model(
            _coordinator_plan("collect_range"),
            _coordinator_plan("escalate"),
        ),
        worker,
        policy,
    )

    state = asyncio.run(coordinator.run(_request()))

    assert state.worker_result is not None
    assert state.worker_result.error_code is not None
    assert state.worker_result.error_code.value == "market_data_error"
    assert (
        store.fetch_daily_candles("RELIANCE", date(2025, 1, 1), date(2025, 1, 3)) == ()
    )


def test_worker_rejects_request_mismatch_before_model_or_ingestion(tmp_path) -> None:
    class _Model:
        requests = 0

        async def generate(self, request):
            self.requests += 1
            return ModelResponse('{"action":"execute_ingestion","task_id":"task-a"}')

    market = _MarketProvider([])
    store = SQLiteDailyCandleStore(tmp_path / "mismatch.sqlite3")
    service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(market), store
    )
    model = _Model()
    worker = ZerodhaIngestionWorkerAgent(
        model, service, KiteIngestionPolicy(maximum_calendar_days=5)
    )

    result = asyncio.run(
        worker.execute(
            _request(symbol="TCS"), KiteIngestionAssignment.for_request(_request())
        )
    )

    assert result.status is WorkerTaskStatus.FAILED
    assert model.requests == 0
    assert market.calls == 0


def test_successful_empty_response_keeps_coverage_unknown(tmp_path) -> None:
    from multi_agent_investment_system.agents.coordinator import CoordinatorAgent

    class _Model:
        def __init__(self, *responses: str) -> None:
            self.responses = iter(responses)

        async def generate(self, request):
            return ModelResponse(next(self.responses))

    market = _MarketProvider([])
    service = DailyMarketDataIngestionService(
        DailyMarketDataExtractionService(market),
        SQLiteDailyCandleStore(tmp_path / "empty.sqlite3"),
    )
    policy = KiteIngestionPolicy(maximum_calendar_days=5)
    worker = ZerodhaIngestionWorkerAgent(
        _Model('{"action":"execute_ingestion","task_id":"task-a"}'),
        service,
        policy,
    )
    coordinator = CoordinatorAgent(
        _Model(
            _coordinator_plan("collect_range"),
            _coordinator_plan("complete"),
        ),
        worker,
        policy,
    )

    state = asyncio.run(coordinator.run(_request()))

    assert state.worker_result is not None
    assert state.worker_result.status is WorkerTaskStatus.SUCCEEDED
    assert state.worker_result.counts is not None
    assert state.worker_result.counts.fetched == 0
    assert state.worker_result.coverage_status is KiteCoverageStatus.UNKNOWN
    assert state.final_observation is not None
    assert state.final_observation.stored_dates == ()
    assert set(state.worker_result.coverage_reasons) == {
        KiteCoverageReason.NO_TRADING_CALENDAR,
        KiteCoverageReason.NO_PROVIDER_COVERAGE_GUARANTEE,
        KiteCoverageReason.HISTORICAL_INTERVALS_NOT_VERIFIED,
    }


def test_runtime_settings_require_explicit_range_bound_and_exclude_secrets() -> None:
    with pytest.raises(ConfigurationError, match="must be configured"):
        KiteWorkflowRuntimeSettings.from_environment({})

    settings = KiteWorkflowRuntimeSettings.from_environment(
        {
            "KITE_INGESTION_MAXIMUM_CALENDAR_DAYS": "14",
            "MARKET_DATA_DATABASE_PATH": "/tmp/kite.sqlite3",
            "KITE_API_KEY": "do-not-copy",
            "KITE_API_SECRET": "do-not-copy-either",
        }
    )
    assert settings.maximum_calendar_days == 14
    assert settings.database_path == Path("/tmp/kite.sqlite3")
    assert "do-not-copy" not in repr(settings)


def test_runtime_factory_wires_dependencies_without_calling_providers(
    tmp_path, monkeypatch
) -> None:
    from multi_agent_investment_system.orchestration import kite_runtime

    class _Model:
        async def generate(self, request):
            raise AssertionError("factory must not call the model")

    class _Provider:
        def __init__(self, credentials, access_token):
            self.credentials = credentials
            self.access_token = access_token

    monkeypatch.setattr(kite_runtime, "KiteConnectMarketDataProvider", _Provider)
    credentials = KiteConnectCredentials("key-secret", "api-secret")
    token = KiteAccessToken("access-secret")
    settings = KiteWorkflowRuntimeSettings(tmp_path / "runtime.sqlite3", 10)

    coordinator = build_kite_coordinator(
        model_provider=_Model(),
        credentials=credentials,
        access_token=token,
        settings=settings,
    )

    assert coordinator is not None
    assert "key-secret" not in repr(coordinator)
    assert "api-secret" not in repr(coordinator)
    assert "access-secret" not in repr(coordinator)
