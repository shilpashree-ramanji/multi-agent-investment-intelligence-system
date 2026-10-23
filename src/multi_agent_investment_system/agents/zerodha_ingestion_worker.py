"""Model-backed worker that delegates all data handling to ingestion services."""

from __future__ import annotations

import json
from datetime import date, datetime

from multi_agent_investment_system.agents._structured_output import parse_action
from multi_agent_investment_system.core.exceptions import (
    DataValidationError,
    MarketDataProviderError,
    StorageError,
)
from multi_agent_investment_system.data.extraction import ClassifiedDailyCandle
from multi_agent_investment_system.data.ingestion import (
    DailyMarketDataIngestionService,
)
from multi_agent_investment_system.llm.provider import (
    ChatMessage,
    ModelProvider,
    ModelRequest,
)
from multi_agent_investment_system.orchestration.kite_ingestion import (
    KiteCoverageReason,
    KiteCoverageStatus,
    KiteIngestionAssignment,
    KiteIngestionCounts,
    KiteIngestionPolicy,
    KiteIngestionRequest,
    KiteMarketDataEvidence,
    KiteWorkerTaskResult,
    KiteWorkflowErrorCode,
    WorkerTaskStatus,
)

_WORKER_SCHEMA = json.dumps(
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "task_id"],
        "properties": {
            "action": {"const": "execute_ingestion"},
            "task_id": {"type": "string"},
        },
    },
    separators=(",", ":"),
)


class ZerodhaIngestionWorkerAgent:
    """Use a model to accept a bounded task, then run deterministic ingestion."""

    def __init__(
        self,
        model_provider: ModelProvider,
        ingestion_service: DailyMarketDataIngestionService,
        policy: KiteIngestionPolicy,
    ) -> None:
        self._model_provider = model_provider
        self._ingestion_service = ingestion_service
        self._policy = policy

    async def execute(
        self,
        request: KiteIngestionRequest,
        assignment: KiteIngestionAssignment,
    ) -> KiteWorkerTaskResult:
        """Execute exactly the caller's assigned symbol and date range."""

        if not isinstance(request, KiteIngestionRequest):
            raise DataValidationError("Worker request is malformed")
        if (
            not isinstance(assignment, KiteIngestionAssignment)
            or assignment.run_id != request.run_id
            or assignment.task_id != request.task_id
            or assignment.request_fingerprint != request.fingerprint
        ):
            return self._failure(
                request,
                KiteWorkflowErrorCode.VALIDATION_ERROR,
            )
        try:
            request.validate_policy(self._policy)
        except DataValidationError:
            return self._failure(
                request,
                KiteWorkflowErrorCode.VALIDATION_ERROR,
            )

        model_request = ModelRequest(
            messages=(
                ChatMessage(
                    "system",
                    "You are the Zerodha ingestion worker. Validate the assigned "
                    "task and authorize only execute_ingestion. Do not produce or "
                    "change symbols, dates, credentials, or market data. Return "
                    "only the requested JSON object.",
                ),
                ChatMessage(
                    "user",
                    json.dumps(
                        {
                            "task_id": request.task_id,
                            "symbol": request.symbol,
                            "start": request.start.isoformat(),
                            "end": request.end.isoformat(),
                            "maximum_calendar_days": self._policy.maximum_calendar_days,
                            "assignment_role": assignment.worker_role,
                        },
                        sort_keys=True,
                    ),
                ),
            ),
            logical_role="zerodha_ingestion_worker",
            output_schema=_WORKER_SCHEMA,
        )
        try:
            response = await self._model_provider.generate(model_request)
        except Exception:
            return self._failure(
                request,
                KiteWorkflowErrorCode.MODEL_PROVIDER_ERROR,
            )
        try:
            parse_action(
                response,
                allowed_actions=frozenset({"execute_ingestion"}),
                task_id=request.task_id,
            )
        except ValueError:
            return self._failure(
                request,
                KiteWorkflowErrorCode.INVALID_MODEL_OUTPUT,
            )

        try:
            result = await self._ingestion_service.ingest_incremental(
                request.symbol,
                request.start,
                request.end,
            )
        except DataValidationError:
            error_code = KiteWorkflowErrorCode.VALIDATION_ERROR
        except MarketDataProviderError:
            error_code = KiteWorkflowErrorCode.MARKET_DATA_ERROR
        except StorageError:
            error_code = KiteWorkflowErrorCode.STORAGE_ERROR
        except Exception:
            error_code = KiteWorkflowErrorCode.INGESTION_ERROR
        else:
            classified_records = (
                *result.extraction_result.valid_records,
                *result.extraction_result.incomplete_records,
                *result.extraction_result.invalid_records,
            )
            evidence = tuple(
                self._evidence(request, candle) for candle in classified_records
            )
            return KiteWorkerTaskResult(
                run_id=request.run_id,
                task_id=request.task_id,
                status=WorkerTaskStatus.SUCCEEDED,
                counts=KiteIngestionCounts(
                    fetched=result.fetched_count,
                    valid=result.valid_count,
                    incomplete=result.incomplete_count,
                    invalid=result.invalid_count,
                    upserted=result.upserted_count,
                    stored=result.stored_count,
                ),
                evidence=evidence,
                coverage_status=KiteCoverageStatus.UNKNOWN,
                coverage_reasons=(
                    KiteCoverageReason.NO_TRADING_CALENDAR,
                    KiteCoverageReason.NO_PROVIDER_COVERAGE_GUARANTEE,
                    KiteCoverageReason.HISTORICAL_INTERVALS_NOT_VERIFIED,
                ),
            )
        return self._failure(request, error_code)

    @staticmethod
    def _evidence(
        request: KiteIngestionRequest,
        candle: ClassifiedDailyCandle,
    ) -> KiteMarketDataEvidence:
        record = candle.record
        source = record.get("source")
        observed_at = record.get("observed_at")
        retrieved_at = record.get("retrieved_at")
        if isinstance(retrieved_at, (datetime, date)):
            retrieved_at_text = retrieved_at.isoformat()
        elif isinstance(retrieved_at, str):
            retrieved_at_text = retrieved_at
        else:
            retrieved_at_text = None
        return KiteMarketDataEvidence(
            symbol=request.symbol,
            source=source if isinstance(source, str) and source.strip() else None,
            observed_at=observed_at if isinstance(observed_at, datetime) else None,
            retrieved_at=retrieved_at_text,
            validation_issue_codes=tuple(
                issue.code.value for issue in candle.validation.issues
            ),
        )

    @staticmethod
    def _failure(
        request: KiteIngestionRequest,
        error_code: KiteWorkflowErrorCode,
    ) -> KiteWorkerTaskResult:
        """Create a failure result without retaining provider/model error text."""

        return KiteWorkerTaskResult(
            run_id=request.run_id,
            task_id=request.task_id,
            status=WorkerTaskStatus.FAILED,
            error_code=error_code,
        )
