"""Coordinator agent for the bounded Kite ingestion workflow."""

from __future__ import annotations

import json

from multi_agent_investment_system.agents._structured_output import parse_action
from multi_agent_investment_system.agents.zerodha_ingestion_worker import (
    ZerodhaIngestionWorkerAgent,
)
from multi_agent_investment_system.core.exceptions import DataValidationError
from multi_agent_investment_system.llm.provider import (
    ChatMessage,
    ModelProvider,
    ModelRequest,
)
from multi_agent_investment_system.orchestration.kite_ingestion import (
    KiteIngestionAssignment,
    KiteIngestionPolicy,
    KiteIngestionRequest,
    KiteIngestionWorkflowState,
    KiteWorkerTaskResult,
    KiteWorkflowErrorCode,
    KiteWorkflowStatus,
    WorkerTaskStatus,
)

_DELEGATION_SCHEMA = json.dumps(
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "task_id"],
        "properties": {
            "action": {"const": "delegate_ingestion"},
            "task_id": {"type": "string"},
        },
    },
    separators=(",", ":"),
)


class CoordinatorAgent:
    """Model-backed coordinator with one allow-listed worker and terminal gate."""

    def __init__(
        self,
        model_provider: ModelProvider,
        worker: ZerodhaIngestionWorkerAgent,
        policy: KiteIngestionPolicy,
    ) -> None:
        self._model_provider = model_provider
        self._worker = worker
        self._policy = policy

    async def run(self, request: KiteIngestionRequest) -> KiteIngestionWorkflowState:
        """Plan delegation, invoke one worker, then choose a permitted next state."""

        if not isinstance(request, KiteIngestionRequest):
            raise DataValidationError("Coordinator request is malformed")
        state = KiteIngestionWorkflowState(request=request).transition(
            KiteWorkflowStatus.COORDINATING
        )
        try:
            request.validate_policy(self._policy)
        except DataValidationError:
            return state.transition(
                KiteWorkflowStatus.FAILED,
                error_code=KiteWorkflowErrorCode.VALIDATION_ERROR,
            )

        try:
            response = await self._model_provider.generate(
                self._planning_request(request)
            )
        except Exception:
            return state.transition(
                KiteWorkflowStatus.FAILED,
                error_code=KiteWorkflowErrorCode.MODEL_PROVIDER_ERROR,
            )
        try:
            parse_action(
                response,
                allowed_actions=frozenset({"delegate_ingestion"}),
                task_id=request.task_id,
            )
        except ValueError:
            return state.transition(
                KiteWorkflowStatus.FAILED,
                error_code=KiteWorkflowErrorCode.INVALID_MODEL_OUTPUT,
            )

        assignment = KiteIngestionAssignment.for_request(request)
        state = state.transition(
            KiteWorkflowStatus.ASSIGNED,
            assignment=assignment,
        ).transition(KiteWorkflowStatus.WORKER_RUNNING)
        try:
            worker_result = await self._worker.execute(request, assignment)
        except Exception:
            return state.transition(
                KiteWorkflowStatus.FAILED,
                error_code=KiteWorkflowErrorCode.INGESTION_ERROR,
            )
        state = state.transition(
            KiteWorkflowStatus.REVIEWING,
            worker_result=worker_result,
        )

        allowed_actions = self._permitted_final_actions(worker_result)
        try:
            response = await self._model_provider.generate(
                self._review_request(request, worker_result, allowed_actions)
            )
        except Exception:
            return state.transition(
                KiteWorkflowStatus.FAILED,
                error_code=KiteWorkflowErrorCode.MODEL_PROVIDER_ERROR,
            )
        try:
            action = parse_action(response, allowed_actions=allowed_actions)
        except ValueError:
            return state.transition(
                KiteWorkflowStatus.FAILED,
                error_code=KiteWorkflowErrorCode.INVALID_MODEL_OUTPUT,
            )

        if action == "complete":
            return state.transition(KiteWorkflowStatus.COMPLETED)
        error_code = worker_result.error_code
        if error_code is None and worker_result.has_data_quality_issues:
            error_code = KiteWorkflowErrorCode.DATA_QUALITY_REVIEW
        return state.transition(
            KiteWorkflowStatus.ESCALATED,
            error_code=error_code,
        )

    @staticmethod
    def _permitted_final_actions(
        worker_result: KiteWorkerTaskResult,
    ) -> frozenset[str]:
        if (
            worker_result.status is WorkerTaskStatus.SUCCEEDED
            and not worker_result.has_data_quality_issues
        ):
            return frozenset({"complete", "escalate"})
        return frozenset({"escalate"})

    def _planning_request(self, request: KiteIngestionRequest) -> ModelRequest:
        return ModelRequest(
            messages=(
                ChatMessage(
                    "system",
                    "You are the workflow Coordinator. Delegate only to the "
                    "Zerodha ingestion worker for the caller's existing task. "
                    "Do not change task IDs, symbols, dates, credentials, or data. "
                    "Return only the requested JSON object.",
                ),
                ChatMessage(
                    "user",
                    json.dumps(
                        {
                            "run_id": request.run_id,
                            "task_id": request.task_id,
                            "symbol": request.symbol,
                            "start": request.start.isoformat(),
                            "end": request.end.isoformat(),
                            "maximum_calendar_days": self._policy.maximum_calendar_days,
                            "available_worker": "zerodha_ingestion_worker",
                        },
                        sort_keys=True,
                    ),
                ),
            ),
            logical_role="coordinator",
            output_schema=_DELEGATION_SCHEMA,
        )

    @staticmethod
    def _review_request(
        request: KiteIngestionRequest,
        worker_result: KiteWorkerTaskResult,
        allowed_actions: frozenset[str],
    ) -> ModelRequest:
        counts = worker_result.counts
        summary = {
            "task_id": request.task_id,
            "worker_status": worker_result.status.value,
            "worker_error_code": (
                worker_result.error_code.value
                if worker_result.error_code is not None
                else None
            ),
            "fetched_count": counts.fetched if counts is not None else 0,
            "valid_count": counts.valid if counts is not None else 0,
            "incomplete_count": counts.incomplete if counts is not None else 0,
            "invalid_count": counts.invalid if counts is not None else 0,
            "stored_count": counts.stored if counts is not None else 0,
            "coverage_status": worker_result.coverage_status.value,
            "coverage_reasons": [
                reason.value for reason in worker_result.coverage_reasons
            ],
            "permitted_actions": sorted(allowed_actions),
        }
        schema = json.dumps(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["action"],
                "properties": {"action": {"enum": sorted(allowed_actions)}},
            },
            separators=(",", ":"),
        )
        return ModelRequest(
            messages=(
                ChatMessage(
                    "system",
                    "You are the workflow Coordinator reviewing deterministic "
                    "worker results. Choose only a permitted terminal action. "
                    "Complete means execution finished; it does not assert full "
                    "historical market-data coverage. "
                    "Do not alter or restate market data as new facts. Return "
                    "only the requested JSON object.",
                ),
                ChatMessage("user", json.dumps(summary, sort_keys=True)),
            ),
            logical_role="coordinator",
            output_schema=schema,
        )
