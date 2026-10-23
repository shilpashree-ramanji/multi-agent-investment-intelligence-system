"""Bounded, observation-driven Coordinator for market-data collection."""

from __future__ import annotations

import json

from multi_agent_investment_system.agents._structured_output import (
    CollectionPlanValidationError,
    parse_collection_plan,
)
from multi_agent_investment_system.agents.zerodha_ingestion_worker import (
    ZerodhaIngestionWorkerAgent,
)
from multi_agent_investment_system.core.exceptions import (
    DataValidationError,
    StorageError,
)
from multi_agent_investment_system.llm.provider import (
    ChatMessage,
    ModelOutputRejectionCategory,
    ModelProvider,
    ModelProviderDiagnostic,
    ModelProviderError,
    ModelRequest,
)
from multi_agent_investment_system.orchestration.kite_ingestion import (
    KiteCollectionAction,
    KiteCollectionActionTrace,
    KiteDataCoverageObservation,
    KiteIngestionAssignment,
    KiteIngestionPolicy,
    KiteIngestionRequest,
    KiteIngestionWorkflowState,
    KiteWorkerTaskResult,
    KiteWorkflowErrorCode,
    KiteWorkflowStatus,
    WorkerTaskStatus,
    safe_action_trace_descriptions,
)

_COMPLETION_CONDITION = "quality_clean_and_persisted"


class CoordinatorAgent:
    """Choose bounded collection actions from observed state, then authorize them.

    The model can select only a fixed action name. Symbol, date range, provider,
    and tools are supplied by deterministic application code and are never
    taken from model output. Market-data access and persistence remain in the
    existing Worker and ingestion service.
    """

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
        """Observe, plan, authorize, execute, inspect, and replan within bounds."""

        if not isinstance(request, KiteIngestionRequest):
            raise DataValidationError("Coordinator request is malformed")
        state = KiteIngestionWorkflowState(request=request).transition(
            KiteWorkflowStatus.COORDINATING
        )
        try:
            request.validate_policy(self._policy)
            observation = await self._worker.observe_coverage(request)
        except DataValidationError:
            return state.transition(
                KiteWorkflowStatus.FAILED,
                error_code=KiteWorkflowErrorCode.VALIDATION_ERROR,
            )
        except StorageError:
            return state.transition(
                KiteWorkflowStatus.FAILED,
                error_code=KiteWorkflowErrorCode.STORAGE_ERROR,
            )
        except Exception:
            return state.transition(
                KiteWorkflowStatus.FAILED,
                error_code=KiteWorkflowErrorCode.INGESTION_ERROR,
            )
        state = state.record_action(
            self._observation_trace(1, "initial_coverage_observed")
        )

        worker_result: KiteWorkerTaskResult | None = None
        assignment: KiteIngestionAssignment | None = None
        collection_attempts = 0

        for iteration in range(1, self._policy.maximum_iterations + 1):
            allowed_actions = self._allowed_actions(
                worker_result, observation, collection_attempts
            )
            try:
                response = await self._model_provider.generate(
                    self._planning_request(
                        request,
                        observation,
                        worker_result,
                        iteration,
                        allowed_actions,
                    )
                )
            except Exception as error:
                diagnostic = (
                    error.diagnostic
                    if isinstance(error, ModelProviderError)
                    else ModelProviderDiagnostic.from_exception_class(
                        type(error).__name__
                    )
                )
                return state.transition(
                    KiteWorkflowStatus.FAILED,
                    error_code=KiteWorkflowErrorCode.MODEL_PROVIDER_ERROR,
                    initial_observation=observation,
                    model_provider_diagnostic=diagnostic,
                )

            try:
                action_text, _, _ = parse_collection_plan(
                    response,
                    allowed_actions=frozenset(
                        action.value for action in allowed_actions
                    ),
                    task_id=request.task_id,
                )
            except CollectionPlanValidationError as error:
                rejected_trace = self._rejected_plan(
                    iteration, rejection_category=error.category
                )
                return state.transition(
                    KiteWorkflowStatus.FAILED,
                    error_code=KiteWorkflowErrorCode.INVALID_MODEL_OUTPUT,
                    initial_observation=observation,
                    final_observation=observation,
                    action_trace=rejected_trace,
                )

            action = KiteCollectionAction(action_text)
            if not self._authorize(
                action, worker_result, observation, collection_attempts
            ):
                rejected_trace = self._rejected_plan(iteration, action)
                return state.transition(
                    KiteWorkflowStatus.FAILED,
                    error_code=KiteWorkflowErrorCode.INVALID_MODEL_OUTPUT,
                    initial_observation=observation,
                    final_observation=observation,
                    action_trace=rejected_trace,
                )

            if action is KiteCollectionAction.ESCALATE:
                reason, expected_outcome = safe_action_trace_descriptions(action)
                return state.transition(
                    KiteWorkflowStatus.ESCALATED,
                    error_code=self._review_error(worker_result),
                    initial_observation=observation,
                    final_observation=observation,
                    action_trace=KiteCollectionActionTrace(
                        iteration,
                        action,
                        reason,
                        expected_outcome,
                        authorized=True,
                        executed=True,
                        outcome_code="escalated_by_coordinator",
                    ),
                )

            if action is KiteCollectionAction.COMPLETE:
                reason, expected_outcome = safe_action_trace_descriptions(action)
                return state.transition(
                    KiteWorkflowStatus.COMPLETED,
                    initial_observation=observation,
                    final_observation=observation,
                    action_trace=KiteCollectionActionTrace(
                        iteration,
                        action,
                        reason,
                        expected_outcome,
                        authorized=True,
                        executed=True,
                        outcome_code="execution_and_persistence_verified_coverage_unknown",
                    ),
                )

            if assignment is None:
                assignment = KiteIngestionAssignment.for_request(request)
                state = state.transition(
                    KiteWorkflowStatus.ASSIGNED,
                    assignment=assignment,
                    initial_observation=observation,
                ).transition(KiteWorkflowStatus.WORKER_RUNNING)
            else:
                state = state.transition(KiteWorkflowStatus.WORKER_RUNNING)

            collection_attempts += 1
            try:
                worker_result = await self._worker.execute(request, assignment)
            except Exception:
                return state.transition(
                    KiteWorkflowStatus.FAILED,
                    error_code=KiteWorkflowErrorCode.INGESTION_ERROR,
                    final_observation=observation,
                )
            try:
                observation = await self._worker.observe_coverage(request)
            except StorageError:
                trace = self._execution_trace(
                    iteration, action, worker_result
                )
                state = state.transition(
                    KiteWorkflowStatus.REVIEWING,
                    worker_result=worker_result,
                    action_trace=trace,
                )
                return state.transition(
                    KiteWorkflowStatus.FAILED,
                    error_code=KiteWorkflowErrorCode.STORAGE_ERROR,
                )
            except Exception:
                return state.transition(
                    KiteWorkflowStatus.FAILED,
                    error_code=KiteWorkflowErrorCode.INGESTION_ERROR,
                    final_observation=observation,
                )

            state = state.transition(
                KiteWorkflowStatus.REVIEWING,
                worker_result=worker_result,
                final_observation=observation,
                action_trace=self._execution_trace(
                    iteration, action, worker_result
                ),
            ).record_action(
                self._observation_trace(iteration, "stored_range_revalidated")
            )

        if state.status is KiteWorkflowStatus.REVIEWING:
            return state.transition(
                KiteWorkflowStatus.ESCALATED,
                error_code=KiteWorkflowErrorCode.ITERATION_BUDGET_EXHAUSTED,
            )
        return state

    def _allowed_actions(
        self,
        worker_result: KiteWorkerTaskResult | None,
        observation: KiteDataCoverageObservation,
        collection_attempts: int,
    ) -> frozenset[KiteCollectionAction]:
        if worker_result is None:
            return frozenset(
                {KiteCollectionAction.COLLECT_RANGE, KiteCollectionAction.ESCALATE}
            )
        if self._can_complete(worker_result, observation):
            return frozenset(
                {KiteCollectionAction.COMPLETE, KiteCollectionAction.ESCALATE}
            )
        can_retry = collection_attempts < self._policy.maximum_collection_attempts
        recoverable_error = (
            worker_result.status is WorkerTaskStatus.FAILED and worker_result.retryable
        )
        quality_issue = worker_result.has_data_quality_issues or bool(
            observation.incomplete_count or observation.invalid_count
        )
        if can_retry and (recoverable_error or quality_issue):
            return frozenset(
                {
                    KiteCollectionAction.RETRY_COLLECTION,
                    KiteCollectionAction.ESCALATE,
                }
            )
        return frozenset({KiteCollectionAction.ESCALATE})

    def _authorize(
        self,
        action: KiteCollectionAction,
        worker_result: KiteWorkerTaskResult | None,
        observation: KiteDataCoverageObservation,
        collection_attempts: int,
    ) -> bool:
        if action in {
            KiteCollectionAction.COLLECT_RANGE,
            KiteCollectionAction.RETRY_COLLECTION,
        }:
            if collection_attempts >= self._policy.maximum_collection_attempts:
                return False
            if action is KiteCollectionAction.RETRY_COLLECTION:
                return action in self._allowed_actions(
                    worker_result, observation, collection_attempts
                )
            return worker_result is None or bool(
                worker_result.has_data_quality_issues
                or observation.incomplete_count
                or observation.invalid_count
            )
        if action is KiteCollectionAction.COMPLETE:
            return worker_result is not None and self._can_complete(
                worker_result, observation
            )
        return action is KiteCollectionAction.ESCALATE

    @staticmethod
    def _can_complete(
        result: KiteWorkerTaskResult,
        observation: KiteDataCoverageObservation,
    ) -> bool:
        if (
            result.status is not WorkerTaskStatus.SUCCEEDED
            or result.has_data_quality_issues
            or result.counts is None
            or result.counts.valid != result.counts.upserted
            or observation.incomplete_count
            or observation.invalid_count
        ):
            return False
        return len(observation.stored_dates) == result.counts.stored

    def _planning_request(
        self,
        request: KiteIngestionRequest,
        observation: KiteDataCoverageObservation,
        worker_result: KiteWorkerTaskResult | None,
        iteration: int,
        allowed_actions: frozenset[KiteCollectionAction],
    ) -> ModelRequest:
        counts = worker_result.counts if worker_result is not None else None
        observed_state: dict[str, object] = {
            "task_id": request.task_id,
            "symbol": request.symbol,
            "requested_start": request.start.isoformat(),
            "requested_end": request.end.isoformat(),
            "iteration": iteration,
            "maximum_iterations": self._policy.maximum_iterations,
            "maximum_calendar_days": self._policy.maximum_calendar_days,
            "collection_attempt_limit": self._policy.maximum_collection_attempts,
            "maximum_symbols": 1,
            "maximum_kite_http_requests": (
                self._policy.maximum_collection_attempts + 1
            ),
            "stored_dates": [item.isoformat() for item in observation.stored_dates],
            "stored_valid_count": observation.valid_count,
            "stored_incomplete_count": observation.incomplete_count,
            "stored_invalid_count": observation.invalid_count,
            "source_capabilities": list(observation.source_capabilities),
            "coverage_can_be_claimed": False,
            "coverage_limitation": (
                "No exchange calendar or provider coverage guarantee"
            ),
            "permitted_actions": sorted(item.value for item in allowed_actions),
        }
        if worker_result is not None:
            observed_state["last_execution"] = {
                "status": worker_result.status.value,
                "error_code": (
                    worker_result.error_code.value
                    if worker_result.error_code is not None
                    else None
                ),
                "retryable": worker_result.retryable,
                "fetched": counts.fetched if counts is not None else 0,
                "valid": counts.valid if counts is not None else 0,
                "incomplete": counts.incomplete if counts is not None else 0,
                "invalid": counts.invalid if counts is not None else 0,
                "upserted": counts.upserted if counts is not None else 0,
                "stored": counts.stored if counts is not None else 0,
                "validation_issue_codes": sorted(
                    {
                        code
                        for evidence in worker_result.evidence
                        for code in evidence.validation_issue_codes
                    }
                ),
            }
        schema = json.dumps(
            {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "action",
                    "task_id",
                    "reason",
                    "expected_outcome",
                    "completion_condition",
                ],
                "properties": {
                    "action": {"enum": sorted(item.value for item in allowed_actions)},
                    "task_id": {"enum": [request.task_id]},
                    "reason": {"type": "string"},
                    "expected_outcome": {"type": "string"},
                    "completion_condition": {
                        "enum": [
                            _COMPLETION_CONDITION,
                            "stop_without_coverage_claim",
                        ]
                    },
                },
            },
            separators=(",", ":"),
        )
        return ModelRequest(
            messages=(
                ChatMessage(
                    "system",
                    "You are the market-data collection Coordinator. Assess only "
                    "the supplied observed state and choose one permitted action. "
                    "Do not alter the symbol or dates. You cannot access tools, "
                    "credentials, files, SQL, or execute code. A successful API "
                    "response does not prove complete trading-session coverage. "
                    "Never infer holidays or missing sessions without a calendar. "
                    "Choose a permitted action and its corresponding completion "
                    "condition: collect_range and retry_collection require "
                    "quality_clean_and_persisted; complete and escalate require "
                    "stop_without_coverage_claim. Select complete only when the "
                    "deterministic completion gate permits it, and never claim "
                    "full historical or session coverage. "
                    "Return only the structured plan requested.",
                ),
                ChatMessage("user", json.dumps(observed_state, sort_keys=True)),
            ),
            logical_role="coordinator",
            output_schema=schema,
        )

    @staticmethod
    def _execution_trace(
        iteration: int,
        action: KiteCollectionAction,
        result: KiteWorkerTaskResult,
    ) -> KiteCollectionActionTrace:
        reason, expected_outcome = safe_action_trace_descriptions(action)
        outcome = (
            result.error_code.value
            if result.error_code is not None
            else "records_classified_and_storage_read_back"
        )
        return KiteCollectionActionTrace(
            iteration,
            action,
            reason,
            expected_outcome,
            authorized=True,
            executed=True,
            outcome_code=outcome,
            counts=result.counts,
        )

    @staticmethod
    def _rejected_plan(
        iteration: int,
        action: KiteCollectionAction = KiteCollectionAction.REJECTED_MODEL_OUTPUT,
        rejection_category: ModelOutputRejectionCategory | None = None,
    ) -> KiteCollectionActionTrace:
        reason, expected_outcome = safe_action_trace_descriptions(action)
        return KiteCollectionActionTrace(
            iteration,
            action,
            reason,
            expected_outcome,
            authorized=False,
            executed=False,
            outcome_code="unsupported_or_malformed_model_action",
            rejection_category=rejection_category,
        )

    @staticmethod
    def _observation_trace(
        iteration: int, outcome_code: str
    ) -> KiteCollectionActionTrace:
        reason, expected_outcome = safe_action_trace_descriptions(
            KiteCollectionAction.INSPECT_COVERAGE
        )
        return KiteCollectionActionTrace(
            iteration,
            KiteCollectionAction.INSPECT_COVERAGE,
            reason,
            expected_outcome,
            authorized=True,
            executed=True,
            outcome_code=outcome_code,
        )

    @staticmethod
    def _review_error(
        worker_result: KiteWorkerTaskResult | None,
    ) -> KiteWorkflowErrorCode | None:
        if worker_result is None:
            return None
        if worker_result.error_code is not None:
            return worker_result.error_code
        if worker_result.has_data_quality_issues:
            return KiteWorkflowErrorCode.DATA_QUALITY_REVIEW
        return None
