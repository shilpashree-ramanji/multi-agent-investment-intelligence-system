"""Validated task and state records for the first Kite ingestion workflow."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import date, datetime
from enum import StrEnum

from multi_agent_investment_system.core.exceptions import (
    DataValidationError,
    MarketDataProviderDiagnostic,
)
from multi_agent_investment_system.llm.provider import (
    ModelOutputRejectionCategory,
    ModelProviderDiagnostic,
)


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{name} must not be empty")


_TRADING_SYMBOL_PATTERN = re.compile(r"[A-Z0-9]+(?:[&-][A-Z0-9]+)*(?:\.(?:NS|BO))?")


class KiteWorkflowStatus(StrEnum):
    """Execution lifecycle; COMPLETED does not imply full data coverage."""

    PENDING = "pending"
    COORDINATING = "coordinating"
    ASSIGNED = "assigned"
    WORKER_RUNNING = "worker_running"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    FAILED = "failed"


class KiteWorkflowErrorCode(StrEnum):
    """Sanitized workflow errors that do not contain provider response text."""

    MODEL_PROVIDER_ERROR = "model_provider_error"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    VALIDATION_ERROR = "validation_error"
    MARKET_DATA_ERROR = "market_data_error"
    STORAGE_ERROR = "storage_error"
    INGESTION_ERROR = "ingestion_error"
    DATA_QUALITY_REVIEW = "data_quality_review"
    ITERATION_BUDGET_EXHAUSTED = "iteration_budget_exhausted"


class WorkerTaskStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class KiteCoverageStatus(StrEnum):
    """Coverage assessment, independent of whether task execution succeeded."""

    UNKNOWN = "unknown"
    NOT_ASSESSED = "not_assessed"


class KiteCoverageReason(StrEnum):
    """Evidence limitations that prevent asserting complete historical coverage."""

    NO_TRADING_CALENDAR = "no_trading_calendar"
    NO_PROVIDER_COVERAGE_GUARANTEE = "no_provider_coverage_guarantee"
    HISTORICAL_INTERVALS_NOT_VERIFIED = "historical_intervals_not_verified"
    EXECUTION_NOT_SUCCESSFUL = "execution_not_successful"


class KiteCollectionAction(StrEnum):
    """Actions and outcomes recorded by a bounded collection run."""

    INSPECT_COVERAGE = "inspect_coverage"
    COLLECT_RANGE = "collect_range"
    RETRY_COLLECTION = "retry_collection"
    COMPLETE = "complete"
    ESCALATE = "escalate"
    REJECTED_MODEL_OUTPUT = "rejected_model_output"


_SAFE_ACTION_TRACE_DESCRIPTIONS = {
    KiteCollectionAction.INSPECT_COVERAGE: (
        "Local stored coverage was inspected.",
        "Stored dates and candle quality were observed.",
    ),
    KiteCollectionAction.COLLECT_RANGE: (
        "A bounded collection action was authorized.",
        "The assigned range is retrieved, validated, and persisted.",
    ),
    KiteCollectionAction.RETRY_COLLECTION: (
        "A bounded retry action was authorized.",
        "The assigned range is retried within workflow limits.",
    ),
    KiteCollectionAction.COMPLETE: (
        "Completion followed deterministic verification.",
        "The verified result is reported without claiming full session coverage.",
    ),
    KiteCollectionAction.ESCALATE: (
        "The workflow was escalated for review.",
        "No further collection proceeds automatically.",
    ),
    KiteCollectionAction.REJECTED_MODEL_OUTPUT: (
        "The model proposal failed deterministic validation.",
        "No unauthorized operation executes.",
    ),
}


def safe_action_trace_descriptions(
    action: KiteCollectionAction,
) -> tuple[str, str]:
    """Return fixed report text; never place model prose into action traces."""

    if not isinstance(action, KiteCollectionAction):
        raise DataValidationError("Collection action is invalid")
    return _SAFE_ACTION_TRACE_DESCRIPTIONS[action]


@dataclass(frozen=True, slots=True)
class KiteDataCoverageObservation:
    """Sanitized local coverage and quality facts supplied to the model."""

    stored_dates: tuple[date, ...]
    valid_count: int
    incomplete_count: int
    invalid_count: int
    source_capabilities: tuple[str, ...] = ("daily_ohlcv", "historical_range")

    def __post_init__(self) -> None:
        if not isinstance(self.stored_dates, tuple) or any(
            not isinstance(item, date) or isinstance(item, datetime)
            for item in self.stored_dates
        ):
            raise DataValidationError("Coverage observation dates are malformed")
        if tuple(sorted(set(self.stored_dates))) != self.stored_dates:
            raise DataValidationError("Coverage observation dates are not unique")
        counts = (self.valid_count, self.incomplete_count, self.invalid_count)
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in counts
        ):
            raise DataValidationError("Coverage observation counts are malformed")
        if sum(counts) != len(self.stored_dates):
            raise DataValidationError(
                "Coverage dates and quality counts do not reconcile"
            )
        if not isinstance(self.source_capabilities, tuple) or any(
            not isinstance(item, str) or not item.strip()
            for item in self.source_capabilities
        ):
            raise DataValidationError("Source capabilities are malformed")


@dataclass(frozen=True, slots=True)
class KiteCollectionActionTrace:
    """Auditable action and deterministic tool authorization outcome."""

    iteration: int
    action: KiteCollectionAction
    reason: str
    expected_outcome: str
    authorized: bool
    executed: bool
    outcome_code: str
    counts: KiteIngestionCounts | None = None
    rejection_category: ModelOutputRejectionCategory | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.iteration, bool)
            or not isinstance(self.iteration, int)
            or self.iteration < 1
        ):
            raise DataValidationError("Collection iteration is invalid")
        if not isinstance(self.action, KiteCollectionAction):
            raise DataValidationError("Collection action is invalid")
        for name in ("reason", "expected_outcome", "outcome_code"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 400:
                raise DataValidationError(
                    f"Collection {name.replace('_', ' ')} is invalid"
                )
        if not isinstance(self.authorized, bool) or not isinstance(self.executed, bool):
            raise DataValidationError("Collection action flags are malformed")
        if self.executed and not self.authorized:
            raise DataValidationError("An unauthorized collection action was executed")
        if self.counts is not None and not isinstance(self.counts, KiteIngestionCounts):
            raise DataValidationError("Collection action counts are malformed")
        if self.rejection_category is not None and (
            not isinstance(
                self.rejection_category, ModelOutputRejectionCategory
            )
            or self.action is not KiteCollectionAction.REJECTED_MODEL_OUTPUT
            or self.authorized
            or self.executed
        ):
            raise DataValidationError("Collection rejection category is invalid")


@dataclass(frozen=True, slots=True)
class KiteIngestionPolicy:
    """Caller-configured bound on the requested inclusive calendar range."""

    maximum_calendar_days: int
    maximum_iterations: int = 5
    maximum_collection_attempts: int = 2

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_calendar_days, bool)
            or not isinstance(self.maximum_calendar_days, int)
            or self.maximum_calendar_days < 1
        ):
            raise DataValidationError(
                "Maximum calendar days must be a positive integer"
            )
        for name in ("maximum_iterations", "maximum_collection_attempts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise DataValidationError(
                    f"{name.replace('_', ' ').title()} must be positive"
                )
        if self.maximum_collection_attempts > self.maximum_iterations:
            raise DataValidationError("Collection attempts exceed iteration limit")


@dataclass(frozen=True, slots=True)
class KiteIngestionRequest:
    """Immutable caller-supplied task; agents cannot rewrite its parameters.

    Symbols use uppercase ASCII alphanumeric segments, optionally separated
    by ``&`` or ``-``. Yahoo-style ``.NS`` and ``.BO`` suffixes and numeric
    exchange scrip codes are syntactically accepted. This checks format only;
    it does not prove an exchange listing exists or is supported by Kite.
    """

    run_id: str
    task_id: str
    symbol: str
    start: date
    end: date

    def __post_init__(self) -> None:
        for name in ("run_id", "task_id", "symbol"):
            _require_text(getattr(self, name), name.replace("_", " ").title())
        if _TRADING_SYMBOL_PATTERN.fullmatch(self.symbol) is None:
            raise DataValidationError("Kite symbol has an unsupported format")
        for name in ("start", "end"):
            value = getattr(self, name)
            if not isinstance(value, date) or isinstance(value, datetime):
                raise DataValidationError(f"Kite {name} must be a date")
        if self.start > self.end:
            raise DataValidationError("Kite start date must not be after end date")

    def validate_policy(self, policy: KiteIngestionPolicy) -> None:
        """Refuse requests that exceed the explicit caller-configured range."""

        if not isinstance(policy, KiteIngestionPolicy):
            raise DataValidationError("Kite ingestion policy is malformed")
        calendar_days = (self.end - self.start).days + 1
        if calendar_days > policy.maximum_calendar_days:
            raise DataValidationError(
                "Kite request exceeds the configured maximum calendar range"
            )

    @property
    def fingerprint(self) -> str:
        """Stable identity for this run/task and its immutable market-data scope."""

        payload = json.dumps(
            {
                "run_id": self.run_id,
                "task_id": self.task_id,
                "symbol": self.symbol,
                "start": self.start.isoformat(),
                "end": self.end.isoformat(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class KiteIngestionAssignment:
    """Coordinator assignment retaining the exact immutable request identity."""

    run_id: str
    task_id: str
    request_fingerprint: str
    worker_role: str = "zerodha_ingestion_worker"

    def __post_init__(self) -> None:
        _require_text(self.run_id, "Assignment run ID")
        _require_text(self.task_id, "Assignment task ID")
        if (
            not isinstance(self.request_fingerprint, str)
            or len(self.request_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.request_fingerprint
            )
        ):
            raise DataValidationError("Assignment request fingerprint is invalid")
        if self.worker_role != "zerodha_ingestion_worker":
            raise DataValidationError("Kite task has an unsupported worker role")

    @classmethod
    def for_request(cls, request: KiteIngestionRequest) -> KiteIngestionAssignment:
        """Bind an assignment to the complete validated request scope."""

        if not isinstance(request, KiteIngestionRequest):
            raise DataValidationError("Assignment request is malformed")
        return cls(request.run_id, request.task_id, request.fingerprint)


@dataclass(frozen=True, slots=True)
class KiteIngestionCounts:
    """Immutable authoritative counts returned by the deterministic service."""

    fetched: int
    valid: int
    incomplete: int
    invalid: int
    upserted: int
    stored: int

    def __post_init__(self) -> None:
        values = (
            self.fetched,
            self.valid,
            self.incomplete,
            self.invalid,
            self.upserted,
            self.stored,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise DataValidationError("Ingestion counts must be non-negative integers")
        if self.fetched != self.valid + self.incomplete + self.invalid:
            raise DataValidationError(
                "Ingestion classification counts do not reconcile"
            )
        if self.upserted > self.valid:
            raise DataValidationError("Upserted count exceeds valid record count")


@dataclass(frozen=True, slots=True)
class KiteMarketDataEvidence:
    """Immutable source and validation references without raw mutable records."""

    symbol: str
    source: str | None
    observed_at: datetime | None
    retrieved_at: str | None
    validation_issue_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.symbol, "Evidence symbol")
        if self.source is not None:
            _require_text(self.source, "Evidence source")
        if self.observed_at is not None and not isinstance(self.observed_at, datetime):
            raise DataValidationError("Evidence observation time is invalid")
        if self.retrieved_at is not None:
            _require_text(self.retrieved_at, "Evidence retrieval time")
        if not isinstance(self.validation_issue_codes, tuple) or any(
            not isinstance(code, str) or not code.strip()
            for code in self.validation_issue_codes
        ):
            raise DataValidationError("Evidence validation codes are malformed")


@dataclass(frozen=True, slots=True)
class KiteWorkerTaskResult:
    """Execution outcome and independent coverage assessment for the coordinator."""

    run_id: str
    task_id: str
    status: WorkerTaskStatus
    counts: KiteIngestionCounts | None = None
    evidence: tuple[KiteMarketDataEvidence, ...] = ()
    error_code: KiteWorkflowErrorCode | None = None
    retryable: bool = False
    market_data_diagnostic: MarketDataProviderDiagnostic | None = None
    coverage_status: KiteCoverageStatus = KiteCoverageStatus.NOT_ASSESSED
    coverage_reasons: tuple[KiteCoverageReason, ...] = (
        KiteCoverageReason.EXECUTION_NOT_SUCCESSFUL,
    )
    model_provider_diagnostic: ModelProviderDiagnostic | None = None

    def __post_init__(self) -> None:
        _require_text(self.run_id, "Worker result run ID")
        _require_text(self.task_id, "Worker result task ID")
        if not isinstance(self.status, WorkerTaskStatus):
            raise DataValidationError("Worker result status is invalid")
        if not isinstance(self.retryable, bool):
            raise DataValidationError("Worker retryability flag is invalid")
        if self.model_provider_diagnostic is not None and not isinstance(
            self.model_provider_diagnostic, ModelProviderDiagnostic
        ):
            raise DataValidationError("Worker model diagnostic is malformed")
        if self.market_data_diagnostic is not None and not isinstance(
            self.market_data_diagnostic, MarketDataProviderDiagnostic
        ):
            raise DataValidationError("Worker market-data diagnostic is malformed")
        if not isinstance(self.coverage_status, KiteCoverageStatus):
            raise DataValidationError("Worker coverage status is invalid")
        if not isinstance(self.coverage_reasons, tuple) or any(
            not isinstance(reason, KiteCoverageReason)
            for reason in self.coverage_reasons
        ):
            raise DataValidationError("Worker coverage reasons are malformed")
        if self.status is WorkerTaskStatus.SUCCEEDED:
            if not isinstance(self.counts, KiteIngestionCounts):
                raise DataValidationError("Successful worker result needs counts")
            if self.error_code is not None:
                raise DataValidationError(
                    "Successful worker result cannot have an error"
                )
            required_coverage_reasons = {
                KiteCoverageReason.NO_TRADING_CALENDAR,
                KiteCoverageReason.NO_PROVIDER_COVERAGE_GUARANTEE,
                KiteCoverageReason.HISTORICAL_INTERVALS_NOT_VERIFIED,
            }
            if (
                self.coverage_status is not KiteCoverageStatus.UNKNOWN
                or len(self.coverage_reasons) != len(required_coverage_reasons)
                or set(self.coverage_reasons) != required_coverage_reasons
            ):
                raise DataValidationError(
                    "Successful execution must report unverified coverage"
                )
        elif self.error_code is None or self.counts is not None or self.evidence:
            raise DataValidationError(
                "Failed worker result needs only a sanitized error"
            )
        elif (
            self.coverage_status is not KiteCoverageStatus.NOT_ASSESSED
            or self.coverage_reasons != (KiteCoverageReason.EXECUTION_NOT_SUCCESSFUL,)
        ):
            raise DataValidationError(
                "Unsuccessful execution cannot claim a coverage assessment"
            )
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(item, KiteMarketDataEvidence) for item in self.evidence
        ):
            raise DataValidationError("Worker evidence is malformed")
        if self.error_code is not None and not isinstance(
            self.error_code, KiteWorkflowErrorCode
        ):
            raise DataValidationError("Worker error code is invalid")
        if self.model_provider_diagnostic is not None and (
            self.status is not WorkerTaskStatus.FAILED
            or self.error_code is not KiteWorkflowErrorCode.MODEL_PROVIDER_ERROR
        ):
            raise DataValidationError("Worker model diagnostic has no model error")
        if self.market_data_diagnostic is not None and (
            self.status is not WorkerTaskStatus.FAILED
            or self.error_code is not KiteWorkflowErrorCode.MARKET_DATA_ERROR
        ):
            raise DataValidationError(
                "Worker market-data diagnostic has no market-data error"
            )
        if self.retryable and (
            self.status is not WorkerTaskStatus.FAILED
            or self.error_code is not KiteWorkflowErrorCode.MARKET_DATA_ERROR
        ):
            raise DataValidationError("Only failed market-data requests may be retried")

    @property
    def has_data_quality_issues(self) -> bool:
        """Whether provider records were incomplete or invalid."""

        if self.counts is None:
            return False
        return bool(self.counts.incomplete or self.counts.invalid)


_ALLOWED_TRANSITIONS = {
    KiteWorkflowStatus.PENDING: frozenset({KiteWorkflowStatus.COORDINATING}),
    KiteWorkflowStatus.COORDINATING: frozenset(
        {
            KiteWorkflowStatus.ASSIGNED,
            KiteWorkflowStatus.FAILED,
            KiteWorkflowStatus.ESCALATED,
        }
    ),
    KiteWorkflowStatus.ASSIGNED: frozenset({KiteWorkflowStatus.WORKER_RUNNING}),
    KiteWorkflowStatus.WORKER_RUNNING: frozenset(
        {KiteWorkflowStatus.REVIEWING, KiteWorkflowStatus.FAILED}
    ),
    KiteWorkflowStatus.REVIEWING: frozenset(
        {
            KiteWorkflowStatus.COORDINATING,
            KiteWorkflowStatus.WORKER_RUNNING,
            KiteWorkflowStatus.COMPLETED,
            KiteWorkflowStatus.ESCALATED,
            KiteWorkflowStatus.FAILED,
        }
    ),
    KiteWorkflowStatus.COMPLETED: frozenset(),
    KiteWorkflowStatus.ESCALATED: frozenset(),
    KiteWorkflowStatus.FAILED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class KiteIngestionWorkflowState:
    """Immutable request, assignment, result, error, and validated transition log."""

    request: KiteIngestionRequest
    status: KiteWorkflowStatus = KiteWorkflowStatus.PENDING
    transitions: tuple[KiteWorkflowStatus, ...] = (KiteWorkflowStatus.PENDING,)
    assignments: tuple[KiteIngestionAssignment, ...] = ()
    worker_result: KiteWorkerTaskResult | None = None
    error_code: KiteWorkflowErrorCode | None = None
    initial_observation: KiteDataCoverageObservation | None = None
    final_observation: KiteDataCoverageObservation | None = None
    action_trace: tuple[KiteCollectionActionTrace, ...] = ()
    model_provider_diagnostic: ModelProviderDiagnostic | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, KiteIngestionRequest):
            raise DataValidationError("Workflow request is malformed")
        if not isinstance(self.status, KiteWorkflowStatus):
            raise DataValidationError("Workflow status is invalid")
        if (
            not isinstance(self.transitions, tuple)
            or not self.transitions
            or any(
                not isinstance(item, KiteWorkflowStatus) for item in self.transitions
            )
            or self.transitions[0] is not KiteWorkflowStatus.PENDING
            or self.transitions[-1] is not self.status
        ):
            raise DataValidationError("Workflow transition history is malformed")
        for current, following in zip(
            self.transitions, self.transitions[1:], strict=False
        ):
            if following not in _ALLOWED_TRANSITIONS[current]:
                raise DataValidationError("Workflow transition history is invalid")
        if not isinstance(self.assignments, tuple) or any(
            not isinstance(item, KiteIngestionAssignment) for item in self.assignments
        ):
            raise DataValidationError("Workflow assignments are malformed")
        if len(self.assignments) > 1:
            raise DataValidationError("Kite workflow permits one worker assignment")
        if self.assignments and KiteWorkflowStatus.ASSIGNED not in self.transitions:
            raise DataValidationError("Workflow assignment is absent from its history")
        if self.assignments and (
            self.assignments[0].run_id != self.request.run_id
            or self.assignments[0].task_id != self.request.task_id
            or self.assignments[0].request_fingerprint != self.request.fingerprint
        ):
            raise DataValidationError("Workflow assignment does not match its request")
        if self.worker_result is not None and (
            self.worker_result.run_id != self.request.run_id
            or self.worker_result.task_id != self.request.task_id
        ):
            raise DataValidationError("Worker result does not match its request")
        if self.worker_result is not None and KiteWorkflowStatus.REVIEWING not in (
            self.transitions
        ):
            raise DataValidationError("Worker result is absent from workflow history")
        for name in ("initial_observation", "final_observation"):
            observation = getattr(self, name)
            if observation is not None and not isinstance(
                observation, KiteDataCoverageObservation
            ):
                raise DataValidationError("Workflow coverage observation is malformed")
        if not isinstance(self.action_trace, tuple) or any(
            not isinstance(item, KiteCollectionActionTrace)
            for item in self.action_trace
        ):
            raise DataValidationError("Workflow action trace is malformed")
        if (
            self.status
            in {
                KiteWorkflowStatus.ASSIGNED,
                KiteWorkflowStatus.WORKER_RUNNING,
                KiteWorkflowStatus.REVIEWING,
                KiteWorkflowStatus.COMPLETED,
            }
            and not self.assignments
        ):
            raise DataValidationError("Workflow state is missing its worker assignment")
        if (
            self.status
            in {
                KiteWorkflowStatus.REVIEWING,
                KiteWorkflowStatus.COMPLETED,
            }
            and self.worker_result is None
        ):
            raise DataValidationError("Workflow state is missing its worker result")
        if self.status is KiteWorkflowStatus.FAILED and self.error_code is None:
            raise DataValidationError("Failed workflow needs a sanitized error code")
        if self.error_code is not None and not isinstance(
            self.error_code, KiteWorkflowErrorCode
        ):
            raise DataValidationError("Workflow error code is invalid")
        if self.error_code is not None and self.status not in {
            KiteWorkflowStatus.FAILED,
            KiteWorkflowStatus.ESCALATED,
        }:
            raise DataValidationError("Workflow error code is invalid for this state")
        if self.model_provider_diagnostic is not None and (
            not isinstance(self.model_provider_diagnostic, ModelProviderDiagnostic)
            or self.status is not KiteWorkflowStatus.FAILED
            or self.error_code is not KiteWorkflowErrorCode.MODEL_PROVIDER_ERROR
        ):
            raise DataValidationError("Workflow model diagnostic is invalid")
        if self.status is KiteWorkflowStatus.COMPLETED and (
            self.worker_result is None
            or self.worker_result.status is not WorkerTaskStatus.SUCCEEDED
            or self.worker_result.has_data_quality_issues
            or self.error_code is not None
            or self.worker_result.counts is None
            or self.worker_result.counts.valid != self.worker_result.counts.upserted
            or self.final_observation is None
            or self.final_observation.incomplete_count
            or self.final_observation.invalid_count
            or len(self.final_observation.stored_dates)
            != self.worker_result.counts.stored
        ):
            raise DataValidationError(
                "Completed workflow lacks verified persistence or has quality issues"
            )

    def transition(
        self,
        status: KiteWorkflowStatus,
        *,
        assignment: KiteIngestionAssignment | None = None,
        worker_result: KiteWorkerTaskResult | None = None,
        error_code: KiteWorkflowErrorCode | None = None,
        initial_observation: KiteDataCoverageObservation | None = None,
        final_observation: KiteDataCoverageObservation | None = None,
        action_trace: KiteCollectionActionTrace | None = None,
        model_provider_diagnostic: ModelProviderDiagnostic | None = None,
    ) -> KiteIngestionWorkflowState:
        """Return a new state only when the transition and attached data are valid."""

        if (
            not isinstance(status, KiteWorkflowStatus)
            or status not in (_ALLOWED_TRANSITIONS[self.status])
        ):
            raise DataValidationError("Workflow transition is not permitted")
        if assignment is not None:
            if status is not KiteWorkflowStatus.ASSIGNED:
                raise DataValidationError(
                    "Assignment is only valid at assignment state"
                )
            if not isinstance(assignment, KiteIngestionAssignment):
                raise DataValidationError("Workflow assignment is malformed")
        if worker_result is not None:
            if status is not KiteWorkflowStatus.REVIEWING:
                raise DataValidationError("Worker result is only valid at review state")
            if not isinstance(worker_result, KiteWorkerTaskResult):
                raise DataValidationError("Worker result is malformed")
        if error_code is not None and not isinstance(error_code, KiteWorkflowErrorCode):
            raise DataValidationError("Workflow error code is invalid")
        if initial_observation is not None and not isinstance(
            initial_observation, KiteDataCoverageObservation
        ):
            raise DataValidationError("Workflow coverage observation is malformed")
        if final_observation is not None and not isinstance(
            final_observation, KiteDataCoverageObservation
        ):
            raise DataValidationError("Workflow coverage observation is malformed")
        if action_trace is not None and not isinstance(
            action_trace, KiteCollectionActionTrace
        ):
            raise DataValidationError("Workflow action trace is malformed")
        if model_provider_diagnostic is not None and (
            not isinstance(model_provider_diagnostic, ModelProviderDiagnostic)
            or status is not KiteWorkflowStatus.FAILED
            or error_code is not KiteWorkflowErrorCode.MODEL_PROVIDER_ERROR
        ):
            raise DataValidationError("Workflow model diagnostic is invalid")
        return replace(
            self,
            status=status,
            transitions=(*self.transitions, status),
            assignments=(
                (*self.assignments, assignment)
                if assignment is not None
                else self.assignments
            ),
            worker_result=worker_result or self.worker_result,
            error_code=error_code,
            initial_observation=(initial_observation or self.initial_observation),
            final_observation=(final_observation or self.final_observation),
            action_trace=(
                (*self.action_trace, action_trace)
                if action_trace is not None
                else self.action_trace
            ),
            model_provider_diagnostic=(
                model_provider_diagnostic or self.model_provider_diagnostic
            ),
        )

    def record_action(
        self, action_trace: KiteCollectionActionTrace
    ) -> KiteIngestionWorkflowState:
        """Append an already-authorized bounded action without changing status."""

        if not isinstance(action_trace, KiteCollectionActionTrace):
            raise DataValidationError("Workflow action trace is malformed")
        return replace(self, action_trace=(*self.action_trace, action_trace))
