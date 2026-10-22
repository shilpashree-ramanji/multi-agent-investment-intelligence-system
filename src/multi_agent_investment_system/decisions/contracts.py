"""Provider-independent contracts for strategy and aggregate decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from multi_agent_investment_system.core.exceptions import DataValidationError
from multi_agent_investment_system.core.types import EvidenceId
from multi_agent_investment_system.decisions.journal import HumanApproval
from multi_agent_investment_system.risk.rules import RiskResult


class StrategyRecommendation(StrEnum):
    """A strategy's proposed action; this enum defines no strategy rules."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    ABSTAIN = "abstain"


class DataQualityStatus(StrEnum):
    """Explicit quality/freshness state for an input used by a strategy."""

    VALID = "valid"
    MISSING = "missing"
    STALE = "stale"
    INVALID = "invalid"
    UNKNOWN = "unknown"


def _require_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{field_name} must not be empty")


def _validate_optional_timestamp(value: datetime | None, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, datetime):
        raise DataValidationError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise DataValidationError(f"{field_name} must be timezone-aware")


def _validate_text_tuple(values: tuple[str, ...], field_name: str) -> None:
    if not isinstance(values, tuple) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise DataValidationError(f"{field_name} must contain non-empty strings")


@dataclass(frozen=True, slots=True)
class SupportingEvidence:
    """A concise strategy claim tied to source evidence and timestamps."""

    evidence_id: EvidenceId
    claim: str
    source: str
    source_reference: str
    observed_at: datetime | None = None
    retrieved_at: datetime | None = None

    def __post_init__(self) -> None:
        for field_name in ("evidence_id", "claim", "source", "source_reference"):
            _require_text(getattr(self, field_name), f"Evidence {field_name}")
        _validate_optional_timestamp(self.observed_at, "Evidence observation time")
        _validate_optional_timestamp(self.retrieved_at, "Evidence retrieval time")


@dataclass(frozen=True, slots=True)
class DataQualityIndicator:
    """One input's explicit quality state; no freshness threshold is implied."""

    name: str
    status: DataQualityStatus
    evidence_id: EvidenceId | None = None
    observed_at: datetime | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.name, "Data-quality input name")
        if not isinstance(self.status, DataQualityStatus):
            raise DataValidationError("Data-quality status is invalid")
        if self.evidence_id is not None:
            _require_text(self.evidence_id, "Data-quality evidence ID")
        if self.detail is not None:
            _require_text(self.detail, "Data-quality detail")
        _validate_optional_timestamp(self.observed_at, "Data observation time")


@dataclass(frozen=True, slots=True)
class StrategyResult:
    """Versioned output from one strategy for one instrument and horizon."""

    strategy_id: str
    strategy_version: str
    exchange: str
    trading_symbol: str
    evaluated_at: datetime
    investment_horizon: str
    recommendation: StrategyRecommendation
    confidence: Decimal
    rationale: str
    supporting_evidence: tuple[SupportingEvidence, ...]
    risk_flags: tuple[str, ...]
    invalidation_conditions: tuple[str, ...]
    data_quality: tuple[DataQualityIndicator, ...]
    input_data_references: tuple[EvidenceId, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "strategy_id",
            "strategy_version",
            "exchange",
            "trading_symbol",
            "investment_horizon",
            "rationale",
        ):
            _require_text(getattr(self, field_name), f"Strategy {field_name}")
        if any(character.isspace() for character in self.trading_symbol):
            raise DataValidationError("Strategy trading symbol is malformed")
        if not isinstance(self.evaluated_at, datetime):
            raise DataValidationError("Strategy evaluation time must be a datetime")
        _validate_optional_timestamp(self.evaluated_at, "Strategy evaluation time")
        if not isinstance(self.recommendation, StrategyRecommendation):
            raise DataValidationError("Strategy recommendation is invalid")
        if not isinstance(self.confidence, Decimal):
            raise DataValidationError("Strategy confidence must be a Decimal")
        if not self.confidence.is_finite() or not Decimal(0) <= self.confidence <= 1:
            raise DataValidationError("Strategy confidence must be between 0 and 1")

        if not isinstance(self.supporting_evidence, tuple) or any(
            not isinstance(item, SupportingEvidence)
            for item in self.supporting_evidence
        ):
            raise DataValidationError("Strategy supporting evidence is malformed")
        if not isinstance(self.data_quality, tuple) or any(
            not isinstance(item, DataQualityIndicator) for item in self.data_quality
        ):
            raise DataValidationError("Strategy data-quality indicators are malformed")
        _validate_text_tuple(self.risk_flags, "Strategy risk flags")
        _validate_text_tuple(
            self.invalidation_conditions, "Strategy invalidation conditions"
        )
        if not isinstance(self.input_data_references, tuple) or any(
            not isinstance(reference, str) or not reference.strip()
            for reference in self.input_data_references
        ):
            raise DataValidationError("Strategy input-data references are malformed")

        evidence_ids = {
            str(evidence.evidence_id) for evidence in self.supporting_evidence
        }
        input_ids = {str(reference) for reference in self.input_data_references}
        if not evidence_ids.issubset(input_ids):
            raise DataValidationError(
                "Supporting evidence must be listed among strategy input references"
            )
        if self.recommendation is not StrategyRecommendation.ABSTAIN:
            if not self.supporting_evidence or not self.input_data_references:
                raise DataValidationError(
                    "Actionable strategy recommendations require traceable evidence"
                )


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """One aggregation request with explicit instrument and evaluation time."""

    exchange: str
    trading_symbol: str
    investment_horizon: str
    evaluated_at: datetime
    strategy_results: tuple[StrategyResult, ...]
    risk_results: tuple[RiskResult, ...]

    def __post_init__(self) -> None:
        _require_text(self.exchange, "Decision exchange")
        _require_text(self.trading_symbol, "Decision trading symbol")
        _require_text(self.investment_horizon, "Decision investment horizon")
        if any(character.isspace() for character in self.trading_symbol):
            raise DataValidationError("Decision trading symbol is malformed")
        if not isinstance(self.evaluated_at, datetime):
            raise DataValidationError("Decision evaluation time must be a datetime")
        _validate_optional_timestamp(self.evaluated_at, "Decision evaluation time")
        if not isinstance(self.strategy_results, tuple) or any(
            not isinstance(result, StrategyResult) for result in self.strategy_results
        ):
            raise DataValidationError("Decision strategy results are malformed")
        if not isinstance(self.risk_results, tuple) or any(
            not isinstance(result, RiskResult) for result in self.risk_results
        ):
            raise DataValidationError("Decision risk results are malformed")


class DecisionFindingCode(StrEnum):
    """Stable explanation codes for aggregate outcomes."""

    MISSING_STRATEGY_RESULTS = "missing_strategy_results"
    MISSING_RISK_RESULTS = "missing_risk_results"
    RISK_VETO = "risk_veto"
    DATA_QUALITY_UNAVAILABLE = "data_quality_unavailable"
    DATA_QUALITY_BLOCK = "data_quality_block"
    INSUFFICIENT_CONFIDENCE = "insufficient_confidence"
    STRATEGY_ABSTENTION = "strategy_abstention"
    INCOMPATIBLE_HORIZONS = "incompatible_horizons"
    CONFLICTING_RECOMMENDATIONS = "conflicting_recommendations"


@dataclass(frozen=True, slots=True)
class DecisionFinding:
    """Structured explanation for a decision-engine outcome."""

    code: DecisionFindingCode
    strategy_ids: tuple[str, ...] = ()
    risk_rule_ids: tuple[str, ...] = ()
    input_name: str | None = None


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Explainable aggregate proposal; it never authorizes order execution."""

    exchange: str
    trading_symbol: str
    investment_horizon: str
    evaluated_at: datetime
    recommendation: StrategyRecommendation
    findings: tuple[DecisionFinding, ...]
    strategy_results: tuple[StrategyResult, ...]
    risk_results: tuple[RiskResult, ...]
    human_approval: HumanApproval
