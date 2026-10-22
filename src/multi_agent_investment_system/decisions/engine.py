"""Deterministic, provider-independent multi-strategy aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Protocol

from multi_agent_investment_system.core.exceptions import DataValidationError
from multi_agent_investment_system.decisions.contracts import (
    DataQualityStatus,
    DecisionFinding,
    DecisionFindingCode,
    DecisionRequest,
    DecisionResult,
    StrategyRecommendation,
    StrategyResult,
)
from multi_agent_investment_system.decisions.journal import HumanApproval
from multi_agent_investment_system.risk.rules import RiskResult


class DecisionEngine(Protocol):
    """Interface for turning strategy outputs into an explainable proposal."""

    def decide(self, request: DecisionRequest) -> DecisionResult:
        """Aggregate one instrument's strategy and deterministic risk outputs."""


class StrictConsensusDecisionEngine:
    """Opt-in reference policy that abstains unless all gates pass and agree.

    Required strategies and the confidence cutoff are explicit constructor
    inputs; no project threshold or strategy list is embedded here. This
    baseline has no vote weights or aggregate-confidence formula. Risk results
    are independent deterministic checks: any failure vetoes an action, and
    missing configured checks also abstain. Every actionable result remains
    pending human approval.
    """

    def __init__(
        self,
        required_strategy_ids: Sequence[str],
        minimum_confidence: Decimal,
        required_risk_rule_ids: Sequence[str] = (),
    ) -> None:
        if (
            isinstance(required_strategy_ids, (str, bytes))
            or not required_strategy_ids
            or any(
                not isinstance(strategy_id, str) or not strategy_id.strip()
                for strategy_id in required_strategy_ids
            )
        ):
            raise DataValidationError(
                "Decision engine requires non-empty strategy identifiers"
            )
        if len(set(required_strategy_ids)) != len(required_strategy_ids):
            raise DataValidationError("Required strategy identifiers must be unique")
        if isinstance(required_risk_rule_ids, (str, bytes)) or any(
            not isinstance(rule_id, str) or not rule_id.strip()
            for rule_id in required_risk_rule_ids
        ):
            raise DataValidationError(
                "Required risk-rule identifiers must be non-empty strings"
            )
        if len(set(required_risk_rule_ids)) != len(required_risk_rule_ids):
            raise DataValidationError("Required risk-rule identifiers must be unique")
        if (
            not isinstance(minimum_confidence, Decimal)
            or not minimum_confidence.is_finite()
            or not Decimal(0) <= minimum_confidence <= 1
        ):
            raise DataValidationError(
                "Minimum decision confidence must be between 0 and 1"
            )

        self._required_strategy_ids = tuple(required_strategy_ids)
        self._required_risk_rule_ids = tuple(required_risk_rule_ids)
        self._minimum_confidence = minimum_confidence

    def decide(self, request: DecisionRequest) -> DecisionResult:
        """Apply fail-closed validation and strict agreement to strategy results."""

        if not isinstance(request, DecisionRequest):
            raise DataValidationError("Decision request is malformed")

        ordered_results = tuple(
            sorted(request.strategy_results, key=lambda result: result.strategy_id)
        )
        ordered_risk_results = tuple(
            sorted(request.risk_results, key=lambda result: result.rule_id)
        )
        self._validate_inputs(request, ordered_results, ordered_risk_results)

        result_by_id = {result.strategy_id: result for result in ordered_results}
        risk_by_id = {result.rule_id: result for result in ordered_risk_results}
        findings: list[DecisionFinding] = []

        missing_strategies = tuple(
            strategy_id
            for strategy_id in self._required_strategy_ids
            if strategy_id not in result_by_id
        )
        if missing_strategies:
            findings.append(
                DecisionFinding(
                    DecisionFindingCode.MISSING_STRATEGY_RESULTS,
                    strategy_ids=missing_strategies,
                )
            )

        missing_risk_rules = tuple(
            rule_id
            for rule_id in self._required_risk_rule_ids
            if rule_id not in risk_by_id
        )
        if missing_risk_rules or not ordered_risk_results:
            findings.append(
                DecisionFinding(
                    DecisionFindingCode.MISSING_RISK_RESULTS,
                    risk_rule_ids=missing_risk_rules,
                )
            )

        failed_risk_rules = tuple(
            result.rule_id for result in ordered_risk_results if not result.passed
        )
        if failed_risk_rules:
            findings.append(
                DecisionFinding(
                    DecisionFindingCode.RISK_VETO,
                    risk_rule_ids=failed_risk_rules,
                )
            )

        for result in ordered_results:
            if not result.data_quality:
                findings.append(
                    DecisionFinding(
                        DecisionFindingCode.DATA_QUALITY_UNAVAILABLE,
                        strategy_ids=(result.strategy_id,),
                    )
                )
                continue
            for indicator in result.data_quality:
                if indicator.status is not DataQualityStatus.VALID:
                    findings.append(
                        DecisionFinding(
                            DecisionFindingCode.DATA_QUALITY_BLOCK,
                            strategy_ids=(result.strategy_id,),
                            input_name=indicator.name,
                        )
                    )

            if result.confidence < self._minimum_confidence:
                findings.append(
                    DecisionFinding(
                        DecisionFindingCode.INSUFFICIENT_CONFIDENCE,
                        strategy_ids=(result.strategy_id,),
                    )
                )
            if result.recommendation is StrategyRecommendation.ABSTAIN:
                findings.append(
                    DecisionFinding(
                        DecisionFindingCode.STRATEGY_ABSTENTION,
                        strategy_ids=(result.strategy_id,),
                    )
                )
            if result.investment_horizon != request.investment_horizon:
                findings.append(
                    DecisionFinding(
                        DecisionFindingCode.INCOMPATIBLE_HORIZONS,
                        strategy_ids=(result.strategy_id,),
                    )
                )

        action_recommendations = {
            result.recommendation
            for result in ordered_results
            if result.recommendation is not StrategyRecommendation.ABSTAIN
        }
        if len(action_recommendations) > 1:
            findings.append(
                DecisionFinding(
                    DecisionFindingCode.CONFLICTING_RECOMMENDATIONS,
                    strategy_ids=tuple(
                        result.strategy_id for result in ordered_results
                    ),
                )
            )

        findings_tuple = tuple(
            sorted(
                findings,
                key=lambda finding: (
                    finding.code.value,
                    finding.strategy_ids,
                    finding.risk_rule_ids,
                    finding.input_name or "",
                ),
            )
        )
        if findings_tuple or len(action_recommendations) != 1:
            recommendation = StrategyRecommendation.ABSTAIN
        else:
            recommendation = next(iter(action_recommendations))

        return DecisionResult(
            exchange=request.exchange,
            trading_symbol=request.trading_symbol,
            investment_horizon=request.investment_horizon,
            evaluated_at=request.evaluated_at,
            recommendation=recommendation,
            findings=findings_tuple,
            strategy_results=ordered_results,
            risk_results=ordered_risk_results,
            human_approval=(
                HumanApproval.PENDING
                if recommendation is not StrategyRecommendation.ABSTAIN
                else HumanApproval.NOT_REQUIRED
            ),
        )

    def _validate_inputs(
        self,
        request: DecisionRequest,
        strategy_results: tuple[StrategyResult, ...],
        risk_results: tuple[RiskResult, ...],
    ) -> None:
        strategy_ids = [result.strategy_id for result in strategy_results]
        if len(strategy_ids) != len(set(strategy_ids)):
            raise DataValidationError("Decision request contains duplicate strategies")
        unexpected = set(strategy_ids) - set(self._required_strategy_ids)
        if unexpected:
            raise DataValidationError(
                "Decision request contains unconfigured strategy results"
            )
        for result in strategy_results:
            if (
                result.exchange.strip().upper() != request.exchange.strip().upper()
                or result.trading_symbol.strip().upper()
                != request.trading_symbol.strip().upper()
            ):
                raise DataValidationError(
                    "Strategy result instrument does not match decision request"
                )
            if result.evaluated_at > request.evaluated_at:
                raise DataValidationError(
                    "Strategy result is later than decision evaluation time"
                )

        risk_ids: list[str] = []
        for result in risk_results:
            if not isinstance(result.rule_id, str) or not result.rule_id.strip():
                raise DataValidationError("Risk result identifier must not be empty")
            if not isinstance(result.passed, bool):
                raise DataValidationError("Risk result status must be boolean")
            if not isinstance(result.reasons, tuple) or any(
                not isinstance(reason, str) or not reason.strip()
                for reason in result.reasons
            ):
                raise DataValidationError("Risk result reasons are malformed")
            risk_ids.append(result.rule_id)
        if len(risk_ids) != len(set(risk_ids)):
            raise DataValidationError(
                "Decision request contains duplicate risk results"
            )
        unexpected_risks = set(risk_ids) - set(self._required_risk_rule_ids)
        if self._required_risk_rule_ids and unexpected_risks:
            raise DataValidationError(
                "Decision request contains unconfigured risk results"
            )
