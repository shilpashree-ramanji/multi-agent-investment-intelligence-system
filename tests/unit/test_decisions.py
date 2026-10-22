"""Tests for shared strategy-result and decision-engine contracts."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from multi_agent_investment_system.core.exceptions import DataValidationError
from multi_agent_investment_system.core.types import EvidenceId
from multi_agent_investment_system.decisions.contracts import (
    DataQualityIndicator,
    DataQualityStatus,
    DecisionFindingCode,
    DecisionRequest,
    StrategyRecommendation,
    StrategyResult,
    SupportingEvidence,
)
from multi_agent_investment_system.decisions.engine import (
    StrictConsensusDecisionEngine,
)
from multi_agent_investment_system.decisions.journal import HumanApproval
from multi_agent_investment_system.risk.rules import RiskResult

_NOW = datetime(2026, 10, 22, 12, tzinfo=UTC)


def _strategy_result(
    strategy_id: str,
    *,
    recommendation: StrategyRecommendation = StrategyRecommendation.BUY,
    confidence: Decimal = Decimal("0.8"),
    quality: tuple[DataQualityIndicator, ...] | None = None,
    horizon: str = "long-term",
    evidence: tuple[SupportingEvidence, ...] | None = None,
) -> StrategyResult:
    evidence_items = evidence if evidence is not None else (
        SupportingEvidence(
            evidence_id=EvidenceId(f"{strategy_id}-input-1"),
            claim="Synthetic supporting observation",
            source="synthetic-source",
            source_reference="synthetic-reference-1",
            observed_at=_NOW - timedelta(days=1),
            retrieved_at=_NOW,
        ),
    )
    quality_items = quality
    if quality_items is None:
        quality_items = (
            (
                DataQualityIndicator(
                    name="daily_prices",
                    status=DataQualityStatus.VALID,
                    evidence_id=evidence_items[0].evidence_id,
                    observed_at=_NOW - timedelta(days=1),
                ),
            )
            if evidence_items
            else ()
        )
    return StrategyResult(
        strategy_id=strategy_id,
        strategy_version="test-v1",
        exchange="NSE",
        trading_symbol="RELIANCE",
        evaluated_at=_NOW,
        investment_horizon=horizon,
        recommendation=recommendation,
        confidence=confidence,
        rationale="Synthetic test rationale",
        supporting_evidence=evidence_items,
        risk_flags=(),
        invalidation_conditions=("Synthetic invalidation condition",),
        data_quality=quality_items,
        input_data_references=tuple(item.evidence_id for item in evidence_items),
    )


def _request(
    strategies: tuple[StrategyResult, ...],
    risks: tuple[RiskResult, ...] = (RiskResult("risk-basic", True),),
    *,
    horizon: str = "long-term",
) -> DecisionRequest:
    return DecisionRequest(
        exchange="NSE",
        trading_symbol="RELIANCE",
        investment_horizon=horizon,
        evaluated_at=_NOW,
        strategy_results=strategies,
        risk_results=risks,
    )


def _engine(
    *strategy_ids: str, required_risk_rule_ids: tuple[str, ...] = ("risk-basic",)
) -> StrictConsensusDecisionEngine:
    return StrictConsensusDecisionEngine(
        required_strategy_ids=strategy_ids,
        minimum_confidence=Decimal("0.6"),
        required_risk_rule_ids=required_risk_rule_ids,
    )


@pytest.mark.parametrize("confidence", [Decimal("0"), Decimal("1")])
def test_confidence_boundary_values_are_valid(confidence: Decimal) -> None:
    result = _strategy_result("strategy-a", confidence=confidence)

    assert result.confidence == confidence


@pytest.mark.parametrize(
    "confidence",
    [Decimal("-0.01"), Decimal("1.01"), Decimal("NaN"), Decimal("Infinity")],
)
def test_out_of_range_or_non_finite_confidence_is_rejected(
    confidence: Decimal,
) -> None:
    with pytest.raises(DataValidationError, match="confidence"):
        _strategy_result("strategy-a", confidence=confidence)


def test_actionable_recommendation_requires_traceable_supporting_evidence() -> None:
    with pytest.raises(DataValidationError, match="traceable evidence"):
        _strategy_result("strategy-a", evidence=())


@pytest.mark.parametrize(
    "overrides",
    [
        {"strategy_id": " "},
        {"strategy_version": ""},
        {"trading_symbol": "RELIANCE NSE"},
        {"evaluated_at": datetime(2026, 10, 22, 12)},
        {"recommendation": "buy"},
        {"rationale": " "},
    ],
)
def test_strategy_contract_rejects_invalid_required_fields(
    overrides: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "strategy_id": "strategy-a",
        "strategy_version": "v1",
        "exchange": "NSE",
        "trading_symbol": "RELIANCE",
        "evaluated_at": _NOW,
        "investment_horizon": "long-term",
        "recommendation": StrategyRecommendation.BUY,
        "confidence": Decimal("0.8"),
        "rationale": "rationale",
        "supporting_evidence": (),
        "risk_flags": (),
        "invalidation_conditions": (),
        "data_quality": (),
        "input_data_references": (),
    }
    values.update(overrides)

    with pytest.raises(DataValidationError):
        StrategyResult(**values)  # type: ignore[arg-type]


def test_engine_returns_action_only_for_complete_unanimous_valid_results() -> None:
    result = _engine("strategy-a", "strategy-b").decide(
        _request((_strategy_result("strategy-b"), _strategy_result("strategy-a")))
    )

    assert result.recommendation is StrategyRecommendation.BUY
    assert result.findings == ()
    assert result.human_approval is HumanApproval.PENDING
    assert tuple(item.strategy_id for item in result.strategy_results) == (
        "strategy-a",
        "strategy-b",
    )


def test_conflicting_strategy_recommendations_abstain() -> None:
    result = _engine("strategy-a", "strategy-b").decide(
        _request(
            (
                _strategy_result(
                    "strategy-a", recommendation=StrategyRecommendation.BUY
                ),
                _strategy_result(
                    "strategy-b", recommendation=StrategyRecommendation.HOLD
                ),
            )
        )
    )

    assert result.recommendation is StrategyRecommendation.ABSTAIN
    assert DecisionFindingCode.CONFLICTING_RECOMMENDATIONS in {
        finding.code for finding in result.findings
    }


def test_missing_strategy_result_abstains_and_names_missing_strategy() -> None:
    result = _engine("strategy-a", "strategy-b").decide(
        _request((_strategy_result("strategy-a"),))
    )

    finding = next(
        finding
        for finding in result.findings
        if finding.code is DecisionFindingCode.MISSING_STRATEGY_RESULTS
    )
    assert result.recommendation is StrategyRecommendation.ABSTAIN
    assert finding.strategy_ids == ("strategy-b",)


def test_insufficient_confidence_abstains() -> None:
    result = _engine("strategy-a").decide(
        _request((_strategy_result("strategy-a", confidence=Decimal("0.59")),))
    )

    assert result.recommendation is StrategyRecommendation.ABSTAIN
    assert DecisionFindingCode.INSUFFICIENT_CONFIDENCE in {
        finding.code for finding in result.findings
    }


def test_strategy_abstention_is_preserved_as_an_aggregate_abstention() -> None:
    result = _engine("strategy-a").decide(
        _request(
            (
                _strategy_result(
                    "strategy-a", recommendation=StrategyRecommendation.ABSTAIN
                ),
            )
        )
    )

    assert result.recommendation is StrategyRecommendation.ABSTAIN
    assert result.human_approval is HumanApproval.NOT_REQUIRED
    assert DecisionFindingCode.STRATEGY_ABSTENTION in {
        finding.code for finding in result.findings
    }


@pytest.mark.parametrize(
    "quality_status",
    [
        DataQualityStatus.MISSING,
        DataQualityStatus.STALE,
        DataQualityStatus.INVALID,
        DataQualityStatus.UNKNOWN,
    ],
)
def test_non_valid_input_quality_abstains(quality_status: DataQualityStatus) -> None:
    result = _engine("strategy-a").decide(
        _request(
            (
                _strategy_result(
                    "strategy-a",
                    quality=(DataQualityIndicator("fundamentals", quality_status),),
                ),
            )
        )
    )

    assert result.recommendation is StrategyRecommendation.ABSTAIN
    assert DecisionFindingCode.DATA_QUALITY_BLOCK in {
        finding.code for finding in result.findings
    }


def test_missing_data_quality_and_missing_risk_results_abstain() -> None:
    result = _engine("strategy-a").decide(
        _request((_strategy_result("strategy-a", quality=()),), risks=())
    )

    finding_codes = {finding.code for finding in result.findings}
    assert result.recommendation is StrategyRecommendation.ABSTAIN
    assert DecisionFindingCode.DATA_QUALITY_UNAVAILABLE in finding_codes
    assert DecisionFindingCode.MISSING_RISK_RESULTS in finding_codes


def test_risk_veto_cannot_be_overridden_by_high_strategy_confidence() -> None:
    result = _engine("strategy-a").decide(
        _request(
            (_strategy_result("strategy-a", confidence=Decimal("1")),),
            risks=(RiskResult("risk-basic", False, ("limit exceeded",)),),
        )
    )

    assert result.recommendation is StrategyRecommendation.ABSTAIN
    assert DecisionFindingCode.RISK_VETO in {
        finding.code for finding in result.findings
    }


def test_horizon_mismatch_abstains_instead_of_combining_outputs() -> None:
    result = _engine("strategy-a").decide(
        _request((_strategy_result("strategy-a", horizon="short-term"),))
    )

    assert result.recommendation is StrategyRecommendation.ABSTAIN
    assert DecisionFindingCode.INCOMPATIBLE_HORIZONS in {
        finding.code for finding in result.findings
    }


def test_engine_rejects_unconfigured_strategies_and_wrong_instruments() -> None:
    with pytest.raises(DataValidationError, match="unconfigured strategy"):
        _engine("strategy-a").decide(_request((_strategy_result("strategy-b"),)))

    wrong_instrument = _strategy_result("strategy-a")
    object.__setattr__(wrong_instrument, "trading_symbol", "TCS")
    with pytest.raises(DataValidationError, match="instrument"):
        _engine("strategy-a").decide(_request((wrong_instrument,)))


def test_decision_engine_configuration_requires_explicit_valid_inputs() -> None:
    with pytest.raises(DataValidationError, match="strategy identifiers"):
        StrictConsensusDecisionEngine((), Decimal("0.5"))
    with pytest.raises(DataValidationError, match="confidence"):
        StrictConsensusDecisionEngine(("strategy-a",), Decimal("1.1"))


def test_aggregation_is_deterministic_across_input_order() -> None:
    engine = _engine(
        "strategy-a", "strategy-b", required_risk_rule_ids=("risk-a", "risk-basic")
    )
    strategy_a = _strategy_result("strategy-a")
    strategy_b = _strategy_result("strategy-b")
    risk_a = RiskResult("risk-a", True)
    risk_b = RiskResult("risk-basic", True)

    first = engine.decide(_request((strategy_a, strategy_b), (risk_a, risk_b)))
    second = engine.decide(_request((strategy_b, strategy_a), (risk_b, risk_a)))

    assert first == second
