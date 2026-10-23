"""Offline tests for foundational data and command-line behavior."""

import logging
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from multi_agent_investment_system.core.exceptions import (
    DataValidationError,
    MarketDataErrorCategory,
    MarketDataProviderDiagnostic,
)
from multi_agent_investment_system.data.schemas import MarketObservation
from multi_agent_investment_system.main import main


def test_market_observation_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(DataValidationError, match="timezone-aware"):
        MarketObservation(
            symbol="TEST",
            observed_at=datetime(2026, 1, 1),
            close=Decimal("10.25"),
            currency="USD",
        )


def test_market_observation_accepts_aware_timestamp() -> None:
    observation = MarketObservation(
        symbol="TEST",
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
        close=Decimal("10.25"),
        currency="USD",
    )

    assert observation.close == Decimal("10.25")


def test_market_data_diagnostic_uses_fixed_category_allowlist() -> None:
    diagnostic = MarketDataProviderDiagnostic("untrusted_category")

    assert diagnostic.category is MarketDataErrorCategory.UNEXPECTED_EXTRACTION_ERROR
    assert diagnostic.to_dict() == {"category": "unexpected_extraction_error"}


@pytest.mark.parametrize("status", [True, 99, 600, "503", None])
def test_market_data_diagnostic_omits_invalid_http_status(status) -> None:
    diagnostic = MarketDataProviderDiagnostic(
        MarketDataErrorCategory.HTTP_ERROR, status
    )

    assert diagnostic.to_dict() == {"category": "http_error"}


def test_cli_reports_initialized_without_external_services(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert main() == 0

    assert "initialized" in caplog.text
