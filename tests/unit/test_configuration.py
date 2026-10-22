"""Tests for secret-safe environment configuration."""

from pathlib import Path

import pytest

from multi_agent_investment_system.core.configuration import (
    KiteConnectCredentials,
    MarketDataDatabaseSettings,
)
from multi_agent_investment_system.core.exceptions import ConfigurationError


def test_kite_credentials_are_optional_when_unconfigured() -> None:
    assert KiteConnectCredentials.from_environment({}) is None
    assert KiteConnectCredentials.from_environment(
        {"KITE_API_KEY": "", "KITE_API_SECRET": "   "}
    ) is None


def test_kite_credentials_load_as_a_pair_and_are_hidden_from_repr() -> None:
    values = {
        "KITE_API_KEY": "test-only-api-key",
        "KITE_API_SECRET": "test-only-api-secret",
    }

    credentials = KiteConnectCredentials.from_environment(values)

    assert credentials is not None
    assert credentials.api_key == values["KITE_API_KEY"]
    assert credentials.api_secret == values["KITE_API_SECRET"]
    assert values["KITE_API_KEY"] not in repr(credentials)
    assert values["KITE_API_SECRET"] not in repr(credentials)


@pytest.mark.parametrize(
    "values",
    [
        {"KITE_API_KEY": "test-only-api-key"},
        {"KITE_API_SECRET": "test-only-api-secret"},
        {"KITE_API_KEY": "test-only-api-key", "KITE_API_SECRET": " "},
    ],
)
def test_partial_kite_credentials_raise_without_exposing_values(
    values: dict[str, str],
) -> None:
    with pytest.raises(ConfigurationError) as error:
        KiteConnectCredentials.from_environment(values)

    assert "test-only-api-key" not in str(error.value)
    assert "test-only-api-secret" not in str(error.value)


def test_market_data_database_settings_use_default_and_environment_override() -> None:
    default_settings = MarketDataDatabaseSettings.from_environment({})
    custom_settings = MarketDataDatabaseSettings.from_environment(
        {"MARKET_DATA_DATABASE_PATH": "./test-data/prices.sqlite3"}
    )

    assert default_settings.path == Path(".local/market_data.sqlite3")
    assert custom_settings.path == Path("./test-data/prices.sqlite3")
