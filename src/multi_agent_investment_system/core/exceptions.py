"""Application exception types for explicit error handling."""


class StockInvestmentError(Exception):
    """Base class for expected application errors."""


class ConfigurationError(StockInvestmentError):
    """Raised when application configuration is invalid."""


class AuthenticationError(StockInvestmentError):
    """Raised when an authentication step fails without exposing credentials."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "unknown_authentication_failure",
    ) -> None:
        super().__init__(message)
        self.category = category


class MarketDataProviderError(StockInvestmentError):
    """Raised when a market-data provider request or response fails."""


class StorageError(StockInvestmentError):
    """Raised when local persistence fails without exposing database details."""


class DataValidationError(StockInvestmentError):
    """Raised when supplied data fails structural or value validation."""


class RiskLimitViolation(StockInvestmentError):
    """Raised when deterministic risk controls reject a proposed action."""
