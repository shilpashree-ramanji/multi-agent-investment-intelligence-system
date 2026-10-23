"""Application exception types for explicit error handling."""

from dataclasses import dataclass
from enum import StrEnum


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


class MarketDataErrorCategory(StrEnum):
    """Fixed, non-sensitive categories for market-data provider failures."""

    HTTP_ERROR = "http_error"
    INSTRUMENT_RESOLUTION_ERROR = "instrument_resolution_error"
    INVALID_PROVIDER_RESPONSE = "invalid_provider_response"
    NETWORK_OR_TIMEOUT = "network_or_timeout"
    UNEXPECTED_EXTRACTION_ERROR = "unexpected_extraction_error"


@dataclass(frozen=True, slots=True)
class MarketDataProviderDiagnostic:
    """Allowlisted market-data failure facts safe to include in a run report."""

    category: MarketDataErrorCategory
    http_status: int | None = None

    def __post_init__(self) -> None:
        try:
            normalized_category = MarketDataErrorCategory(self.category)
        except (TypeError, ValueError):
            normalized_category = MarketDataErrorCategory.UNEXPECTED_EXTRACTION_ERROR
        object.__setattr__(self, "category", normalized_category)
        if (
            type(self.http_status) is not int
            or not 100 <= self.http_status <= 599
        ):
            object.__setattr__(self, "http_status", None)

    def to_dict(self) -> dict[str, str | int]:
        """Serialize only the fixed category and valid HTTP status, if any."""

        result: dict[str, str | int] = {"category": self.category.value}
        if self.http_status is not None:
            result["http_status"] = self.http_status
        return result


class MarketDataProviderError(StockInvestmentError):
    """Raised when a market-data provider request or response fails."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        category: MarketDataErrorCategory | str = (
            MarketDataErrorCategory.UNEXPECTED_EXTRACTION_ERROR
        ),
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable is True
        self.diagnostic = MarketDataProviderDiagnostic(category, http_status)


class CandidateUniverseProviderError(StockInvestmentError):
    """Raised when the authorized equity-universe provider fails."""


class FundamentalsProviderError(StockInvestmentError):
    """Raised when a company-fundamentals provider fails."""


class StorageError(StockInvestmentError):
    """Raised when local persistence fails without exposing database details."""


class DataValidationError(StockInvestmentError):
    """Raised when supplied data fails structural or value validation."""


class RiskLimitViolation(StockInvestmentError):
    """Raised when deterministic risk controls reject a proposed action."""
