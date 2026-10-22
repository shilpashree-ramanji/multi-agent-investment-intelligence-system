"""Focused tests for provider-neutral daily market-data validation."""

import asyncio
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from types import ModuleType

import pytest

from multi_agent_investment_system.core.exceptions import DataValidationError
from multi_agent_investment_system.data.extraction import (
    normalize_and_validate_daily_candles,
)
from multi_agent_investment_system.data.schemas import (
    CandleIssueCode,
    validate_daily_ohlcv,
)
from multi_agent_investment_system.data.service import (
    DailyMarketDataExtractionService,
)
from multi_agent_investment_system.tools.market_data import MarketDataProvider
from multi_agent_investment_system.tools.yahoo_finance import (
    YahooFinanceMarketDataProvider,
)


def _valid_candle() -> dict[str, object]:
    return {
        "symbol": "RELIANCE.NS",
        "observed_at": datetime(2026, 10, 20, tzinfo=UTC),
        "open": Decimal("100.00"),
        "high": Decimal("105.00"),
        "low": Decimal("99.00"),
        "close": Decimal("103.00"),
        "volume": 1000,
    }


class FakeMarketDataProvider:
    def __init__(
        self,
        records: Sequence[Mapping[str, object]] = (),
        error: Exception | None = None,
    ) -> None:
        self.records = records
        self.error = error
        self.calls: list[tuple[str, date, date]] = []

    async def get_daily_candles(
        self, symbol: str, start: date, end: date
    ) -> Sequence[Mapping[str, object]]:
        self.calls.append((symbol, start, end))
        if self.error is not None:
            raise self.error
        return self.records


def test_valid_daily_candle_preserves_existing_context() -> None:
    candle = _valid_candle()

    result = validate_daily_ohlcv(candle)

    assert result.is_complete
    assert result.is_valid
    assert result.issues == ()
    assert result.symbol == candle["symbol"]
    assert result.observed_at == candle["observed_at"]


def test_missing_close_marks_candle_incomplete_without_imputation() -> None:
    candle = _valid_candle()
    del candle["close"]

    result = validate_daily_ohlcv(candle)

    assert not result.is_complete
    assert not result.is_valid
    assert [(issue.code, issue.field) for issue in result.issues] == [
        (CandleIssueCode.MISSING_FIELD, "close")
    ]
    assert "close" not in candle


def test_null_required_value_marks_candle_incomplete() -> None:
    candle = _valid_candle()
    candle["close"] = None

    result = validate_daily_ohlcv(candle)

    assert not result.is_complete
    assert not result.is_valid
    assert [(issue.code, issue.field) for issue in result.issues] == [
        (CandleIssueCode.MISSING_VALUE, "close")
    ]


def test_nan_required_value_marks_candle_incomplete_and_non_finite() -> None:
    candle = _valid_candle()
    candle["close"] = float("nan")

    result = validate_daily_ohlcv(candle)

    assert not result.is_complete
    assert not result.is_valid
    assert [(issue.code, issue.field) for issue in result.issues] == [
        (CandleIssueCode.MISSING_VALUE, "close"),
        (CandleIssueCode.NON_FINITE, "close"),
    ]


def test_non_numeric_non_finite_and_nonpositive_prices_are_reported() -> None:
    candle = _valid_candle()
    candle.update({"open": "not-a-number", "high": float("inf"), "low": 0})

    result = validate_daily_ohlcv(candle)

    assert result.is_complete
    assert not result.is_valid
    assert [(issue.code, issue.field) for issue in result.issues] == [
        (CandleIssueCode.NON_NUMERIC, "open"),
        (CandleIssueCode.NON_FINITE, "high"),
        (CandleIssueCode.NON_POSITIVE_PRICE, "low"),
    ]


def test_negative_volume_is_reported() -> None:
    candle = _valid_candle()
    candle["volume"] = -1

    result = validate_daily_ohlcv(candle)

    assert result.is_complete
    assert not result.is_valid
    assert [(issue.code, issue.field) for issue in result.issues] == [
        (CandleIssueCode.NEGATIVE_VOLUME, "volume")
    ]


def test_inconsistent_ohlc_relationships_are_reported() -> None:
    candle = _valid_candle()
    candle.update({"open": 10, "high": 9, "low": 11, "close": 12})

    result = validate_daily_ohlcv(candle)

    assert result.is_complete
    assert not result.is_valid
    assert [issue.code for issue in result.issues] == [
        CandleIssueCode.HIGH_BELOW_LOW,
        CandleIssueCode.HIGH_BELOW_OPEN_OR_CLOSE,
        CandleIssueCode.LOW_ABOVE_OPEN_OR_CLOSE,
    ]


def test_extraction_normalizes_numbers_and_preserves_available_provenance() -> None:
    candle = _valid_candle()
    candle.update(
        {
            "open": 100.25,
            "high": "105",
            "low": 99,
            "close": "103.5",
            "volume": "1000",
            "source": "test-source",
            "retrieved_at": datetime(2026, 10, 21, tzinfo=UTC),
        }
    )

    result = normalize_and_validate_daily_candles([candle])

    assert len(result.valid_records) == 1
    assert result.incomplete_records == ()
    assert result.invalid_records == ()
    classified = result.valid_records[0]
    assert classified.record["open"] == Decimal("100.25")
    assert classified.record["high"] == Decimal("105")
    assert classified.record["close"] == Decimal("103.5")
    assert classified.record["volume"] == Decimal("1000")
    assert classified.record["symbol"] == "RELIANCE.NS"
    assert classified.record["observed_at"] == candle["observed_at"]
    assert classified.record["source"] == "test-source"
    assert classified.record["retrieved_at"] == candle["retrieved_at"]


def test_extraction_separates_valid_incomplete_and_invalid_records() -> None:
    valid = _valid_candle()
    incomplete = _valid_candle()
    del incomplete["close"]
    incomplete["volume"] = -1
    invalid = _valid_candle()
    invalid["volume"] = -1

    result = normalize_and_validate_daily_candles([valid, incomplete, invalid])

    assert len(result.valid_records) == 1
    assert len(result.incomplete_records) == 1
    assert len(result.invalid_records) == 1
    assert result.incomplete_records[0].validation.is_complete is False
    assert [issue.code for issue in result.incomplete_records[0].validation.issues] == [
        CandleIssueCode.MISSING_FIELD,
        CandleIssueCode.NEGATIVE_VOLUME,
    ]
    assert [issue.code for issue in result.invalid_records[0].validation.issues] == [
        CandleIssueCode.NEGATIVE_VOLUME
    ]
    assert "close" not in result.incomplete_records[0].record


def test_malformed_records_are_classified_and_do_not_stop_the_batch() -> None:
    missing = _valid_candle()
    missing["close"] = None
    nan_value = _valid_candle()
    nan_value["close"] = float("nan")
    infinity = _valid_candle()
    infinity["close"] = "Infinity"
    boolean = _valid_candle()
    boolean["open"] = True
    malformed_string = _valid_candle()
    malformed_string["close"] = "not-a-number"
    valid_after_errors = _valid_candle()

    result = normalize_and_validate_daily_candles(
        [missing, nan_value, infinity, boolean, malformed_string, valid_after_errors]
    )

    assert len(result.valid_records) == 1
    assert len(result.incomplete_records) == 2
    assert len(result.invalid_records) == 3
    assert [
        [issue.code for issue in item.validation.issues]
        for item in result.invalid_records
    ] == [
        [CandleIssueCode.NON_FINITE],
        [CandleIssueCode.NON_NUMERIC],
        [CandleIssueCode.NON_NUMERIC],
    ]


def test_provider_contract_output_flows_into_candle_preparation() -> None:
    class FakeMarketDataProvider:
        async def get_daily_candles(
            self, symbol: str, start: date, end: date
        ) -> Sequence[Mapping[str, object]]:
            assert symbol == "RELIANCE.NS"
            observation_date = date(2026, 10, 20)
            assert start <= observation_date <= end
            return (
                {
                    "symbol": symbol,
                    "observed_at": datetime(2026, 10, 20, tzinfo=UTC),
                    "source": "fake-provider",
                    "retrieved_at": datetime(2026, 10, 21, tzinfo=UTC),
                    "open": 100,
                    "high": 105,
                    "low": 99,
                    "close": 103,
                    "volume": 1000,
                },
            )

    provider: MarketDataProvider = FakeMarketDataProvider()
    records = asyncio.run(
        provider.get_daily_candles(
            "RELIANCE.NS",
            date(2026, 10, 20),
            date(2026, 10, 20),
        )
    )
    result = normalize_and_validate_daily_candles(records)

    assert len(result.valid_records) == 1
    assert result.incomplete_records == ()
    assert result.invalid_records == ()
    assert result.valid_records[0].record["source"] == "fake-provider"
    assert "retrieved_at" in result.valid_records[0].record


def test_provider_success_with_no_matching_candles_returns_empty_batch() -> None:
    class EmptyMarketDataProvider:
        async def get_daily_candles(
            self, symbol: str, start: date, end: date
        ) -> Sequence[Mapping[str, object]]:
            return ()

    provider: MarketDataProvider = EmptyMarketDataProvider()
    records = asyncio.run(
        provider.get_daily_candles(
            "RELIANCE.NS", date(2026, 10, 20), date(2026, 10, 20)
        )
    )

    result = normalize_and_validate_daily_candles(records)

    assert records == ()
    assert result.valid_records == ()
    assert result.incomplete_records == ()
    assert result.invalid_records == ()


def test_provider_retrieval_failure_is_not_converted_to_empty_result() -> None:
    class FailingMarketDataProvider:
        async def get_daily_candles(
            self, symbol: str, start: date, end: date
        ) -> Sequence[Mapping[str, object]]:
            raise RuntimeError("provider unavailable")

    provider: MarketDataProvider = FailingMarketDataProvider()

    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(
            provider.get_daily_candles(
                "RELIANCE.NS", date(2026, 10, 20), date(2026, 10, 20)
            )
        )


def test_extraction_service_returns_valid_records_and_preserves_metadata() -> None:
    candle = _valid_candle()
    candle.update(
        {
            "source": "test-provider",
            "retrieved_at": datetime(2026, 10, 21, tzinfo=UTC),
        }
    )
    provider = FakeMarketDataProvider([candle])
    service = DailyMarketDataExtractionService(provider)
    start = date(2026, 10, 20)
    end = date(2026, 10, 20)

    result = asyncio.run(service.extract("RELIANCE.NS", start, end))

    assert provider.calls == [("RELIANCE.NS", start, end)]
    assert len(result.valid_records) == 1
    assert result.incomplete_records == ()
    assert result.invalid_records == ()
    record = result.valid_records[0].record
    assert record["symbol"] == "RELIANCE.NS"
    assert record["observed_at"] == candle["observed_at"]
    assert record["source"] == "test-provider"
    assert record["retrieved_at"] == candle["retrieved_at"]


def test_extraction_service_treats_empty_retrieval_as_success() -> None:
    provider = FakeMarketDataProvider()
    service = DailyMarketDataExtractionService(provider)

    result = asyncio.run(
        service.extract(
            "RELIANCE.NS", date(2026, 10, 20), date(2026, 10, 20)
        )
    )

    assert len(provider.calls) == 1
    assert result.valid_records == ()
    assert result.incomplete_records == ()
    assert result.invalid_records == ()


def test_extraction_service_returns_incomplete_candles_without_dropping_them() -> None:
    candle = _valid_candle()
    del candle["close"]
    provider = FakeMarketDataProvider([candle])
    service = DailyMarketDataExtractionService(provider)

    result = asyncio.run(
        service.extract(
            "RELIANCE.NS", date(2026, 10, 20), date(2026, 10, 20)
        )
    )

    assert result.valid_records == ()
    assert len(result.incomplete_records) == 1
    assert result.invalid_records == ()
    assert "close" not in result.incomplete_records[0].record


def test_extraction_service_returns_invalid_candles_without_dropping_them() -> None:
    candle = _valid_candle()
    candle["volume"] = -1
    provider = FakeMarketDataProvider([candle])
    service = DailyMarketDataExtractionService(provider)

    result = asyncio.run(
        service.extract(
            "RELIANCE.NS", date(2026, 10, 20), date(2026, 10, 20)
        )
    )

    assert result.valid_records == ()
    assert result.incomplete_records == ()
    assert len(result.invalid_records) == 1
    assert result.invalid_records[0].record["volume"] == Decimal("-1")


def test_extraction_service_propagates_provider_failure() -> None:
    failure = RuntimeError("provider unavailable")
    provider = FakeMarketDataProvider(error=failure)
    service = DailyMarketDataExtractionService(provider)

    with pytest.raises(RuntimeError, match="provider unavailable") as raised:
        asyncio.run(
            service.extract(
                "RELIANCE.NS", date(2026, 10, 20), date(2026, 10, 20)
            )
        )

    assert raised.value is failure


@pytest.mark.parametrize(
    ("symbol", "start", "end", "message"),
    [
        ("", date(2026, 10, 20), date(2026, 10, 21), "symbol"),
        ("   ", date(2026, 10, 20), date(2026, 10, 21), "symbol"),
        ("RELIANCE.NS", date(2026, 10, 21), date(2026, 10, 20), "after end"),
        (
            "RELIANCE.NS",
            datetime(2026, 10, 20, tzinfo=UTC),
            date(2026, 10, 21),
            "start must be a date",
        ),
    ],
)
def test_extraction_service_rejects_invalid_request_arguments(
    symbol: str, start: date, end: date, message: str
) -> None:
    provider = FakeMarketDataProvider()
    service = DailyMarketDataExtractionService(provider)

    with pytest.raises(DataValidationError, match=message):
        asyncio.run(service.extract(symbol, start, end))

    assert provider.calls == []


class FakeYahooHistory:
    def __init__(
        self,
        rows: Sequence[tuple[datetime, Mapping[str, object]]],
        columns: Sequence[str],
    ) -> None:
        self._rows = rows
        self.columns = columns
        self.empty = not rows

    def iterrows(self):
        return iter(self._rows)


def _install_fake_yfinance(
    monkeypatch: pytest.MonkeyPatch,
    history: FakeYahooHistory | None = None,
    error: Exception | None = None,
) -> list[tuple[str, dict[str, object]]]:
    calls: list[tuple[str, dict[str, object]]] = []
    module = ModuleType("yfinance")

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, **kwargs: object) -> FakeYahooHistory:
            calls.append((self.symbol, kwargs))
            if error is not None:
                raise error
            assert history is not None
            return history

    module.Ticker = FakeTicker
    monkeypatch.setitem(sys.modules, "yfinance", module)
    return calls


def test_yahoo_provider_maps_fields_and_requests_inclusive_end_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_at = datetime(2026, 10, 20, tzinfo=UTC)
    frame = FakeYahooHistory(
        [
            (
                observed_at,
                {"Open": 100, "High": 105, "Low": 99, "Close": 103, "Volume": 1000},
            )
        ],
        ["Open", "High", "Low", "Close", "Volume"],
    )
    calls = _install_fake_yfinance(monkeypatch, history=frame)
    provider = YahooFinanceMarketDataProvider()

    records = asyncio.run(
        provider.get_daily_candles(
            "RELIANCE.NS", date(2026, 10, 1), date(2026, 10, 20)
        )
    )

    assert calls == [
        (
            "RELIANCE.NS",
            {
                "start": "2026-10-01",
                "end": "2026-10-21",
                "interval": "1d",
                "auto_adjust": False,
                "actions": False,
                "raise_errors": True,
            },
        )
    ]
    assert records == (
        {
            "symbol": "RELIANCE.NS",
            "observed_at": observed_at,
            "open": 100,
            "high": 105,
            "low": 99,
            "close": 103,
            "volume": 1000,
        },
    )


def test_yahoo_provider_returns_empty_sequence_for_empty_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    columns = ["Open", "High", "Low", "Close", "Volume"]
    _install_fake_yfinance(
        monkeypatch, history=FakeYahooHistory([], columns)
    )
    provider = YahooFinanceMarketDataProvider()

    records = asyncio.run(
        provider.get_daily_candles(
            "RELIANCE.NS", date(2026, 10, 20), date(2026, 10, 20)
        )
    )

    assert records == ()


def test_yahoo_provider_preserves_missing_values_without_imputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = FakeYahooHistory(
        [
            (
                datetime(2026, 10, 20, tzinfo=UTC),
                {
                    "Open": 100,
                    "High": 105,
                    "Low": 99,
                    "Close": None,
                    "Volume": 1000,
                },
            )
        ],
        ["Open", "High", "Low", "Close", "Volume"],
    )
    _install_fake_yfinance(monkeypatch, history=frame)
    provider = YahooFinanceMarketDataProvider()

    records = asyncio.run(
        provider.get_daily_candles(
            "RELIANCE.NS", date(2026, 10, 20), date(2026, 10, 20)
        )
    )

    assert len(records) == 1
    assert records[0]["close"] is None


def test_yahoo_provider_omits_absent_fields_instead_of_fabricating_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = FakeYahooHistory(
        [
            (
                datetime(2026, 10, 20, tzinfo=UTC),
                {"Open": 100, "High": 105, "Low": 99, "Volume": 1000},
            )
        ],
        ["Open", "High", "Low", "Volume"],
    )
    _install_fake_yfinance(monkeypatch, history=frame)
    provider = YahooFinanceMarketDataProvider()

    records = asyncio.run(
        provider.get_daily_candles(
            "RELIANCE.NS", date(2026, 10, 20), date(2026, 10, 20)
        )
    )

    assert len(records) == 1
    assert "close" not in records[0]


def test_yahoo_provider_propagates_retrieval_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("Yahoo Finance request failed")
    calls = _install_fake_yfinance(monkeypatch, error=failure)
    provider = YahooFinanceMarketDataProvider()

    with pytest.raises(RuntimeError, match="Yahoo Finance request failed") as raised:
        asyncio.run(
            provider.get_daily_candles(
                "RELIANCE.NS", date(2026, 10, 20), date(2026, 10, 20)
            )
        )

    assert raised.value is failure
    assert len(calls) == 1
