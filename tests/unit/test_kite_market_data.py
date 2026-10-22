"""Offline tests for the Kite Connect daily market-data adapter."""

from __future__ import annotations

import asyncio
import gzip
import json
from datetime import date
from io import BytesIO
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest

from multi_agent_investment_system.core.configuration import KiteConnectCredentials
from multi_agent_investment_system.core.exceptions import MarketDataProviderError
from multi_agent_investment_system.data.service import DailyMarketDataExtractionService
from multi_agent_investment_system.tools import kite_market_data
from multi_agent_investment_system.tools.kite_authentication import KiteAccessToken
from multi_agent_investment_system.tools.kite_market_data import (
    KiteConnectMarketDataProvider,
)

_API_KEY = "dummy-api-key"
_ACCESS_TOKEN = "dummy-access-token"
_NSE_CSV = (
    b"instrument_token,tradingsymbol,instrument_type,exchange\n"
    b"111,ACME,FUT,NSE\n"
    b"222,ACME,EQ,BSE\n"
    b"333,ACME,EQ,NSE\n"
)


class FakeResponse(BytesIO):
    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _provider() -> KiteConnectMarketDataProvider:
    return KiteConnectMarketDataProvider(
        KiteConnectCredentials(api_key=_API_KEY, api_secret="dummy-api-secret"),
        KiteAccessToken(_ACCESS_TOKEN),
    )


def _mock_urlopen(monkeypatch: pytest.MonkeyPatch, candles: list[object]):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        if urlparse(request.full_url).path == "/instruments/NSE":
            return FakeResponse(gzip.compress(_NSE_CSV))
        return FakeResponse(
            json.dumps(
                {"status": "success", "data": {"candles": candles}}
            ).encode()
        )

    monkeypatch.setattr(kite_market_data, "urlopen", fake_urlopen)
    return requests


def test_resolves_exact_nse_equity_and_maps_canonical_daily_candle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = _mock_urlopen(
        monkeypatch,
        [["2026-01-02T00:00:00+05:30", 100, 110, 95, 105, 1200]],
    )

    records = asyncio.run(
        _provider().get_daily_candles(
            "ACME", date(2026, 1, 2), date(2026, 1, 2)
        )
    )

    historical_request = requests[1][0]
    assert "/historical/333/day?" in historical_request.full_url
    assert historical_request.get_header("Authorization") == (
        f"token {_API_KEY}:{_ACCESS_TOKEN}"
    )
    assert set(records[0]) >= {
        "symbol",
        "observed_at",
        "source",
        "instrument_token",
        "retrieved_at",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    assert records[0]["symbol"] == "ACME"
    assert records[0]["instrument_token"] == 333
    assert records[0]["source"] == "kite_connect"
    assert records[0]["observed_at"].utcoffset().total_seconds() == 5.5 * 3600
    assert [records[0][key] for key in ("open", "high", "low", "close", "volume")] == [
        100,
        110,
        95,
        105,
        1200,
    ]
    assert _ACCESS_TOKEN not in repr(_provider())


def test_date_bounds_are_inclusive_and_out_of_range_candles_are_filtered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = _mock_urlopen(
        monkeypatch,
        [
            ["2026-01-01T00:00:00+05:30", 1, 2, 1, 2, 1],
            ["2026-01-02T00:00:00+05:30", 2, 3, 2, 3, 1],
            ["2026-01-03T00:00:00+05:30", 3, 4, 3, 4, 1],
        ],
    )

    records = asyncio.run(
        _provider().get_daily_candles(
            "ACME", date(2026, 1, 1), date(2026, 1, 2)
        )
    )

    query = parse_qs(urlparse(requests[1][0].full_url).query)
    assert query["from"] == ["2026-01-01 00:00:00"]
    assert query["to"] == ["2026-01-02 23:59:59"]
    assert [record["observed_at"].date() for record in records] == [
        date(2026, 1, 1),
        date(2026, 1, 2),
    ]


def test_empty_candles_are_a_successful_empty_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_urlopen(monkeypatch, [])

    records = asyncio.run(
        _provider().get_daily_candles(
            "ACME", date(2026, 1, 1), date(2026, 1, 2)
        )
    )

    assert records == ()


def test_missing_candle_fields_reach_shared_pipeline_as_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_urlopen(monkeypatch, [["2026-01-02T00:00:00+05:30", 100, 110, 95]])
    service = DailyMarketDataExtractionService(_provider())

    result = asyncio.run(
        service.extract("ACME", date(2026, 1, 2), date(2026, 1, 2))
    )

    assert len(result.incomplete_records) == 1
    assert result.incomplete_records[0].record["close"] is None
    assert result.incomplete_records[0].record["volume"] is None


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json",
        json.dumps({"status": "error", "data": {"candles": []}}).encode(),
        json.dumps({"status": "success", "data": {"candles": ["bad"]}}).encode(),
    ],
)
def test_malformed_historical_responses_raise_provider_error(
    monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> None:
    def fake_urlopen(request, timeout):
        if urlparse(request.full_url).path == "/instruments/NSE":
            return FakeResponse(gzip.compress(_NSE_CSV))
        return FakeResponse(payload)

    monkeypatch.setattr(kite_market_data, "urlopen", fake_urlopen)

    with pytest.raises(MarketDataProviderError, match="response"):
        asyncio.run(
            _provider().get_daily_candles(
                "ACME", date(2026, 1, 1), date(2026, 1, 2)
            )
        )


def test_unknown_symbol_does_not_guess_a_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = _mock_urlopen(monkeypatch, [])

    with pytest.raises(MarketDataProviderError, match="not found"):
        asyncio.run(
            _provider().get_daily_candles(
                "UNKNOWN", date(2026, 1, 1), date(2026, 1, 2)
            )
        )

    assert len(requests) == 1


def test_api_failure_is_sanitized_and_not_converted_to_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(request, timeout):
        raise HTTPError(
            request.full_url,
            403,
            f"failure {_ACCESS_TOKEN}",
            hdrs=None,
            fp=BytesIO(f"failure {_ACCESS_TOKEN}".encode()),
        )

    monkeypatch.setattr(kite_market_data, "urlopen", fail)

    with pytest.raises(MarketDataProviderError) as raised:
        asyncio.run(
            _provider().get_daily_candles(
                "ACME", date(2026, 1, 1), date(2026, 1, 2)
            )
        )

    assert str(raised.value) == "Kite market-data request failed with HTTP status 403"
    assert _ACCESS_TOKEN not in str(raised.value)
    assert raised.value.__cause__ is None
