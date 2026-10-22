"""Kite Connect adapter for provider-neutral daily market candles."""

from __future__ import annotations

import asyncio
import csv
import gzip
import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from io import StringIO
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from multi_agent_investment_system.core.configuration import KiteConnectCredentials
from multi_agent_investment_system.core.exceptions import (
    DataValidationError,
    MarketDataProviderError,
)
from multi_agent_investment_system.tools.kite_authentication import KiteAccessToken
from multi_agent_investment_system.tools.market_data import MarketDataProvider

_API_ROOT = "https://api.kite.trade"
_REQUIRED_INSTRUMENT_COLUMNS = {
    "exchange",
    "tradingsymbol",
    "instrument_type",
    "instrument_token",
}
_HISTORICAL_INTERVAL_SECONDS = 1 / 3


class KiteConnectMarketDataProvider(MarketDataProvider):
    """Retrieve NSE daily candles with exact instrument-master resolution.

    The provider caches the NSE instrument master in memory for its lifetime.
    All requests are paced conservatively to stay within the documented
    historical endpoint limit. No credentials or tokens are included in its
    representation, logs, or raised provider-error messages.
    """

    def __init__(
        self,
        credentials: KiteConnectCredentials,
        access_token: KiteAccessToken,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        self._credentials = credentials
        self._access_token = access_token
        self._timeout_seconds = timeout_seconds
        self._instrument_tokens: dict[str, int] | None = None
        self._instrument_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._last_request_at: float | None = None

    async def get_daily_candles(
        self, symbol: str, start: date, end: date
    ) -> Sequence[Mapping[str, object]]:
        """Get NSE daily candles within inclusive requested calendar dates."""

        if not isinstance(symbol, str) or not symbol.strip():
            raise DataValidationError("Kite NSE trading symbol must not be empty")
        if not isinstance(start, date) or isinstance(start, datetime):
            raise DataValidationError("Market-data start must be a date")
        if not isinstance(end, date) or isinstance(end, datetime):
            raise DataValidationError("Market-data end must be a date")
        if start > end:
            raise DataValidationError("Market-data start must not be after end")

        normalized_symbol = symbol.strip()
        instrument_token = await self._resolve_instrument_token(normalized_symbol)
        query = urlencode(
            {
                "from": f"{start.isoformat()} 00:00:00",
                "to": f"{end.isoformat()} 23:59:59",
                "oi": "0",
            }
        )
        url = (
            f"{_API_ROOT}/instruments/historical/{instrument_token}/day?{query}"
        )
        payload_bytes = await self._request_bytes(url)
        candles = self._decode_candles(payload_bytes)
        retrieved_at = datetime.now(UTC)
        records: list[Mapping[str, object]] = []

        for candle in candles:
            if not isinstance(candle, list) or len(candle) > 7:
                raise MarketDataProviderError(
                    "Kite historical data response contained a malformed candle"
                )

            record: dict[str, object] = {
                "symbol": normalized_symbol,
                "source": "kite_connect",
                "instrument_token": instrument_token,
                "retrieved_at": retrieved_at,
            }
            if candle and candle[0] is not None:
                try:
                    observed_at = datetime.fromisoformat(str(candle[0]))
                except ValueError:
                    raise MarketDataProviderError(
                        "Kite historical data response contained an invalid timestamp"
                    ) from None
                if observed_at.tzinfo is None or observed_at.utcoffset() is None:
                    raise MarketDataProviderError(
                        "Kite historical data response contained an invalid timestamp"
                    )
                if not start <= observed_at.date() <= end:
                    continue
                record["observed_at"] = observed_at

            for field, index in (
                ("open", 1),
                ("high", 2),
                ("low", 3),
                ("close", 4),
                ("volume", 5),
            ):
                record[field] = candle[index] if len(candle) > index else None
            if len(candle) == 7:
                record["open_interest"] = candle[6]
            records.append(record)

        return tuple(records)

    async def _resolve_instrument_token(self, symbol: str) -> int:
        async with self._instrument_lock:
            if self._instrument_tokens is None:
                self._instrument_tokens = await self._load_nse_instruments()

        token = self._instrument_tokens.get(symbol)
        if token is None:
            raise MarketDataProviderError(
                "NSE trading symbol was not found in the Kite instrument list"
            )
        return token

    async def _load_nse_instruments(self) -> dict[str, int]:
        payload = await self._request_bytes(f"{_API_ROOT}/instruments/NSE")
        try:
            if payload.startswith(b"\x1f\x8b"):
                payload = gzip.decompress(payload)
            rows = csv.DictReader(StringIO(payload.decode("utf-8")))
            headers = set(rows.fieldnames or ())
            if not _REQUIRED_INSTRUMENT_COLUMNS.issubset(headers):
                raise ValueError

            matches: dict[str, list[int]] = {}
            for row in rows:
                if row.get("exchange") != "NSE" or row.get("instrument_type") != "EQ":
                    continue
                trading_symbol = row.get("tradingsymbol", "")
                token_text = row.get("instrument_token", "")
                if not trading_symbol or not token_text.isdigit():
                    continue
                matches.setdefault(trading_symbol, []).append(int(token_text))
        except (UnicodeDecodeError, csv.Error, OSError, ValueError):
            raise MarketDataProviderError(
                "Kite NSE instrument list response was malformed"
            ) from None

        if any(len(tokens) != 1 for tokens in matches.values()):
            raise MarketDataProviderError(
                "Kite NSE instrument list contained ambiguous equity symbols"
            )
        return {symbol: tokens[0] for symbol, tokens in matches.items()}

    @staticmethod
    def _decode_candles(payload: bytes) -> list[object]:
        try:
            response = json.loads(payload.decode("utf-8"))
            candles = response["data"]["candles"]
            if response.get("status") != "success" or not isinstance(candles, list):
                raise ValueError
            return candles
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            raise MarketDataProviderError(
                "Kite historical data response was malformed or unsuccessful"
            ) from None

    async def _request_bytes(self, url: str) -> bytes:
        async with self._request_lock:
            if self._last_request_at is not None:
                elapsed = time.monotonic() - self._last_request_at
                wait = _HISTORICAL_INTERVAL_SECONDS - elapsed
                if wait > 0:
                    await asyncio.sleep(wait)

            request = Request(
                url,
                headers={
                    "Authorization": (
                        "token "
                        f"{self._credentials.api_key}:{self._access_token.value}"
                    ),
                    "X-Kite-Version": "3",
                },
                method="GET",
            )
            try:
                return await asyncio.to_thread(self._read_response, request)
            except HTTPError as error:
                raise MarketDataProviderError(
                    f"Kite market-data request failed with HTTP status {error.code}"
                ) from None
            except Exception:
                raise MarketDataProviderError(
                    "Kite market-data request failed"
                ) from None
            finally:
                self._last_request_at = time.monotonic()

    def _read_response(self, request: Request) -> bytes:
        with urlopen(request, timeout=self._timeout_seconds) as response:
            return response.read()
