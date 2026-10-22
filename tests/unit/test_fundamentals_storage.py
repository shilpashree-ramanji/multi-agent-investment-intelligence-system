"""Tests for append-only SQLite company-fundamentals storage."""

import sqlite3
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from multi_agent_investment_system.core.exceptions import (
    DataValidationError,
    StorageError,
)
from multi_agent_investment_system.data.fundamentals import (
    CompanyFundamentalsPeriod,
    FinancialStatementBasis,
    FundamentalMetric,
    FundamentalValueStatus,
)
from multi_agent_investment_system.data.fundamentals_storage import (
    SQLiteCompanyFundamentalsStore,
)


def _period(
    *,
    symbol: str = "RELIANCE",
    basis: FinancialStatementBasis = FinancialStatementBasis.CONSOLIDATED,
    source_record_id: str = "filing-1",
    revision_of: str | None = None,
    retrieved_at: datetime | None = None,
    revenue: object = Decimal("1250.00"),
    metrics: tuple[FundamentalMetric, ...] | None = None,
) -> CompanyFundamentalsPeriod:
    return CompanyFundamentalsPeriod(
        exchange="NSE",
        trading_symbol=symbol,
        source="synthetic-source",
        retrieved_at=retrieved_at or datetime(2026, 10, 22, 12, tzinfo=UTC),
        reporting_period="FY 2025-26",
        period_start=date(2025, 4, 1),
        period_end=date(2026, 3, 31),
        basis=basis,
        company_name="Synthetic Company",
        company_id="INE002A01018",
        company_id_type="ISIN",
        source_record_id=source_record_id,
        published_at=datetime(2026, 5, 1, 9, tzinfo=UTC),
        revision_of=revision_of,
        source_references=("synthetic-filing-reference",),
        source_metadata={"document": "synthetic", "page": 4},
        metrics=metrics
        if metrics is not None
        else (
            FundamentalMetric(
                name="Revenue",
                status=FundamentalValueStatus.REPORTED,
                value=revenue,
                currency="INR",
                unit="crore",
                provenance={"tag": "RevenueFromOperations", "scale": 7},
                source_references=("synthetic-page-4",),
            ),
        ),
    )


def test_store_initializes_fundamentals_table_and_is_idempotent(tmp_path) -> None:
    path = tmp_path / "nested" / "fundamentals.sqlite3"

    store = SQLiteCompanyFundamentalsStore(path)

    assert store.database_path == path
    assert store.count_records() == 0
    with sqlite3.connect(path) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    assert ("company_fundamentals",) in tables
    store.initialize()
    assert store.count_records() == 0


def test_round_trip_preserves_period_identity_metrics_and_provenance(tmp_path) -> None:
    record = _period(
        retrieved_at=datetime(
            2026, 10, 22, 17, tzinfo=timezone(timedelta(hours=5, minutes=30))
        ),
        metrics=(
            FundamentalMetric(
                name="Revenue",
                status=FundamentalValueStatus.REPORTED,
                value=Decimal("1250.00"),
                currency="INR",
                unit="crore",
                provenance={"taxonomy": "tag-1", "context": ("duration", 1)},
                source_references=("filing-page-4",),
            ),
            FundamentalMetric(
                name="Operating cash flow",
                status=FundamentalValueStatus.MISSING,
            ),
            FundamentalMetric(
                name="Free cash flow",
                status=FundamentalValueStatus.UNAVAILABLE,
            ),
        ),
    )
    store = SQLiteCompanyFundamentalsStore(tmp_path / "fundamentals.sqlite3")

    assert store.insert_validated((record,)) == 1
    [loaded] = store.fetch_company_periods("NSE", "RELIANCE", "FY 2025-26")

    assert loaded == record
    assert loaded.metrics[0].value == Decimal("1250.00")
    assert loaded.metrics[0].provenance == {
        "taxonomy": "tag-1",
        "context": ("duration", 1),
    }
    assert loaded.metrics[1].status is FundamentalValueStatus.MISSING
    assert loaded.metrics[2].status is FundamentalValueStatus.UNAVAILABLE
    assert loaded.published_at == record.published_at
    assert loaded.retrieved_at == record.retrieved_at


def test_identical_reingestion_is_ignored_without_updating_retrieval_time(
    tmp_path,
) -> None:
    first = _period()
    later_retrieval = _period(
        retrieved_at=datetime(2026, 10, 23, 12, tzinfo=UTC)
    )
    store = SQLiteCompanyFundamentalsStore(tmp_path / "fundamentals.sqlite3")

    assert store.insert_validated((first,)) == 1
    assert store.insert_validated((later_retrieval,)) == 0
    [stored] = store.fetch_company_periods("NSE", "RELIANCE")

    assert store.count_records() == 1
    assert stored.retrieved_at == first.retrieved_at


def test_distinct_revision_and_changed_same_id_content_remain_historical(
    tmp_path,
) -> None:
    original = _period()
    revision = _period(
        source_record_id="filing-2",
        revision_of="filing-1",
        revenue=Decimal("1300.00"),
    )
    corrected_same_id = _period(revenue=Decimal("1260.00"))
    store = SQLiteCompanyFundamentalsStore(tmp_path / "fundamentals.sqlite3")

    assert store.insert_validated((original, revision, corrected_same_id)) == 3
    history = store.fetch_company_periods("NSE", "RELIANCE", "FY 2025-26")

    assert len(history) == 3
    assert [item.metrics[0].value for item in history] == [
        Decimal("1250.00"),
        Decimal("1300.00"),
        Decimal("1260.00"),
    ]
    assert history[1].revision_of == "filing-1"
    assert history[2].source_record_id == "filing-1"


def test_standalone_and_consolidated_statements_are_separate(tmp_path) -> None:
    consolidated = _period()
    standalone = _period(basis=FinancialStatementBasis.STANDALONE)
    store = SQLiteCompanyFundamentalsStore(tmp_path / "fundamentals.sqlite3")

    assert store.insert_validated((consolidated, standalone)) == 2
    all_versions = store.fetch_company_periods("NSE", "RELIANCE", "FY 2025-26")
    standalone_only = store.fetch_company_periods(
        "NSE",
        "RELIANCE",
        "FY 2025-26",
        FinancialStatementBasis.STANDALONE,
    )

    assert [item.basis for item in all_versions] == [
        FinancialStatementBasis.CONSOLIDATED,
        FinancialStatementBasis.STANDALONE,
    ]
    assert standalone_only == (standalone,)


def test_records_are_revalidated_before_any_write(tmp_path) -> None:
    valid = _period()
    corrupted = _period(source_record_id="bad-filing")
    object.__setattr__(corrupted, "retrieved_at", datetime(2026, 10, 22, 12))
    store = SQLiteCompanyFundamentalsStore(tmp_path / "fundamentals.sqlite3")

    with pytest.raises(DataValidationError, match="timezone-aware"):
        store.insert_validated((valid, corrupted))

    assert store.count_records() == 0


def test_unsupported_metric_value_is_rejected_before_write(tmp_path) -> None:
    record = _period(revenue=object())
    store = SQLiteCompanyFundamentalsStore(tmp_path / "fundamentals.sqlite3")

    with pytest.raises(DataValidationError, match="unsupported by storage"):
        store.insert_validated((record,))

    assert store.count_records() == 0


def test_sqlite_write_failure_rolls_back_batch_and_is_sanitized(tmp_path) -> None:
    path = tmp_path / "fundamentals.sqlite3"
    store = SQLiteCompanyFundamentalsStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_fundamentals_write
            BEFORE INSERT ON company_fundamentals
            WHEN NEW.trading_symbol = 'TCS'
            BEGIN
                SELECT RAISE(ABORT, 'private database detail');
            END;
            """
        )

    with pytest.raises(StorageError) as exc_info:
        store.insert_validated((_period(), _period(symbol="TCS")))

    assert str(exc_info.value) == "Could not write local company fundamentals"
    assert "private database detail" not in str(exc_info.value)
    assert store.count_records() == 0


def test_sqlite_read_failure_is_sanitized(tmp_path) -> None:
    store = SQLiteCompanyFundamentalsStore(tmp_path / "fundamentals.sqlite3")
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("DROP TABLE company_fundamentals")

    with pytest.raises(StorageError) as exc_info:
        store.fetch_company_periods("NSE", "RELIANCE")

    assert str(exc_info.value) == "Could not read local company fundamentals"


def test_storage_open_failure_is_sanitized(tmp_path) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("synthetic")

    with pytest.raises(StorageError, match="Could not open local fundamentals storage"):
        SQLiteCompanyFundamentalsStore(blocker / "fundamentals.sqlite3")
