"""Validated, provenance-preserving ingestion for candidate-screen CSV files.

This module imports source-reported candidate data only. It does not apply
screening criteria or create investment recommendations.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path

from multi_agent_investment_system.core.exceptions import DataValidationError

_NUMERIC_COLUMNS = frozenset(
    {
        "source_rank",
        "current_price_inr",
        "pe_ratio",
        "market_cap_crore",
        "dividend_yield_pct",
        "net_profit_quarter_crore",
        "quarterly_profit_growth_pct",
        "quarterly_sales_crore",
        "quarterly_sales_growth_pct",
        "roce_pct",
        "sales_growth_5y_pct",
        "profit_growth_5y_pct",
        "debt_to_equity",
    }
)
_SCREEN_NUMERIC_FIELDS = {
    "sales_growth_5y_pct": "Five-year sales growth",
    "profit_growth_5y_pct": "Five-year profit growth",
    "roce_pct": "Return on capital employed",
    "debt_to_equity": "Debt-to-equity",
    "market_cap_crore": "Market capitalization",
}


class CandidateCsvIssueSeverity(StrEnum):
    """Whether an import finding needs correction or is informational."""

    ERROR = "error"
    WARNING = "warning"


class CandidateCsvIssueCode(StrEnum):
    """Stable codes for candidate CSV validation findings."""

    EMPTY_ROW = "empty_row"
    EXTRA_COLUMNS = "extra_columns"
    BLANK_COMPANY_NAME = "blank_company_name"
    INVALID_NUMERIC_VALUE = "invalid_numeric_value"
    INVALID_EXTRACTION_TIMESTAMP = "invalid_extraction_timestamp"
    DUPLICATE_RECORD = "duplicate_record"
    MALFORMED_CSV = "malformed_csv"
    MISSING_SCREEN_INPUT = "missing_screen_input"


@dataclass(frozen=True, slots=True)
class CandidateCsvIssue:
    """One actionable import finding, optionally tied to a source row."""

    code: CandidateCsvIssueCode
    severity: CandidateCsvIssueSeverity
    message: str
    row_number: int | None = None
    field: str | None = None


@dataclass(frozen=True, slots=True)
class ImportedCandidateCsvRow:
    """One input row retained with both original and parsed values."""

    source_file: str
    source_identifier: str | None
    row_number: int
    original_rank: str | None
    company_name: str | None
    extracted_at: datetime | None
    original_headers: tuple[str, ...]
    original_values: tuple[str, ...]
    values: Mapping[str, str]
    numeric_values: Mapping[str, Decimal | None]
    extra_values: tuple[str, ...]
    issues: tuple[CandidateCsvIssue, ...]


@dataclass(frozen=True, slots=True)
class CandidateCsvImportResult:
    """Complete import result; invalid rows remain available for review."""

    source_file: str
    source_identifier: str | None
    headers: tuple[str, ...]
    rows: tuple[ImportedCandidateCsvRow, ...]
    issues: tuple[CandidateCsvIssue, ...]
    screening_limitations: tuple[str, ...]

    @property
    def can_evaluate_all_screen_criteria(self) -> bool:
        """Whether the file has both debt/equity and reliable listing identity."""

        return bool(self.rows) and not self.screening_limitations and all(
            row.company_name is not None for row in self.rows
        )


def _normalize_header(header: str) -> str:
    return "_".join(header.strip().casefold().split())


def _parse_timestamp(raw_value: str | None) -> datetime | None:
    if raw_value is None or not raw_value.strip():
        return None
    try:
        value = datetime.fromisoformat(raw_value.strip())
    except ValueError:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return None
    return value


def ingest_candidate_csv(
    path: str | Path, *, source_identifier: str | None = None
) -> CandidateCsvImportResult:
    """Import a caller-supplied CSV without screening or dropping any row.

    ``company_name`` is the minimum required header. Other recognized fields
    are retained as source text and parsed as decimals when present. Blank
    numeric cells remain explicit ``None`` values. Validation findings are
    returned alongside every parsed row, including malformed or duplicate
    records. Extraction timestamps are parsed only when timezone-aware.
    """

    if source_identifier is not None and (
        not isinstance(source_identifier, str) or not source_identifier.strip()
    ):
        raise DataValidationError("Candidate CSV source identifier must not be empty")

    source_path = Path(path)
    file_source_identifier = (
        source_identifier.strip() if source_identifier is not None else None
    )
    all_issues: list[CandidateCsvIssue] = []
    rows: list[ImportedCandidateCsvRow] = []

    with source_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file, strict=True)
        raw_headers = next(reader, None)
        if raw_headers is None:
            raise DataValidationError("Candidate CSV is empty and has no header row")

        headers = tuple(header.strip() for header in raw_headers)
        normalized_headers = tuple(_normalize_header(header) for header in headers)
        if any(not header for header in normalized_headers):
            raise DataValidationError("Candidate CSV headers must not be blank")
        if len(set(normalized_headers)) != len(normalized_headers):
            raise DataValidationError("Candidate CSV headers must be unique")
        if "company_name" not in normalized_headers:
            raise DataValidationError(
                "Candidate CSV is missing required header: company_name"
            )

        column_indexes = {name: index for index, name in enumerate(normalized_headers)}
        normalized_header_names = set(normalized_headers)
        row_number = 2
        try:
            while True:
                row_start_line = reader.line_num + 1
                try:
                    raw_row = next(reader)
                except StopIteration:
                    break
                row_number = row_start_line
                row_issues: list[CandidateCsvIssue] = []

                if not raw_row:
                    issue = CandidateCsvIssue(
                        CandidateCsvIssueCode.EMPTY_ROW,
                        CandidateCsvIssueSeverity.ERROR,
                        "CSV row is blank and contains no candidate values.",
                        row_number=row_number,
                    )
                    all_issues.append(issue)
                    rows.append(
                        ImportedCandidateCsvRow(
                            source_file=source_path.name,
                            source_identifier=file_source_identifier,
                            row_number=row_number,
                            original_rank=None,
                            company_name=None,
                            extracted_at=None,
                            original_headers=headers,
                            original_values=(),
                            values={},
                            numeric_values={},
                            extra_values=(),
                            issues=(issue,),
                        )
                    )
                    continue

                if len(raw_row) > len(headers):
                    issue = CandidateCsvIssue(
                        CandidateCsvIssueCode.EXTRA_COLUMNS,
                        CandidateCsvIssueSeverity.ERROR,
                        (
                            f"Row has {len(raw_row) - len(headers)} value(s) "
                            "beyond the header."
                        ),
                        row_number=row_number,
                    )
                    row_issues.append(issue)
                    all_issues.append(issue)

                padded_row = [*raw_row[: len(headers)]]
                padded_row.extend([""] * (len(headers) - len(padded_row)))
                values = dict(zip(normalized_headers, padded_row, strict=True))
                company_name_value = values["company_name"].strip()
                company_name = company_name_value or None
                if company_name is None:
                    issue = CandidateCsvIssue(
                        CandidateCsvIssueCode.BLANK_COMPANY_NAME,
                        CandidateCsvIssueSeverity.ERROR,
                        (
                            "Company name is blank; provide the source-reported "
                            "company name."
                        ),
                        row_number=row_number,
                        field="company_name",
                    )
                    row_issues.append(issue)
                    all_issues.append(issue)

                numeric_values: dict[str, Decimal | None] = {}
                for field_name in sorted(_NUMERIC_COLUMNS & normalized_header_names):
                    raw_value = values[field_name]
                    if not raw_value.strip():
                        numeric_values[field_name] = None
                        continue
                    try:
                        number = Decimal(raw_value.strip())
                    except InvalidOperation:
                        number = Decimal("NaN")
                    if not number.is_finite():
                        issue = CandidateCsvIssue(
                            CandidateCsvIssueCode.INVALID_NUMERIC_VALUE,
                            CandidateCsvIssueSeverity.ERROR,
                            f"Value {raw_value!r} is not a finite numeric value.",
                            row_number=row_number,
                            field=field_name,
                        )
                        row_issues.append(issue)
                        all_issues.append(issue)
                        numeric_values[field_name] = None
                    else:
                        numeric_values[field_name] = number

                timestamp_field = next(
                    (
                        field_name
                        for field_name in ("extracted_at", "extraction_timestamp")
                        if field_name in values
                    ),
                    None,
                )
                raw_timestamp = values[timestamp_field] if timestamp_field else None
                extracted_at = _parse_timestamp(raw_timestamp)
                if raw_timestamp and raw_timestamp.strip() and extracted_at is None:
                    issue = CandidateCsvIssue(
                        CandidateCsvIssueCode.INVALID_EXTRACTION_TIMESTAMP,
                        CandidateCsvIssueSeverity.WARNING,
                        (
                            "Extraction timestamp is not timezone-aware ISO 8601; "
                            "raw value is retained."
                        ),
                        row_number=row_number,
                        field=timestamp_field,
                    )
                    row_issues.append(issue)
                    all_issues.append(issue)

                rank_index = column_indexes.get("source_rank")
                original_rank = (
                    raw_row[rank_index]
                    if rank_index is not None and rank_index < len(raw_row)
                    else None
                )
                row_source_identifier = file_source_identifier
                if row_source_identifier is None and values.get("source", "").strip():
                    row_source_identifier = values["source"].strip()

                rows.append(
                    ImportedCandidateCsvRow(
                        source_file=source_path.name,
                        source_identifier=row_source_identifier,
                        row_number=row_number,
                        original_rank=original_rank,
                        company_name=company_name,
                        extracted_at=extracted_at,
                        original_headers=headers,
                        original_values=tuple(raw_row),
                        values=values,
                        numeric_values=numeric_values,
                        extra_values=tuple(raw_row[len(headers) :]),
                        issues=tuple(row_issues),
                    )
                )
        except csv.Error as exc:
            issue = CandidateCsvIssue(
                CandidateCsvIssueCode.MALFORMED_CSV,
                CandidateCsvIssueSeverity.ERROR,
                (
                    f"Malformed CSV near source row "
                    f"{max(row_number, reader.line_num + 1)}: {exc}"
                ),
                row_number=max(row_number, reader.line_num + 1),
            )
            all_issues.append(issue)

    rows_by_identity: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.company_name is not None:
            duplicate_key = " ".join(row.company_name.casefold().split())
            rows_by_identity[duplicate_key].append(index)

    for indexes in rows_by_identity.values():
        if len(indexes) < 2:
            continue
        for index in indexes:
            row = rows[index]
            issue = CandidateCsvIssue(
                CandidateCsvIssueCode.DUPLICATE_RECORD,
                CandidateCsvIssueSeverity.WARNING,
                "Company name repeats in this file; rows are retained for review.",
                row_number=row.row_number,
                field="company_name",
            )
            all_issues.append(issue)
            rows[index] = ImportedCandidateCsvRow(
                source_file=row.source_file,
                source_identifier=row.source_identifier,
                row_number=row.row_number,
                original_rank=row.original_rank,
                company_name=row.company_name,
                extracted_at=row.extracted_at,
                original_headers=row.original_headers,
                original_values=row.original_values,
                values=row.values,
                numeric_values=row.numeric_values,
                extra_values=row.extra_values,
                issues=(*row.issues, issue),
            )

    limitations: list[str] = []
    for field_name, display_name in _SCREEN_NUMERIC_FIELDS.items():
        if field_name not in normalized_header_names:
            limitations.append(
                f"{display_name} data is absent, so this candidate screen "
                "criterion cannot be evaluated."
            )
        elif any(row.numeric_values.get(field_name) is None for row in rows):
            limitations.append(
                f"{display_name} is missing or invalid for one or more rows."
            )

    has_listing_identity_headers = {
        "exchange",
        "trading_symbol",
    }.issubset(normalized_header_names)
    if not has_listing_identity_headers:
        limitations.append(
            "Exchange and trading-symbol identifiers are absent; company names "
            "are not resolved to listings."
        )
    elif any(
        not row.values.get("exchange", "").strip()
        or not row.values.get("trading_symbol", "").strip()
        for row in rows
    ):
        limitations.append(
            "Exchange or trading-symbol identifiers are missing for one or more rows."
        )

    for message in limitations:
        all_issues.append(
            CandidateCsvIssue(
                CandidateCsvIssueCode.MISSING_SCREEN_INPUT,
                CandidateCsvIssueSeverity.WARNING,
                message,
            )
        )

    return CandidateCsvImportResult(
        source_file=source_path.name,
        source_identifier=file_source_identifier,
        headers=headers,
        rows=tuple(rows),
        issues=tuple(all_issues),
        screening_limitations=tuple(limitations),
    )
