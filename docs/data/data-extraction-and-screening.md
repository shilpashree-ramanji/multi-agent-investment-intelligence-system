# Data Extraction and Stock Screening Approach

## Purpose and status

This document records a proposed, provider-neutral approach for discovering stock candidates and acquiring data for investment research. The sources and sequence below are options for discussion, not approved integrations or configured requirements. The initial implementation should remain small, testable, traceable, and read-only.

Stock screening generates candidates for further research. A screening result is not a final investment decision or recommendation. Candidate selection does not replace independent data validation, strategy analysis, risk controls, evidence review, or human approval.

The four investment strategies and their data requirements are not defined here. Their specifications must be available and confirmed before strategy-specific data requirements or analysis are implemented.

## Proposed data pipeline

1. **Screen and generate candidates.** Apply explicitly configured screening conditions to a defined market universe. Screening may use fundamental data, technical data, or both, once the relevant methods are selected.
2. **Create a candidate shortlist.** Retain each candidate with the screening source, conditions or criteria applied, screening result, and screening timestamp. Preserve enough information to reproduce or explain why it entered the shortlist.
3. **Validate candidates independently.** Check shortlisted securities and relevant observations against appropriate exchange sources or authorized data providers. A screening source alone should not be treated as independent verification.
4. **Validate, normalize, and record provenance.** Check data quality, normalize representations and units where requirements are known, and retain source, time, adjustment, and validation details.
5. **Prepare strategy-specific analysis inputs.** After the four strategy specifications and their data requirements are confirmed, provide each strategy with the appropriate data types, lookback periods, and frequencies.

Missing, stale, conflicting, or invalid data should be surfaced for review rather than silently accepted. The pipeline described here does not produce recommendations or execute trades.

## Candidate screening source options

The following sources are candidates only. No integration, automated extraction method, subscription, data license, or permission to scrape has been approved by this document.

| Candidate source | Potential responsibility | Considerations to resolve |
|---|---|---|
| **Screener.in** | Potential fundamental screening, such as revenue and profit growth, profitability, debt, and valuation ratios. | Available fields, permitted access method, terms, licensing, coverage, freshness, and cost. |
| **Chartink** | Potential technical screening, such as price action, moving-average conditions, volume spikes, and multi-timeframe conditions. | Available conditions and frequencies, permitted access method, terms, licensing, coverage, freshness, and cost. |
| **NSE** | Potential official source for Indian exchange information and market data. | The relevant data product, access method, applicable terms, permissions, and cost. |
| **BSE** | Potential additional exchange source for security coverage or cross-checking. | Whether its coverage or data is needed, plus the applicable product, access method, terms, permissions, and cost. |
| **Authorized market-data or broker APIs** | Potential sources for historical or intraday data, depending on availability, permissions, cost, and strategy requirements. | Provider selection, supported instruments and frequencies, usage rights, rate limits, cost, and reliability. |
| **Company filings and investor-relations publications** | Potential sources for financial statements, disclosures, and corporate events. | Source identity, publication time, document versions, extraction quality, and applicable use terms. |

## Data categories to evaluate

The eventual data requirements may include:

- Security identifiers and exchange listings.
- Historical open, high, low, close, and volume (OHLCV) prices.
- Intraday prices where required.
- Corporate actions and adjusted-price considerations.
- Financial statements and fundamental metrics.
- Company announcements and relevant news.
- Screening results, applied conditions, and screening timestamps.

These categories do not imply that every strategy needs every category. The four strategy specifications must determine the required fields, lookback periods, and frequencies.

## Traceability and logical data layers

For each retrieved record or observation, propose retaining the applicable details below:

- Source identity and, when available, source record identifier.
- Retrieval or capture timestamp.
- Market observation timestamp or source publication timestamp.
- Relevant adjustment status, including whether prices are adjusted and the basis when known.
- Validation status and any data-quality issues.
- Screening conditions and screening timestamp for candidate-generation results.

Treat data in three proposed logical layers:

1. **Raw:** Source material or source-shaped records retained as received, with source identity and capture time where permitted.
2. **Normalized:** Validated records represented in consistent formats, with transformations and validation outcomes traceable to their inputs.
3. **Derived:** Screening results, indicators, and other computed features, with their input references, calculation definitions or versions, and calculation times recorded.

These are logical distinctions, not a decision about physical storage. The database, file formats, retention policy, and any vendor-specific storage remain open.

## Proposed implementation sequence

1. Confirm the market universe and the data requirements of the four strategy specifications.
2. Select permitted screening and data-access methods, including applicable terms and usage rights.
3. Build a minimal screening-result ingestion path.
4. Validate shortlisted securities and observations against independent appropriate sources.
5. Normalize and store validated data using a justified, reviewable design.
6. Add only the technical indicators and fundamental features required by the confirmed strategies.
7. Expand extraction frequency and source coverage when demonstrated requirements justify the added complexity and cost.

The first implementation should be read-only. It should not generate investment recommendations, place orders, or assume access to a source that has not been authorized.

## Initial Prototype Decision — Automated Market Data Extraction

- **Initial prototype data source:** Yahoo Finance is the proposed starting source for historical daily prices for Indian equities. Example symbols include `RELIANCE.NS` and `TCS.NS`.
- **Automation:** The application is intended to retrieve data through code. Manual CSV uploads are not the intended workflow.
- **Zerodha:** Deferred because the user is currently experiencing an account issue. Keep the data-provider architecture independent so Zerodha can be reconsidered later.
- **Validation:** Validate retrieved records, timestamps, missing values, and source provenance before making data available downstream.
- **Storage:** Preserve source and retrieval timestamps and store validated records. The final storage technology remains undecided.
- **Future enhancements:** Evaluate permitted official or authorized sources for intraday prices, fundamentals, corporate actions, and improved production reliability.
- **Status:** Yahoo Finance is a proposed starting source for the prototype, not a verified or finalized production source. Check availability, coverage, reliability, and applicable usage terms before implementation.
- **Scope boundary:** This decision concerns data extraction only. It does not authorize or define stock strategies, AI agents, orchestration, recommendations, or trading.

## Open Questions

These decisions remain unresolved and require user input; this document does not answer them:

- Is the initial market universe NSE only, NSE and BSE, or broader?
- Will initial screening use Screener.in, Chartink, both, or another permitted method?
- Will screening results initially be supplied manually or obtained through a permitted automated integration?
- Which data providers and access methods are acceptable?
- What historical lookback periods and data frequencies does each of the four strategies require?
- Which corporate-action adjustments and financial fields are required?
- What storage technology and retention policy are appropriate?
- What costs, usage rights, and licensing constraints apply to the selected sources and methods?

For historical daily-price extraction in the initial Yahoo Finance prototype, code-driven retrieval is the intended workflow. The open question about manual or automated screening results remains relevant to candidate screening methods; it does not change the extraction decision above.

The four strategy specifications are not present in the current repository documentation. Their data requirements cannot be finalized until those specifications are provided or otherwise confirmed.
