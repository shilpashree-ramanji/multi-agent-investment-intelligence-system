# Multi-Agent Investment System

## 1. Vision

Build a modular, evidence-grounded investment research platform in which specialized AI agents independently analyze investment opportunities, execute distinct investment strategies, challenge one another's conclusions, and produce risk-aware decision recommendations.

The system is designed to support human investment decisions. It must explain its reasoning through traceable evidence, explicit assumptions, uncertainty, and risk assessments.

## 2. Core Principles

- **Agent-first architecture:** Agents must have defined responsibilities, inputs, outputs, tools, and execution boundaries.
- **Strategy independence:** Each investment strategy must have its own rules, analysis process, and evaluation criteria.
- **Evidence before conclusions:** Recommendations must reference source data, timestamps, assumptions, and data-quality checks.
- **Independent review:** Review agents should challenge unsupported claims, conflicting evidence, and weak investment theses.
- **Risk before allocation:** Position sizing and capital allocation must respect configurable portfolio and strategy risk limits.
- **Human oversight:** The system produces decision support; users retain control over investment decisions and order execution.
- **Reproducibility:** Decisions should record the data, configuration, strategy version, and agent outputs used.
- **Testability:** Agent behavior, strategy rules, orchestration, and risk controls must be independently testable.
- **Secure configuration:** Credentials must remain outside source control.

## 3. Intended Agent Architecture

The planned system will contain specialized agent roles:

1. **Discovery agents** — identify candidate stocks and investment opportunities.
2. **Research agents** — gather and evaluate company, financial, market, and relevant news evidence.
3. **Strategy agents** — independently evaluate opportunities against their assigned strategy.
4. **Review and critic agents** — challenge theses, check evidence, and identify contradictions.
5. **Risk agents** — evaluate downside, exposure, liquidity, concentration, and proposed position sizing.
6. **Decision orchestration** — coordinate agent execution and consolidate validated outputs.
7. **Portfolio intelligence** — evaluate capital allocation and interactions among positions.
8. **Monitoring agents** — identify changes that may invalidate an existing thesis or decision.

These are planned responsibilities, not a claim that the agents have already been implemented.

## 4. Investment Strategies

The system is intended to support four separately implemented investment strategies.

Each strategy must have a documented specification, independent evaluation logic, explicit entry and exit conditions, risk rules, and measurable performance criteria.

The exact definitions of all four strategies must be recovered from the project's existing strategy documentation and confirmed before implementation. They must not be invented merely to fill the architecture.

## 5. High-Level Decision Workflow

1. Discover candidate opportunities.
2. Collect and validate relevant data.
3. Run the applicable strategy agents.
4. Review evidence, assumptions, and conflicting conclusions.
5. Evaluate risk and portfolio constraints.
6. Generate a traceable decision proposal with confidence and supporting evidence.
7. Present the proposal for human review.
8. Record the user's decision and subsequent outcome.
9. Monitor changes and use observed outcomes to evaluate system performance.

## 6. Safety and Trading Boundaries

- Paper trading is the default.
- Live trading is disabled by default.
- No live order execution is permitted without a separately designed, tested, and explicitly authorized execution layer.
- Risk limits must not be bypassed because an agent reports high confidence.
- Missing, stale, or conflicting data must be surfaced rather than silently treated as reliable.
- Backtests must account for transaction costs, slippage, and relevant execution constraints.
- Historical performance does not guarantee future results.

## 7. Development Approach

Development will proceed incrementally:

1. Establish the repository, development environment, and quality checks.
2. Confirm the architecture and recover the four strategy specifications.
3. Implement shared agent contracts and evidence models.
4. Implement orchestration and tool interfaces.
5. Implement the four strategy agents independently.
6. Add review, risk, decision, and portfolio components.
7. Add integration tests, historical evaluation, and paper-trading workflows.
8. Add monitoring and a user interface after the core decision pipeline is operational.

Every substantial feature will be developed on a feature branch, tested, reviewed, and merged into `main` only after approval.

## 8. Current Implementation Status

**Phase:** Market-data collection foundation.

The repository includes a Kite Connect historical daily-candle adapter, deterministic OHLCV validation, SQLite persistence, incremental ingestion services, and a bounded Coordinator/Worker workflow. The Coordinator now selects from a small allow-listed action set using stored-coverage and execution observations; deterministic policy authorizes each action. This is a prototype data-collection workflow, not a strategy agent or an investment recommendation system. Exchange-session coverage remains unverified because there is no configured trading calendar or provider coverage guarantee. The four strategy agents, fundamentals/news integrations, and trading execution are not implemented.

## 9. Technology and Configuration

The project uses Python and a `src/` package layout. Development quality checks use pytest and Ruff.

AI model providers, market-data sources, storage, and other integrations will be selected and configured through explicit interfaces rather than hard-coded throughout the codebase.

## 10. Running the Foundation

Use Python 3.12 or newer. Install the project and its already-declared development tools with:

```bash
python -m pip install -e ".[dev]"
```

Run the initialization command, safe CLI modes, tests, and lint check with:

```bash
python -m multi_agent_investment_system.main
python -m multi_agent_investment_system.main --mode dry-run --symbol RELIANCE --start 2026-10-19 --end 2026-10-23
python -m pytest
python -m ruff check .
```

The CLI reads configuration from the process environment; it does not load `.env` or the YAML templates. For a live, read-only data collection, configure `KITE_API_KEY`, `KITE_API_SECRET`, `KITE_ACCESS_TOKEN`, `OPENAI_API_KEY`, and `KITE_INGESTION_MAXIMUM_CALENDAR_DAYS` locally, then explicitly pass both live opt-ins:

```bash
python -m multi_agent_investment_system.main --mode live --confirm-live --symbol RELIANCE --start 2026-10-19 --end 2026-10-23
```

Live mode may call OpenAI for structured Coordinator/Worker decisions and Kite for daily market data. It does not place orders. Successful execution and persistence do not establish complete exchange-session coverage; inspect the report's coverage status and limitations. `.env.example` remains a credential-free template; paper trading defaults to enabled, and live trading defaults to disabled.

## 11. Guiding Rule

Build the investment intelligence and decision pipeline first. A user interface is a supporting layer, not a substitute for working agents, strategy implementations, evidence validation, and risk controls.
