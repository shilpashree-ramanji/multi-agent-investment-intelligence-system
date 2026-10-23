"""Opt-in composition root for Coordinator-led Kite market-data ingestion."""

from __future__ import annotations

from multi_agent_investment_system.agents.coordinator import CoordinatorAgent
from multi_agent_investment_system.agents.zerodha_ingestion_worker import (
    ZerodhaIngestionWorkerAgent,
)
from multi_agent_investment_system.core.configuration import KiteConnectCredentials
from multi_agent_investment_system.core.exceptions import ConfigurationError
from multi_agent_investment_system.core.kite_runtime_settings import (
    KiteWorkflowRuntimeSettings,
)
from multi_agent_investment_system.data.ingestion import (
    DailyMarketDataIngestionService,
)
from multi_agent_investment_system.data.service import (
    DailyMarketDataExtractionService,
)
from multi_agent_investment_system.data.storage import SQLiteDailyCandleStore
from multi_agent_investment_system.llm.provider import ModelProvider
from multi_agent_investment_system.orchestration.kite_ingestion import (
    KiteIngestionPolicy,
)
from multi_agent_investment_system.tools.kite_authentication import KiteAccessToken
from multi_agent_investment_system.tools.kite_market_data import (
    KiteConnectMarketDataProvider,
)
from multi_agent_investment_system.tools.kite_symbol_guard import (
    RequestedSymbolGuard,
)


def build_kite_coordinator(
    *,
    model_provider: ModelProvider,
    credentials: KiteConnectCredentials,
    access_token: KiteAccessToken,
    settings: KiteWorkflowRuntimeSettings,
) -> CoordinatorAgent:
    """Wire agents and deterministic services without performing API calls.

    Authentication is supplied in memory by the caller. Secrets are never
    copied into settings, model requests, or workflow result records.
    """

    if not callable(getattr(model_provider, "generate", None)):
        raise ConfigurationError("A model provider must be supplied")
    if not isinstance(credentials, KiteConnectCredentials):
        raise ConfigurationError("Kite credentials must be supplied")
    if not isinstance(access_token, KiteAccessToken):
        raise ConfigurationError("A Kite access token must be supplied")
    if not isinstance(settings, KiteWorkflowRuntimeSettings):
        raise ConfigurationError("Kite workflow settings are malformed")

    policy = KiteIngestionPolicy(settings.maximum_calendar_days)
    provider = RequestedSymbolGuard(
        KiteConnectMarketDataProvider(credentials, access_token)
    )
    extraction = DailyMarketDataExtractionService(provider)
    store = SQLiteDailyCandleStore(settings.database_path)
    ingestion = DailyMarketDataIngestionService(extraction, store)
    worker = ZerodhaIngestionWorkerAgent(model_provider, ingestion, policy)
    return CoordinatorAgent(model_provider, worker, policy)
