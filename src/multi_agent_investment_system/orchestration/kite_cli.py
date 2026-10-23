"""Safe CLI modes and explicit live execution for Kite ingestion."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from multi_agent_investment_system.core.configuration import KiteConnectCredentials
from multi_agent_investment_system.core.exceptions import (
    ConfigurationError,
    DataValidationError,
)
from multi_agent_investment_system.core.kite_runtime_settings import (
    KiteWorkflowRuntimeSettings,
)
from multi_agent_investment_system.core.logging_config import configure_logging
from multi_agent_investment_system.llm.openai_provider import (
    DEFAULT_OPENAI_MODEL,
    OpenAIModelProvider,
)
from multi_agent_investment_system.orchestration.kite_ingestion import (
    KiteIngestionPolicy,
    KiteIngestionRequest,
    KiteWorkflowErrorCode,
    KiteWorkflowStatus,
    safe_action_trace_descriptions,
)
from multi_agent_investment_system.orchestration.kite_runtime import (
    build_kite_coordinator,
)
from multi_agent_investment_system.tools.kite_authentication import KiteAccessToken

_MODES = ("validate-config", "dry-run", "live")


@dataclass(frozen=True, slots=True)
class _RuntimeConfiguration:
    settings: KiteWorkflowRuntimeSettings
    credentials: KiteConnectCredentials = field(repr=False)
    access_token: KiteAccessToken = field(repr=False)
    model: str


def _parse_date(value: str, label: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise DataValidationError(f"{label} must use ISO format YYYY-MM-DD") from None
    if parsed.isoformat() != value:
        raise DataValidationError(f"{label} must use ISO format YYYY-MM-DD")
    return parsed


def _load_configuration() -> _RuntimeConfiguration:
    settings = KiteWorkflowRuntimeSettings.from_environment()
    credentials = KiteConnectCredentials.from_environment()
    if credentials is None:
        raise ConfigurationError("Kite API credentials must be configured")
    access_token_value = os.environ.get("KITE_ACCESS_TOKEN", "").strip()
    if not access_token_value:
        raise ConfigurationError("KITE_ACCESS_TOKEN must be configured")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError("OPENAI_API_KEY must be configured")
    model = os.environ.get("OPENAI_MODEL", "").strip() or DEFAULT_OPENAI_MODEL
    if not model:
        raise ConfigurationError("OPENAI_MODEL must not be empty")
    return _RuntimeConfiguration(
        settings=settings,
        credentials=credentials,
        access_token=KiteAccessToken(access_token_value),
        model=model,
    )


def _build_request(args: argparse.Namespace) -> KiteIngestionRequest:
    return KiteIngestionRequest(
        run_id="cli-validation",
        task_id="cli-validation",
        symbol=args.symbol,
        start=_parse_date(args.start, "Start date"),
        end=_parse_date(args.end, "End date"),
    )


def _plan(
    configuration: _RuntimeConfiguration, request: KiteIngestionRequest, mode: str
) -> dict[str, object]:
    policy = KiteIngestionPolicy(configuration.settings.maximum_calendar_days)
    request.validate_policy(policy)
    return {
        "mode": mode,
        "symbol": request.symbol,
        "start": request.start.isoformat(),
        "end": request.end.isoformat(),
        "maximum_calendar_days": configuration.settings.maximum_calendar_days,
        "database_target": configuration.settings.database_path.name,
        "model": configuration.model,
        "maximum_coordinator_iterations": policy.maximum_iterations,
        "maximum_collection_attempts": policy.maximum_collection_attempts,
        "maximum_symbols": 1,
        "maximum_kite_http_requests": policy.maximum_collection_attempts + 1,
        "maximum_model_calls": (
            policy.maximum_iterations + policy.maximum_collection_attempts
        ),
        "coverage": (
            "Successful ingestion does not establish historical coverage; "
            "coverage remains unknown unless independently verified."
        ),
    }


def _print_json(value: dict[str, object]) -> None:
    print(json.dumps(value, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multi-agent-investment-system",
        description="Validate, preview, or explicitly run bounded Kite ingestion.",
    )
    parser.add_argument("--mode", choices=_MODES, default="validate-config")
    parser.add_argument("--symbol", help="NSE trading symbol")
    parser.add_argument("--start", help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="Inclusive end date (YYYY-MM-DD)")
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required in addition to --mode live before any API calls",
    )
    return parser


def _run_live(
    configuration: _RuntimeConfiguration, request: KiteIngestionRequest
) -> int:
    if importlib.util.find_spec("openai") is None:
        raise ConfigurationError(
            "OpenAI SDK is unavailable; install the optional 'openai' dependency"
        )
    model_provider = OpenAIModelProvider(model=configuration.model)
    coordinator = build_kite_coordinator(
        model_provider=model_provider,
        credentials=configuration.credentials,
        access_token=configuration.access_token,
        settings=configuration.settings,
    )
    request = KiteIngestionRequest(
        run_id=str(uuid.uuid4()),
        task_id=str(uuid.uuid4()),
        symbol=request.symbol,
        start=request.start,
        end=request.end,
    )
    state = asyncio.run(coordinator.run(request))
    worker_result = state.worker_result
    counts = getattr(worker_result, "counts", None)
    final_observation = getattr(state, "final_observation", None)
    initial_observation = getattr(state, "initial_observation", None)
    action_trace = getattr(state, "action_trace", ())
    model_diagnostic = getattr(state, "model_provider_diagnostic", None)
    if model_diagnostic is None and worker_result is not None:
        model_diagnostic = getattr(
            worker_result, "model_provider_diagnostic", None
        )
    market_data_diagnostic = (
        getattr(worker_result, "market_data_diagnostic", None)
        if worker_result is not None
        else None
    )
    policy = KiteIngestionPolicy(configuration.settings.maximum_calendar_days)
    attempt_counts = [
        item.counts for item in action_trace if item.counts is not None
    ]
    coverage_status = (
        worker_result.coverage_status.value
        if worker_result is not None
        else "not_assessed"
    )
    coverage_reasons = (
        [reason.value for reason in worker_result.coverage_reasons]
        if worker_result is not None
        else []
    )
    _print_json(
        {
            "mode": "live",
            "status": state.status.value,
            "error_code": state.error_code.value if state.error_code else None,
            "model_provider_diagnostic": (
                model_diagnostic.to_dict() if model_diagnostic is not None else None
            ),
            "market_data_diagnostic": (
                {
                    **market_data_diagnostic.to_dict(),
                    "retryable": worker_result.retryable,
                }
                if market_data_diagnostic is not None
                else None
            ),
            "requested": {
                "symbol": request.symbol,
                "start": request.start.isoformat(),
                "end": request.end.isoformat(),
            },
            "counts": {
                "fetched_across_attempts": sum(item.fetched for item in attempt_counts),
                "valid_across_attempts": sum(item.valid for item in attempt_counts),
                "incomplete_across_attempts": sum(
                    item.incomplete for item in attempt_counts
                ),
                "invalid_across_attempts": sum(
                    item.invalid for item in attempt_counts
                ),
                "upsert_operations_across_attempts": sum(
                    item.upserted for item in attempt_counts
                ),
                "stored": counts.stored if counts is not None else 0,
            },
            "coverage_observed_dates": (
                [item.isoformat() for item in final_observation.stored_dates]
                if final_observation is not None
                else []
            ),
            "initial_coverage_dates": (
                [item.isoformat() for item in initial_observation.stored_dates]
                if initial_observation is not None
                else []
            ),
            "newly_covered_dates": (
                sorted(
                    set(item.isoformat() for item in final_observation.stored_dates)
                    - set(item.isoformat() for item in initial_observation.stored_dates)
                )
                if final_observation is not None and initial_observation is not None
                else []
            ),
            "coverage_quality": (
                {
                    "valid": final_observation.valid_count,
                    "incomplete": final_observation.incomplete_count,
                    "invalid": final_observation.invalid_count,
                }
                if final_observation is not None
                else None
            ),
            "missing_sessions": "not_assessed_no_exchange_calendar",
            "unresolved_gaps": "not_assessed_no_exchange_calendar",
            "duplicate_records": "not_measured_by_current_ingestion_result",
            "duplicate_storage_behavior": (
                "SQLite upserts by symbol and trading date; repeated provider rows "
                "for one date are not separately counted"
            ),
            "coverage_status": coverage_status,
            "coverage_reasons": coverage_reasons,
            "source_error_code": (
                getattr(worker_result, "error_code", None).value
                if worker_result is not None
                and getattr(worker_result, "error_code", None)
                is KiteWorkflowErrorCode.MARKET_DATA_ERROR
                else None
            ),
            "source_error_codes": sorted(
                {
                    item.outcome_code
                    for item in action_trace
                    if item.outcome_code
                    == KiteWorkflowErrorCode.MARKET_DATA_ERROR.value
                }
            ),
            "remaining_limitations": [
                "No exchange trading calendar is configured",
                "The Kite provider has no verified historical coverage guarantee",
                "Duplicate provider rows within one response are not counted "
                "separately",
            ],
            "actions": [
                {
                    "iteration": item.iteration,
                    "action": item.action.value,
                    "reason": safe_action_trace_descriptions(item.action)[0],
                    "expected_outcome": safe_action_trace_descriptions(item.action)[
                        1
                    ],
                    "authorized": item.authorized,
                    "executed": item.executed,
                    "outcome_code": item.outcome_code,
                    "rejection_category": (
                        item.rejection_category.value
                        if getattr(item, "rejection_category", None) is not None
                        else None
                    ),
                    "counts": (
                        {
                            "fetched": item.counts.fetched,
                            "valid": item.counts.valid,
                            "incomplete": item.counts.incomplete,
                            "invalid": item.counts.invalid,
                            "upserted": item.counts.upserted,
                            "stored": item.counts.stored,
                        }
                        if item.counts is not None
                        else None
                    ),
                }
                for item in action_trace
            ],
            "proposed_actions": [item.action.value for item in action_trace],
            "authorized_actions": [
                item.action.value for item in action_trace if item.authorized
            ],
            "executed_actions": [
                item.action.value for item in action_trace if item.executed
            ],
            "rejected_actions": [
                item.outcome_code for item in action_trace if not item.authorized
            ],
            "retry_count": sum(
                item.action.value == "retry_collection" and item.executed
                for item in action_trace
            ),
            "execution_and_persistence_verified": state.status
            is KiteWorkflowStatus.COMPLETED,
            "full_requested_session_coverage_verified": False,
            "completion_criteria_satisfied": False,
            "recommended_next_action": (
                "verify exchange-session coverage when an authorized calendar "
                "is available"
                if coverage_status == "unknown"
                else "review the sanitized workflow error"
            ),
            "maximum_model_calls": (
                policy.maximum_iterations + policy.maximum_collection_attempts
            ),
            "maximum_symbols": 1,
            "maximum_kite_http_requests": policy.maximum_collection_attempts + 1,
        }
    )
    return 0 if state.status is KiteWorkflowStatus.COMPLETED else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run initialization with no arguments or an explicitly selected CLI mode."""

    supplied_args = list(sys.argv[1:] if argv is None else argv)
    if not supplied_args:
        configure_logging()
        logging.getLogger(__name__).info("Multi-Agent Investment System initialized")
        return 0

    parser = _parser()
    args = parser.parse_args(supplied_args)
    if args.mode == "live" and not args.confirm_live:
        parser.error("live mode requires the separate --confirm-live flag")
    if args.mode != "live" and args.confirm_live:
        parser.error("--confirm-live is valid only with --mode live")
    missing = [name for name in ("symbol", "start", "end") if not getattr(args, name)]
    if missing:
        parser.error(
            f"--symbol, --start, and --end are required ({', '.join(missing)} missing)"
        )

    try:
        configuration = _load_configuration()
        request = _build_request(args)
        plan = _plan(configuration, request, args.mode)
    except (ConfigurationError, DataValidationError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    if args.mode == "validate-config":
        _print_json(
            {
                **plan,
                "result": "configuration_valid",
                "network_calls": 0,
                "openai_api_calls": 0,
                "zerodha_api_calls": 0,
            }
        )
        return 0
    if args.mode == "dry-run":
        _print_json(
            {
                **plan,
                "result": "dry_run_only",
                "network_calls": 0,
                "openai_api_calls": 0,
                "zerodha_api_calls": 0,
            }
        )
        return 0

    _print_json(plan)
    try:
        return _run_live(configuration, request)
    except ConfigurationError as error:
        print(f"Runtime configuration error: {error}", file=sys.stderr)
        return 2
    except Exception:
        print("Runtime execution failed; details omitted", file=sys.stderr)
        return 1
