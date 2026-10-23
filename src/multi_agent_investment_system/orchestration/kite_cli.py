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
    KiteWorkflowStatus,
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
    request.validate_policy(
        KiteIngestionPolicy(configuration.settings.maximum_calendar_days)
    )
    return {
        "mode": mode,
        "symbol": request.symbol,
        "start": request.start.isoformat(),
        "end": request.end.isoformat(),
        "maximum_calendar_days": configuration.settings.maximum_calendar_days,
        "database_target": configuration.settings.database_path.name,
        "model": configuration.model,
        "expected_model_calls_per_successful_task": 3,
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
            "coverage_status": coverage_status,
            "coverage_reasons": coverage_reasons,
            "expected_model_calls_per_successful_task": 3,
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
