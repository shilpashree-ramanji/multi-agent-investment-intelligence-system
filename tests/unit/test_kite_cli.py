"""Offline tests for safe and live Kite CLI modes."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pytest

from multi_agent_investment_system.core.exceptions import (
    MarketDataErrorCategory,
    MarketDataProviderDiagnostic,
)
from multi_agent_investment_system.llm.provider import (
    ModelOutputRejectionCategory,
    ModelProviderDiagnostic,
)
from multi_agent_investment_system.orchestration import kite_cli
from multi_agent_investment_system.orchestration.kite_ingestion import (
    KiteCollectionAction,
    KiteCoverageReason,
    KiteCoverageStatus,
    KiteWorkflowStatus,
)


def _configure_environment(monkeypatch, tmp_path) -> None:
    for key, value in {
        "OPENAI_API_KEY": "test-openai-secret",
        "OPENAI_MODEL": "test-model",
        "KITE_API_KEY": "test-kite-key",
        "KITE_API_SECRET": "test-kite-secret",
        "KITE_ACCESS_TOKEN": "test-access-token",
        "KITE_INGESTION_MAXIMUM_CALENDAR_DAYS": "31",
        "MARKET_DATA_DATABASE_PATH": str(tmp_path / "market.sqlite3"),
    }.items():
        monkeypatch.setenv(key, value)


@pytest.mark.parametrize("mode", ["validate-config", "dry-run"])
def test_safe_modes_do_not_construct_clients_or_open_database(
    mode, monkeypatch, tmp_path, capsys
) -> None:
    _configure_environment(monkeypatch, tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("safe mode must not construct runtime dependencies")

    monkeypatch.setattr(kite_cli, "OpenAIModelProvider", forbidden)
    monkeypatch.setattr(kite_cli, "build_kite_coordinator", forbidden)
    monkeypatch.setattr(kite_cli, "_run_live", forbidden)
    result = kite_cli.main(
        [
            "--mode",
            mode,
            "--symbol",
            "INFY",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-31",
        ]
    )

    assert result == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["network_calls"] == 0
    assert output["openai_api_calls"] == 0
    assert output["zerodha_api_calls"] == 0
    assert output["maximum_model_calls"] == 7
    assert "coverage" in output
    assert output["database_target"] == "market.sqlite3"
    assert all(
        secret not in captured.out
        for secret in ("test-openai-secret", "test-kite-secret", "test-access-token")
    )
    assert not (tmp_path / "market.sqlite3").exists()


def test_mode_defaults_to_validate_config_when_task_arguments_are_supplied(
    monkeypatch, tmp_path, capsys
) -> None:
    _configure_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(
        kite_cli,
        "build_kite_coordinator",
        lambda **kwargs: pytest.fail("factory called"),
    )

    result = kite_cli.main(
        ["--symbol", "INFY", "--start", "2025-01-01", "--end", "2025-01-02"]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "validate-config"


def test_no_arguments_preserve_initialization_behavior(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        kite_cli,
        "configure_logging",
        lambda: calls.append("logging"),
    )
    assert kite_cli.main([]) == 0
    assert calls == ["logging"]


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "--mode",
            "live",
            "--symbol",
            "INFY",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-02",
        ],
        [
            "--confirm-live",
            "--symbol",
            "INFY",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-02",
        ],
    ],
)
def test_live_execution_requires_both_opt_ins_before_configuration_or_clients(
    arguments, monkeypatch
) -> None:
    monkeypatch.setattr(
        kite_cli, "_load_configuration", lambda: pytest.fail("configuration loaded")
    )
    monkeypatch.setattr(
        kite_cli, "OpenAIModelProvider", lambda **kwargs: pytest.fail("client created")
    )
    monkeypatch.setattr(
        kite_cli,
        "build_kite_coordinator",
        lambda **kwargs: pytest.fail("coordinator constructed"),
    )

    with pytest.raises(SystemExit) as error:
        kite_cli.main(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize(
    ("start", "end"),
    [("2025-02-30", "2025-03-01"), ("2025-03-02", "2025-03-01")],
)
def test_invalid_dates_fail_before_runtime_construction(
    start, end, monkeypatch, tmp_path, capsys
) -> None:
    _configure_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(
        kite_cli,
        "OpenAIModelProvider",
        lambda **kwargs: pytest.fail("model client constructed"),
    )
    monkeypatch.setattr(
        kite_cli,
        "build_kite_coordinator",
        lambda **kwargs: pytest.fail("runtime factory called"),
    )

    result = kite_cli.main(
        [
            "--mode",
            "live",
            "--confirm-live",
            "--symbol",
            "INFY",
            "--start",
            start,
            "--end",
            end,
        ]
    )

    assert result == 2
    assert "Configuration error" in capsys.readouterr().err


def test_invalid_symbol_fails_before_runtime_construction(
    monkeypatch, tmp_path, capsys
) -> None:
    _configure_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(
        kite_cli,
        "OpenAIModelProvider",
        lambda **kwargs: pytest.fail("model client constructed"),
    )

    result = kite_cli.main(
        [
            "--mode",
            "live",
            "--confirm-live",
            "--symbol",
            "INFY!!!",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-02",
        ]
    )

    assert result == 2
    assert "unsupported format" in capsys.readouterr().err


@pytest.mark.parametrize(
    "symbol", ["RELIANCE", "M&M", "BAJAJ-AUTO", "RELIANCE.NS", "RELIANCE.BO", "500209"]
)
def test_supported_exchange_symbol_formats_are_accepted(symbol: str) -> None:
    request = kite_cli.KiteIngestionRequest(
        "run", "task", symbol, date(2025, 1, 1), date(2025, 1, 2)
    )
    assert request.symbol == symbol


def test_calendar_limit_is_checked_before_live_clients(
    monkeypatch, tmp_path, capsys
) -> None:
    _configure_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(
        kite_cli, "OpenAIModelProvider", lambda **kwargs: pytest.fail("client created")
    )

    result = kite_cli.main(
        [
            "--mode",
            "live",
            "--confirm-live",
            "--symbol",
            "INFY",
            "--start",
            "2025-01-01",
            "--end",
            "2025-02-01",
        ]
    )

    assert result == 2
    assert "calendar range" in capsys.readouterr().err


def test_live_composition_injects_provider_and_reports_unknown_coverage(
    monkeypatch, tmp_path, capsys
) -> None:
    _configure_environment(monkeypatch, tmp_path)
    configuration = kite_cli._load_configuration()
    request = kite_cli.KiteIngestionRequest(
        "input-run", "input-task", "INFY", date(2025, 1, 1), date(2025, 1, 2)
    )
    provider = object()
    observed: dict[str, object] = {}
    hostile_model_text = (
        "fake_api_key_value fake_access_token_value "
        "Authorization: Bearer fake_header_value prompt-response-body-sentinel"
    )

    class _Coordinator:
        async def run(self, live_request):
            observed["request"] = live_request
            return SimpleNamespace(
                status=KiteWorkflowStatus.ESCALATED,
                error_code=None,
                model_provider_diagnostic=ModelProviderDiagnostic(
                    exception_class="APIStatusError",
                    http_status=400,
                    provider_error_code="invalid_request_error",
                    provider_error_type="invalid_request_error",
                ),
                action_trace=(
                    SimpleNamespace(
                        iteration=1,
                        action=KiteCollectionAction.COLLECT_RANGE,
                        reason=hostile_model_text,
                        expected_outcome=hostile_model_text,
                        authorized=True,
                        executed=True,
                        outcome_code="records_classified_and_storage_read_back",
                        counts=None,
                    ),
                ),
                worker_result=SimpleNamespace(
                    coverage_status=KiteCoverageStatus.UNKNOWN,
                    coverage_reasons=(
                        KiteCoverageReason.NO_TRADING_CALENDAR,
                        KiteCoverageReason.NO_PROVIDER_COVERAGE_GUARANTEE,
                        KiteCoverageReason.HISTORICAL_INTERVALS_NOT_VERIFIED,
                    ),
                ),
            )

    def fake_model_provider(*, model):
        observed["model"] = model
        return provider

    def fake_factory(**kwargs):
        observed["factory_args"] = kwargs
        return _Coordinator()

    monkeypatch.setattr(kite_cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(kite_cli, "OpenAIModelProvider", fake_model_provider)
    monkeypatch.setattr(kite_cli, "build_kite_coordinator", fake_factory)

    result = kite_cli._run_live(configuration, request)

    assert result == 1
    assert observed["model"] == "test-model"
    factory_args = observed["factory_args"]
    assert factory_args["model_provider"] is provider
    assert factory_args["credentials"] is configuration.credentials
    assert factory_args["access_token"] is configuration.access_token
    assert observed["request"].symbol == "INFY"
    assert observed["request"].run_id != "input-run"
    assert observed["request"].task_id != "input-task"
    serialized_output = capsys.readouterr().out
    output = json.loads(serialized_output)
    assert output["coverage_status"] == "unknown"
    assert output["model_provider_diagnostic"] == {
        "exception_class": "APIStatusError",
        "http_status": 400,
        "provider_error_code": "invalid_request_error",
        "provider_error_type": "invalid_request_error",
    }
    assert output["maximum_model_calls"] == 7
    assert output["full_requested_session_coverage_verified"] is False
    assert output["missing_sessions"] == "not_assessed_no_exchange_calendar"
    assert output["full_requested_session_coverage_verified"] is False
    assert output["missing_sessions"] == "not_assessed_no_exchange_calendar"
    assert hostile_model_text not in serialized_output
    action_report = output["actions"][0]
    assert action_report["action"] == "collect_range"
    assert action_report["authorized"] is True
    assert action_report["executed"] is True
    assert action_report["outcome_code"] == "records_classified_and_storage_read_back"
    assert action_report["reason"] == "A bounded collection action was authorized."
    assert "test-access-token" not in repr(observed)


def test_live_report_includes_only_fixed_model_rejection_category(
    monkeypatch, tmp_path, capsys
) -> None:
    _configure_environment(monkeypatch, tmp_path)
    configuration = kite_cli._load_configuration()
    request = kite_cli.KiteIngestionRequest(
        "input-run", "input-task", "INFY", date(2025, 1, 1), date(2025, 1, 2)
    )
    raw_model_output = 'malformed fake_api_key prompt and response contents "'

    class _Coordinator:
        async def run(self, live_request):
            return SimpleNamespace(
                status=KiteWorkflowStatus.FAILED,
                error_code=kite_cli.KiteWorkflowErrorCode.INVALID_MODEL_OUTPUT,
                model_provider_diagnostic=None,
                action_trace=(
                    SimpleNamespace(
                        iteration=1,
                        action=KiteCollectionAction.REJECTED_MODEL_OUTPUT,
                        reason=raw_model_output,
                        expected_outcome=raw_model_output,
                        authorized=False,
                        executed=False,
                        outcome_code="unsupported_or_malformed_model_action",
                        rejection_category=ModelOutputRejectionCategory.INVALID_JSON,
                        counts=None,
                    ),
                ),
                worker_result=None,
                initial_observation=None,
                final_observation=None,
            )

    monkeypatch.setattr(kite_cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(kite_cli, "OpenAIModelProvider", lambda **kwargs: object())
    monkeypatch.setattr(
        kite_cli, "build_kite_coordinator", lambda **kwargs: _Coordinator()
    )

    result = kite_cli._run_live(configuration, request)

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert result == 1
    assert output["error_code"] == "invalid_model_output"
    assert output["actions"][0]["rejection_category"] == "invalid_json"
    assert raw_model_output not in captured.out


def test_live_report_includes_sanitized_market_data_diagnostic(
    monkeypatch, tmp_path, capsys
) -> None:
    _configure_environment(monkeypatch, tmp_path)
    configuration = kite_cli._load_configuration()
    request = kite_cli.KiteIngestionRequest(
        "input-run", "input-task", "INFY", date(2025, 1, 1), date(2025, 1, 2)
    )
    sensitive_text = "private response body fake-token authorization header"
    market_diagnostic = MarketDataProviderDiagnostic(
        MarketDataErrorCategory.HTTP_ERROR, 403
    )

    class _Coordinator:
        async def run(self, live_request):
            return SimpleNamespace(
                status=KiteWorkflowStatus.ESCALATED,
                error_code=kite_cli.KiteWorkflowErrorCode.MARKET_DATA_ERROR,
                model_provider_diagnostic=None,
                action_trace=(
                    SimpleNamespace(
                        iteration=1,
                        action=KiteCollectionAction.COLLECT_RANGE,
                        reason=sensitive_text,
                        expected_outcome=sensitive_text,
                        authorized=True,
                        executed=True,
                        outcome_code="market_data_error",
                        counts=None,
                    ),
                ),
                worker_result=SimpleNamespace(
                    error_code=kite_cli.KiteWorkflowErrorCode.MARKET_DATA_ERROR,
                    retryable=False,
                    market_data_diagnostic=market_diagnostic,
                    coverage_status=KiteCoverageStatus.NOT_ASSESSED,
                    coverage_reasons=(KiteCoverageReason.EXECUTION_NOT_SUCCESSFUL,),
                    counts=None,
                ),
                initial_observation=None,
                final_observation=None,
            )

    monkeypatch.setattr(kite_cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(kite_cli, "OpenAIModelProvider", lambda **kwargs: object())
    monkeypatch.setattr(
        kite_cli, "build_kite_coordinator", lambda **kwargs: _Coordinator()
    )

    result = kite_cli._run_live(configuration, request)

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert result == 1
    assert output["error_code"] == "market_data_error"
    assert output["market_data_diagnostic"] == {
        "category": "http_error",
        "http_status": 403,
        "retryable": False,
    }
    assert sensitive_text not in captured.out


def test_configuration_errors_do_not_print_secret_values(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-print-this")
    monkeypatch.delenv("KITE_ACCESS_TOKEN", raising=False)
    result = kite_cli.main(
        ["--symbol", "INFY", "--start", "2025-01-01", "--end", "2025-01-02"]
    )
    captured = capsys.readouterr()
    assert result == 2
    assert "do-not-print-this" not in captured.out + captured.err


def _live_arguments() -> list[str]:
    return [
        "--mode",
        "live",
        "--confirm-live",
        "--symbol",
        "INFY",
        "--start",
        "2025-01-01",
        "--end",
        "2025-01-02",
    ]


def _assert_no_credentials_in_output(output: str) -> None:
    for secret in (
        "test-openai-secret",
        "test-kite-secret",
        "test-access-token",
    ):
        assert secret not in output


def test_missing_openai_api_key_fails_before_client_construction(
    monkeypatch, tmp_path, capsys
) -> None:
    _configure_environment(monkeypatch, tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setattr(
        kite_cli,
        "OpenAIModelProvider",
        lambda **kwargs: pytest.fail("client constructed without API key"),
    )

    result = kite_cli.main(_live_arguments())

    captured = capsys.readouterr()
    assert result == 2
    assert "OPENAI_API_KEY must be configured" in captured.err
    _assert_no_credentials_in_output(captured.out + captured.err)


@pytest.mark.parametrize("limit", ["not-a-number", "0", "-1"])
def test_invalid_calendar_day_configuration_fails_before_clients(
    limit, monkeypatch, tmp_path, capsys
) -> None:
    _configure_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("KITE_INGESTION_MAXIMUM_CALENDAR_DAYS", limit)
    monkeypatch.setattr(
        kite_cli,
        "OpenAIModelProvider",
        lambda **kwargs: pytest.fail("client constructed with invalid settings"),
    )
    monkeypatch.setattr(
        kite_cli,
        "build_kite_coordinator",
        lambda **kwargs: pytest.fail("runtime factory called with invalid settings"),
    )

    result = kite_cli.main(_live_arguments())

    captured = capsys.readouterr()
    assert result == 2
    assert "positive integer" in captured.err
    _assert_no_credentials_in_output(captured.out + captured.err)


def test_openai_client_construction_failure_is_sanitized(
    monkeypatch, tmp_path, capsys
) -> None:
    _configure_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(kite_cli.importlib.util, "find_spec", lambda name: object())

    def fail_openai_client(**kwargs):
        raise kite_cli.ConfigurationError("Could not initialize the OpenAI client")

    monkeypatch.setattr(kite_cli, "OpenAIModelProvider", fail_openai_client)
    monkeypatch.setattr(
        kite_cli,
        "build_kite_coordinator",
        lambda **kwargs: pytest.fail("factory called after OpenAI init failed"),
    )

    result = kite_cli.main(_live_arguments())

    captured = capsys.readouterr()
    assert result == 2
    assert "Could not initialize the OpenAI client" in captured.err
    _assert_no_credentials_in_output(captured.out + captured.err)


def test_kite_runtime_construction_failure_is_sanitized(
    monkeypatch, tmp_path, capsys
) -> None:
    _configure_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(kite_cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(kite_cli, "OpenAIModelProvider", lambda **kwargs: object())

    def fail_runtime(**kwargs):
        raise RuntimeError(
            "test-openai-secret test-kite-secret test-access-token runtime details"
        )

    monkeypatch.setattr(kite_cli, "build_kite_coordinator", fail_runtime)

    result = kite_cli.main(_live_arguments())

    captured = capsys.readouterr()
    assert result == 1
    assert "Runtime execution failed; details omitted" in captured.err
    _assert_no_credentials_in_output(captured.out + captured.err)
    assert "runtime details" not in captured.err


def test_live_execution_exception_is_sanitized(monkeypatch, tmp_path, capsys) -> None:
    _configure_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(kite_cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(kite_cli, "OpenAIModelProvider", lambda **kwargs: object())

    class _FailingCoordinator:
        async def run(self, request):
            raise RuntimeError(
                "test-openai-secret test-kite-secret test-access-token failure"
            )

    monkeypatch.setattr(
        kite_cli, "build_kite_coordinator", lambda **kwargs: _FailingCoordinator()
    )

    result = kite_cli.main(_live_arguments())

    captured = capsys.readouterr()
    assert result == 1
    assert "Runtime execution failed; details omitted" in captured.err
    _assert_no_credentials_in_output(captured.out + captured.err)
