"""Provider policy invariants: moved values are unchanged and unsafe values fail closed."""
from __future__ import annotations

import pytest


def _policy():  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config

    return load_provider_policy(load_runtime_config())


def test_moved_values_are_unchanged() -> None:
    provider = _policy()

    primary = provider.dart_key("OPENDART_API_KEY")
    assert (primary.daily_limit, primary.daily_budget, primary.daily_reserve) == (20000, 16000, 400)
    assert (primary.min_interval_seconds, primary.max_workers) == (0.2, 2)

    secondary = provider.dart_key("OPENDART_API_KEY_2")
    assert (secondary.daily_limit, secondary.daily_budget, secondary.daily_reserve) == (20000, 19500, 500)
    assert (secondary.min_interval_seconds, secondary.max_workers) == (0.2, 2)

    assert provider.dart.circuit_threshold == 3
    assert provider.dart.requests_per_identity == 3
    assert provider.dart.batch_identities == 500
    assert [tuple(window) for window in provider.dart.shared_ip_avoid_windows_kst] == [("21:25", "21:50")]

    assert provider.primary_key_env == "OPENDART_API_KEY"
    assert provider.default_key_env == "OPENDART_API_KEY_2"
    assert provider.dart_key(provider.default_key_env).default is True
    assert provider.dart_key(provider.primary_key_env).default is False
    assert tuple(provider.dart.disclosure_types) == ("A", "I001")
    assert (provider.kis.app_key_env, provider.kis.app_secret_env) == ("KIS_APP_KEY", "KIS_APP_SECRET")
    assert provider.kis.min_interval_seconds == 1.0
    assert provider.kis.max_attempts == 3
    assert provider.kis.circuit_threshold == 3
    assert provider.kis.daily_limit == 20000
    assert provider.kis.investor_flow_rows_per_page == 30
    assert provider.krx.circuit_threshold == 3
    assert provider.krx.daily_limit == 10000
    assert provider.ls.circuit_threshold == 3
    assert provider.ls.max_sessions_per_request == 700
    assert provider.ls.daily_limit == 10000


def test_undeclared_key_is_rejected() -> None:
    from src.config import ConfigError

    with pytest.raises(ConfigError):
        _policy().dart_key("OPENDART_API_KEY_9")


def test_unsafe_interval_is_rejected() -> None:
    import pydantic

    from src.config.providers import DartKeyPolicy

    with pytest.raises(pydantic.ValidationError):
        DartKeyPolicy(
            daily_limit=20000,
            daily_budget=1000,
            daily_reserve=10,
            min_interval_seconds=0.01,
            max_workers=4,
        )


def test_key_budget_and_reserve_are_validated() -> None:
    import pydantic

    from src.config.providers import DartKeyPolicy

    with pytest.raises(pydantic.ValidationError, match="must be below"):
        DartKeyPolicy(
            daily_limit=20000,
            daily_budget=20000,
            daily_reserve=10,
            min_interval_seconds=0.1,
            max_workers=4,
        )
    with pytest.raises(pydantic.ValidationError, match="daily_reserve"):
        DartKeyPolicy(
            daily_limit=20000,
            daily_budget=1000,
            daily_reserve=1000,
            min_interval_seconds=0.1,
            max_workers=4,
        )


def test_default_key_falls_back_to_primary_without_marker() -> None:
    import pydantic

    import pytest

    from src.config import load_provider_policy, load_runtime_config
    from src.config.providers import DartPolicy

    provider = load_provider_policy(load_runtime_config())
    assert provider.default_key_env == "OPENDART_API_KEY_2"
    unmarked = provider.model_copy(
        update={
            "dart": provider.dart.model_copy(
                update={
                    "keys": {
                        key_env: policy.model_copy(update={"default": False})
                        for key_env, policy in provider.dart.keys.items()
                    }
                }
            )
        }
    )
    assert unmarked.default_key_env == unmarked.primary_key_env

    with pytest.raises(pydantic.ValidationError, match="disclosure_types"):
        DartPolicy(
            circuit_threshold=3,
            requests_per_identity=3,
            batch_identities=500,
            shared_ip_avoid_windows_kst=[],
            disclosure_types=["  "],
            keys=dict(provider.dart.keys),
        )


def _dart_kwargs(provider):  # type: ignore[no-untyped-def]
    return {
        "circuit_threshold": 3,
        "requests_per_identity": 3,
        "batch_identities": 500,
        "shared_ip_avoid_windows_kst": [],
        "keys": dict(provider.dart.keys),
    }


def test_disclosure_type_code_maps_to_pblntf_ty() -> None:
    from src.config.providers import DartPolicy

    provider = _policy()
    policy = DartPolicy(**{**_dart_kwargs(provider), "disclosure_types": ["A"]})

    (single,) = policy.disclosure_filters

    assert (single.code, single.parameter) == ("A", "pblntf_ty")


def test_disclosure_detail_code_maps_to_pblntf_detail_ty() -> None:
    from src.config.providers import DartPolicy

    provider = _policy()
    policy = DartPolicy(**{**_dart_kwargs(provider), "disclosure_types": ["I001"]})

    (single,) = policy.disclosure_filters

    assert (single.code, single.parameter) == ("I001", "pblntf_detail_ty")


def test_malformed_disclosure_code_rejected() -> None:
    import pytest

    from src.config import ConfigError, load_provider_policy, load_runtime_config
    from src.config.providers import DartPolicy
    from src.config.runtime import RuntimeConfig

    provider = _policy()
    import pydantic

    for bad in ("A1", "a"):
        with pytest.raises((ConfigError, pydantic.ValidationError)):
            DartPolicy(**{**_dart_kwargs(provider), "disclosure_types": [bad]})

    runtime = load_runtime_config()

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        (tmp_root / "config").mkdir()
        lines = ["[dart]", "disclosure_types = ['A1']", "circuit_threshold = 3",
                 "requests_per_identity = 3", "batch_identities = 500"]
        (tmp_root / "config" / "providers.toml").write_text("\n".join(lines), encoding="utf-8")
        with pytest.raises(ConfigError):
            load_provider_policy(RuntimeConfig(**{**runtime.model_dump(), "repo_root": tmp_root}))


def test_corp_codes_max_age_defaults_to_30_days() -> None:
    assert _policy().dart.corp_codes_max_age_days == 30


def test_disclosure_filter_helper_rejects_malformed_code() -> None:
    import pytest

    from src.config import ConfigError
    from src.config.providers import disclosure_filter_for_code

    with pytest.raises(ConfigError):
        disclosure_filter_for_code("A1")


def test_invalid_provider_policy_fails_closed() -> None:
    import pydantic

    import pytest

    from src.config import load_provider_policy, load_runtime_config
    from src.config.providers import DartPolicy, KisPolicy

    provider = load_provider_policy(load_runtime_config())
    with pytest.raises(pydantic.ValidationError, match="at least one key"):
        DartPolicy(**{**_dart_kwargs(provider), "keys": {}})
    with pytest.raises(pydantic.ValidationError, match="HH:MM"):
        DartPolicy(**{**_dart_kwargs(provider), "shared_ip_avoid_windows_kst": [["9:00", "10:00"]]})
    with pytest.raises(pydantic.ValidationError, match="non-empty"):
        KisPolicy(
            app_key_env="  ",
            app_secret_env="S",
            account_no_env="A",
            account_product_code_env="P",
            env_env="E",
            circuit_threshold=3,
            daily_limit=20000,
            investor_flow_rows_per_page=30,
            min_interval_seconds=1.0,
            max_attempts=3,
        )


def test_missing_or_invalid_provider_file_fails_closed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import pytest

    from src.config import ConfigError, load_provider_policy, load_runtime_config
    from src.config.runtime import RuntimeConfig

    runtime = load_runtime_config()
    empty_root = tmp_path / "empty-repo"
    empty_root.mkdir()
    with pytest.raises(ConfigError, match="missing"):
        load_provider_policy(RuntimeConfig(**{**runtime.model_dump(), "repo_root": empty_root}))

    bad_toml_root = tmp_path / "bad-toml-repo"
    config_dir = bad_toml_root / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "providers.toml").write_text("{broken", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_provider_policy(RuntimeConfig(**{**runtime.model_dump(), "repo_root": bad_toml_root}))

    bad_values_root = tmp_path / "bad-values-repo"
    bad_config_dir = bad_values_root / "config"
    bad_config_dir.mkdir(parents=True)
    (bad_config_dir / "providers.toml").write_text("[dart]\ncircuit_threshold = 0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid"):
        load_provider_policy(RuntimeConfig(**{**runtime.model_dump(), "repo_root": bad_values_root}))


def test_secrets_are_never_echoed(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config import ConfigError, read_secret

    monkeypatch.delenv("OPENDART_API_KEY_MISSING_FOR_TEST", raising=False)
    with pytest.raises(ConfigError) as exc_info:
        read_secret("OPENDART_API_KEY_MISSING_FOR_TEST")

    assert str(exc_info.value) == "secret OPENDART_API_KEY_MISSING_FOR_TEST is not set (source ~/.quant_env.sh)"

    monkeypatch.setenv("OPENDART_API_KEY_MISSING_FOR_TEST", "s3cr3t-value")
    assert read_secret("OPENDART_API_KEY_MISSING_FOR_TEST") == "s3cr3t-value"


def test_blank_env_names_are_rejected() -> None:
    import pytest

    from src.config.providers import KrxPolicy, LsPolicy

    with pytest.raises(ValueError, match="KRX env names must be non-empty"):
        KrxPolicy(api_key_env="  ", circuit_threshold=3, min_interval_seconds=1.5, daily_limit=10000)
    with pytest.raises(ValueError, match="LS env names must be non-empty"):
        LsPolicy(
            app_key_env="L1",
            app_secret_env="",
            circuit_threshold=3,
            max_sessions_per_request=700,
            min_interval_seconds=1.05,
            daily_limit=10000,
        )


def test_runner_policy_per_provider() -> None:
    from src.config import ConfigError

    provider = _policy()

    dart = provider.runner("dart")
    assert (dart.quota_provider, dart.daily_budget, dart.daily_reserve) == ("DART", 19500, 500)
    assert dart.circuit_threshold == 3
    assert tuple(dart.avoid_windows_kst) == (("21:25", "21:50"),)
    primary = provider.runner("dart", key_env=provider.primary_key_env)
    assert (primary.daily_budget, primary.daily_reserve) == (16000, 400)

    ls = provider.runner("ls")
    assert (ls.quota_provider, ls.daily_budget, ls.daily_reserve) == ("LS", 10000, 0)
    assert ls.circuit_threshold == 3
    assert ls.avoid_windows_kst == ()

    krx = provider.runner("krx")
    assert (krx.quota_provider, krx.daily_budget, krx.daily_reserve) == ("KRX", 10000, 0)
    assert krx.circuit_threshold == 3
    assert krx.avoid_windows_kst == ()

    kis = provider.runner("kis")
    assert (kis.quota_provider, kis.daily_budget, kis.daily_reserve) == ("KIS", 20000, 0)
    assert kis.circuit_threshold == 3
    assert kis.avoid_windows_kst == ()

    with pytest.raises(ConfigError, match="unknown provider"):
        provider.runner("nyse")
    with pytest.raises(ConfigError, match="no declared policy"):
        provider.runner("dart", key_env="OPENDART_API_KEY_9")


def test_document_parser_defaults_to_untrusted_gate() -> None:
    """Fail closed: a policy that does not say so never trusts document facts.

    The shipped file flips ``trusted`` only after the precision benchmark passes,
    so the default is checked on the model and the gate values on the shipped file.
    """
    from src.config.providers import DocumentParserPolicy

    assert DocumentParserPolicy().trusted is False
    provider = _policy()
    assert provider.dart.document_parser.min_precision == 0.995
    assert provider.dart.document_parser.benchmark_sample == 300


def test_document_parser_rejects_bad_precision() -> None:
    import pydantic
    import pytest

    from src.config.providers import DocumentParserPolicy

    with pytest.raises(pydantic.ValidationError):
        DocumentParserPolicy(trusted=False, min_precision=0.0, benchmark_sample=300)
    with pytest.raises(pydantic.ValidationError):
        DocumentParserPolicy(trusted=False, min_precision=1.5, benchmark_sample=300)


def test_document_parser_rejects_bad_sample() -> None:
    import pydantic
    import pytest

    from src.config.providers import DocumentParserPolicy

    with pytest.raises(pydantic.ValidationError):
        DocumentParserPolicy(trusted=False, min_precision=0.995, benchmark_sample=0)
