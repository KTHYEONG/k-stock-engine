"""KIS business-error and storage-free collector invariants."""


def test_business_error_is_terminal(tmp_path) -> None:
    import pytest
    from types import SimpleNamespace

    from src.integrations.errors import ProviderTerminalError
    from src.integrations.kis.client import KisClient, KisCredentials

    calls = {"n": 0}

    def fake_get(*_args, **_kwargs):
        calls["n"] += 1
        return SimpleNamespace(
            status_code=200,
            headers={},
            json=lambda: {"rt_cd": "1", "msg_cd": "999999", "msg1": "non-retryable business rejection"},
        )

    client = KisClient(
        KisCredentials(app_key="k", app_secret="s", account_no="1", account_product_code="01", env="demo"),
        token_cache_dir=tmp_path,
    )
    client.ensure_token = lambda: "token"  # type: ignore[method-assign]
    client.session = SimpleNamespace(get=fake_get, post=lambda *a, **k: None)  # type: ignore[assignment]
    client._sync_session()

    with pytest.raises(ProviderTerminalError):
        client.inquire_price("005930")
    assert calls["n"] == 1


def test_collectors_do_not_write_storage(tmp_path) -> None:
    from datetime import date

    from src.integrations.kis.investor_flow import KisInvestorFlowCollector

    class _FakeClient:
        def inquire_investor_trade_by_stock_daily(self, symbol, anchor):
            return (
                {
                    "stck_bsop_date": anchor.strftime("%Y%m%d"),
                    "frgn_shnu_tr_pbmn": "1",
                    "frgn_seln_tr_pbmn": "0",
                    "frgn_ntby_tr_pbmn": "1",
                    "orgn_ntby_tr_pbmn": "0",
                    "prsn_ntby_tr_pbmn": "-1",
                },
            )

    root = tmp_path / "workspace"
    response = KisInvestorFlowCollector(client=_FakeClient()).fetch("005930", date(2024, 1, 2))

    assert len(response.rows) == 1
    assert list(root.rglob("*")) == []


def _policy(**overrides):
    from src.config.providers import KisPolicy

    params = {
        "app_key_env": "KIS_APP_KEY",
        "app_secret_env": "KIS_APP_SECRET",
        "account_no_env": "KIS_ACCOUNT_NO",
        "account_product_code_env": "KIS_ACCOUNT_PRODUCT_CODE",
        "env_env": "KIS_ENV",
        "circuit_threshold": 3,
        "daily_limit": 20000,
        "investor_flow_rows_per_page": 30,
        "min_interval_seconds": 0.001,
        "max_attempts": 3,
    }
    params.update(overrides)
    return KisPolicy(**params)


def test_credentials_from_env_variants(monkeypatch) -> None:
    import pytest

    from src.integrations.kis.client import KisCredentials

    monkeypatch.setenv("KIS_APP_KEY", "k")
    monkeypatch.setenv("KIS_APP_SECRET", "s")
    monkeypatch.setenv("KIS_ACCOUNT_NO", "87654321")
    monkeypatch.setenv("KIS_ACCOUNT_PRODUCT_CODE", "02")
    monkeypatch.setenv("KIS_ENV", "real")

    creds = KisCredentials.from_env(_policy(), env="demo")

    assert creds.env == "demo"
    assert (creds.account_no, creds.account_product_code) == ("87654321", "02")

    monkeypatch.setenv("KIS_ACCOUNT_NO", "12345678")
    monkeypatch.setenv("KIS_ACCOUNT_PRODUCT_CODE", "")

    with pytest.raises(ValueError, match="account number"):
        KisCredentials.from_env(_policy())


def test_default_policy_fallback_when_config_missing(monkeypatch, tmp_path) -> None:
    import src.integrations.kis.client as kis_client
    from src.integrations.kis.client import KisClient, KisCredentials

    monkeypatch.setattr(kis_client, "load_runtime_config", lambda: (_ for _ in ()).throw(RuntimeError("no cfg")))

    client = KisClient(
        KisCredentials(app_key="k", app_secret="s", account_no="1", account_product_code="01", env="demo"),
        token_cache_dir=tmp_path,
    )

    assert client._retryable_codes == ()


def test_min_interval_explicit_and_policy_pace(tmp_path) -> None:
    from src.integrations.kis.client import KisClient, KisCredentials

    creds = KisCredentials(app_key="k", app_secret="s", account_no="1", account_product_code="01", env="demo")
    assert KisClient(creds, token_cache_dir=tmp_path, min_interval_seconds=0.5)._transport._min_interval == 0.5
    assert KisClient(creds, token_cache_dir=tmp_path, policy=_policy(min_interval_seconds=2.0))._transport._min_interval == 2.0


def test_token_roundtrip_rejects_insecure_and_corrupt(tmp_path) -> None:
    from datetime import datetime, timedelta

    from src.integrations.kis.client import KisClient, KisCredentials

    client = KisClient(
        KisCredentials(app_key="k", app_secret="s", account_no="1", account_product_code="01", env="demo"),
        token_cache_dir=tmp_path,
        policy=_policy(),
    )
    assert client._token_cache_path.parent == tmp_path

    client._access_token = None
    client._load_cached_token()
    assert client._access_token is None

    expire_at = datetime.now() + timedelta(hours=1)
    client._save_cached_token("tok", expire_at)
    assert (client._token_cache_path.stat().st_mode & 0o777) == 0o600
    client._access_token = None
    client._token_expire_at = None
    client._load_cached_token()
    assert client._access_token == "tok"

    client._token_cache_path.write_text("not json", encoding="utf-8")
    client._token_cache_path.chmod(0o600)
    client._access_token = None
    client._load_cached_token()
    assert client._access_token is None

    client._token_cache_path.write_text("{}", encoding="utf-8")
    client._token_cache_path.chmod(0o600)
    client._load_cached_token()
    assert client._access_token is None

    client._save_cached_token("tok", datetime.now() - timedelta(hours=1))
    client._access_token = None
    client._load_cached_token()
    assert client._access_token is None

    client._token_cache_path.write_text("{}", encoding="utf-8")
    client._token_cache_path.chmod(0o644)
    client._access_token = None
    client._load_cached_token()
    assert client._access_token is None


def test_request_new_token_paths(tmp_path) -> None:
    import pytest
    from types import SimpleNamespace

    from src.integrations.errors import ProviderTerminalError
    from src.integrations.kis.client import KisClient, KisCredentials

    client = KisClient(
        KisCredentials(app_key="k", app_secret="s", account_no="1", account_product_code="01", env="demo"),
        token_cache_dir=tmp_path,
        policy=_policy(),
    )

    client.session = SimpleNamespace(
        post=lambda *a, **k: SimpleNamespace(
            status_code=200, raise_for_status=lambda: None, json=lambda: {"access_token": "t", "expires_in": 3600}
        )
    )  # type: ignore[assignment]
    assert client._request_new_token() == "t"
    assert client.ensure_token() == "t"

    client.session = SimpleNamespace(
        post=lambda *a, **k: SimpleNamespace(
            status_code=200, raise_for_status=lambda: None, json=lambda: {"expires_in": 3600}
        )
    )  # type: ignore[assignment]
    client._access_token = None
    client._token_expire_at = None
    with pytest.raises(ProviderTerminalError, match="token"):
        client._request_new_token()


def test_call_retries_retryable_business_code(tmp_path) -> None:
    from types import SimpleNamespace

    from src.integrations.kis.client import KisClient, KisCredentials

    client = KisClient(
        KisCredentials(app_key="k", app_secret="s", account_no="1", account_product_code="01", env="demo"),
        token_cache_dir=tmp_path,
        policy=_policy(retryable_msg_codes=("900",)),
    )
    client.ensure_token = lambda: "token"  # type: ignore[method-assign]
    calls = {"n": 0}

    def fake_get(*_a, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return SimpleNamespace(status_code=200, headers={}, json=lambda: {"rt_cd": "1", "msg_cd": "900", "msg1": "900 try later"})
        return SimpleNamespace(status_code=200, headers={}, json=lambda: {"rt_cd": "0", "output": {"a": 1}})

    client.session = SimpleNamespace(get=fake_get, post=lambda *a, **k: None)  # type: ignore[assignment]
    client._sync_session()

    assert client.inquire_price("005930") == {"a": 1}
    assert calls["n"] == 2


def test_call_rejects_invalid_json_and_non_object(tmp_path) -> None:
    import pytest
    from types import SimpleNamespace

    from src.integrations.errors import ProviderRetryableError, ProviderTerminalError
    from src.integrations.kis.client import KisClient, KisCredentials

    client = KisClient(
        KisCredentials(app_key="k", app_secret="s", account_no="1", account_product_code="01", env="demo"),
        token_cache_dir=tmp_path,
        policy=_policy(),
    )
    client.ensure_token = lambda: "token"  # type: ignore[method-assign]

    def bad_json():
        raise ValueError("no json")

    client.session = SimpleNamespace(get=lambda *_a, **_k: SimpleNamespace(status_code=200, headers={}, json=bad_json), post=lambda *a, **k: None)  # type: ignore[assignment]
    client._sync_session()
    with pytest.raises(ProviderRetryableError, match="invalid JSON"):
        client.inquire_price("005930")

    client.session = SimpleNamespace(get=lambda *_a, **_k: SimpleNamespace(status_code=200, headers={}, json=lambda: ["x"]), post=lambda *a, **k: None)  # type: ignore[assignment]
    client._sync_session()
    with pytest.raises(ProviderTerminalError, match="must be an object"):
        client.inquire_price("005930")


def test_post_paths_and_hashkey(tmp_path) -> None:
    import pytest
    from types import SimpleNamespace

    from src.integrations.errors import ProviderTerminalError
    from src.integrations.kis.client import KisClient, KisCredentials

    client = KisClient(
        KisCredentials(app_key="k", app_secret="s", account_no="1", account_product_code="01", env="demo"),
        token_cache_dir=tmp_path,
        policy=_policy(),
    )
    client.ensure_token = lambda: "token"  # type: ignore[method-assign]
    posts: list[dict] = []

    def fake_post(*_a, **_kw):
        posts.append(dict(_kw.get("json") or {}))
        return SimpleNamespace(status_code=200, headers={}, json=lambda: {"rt_cd": "0", "HASH": "h"})

    client.session = SimpleNamespace(get=lambda *_a, **_k: None, post=fake_post)  # type: ignore[assignment]
    client._sync_session()

    assert client.get_hashkey({"a": 1}) == "h"

    def no_hash(*_a, **_kw):
        return SimpleNamespace(status_code=200, headers={}, json=lambda: {"rt_cd": "0"})

    client.session = SimpleNamespace(get=lambda *_a, **_k: None, post=no_hash)  # type: ignore[assignment]
    client._sync_session()
    with pytest.raises(ProviderTerminalError, match="hashkey"):
        client.get_hashkey({"a": 1})


def test_malformed_outputs_and_balance_and_psbl(tmp_path) -> None:
    import pytest
    from types import SimpleNamespace

    from src.integrations.errors import ProviderTerminalError
    from src.integrations.kis.client import KisClient, KisCredentials

    def _client(payload):
        client = KisClient(
            KisCredentials(app_key="k", app_secret="s", account_no="1", account_product_code="01", env="demo"),
            token_cache_dir=tmp_path,
            policy=_policy(),
        )
        client.ensure_token = lambda: "token"  # type: ignore[method-assign]
        client.session = SimpleNamespace(get=lambda *_a, **_k: SimpleNamespace(status_code=200, headers={}, json=lambda: dict(payload)), post=lambda *a, **k: None)  # type: ignore[assignment]
        client._sync_session()
        return client

    with pytest.raises(ProviderTerminalError, match="inquire_price"):
        _client({"output": ["x"]}).inquire_price("005930")
    with pytest.raises(ProviderTerminalError, match="output2 must be a list"):
        _client({"output2": {}}).inquire_investor_trade_by_stock_daily("005930", __import__("datetime").date(2024, 1, 2))
    with pytest.raises(ProviderTerminalError, match="output1 malformed"):
        _client({"output1": {}, "output2": []}).inquire_balance()
    with pytest.raises(ProviderTerminalError, match="output2 malformed"):
        _client({"output1": [], "output2": {}}).inquire_balance()
    with pytest.raises(ProviderTerminalError, match="inquire_psbl_order"):
        _client({"output": []}).inquire_psbl_order("005930")


def test_limit_order_math_and_post(tmp_path) -> None:
    from datetime import date
    from pathlib import Path

    from src.core.market_rules import KrxMarket, load_krx_market_rules
    from src.integrations.kis.client import KisClient, KisCredentials

    rules = load_krx_market_rules(Path("config/market/krx_market_rules.toml"))
    client = KisClient(
        KisCredentials(app_key="k", app_secret="s", account_no="1", account_product_code="01", env="demo"),
        token_cache_dir=tmp_path,
        policy=_policy(),
    )
    bodies: list[dict] = []

    def fake_call(*, method, path, tr_id=None, params=None, body=None, include_auth=True, use_hashkey=False):
        bodies.append(dict(body or {}))
        return {"rt_cd": "0", "output": {}}

    client._call = fake_call  # type: ignore[method-assign]
    client.place_order(symbol="005930", side="buy", qty=1, price=4993.0, order_type="limit",
                       market=KrxMarket.KOSPI, session=date(2024, 1, 2), rules=rules)

    assert bodies[0]["ORD_UNPR"] == "4990"


def test_numeric_extractors_skip_unparseable(tmp_path) -> None:
    from src.integrations.kis.client import KisClient

    assert KisClient.extract_cash({"dnca_tot_amt": "xx", "nxdy_excc_amt": "5"}) == 5.0
    assert KisClient.extract_total_equity({"tot_evlu_amt": "xx"}) == 0.0
    assert KisClient.extract_current_price({"stck_prpr": "xx", "cur_prc": "7"}) == 7.0

from pathlib import Path


def _order_rules():
    from datetime import date

    from src.core.market_rules import KrxMarket, load_krx_market_rules

    rules = load_krx_market_rules(Path("config/market/krx_market_rules.toml"))
    return KrxMarket.KOSPI, date(2024, 1, 2), rules


_MARKET, _SESSION, _RULES = _order_rules()


def test_kis_client_rejects_invalid_order_without_network(tmp_path: Path) -> None:
    import pytest

    from src.integrations.kis.client import KisClient, KisCredentials

    credentials = KisCredentials(
        app_key='key',
        app_secret='secret',
        account_no='12345678',
        account_product_code='01',
        env='demo',
    )
    client = KisClient(credentials, token_cache_dir=tmp_path)

    with pytest.raises(ValueError, match='qty'):
        client.place_order(
            symbol='005930', side='buy', qty=0,
            market=_MARKET, session=_SESSION, rules=_RULES,
        )


def test_kis_token_cache_rejects_insecure_file(tmp_path) -> None:
    from datetime import datetime, timedelta

    from src.integrations.kis.client import KisClient, KisCredentials

    client = KisClient(KisCredentials('key', 'secret', '12345678', '01', env='demo'))
    client._token_cache_path = tmp_path / 'token.json'
    client._token_cache_path.write_text(
        '{"access_token": "cached", "expire_at": "'
        + (datetime.now() + timedelta(hours=1)).isoformat()
        + '"}',
        encoding='utf-8',
    )
    client._token_cache_path.chmod(0o644)

    client._load_cached_token()

    assert client._access_token is None


def _search_client(monkeypatch=None, response=None):
    from src.integrations.kis.client import KisClient, KisCredentials

    client = KisClient(KisCredentials('key', 'secret', '12345678', '01', env='demo'))
    calls: list[dict] = []
    payload = {'output': response} if response is not None else {'output': {'std_idst_clsf_cd': '032604'}}

    def fake_call(*, method, path, tr_id=None, params=None, body=None, include_auth=True, use_hashkey=False):
        calls.append({'method': method, 'path': path, 'tr_id': tr_id, 'params': dict(params or {})})
        return dict(payload)

    client._call = fake_call  # type: ignore[method-assign]
    return client, calls


def test_search_stock_info_sends_documented_request() -> None:
    raw = {'std_idst_clsf_cd': '032604', 'std_idst_clsf_cd_name': '통신 및 방송 장비 제조업'}
    client, calls = _search_client(response=raw)

    assert client.search_stock_info('005930') == raw
    assert len(calls) == 1
    assert calls[0]['method'] == 'GET'
    assert calls[0]['path'] == '/uapi/domestic-stock/v1/quotations/search-stock-info'
    assert calls[0]['tr_id'] == 'CTPF1002R'
    assert calls[0]['params'] == {'PRDT_TYPE_CD': '300', 'PDNO': '005930'}


def test_search_stock_info_rejects_blank_symbol() -> None:
    import pytest

    client, calls = _search_client(response={})
    with pytest.raises(ValueError, match='symbol is required'):
        client.search_stock_info('   ')
    assert calls == []


def test_search_stock_info_rejects_malformed_output() -> None:
    import pytest

    client, _ = _search_client(response=['not-a-dict'])
    with pytest.raises(RuntimeError, match='search-stock-info'):
        client.search_stock_info('005930')
