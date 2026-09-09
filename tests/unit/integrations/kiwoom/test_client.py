def test_kiwoom_credentials_from_env_fails_closed_without_keys(monkeypatch) -> None:
    import pytest

    from src.core.pit import PITDataError as CorePITDataError
    from src.integrations.kiwoom.client import KiwoomCredentials
    from src.integrations.kiwoom.client import PITDataError as ClientPITDataError

    for var in ("KIWOM_APP_KEY", "KIWOOM_APP_KEY", "KIWOM_SECRET_KEY", "KIWOOM_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(ClientPITDataError, match="missing required Kiwoom credentials"):
        KiwoomCredentials.from_env()

    assert ClientPITDataError is CorePITDataError
