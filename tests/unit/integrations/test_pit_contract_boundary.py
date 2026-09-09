def test_provider_and_domain_pit_consumers_import_from_core() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    consumers = (
        "src/domain/stock_data.py",
        "src/integrations/dart/xbrl.py",
        "src/integrations/investor_flow_router.py",
        "src/integrations/kis/investor_flow.py",
        "src/integrations/kiwoom/client.py",
        "src/integrations/kiwoom/investor_flow.py",
        "src/integrations/krx/historical.py",
        "src/integrations/ls/client.py",
        "src/integrations/ls/investor_flow.py",
    )

    for relative_path in consumers:
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "from src.core.pit import" in source, relative_path
        assert "src.data.schemas" not in source, relative_path
