def test_provider_and_domain_pit_consumers_import_from_core() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    consumers = (
        "src/integrations/dart/xbrl.py",
        "src/integrations/kis/investor_flow.py",
    )

    for relative_path in consumers:
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "from src.core.pit import" in source, relative_path
        assert "src.data.schemas" not in source, relative_path
