def test_model_selection_screening_leaf_preserves_chronological_sampling() -> None:
    import polars as pl
    from src.stocks.ml import model_selection_screening as screening
    from src.stocks.ml import model_selection_study as study

    frame = pl.DataFrame({"session": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"], "instrument_id": ["B", "A", "D", "C"], "adtv_20d": [1.0, 2.0, 1.0, 3.0]})
    expected = screening.sample_labeled_screen_rows(frame, 3).tolist()
    actual = study.sample_labeled_screen_rows(frame, 3).tolist()
    assert expected == [1, 3, 0]
    assert actual == expected

def test_model_selection_compatibility_monkeypatch_reaches_leaf(monkeypatch) -> None:
    from src.stocks.ml import model_selection as facade
    from src.stocks.ml import model_selection_study as study

    sentinel = object()
    monkeypatch.setattr(facade, "ElasticNet", sentinel)
    study._sync_legacy_selection_hooks()
    assert study.ElasticNet is sentinel
