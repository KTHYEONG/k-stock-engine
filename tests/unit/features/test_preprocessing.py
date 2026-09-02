def test_normalize_component_scores_keeps_ties_sector_isolation_and_small_sector_fail_closed() -> None:
    import polars as pl

    from src.features.contracts import QvefFeaturePolicy
    from src.features.preprocessing import normalize_component_scores

    rows = pl.DataFrame({
        'instrument_id': [f'T{i}' for i in range(10)] + [f'H{i}' for i in range(10)] + ['S0'],
        'sector': ['Technology'] * 10 + ['Health'] * 10 + ['Small'],
        'raw_value': [float(i) for i in range(1, 11)] + [5.0] * 10 + [1000.0],
    })

    result = normalize_component_scores(rows, policy=QvefFeaturePolicy())

    technology = result.filter(pl.col('sector') == 'Technology').sort('raw_value')
    assert technology['normalized_score'].to_list()[0] == -1.0
    assert technology['normalized_score'].to_list()[-1] == 1.0
    assert set(result.filter(pl.col('sector') == 'Health')['normalized_score'].to_list()) == {0.0}
    small = result.filter(pl.col('instrument_id') == 'S0').row(0, named=True)
    assert small['normalized_score'] is None
    assert small['score_available'] is False
    assert small['score_reason'] == 'sector_too_small'
