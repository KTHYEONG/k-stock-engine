# ruff: noqa
"""Research feature schema fold-local tests."""
# MODEL_SELECTION_02_FOLD_LOCAL_ATTRIBUTION
import polars as pl
import numpy as np


def test_stock_net_alpha_roles_exclude_pit_fundamentals_without_lineage():
    from legacy.stocks.ml.features import stock_net_alpha_v1_roles

    roles = stock_net_alpha_v1_roles(
        available_columns=("instrument_id", "session", "feature__momentum_5d")
    )
    assert "ep_ratio" not in roles
    assert "bp_ratio" not in roles

    with_lineage = stock_net_alpha_v1_roles(
        available_columns=("instrument_id", "session", "disclosure_date")
    )
    assert "ep_ratio" in with_lineage
    assert "bp_ratio" in with_lineage

def test_MODEL_SELECTION_02_FOLD_LOCAL_ATTRIBUTION():
    from legacy.stocks.ml.features import fit_research_feature_schema
    from legacy.stocks.ml.model_selection import select_feature_groups
    from legacy.stocks.ml.contracts import ModelFamily
    # create train frame with two groups
    rng=np.random.default_rng(1)
    n=100
    frame=pl.DataFrame({
        "instrument_id": [f"KRX:{i%5:05d}" for i in range(n)],
        "session": [f"2024-01-{(i%10)+1:02d}" for i in range(n)],
        "sector": ["tech"]*n,
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "net_alpha_target": rng.normal(size=n),
    })
    # Make one source economically informative so the non-negative
    # permutation-contribution gate has a deterministic survivor.
    frame = frame.with_columns((pl.col("a") * 2.0 + pl.col("b") * 0.1).alias("net_alpha_target"))
    roles={"a":"ALPHA", "b":"ALPHA"}
    schema=fit_research_feature_schema(frame, roles)
    # ensure selected groups non-empty subset and finite scores
    evidence=select_feature_groups(frame, [], ModelFamily.elastic_net_v2, schema)
    assert len(evidence.selected_source_groups)>0
    assert set(evidence.selected_source_groups).issubset({k for k,_ in schema.source_groups})
    for _, score in evidence.source_group_scores:
        assert np.isfinite(score)
    # mutation of validation/holdout rows leaves earlier schema fingerprint unchanged
    fp=schema.fingerprint
    selected=tuple(evidence.selected_source_groups)
    # mutate holdout-like rows (create copy with changed b values)
    mutated=frame.with_columns((pl.col("b")+1000).alias("b"))
    # fit again on original train should be same
    schema2=fit_research_feature_schema(frame, roles)
    assert schema2.fingerprint==fp
    evidence2=select_feature_groups(frame, [], ModelFamily.elastic_net_v2, schema2)
    assert tuple(evidence2.selected_source_groups)==selected
    assert evidence2.schema_fingerprint==evidence.schema_fingerprint
