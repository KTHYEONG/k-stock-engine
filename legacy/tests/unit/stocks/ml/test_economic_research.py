"""Economic family study contract: causal selection and all-cash outcomes."""
from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest

from legacy.stocks.ml import economic_research
from legacy.stocks.ml.contracts import (
    ELASTIC_NET_FAMILY,
    TAIL_LAMBDARANK_FAMILY,
    EconomicFamilyStudySettings,
    NetAlphaTrainingRequest,
)
from legacy.stocks.ml.economic_objective import InvalidOofEconomicUtilityError, build_route_tail_relevance, measure_tail_capture
from legacy.stocks.ml.economic_research import (
    evaluate_economic_family_study,
    evaluate_economic_window_candidate,
)

ECONOMIC_FAMILY_03_CAUSAL_FAMILY_SELECTION = "ECONOMIC_FAMILY_03_CAUSAL_FAMILY_SELECTION"
ECONOMIC_FAMILY_04_ALL_CASH_ON_INSUFFICIENT_ECONOMICS = (
    "ECONOMIC_FAMILY_04_ALL_CASH_ON_INSUFFICIENT_ECONOMICS"
)
_LABEL_ID = "instrument_id"
_LABEL_SESSION = "session"

_FAILURE_ACTIONS = (
    "no-label-capacity",
    "tail-capture-insufficient",
    "execution-economics-insufficient",
)

_REGISTRY_SENTINEL = SimpleNamespace(name="registry")


def _request(**overrides) -> NetAlphaTrainingRequest:
    from dataclasses import replace

    base = NetAlphaTrainingRequest(
        artifact_id="econ_study",
        candidate_horizon_sessions=(10,),
        forward_holdout_sessions=252,
    )
    return replace(base, **overrides)


def _cost_bound_request(**overrides) -> NetAlphaTrainingRequest:
    return _request(
        base_cost_schedule=SimpleNamespace(kind="base"),
        stress_cost_schedule=SimpleNamespace(kind="stress"),
        liquidity_model=SimpleNamespace(name="base_liq"),
        stress_liquidity_model=SimpleNamespace(name="stress_liq"),
        **overrides,
    )


def _settings(**overrides) -> EconomicFamilyStudySettings:
    kwargs: dict[str, object] = {"candidate_lookback_sessions": (504, 756)}
    kwargs.update(overrides)
    return EconomicFamilyStudySettings(**kwargs)


def _data(total_sessions: int, holdout_flip: bool = False) -> SimpleNamespace:
    sessions = list(range(total_sessions))
    if holdout_flip:
        # Mutate only the locked newest holdout sessions; the pre-holdout
        # research surface stays byte-identical.
        boundary = total_sessions - 252
        sessions = [
            -s if index >= boundary else s for index, s in enumerate(sessions)
        ]
    return SimpleNamespace(feature_frame=pl.DataFrame({"session": sessions}))


def _certificate(
    *,
    passed: bool = True,
    base: float | None = 0.08,
    stress: float | None = None,
    matched: float | None = None,
) -> dict[str, object]:
    return {
        "passed": passed,
        "reasons": [] if passed else ["non-positive-base-lower-cagr"],
        "base_lower_cagr": base,
        "stress_lower_cagr": stress if stress is not None else 0.05,
        "matched_lower_excess_cagr": matched if matched is not None else 0.02,
        "mdd": 0.21,
        "observed_intervals": 720,
        "invested_intervals": 650,
        "filled_orders": 189,
    }


def _failing_certificate() -> dict[str, object]:
    return {
        "passed": False,
        "reasons": ["no-filled-orders"],
        "base_lower_cagr": None,
        "stress_lower_cagr": None,
        "matched_lower_excess_cagr": None,
        "mdd": None,
        "observed_intervals": 720,
        "invested_intervals": 0,
        "filled_orders": 0,
    }


def _family_summary(
    family: str,
    *,
    qualified: bool,
    failure_stage: str | None,
    certificate: dict[str, object] | None,
    tail_bound: float = 0.01,
) -> dict[str, object]:
    return {
        "model_family": family,
        "admitted_candidate_count": 1 if qualified else 0,
        "oracle_capacity_observed": failure_stage != "no-label-capacity",
        "tail_gate_observed": failure_stage not in (
            "no-label-capacity",
            "tail-capture-insufficient",
        ),
        "best_tail_excess_lower_bound": tail_bound if qualified else -0.004,
        "best_tail_capture_ratio": None,
        "certificate": certificate,
        "growth_route": {
            "version": "v1",
            "candidate_count": 1 if qualified else 0,
            "selected_policy": None,
            "rejection_reason_counts": {},
        },
        "qualified": qualified,
        "failure_stage": failure_stage,
        "rejection_reason_counts": {},
    }


def _window_result(
    summaries: dict[str, dict[str, object]],
    *,
    selected_family: str | None = None,
    certificate: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "status": "RESEARCH_ONLY",
        "artifact_published": False,
        "study_complete": True,
        "fold_count": 3,
        "candidates_evaluated": len(summaries),
        "selected_family": selected_family,
        "certificate": certificate,
        "families": summaries,
        "rejection_reason_counts": {},
    }


def _install_candidate(monkeypatch, responder) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake(data, request, settings, *, registry):
        calls.append(
            {
                "lookback": request.max_training_lookback_sessions,
                "request_alpha": request.bootstrap_alpha,
                "request_resamples": request.bootstrap_resamples,
                "alpha": request.compounding.bootstrap_alpha,
                "resamples": request.compounding.bootstrap_resamples,
                "fold_count": request.fold_count,
                "registry": registry,
            }
        )
        return responder(request.max_training_lookback_sessions)

    monkeypatch.setattr(
        economic_research, "evaluate_economic_window_candidate", fake
    )
    return calls


def test_economic_family_03_causal_family_selection(monkeypatch) -> None:
    """ECONOMIC_FAMILY_03_CAUSAL_FAMILY_SELECTION.

    Selection is invariant to locked-holdout label changes, every candidate
    shares one fold calendar and family-adjusted alpha, and only a candidate
    with positive base/stress/matched/tail lower bounds is ever selected.
    """
    responses = {
        504: _window_result(
            {
                ELASTIC_NET_FAMILY: _family_summary(
                    ELASTIC_NET_FAMILY,
                    qualified=False,
                    failure_stage="execution-economics-insufficient",
                    certificate=_passing_negative_stress(),
                ),
                TAIL_LAMBDARANK_FAMILY: _family_summary(
                    TAIL_LAMBDARANK_FAMILY,
                    qualified=True,
                    failure_stage=None,
                    certificate=_certificate(stress=0.05, matched=0.02),
                ),
            },
            selected_family=TAIL_LAMBDARANK_FAMILY,
            certificate=_certificate(stress=0.05, matched=0.02),
        ),
        756: _window_result(
            {
                ELASTIC_NET_FAMILY: _family_summary(
                    ELASTIC_NET_FAMILY,
                    qualified=False,
                    failure_stage="tail-capture-insufficient",
                    certificate=None,
                ),
                TAIL_LAMBDARANK_FAMILY: _family_summary(
                    TAIL_LAMBDARANK_FAMILY,
                    qualified=True,
                    failure_stage=None,
                    certificate=_certificate(stress=0.01, matched=0.09),
                ),
            },
            selected_family=TAIL_LAMBDARANK_FAMILY,
            certificate=_certificate(stress=0.01, matched=0.09),
        ),
    }
    calls = _install_candidate(monkeypatch, lambda lookback: responses[lookback])

    baseline = evaluate_economic_family_study(
        _data(1952), _cost_bound_request(), _settings(), registry=_REGISTRY_SENTINEL
    )
    flipped = evaluate_economic_family_study(
        _data(1952, holdout_flip=True),
        _cost_bound_request(),
        _settings(),
        registry=_REGISTRY_SENTINEL,
    )

    assert [call["lookback"] for call in calls] == [504, 756, 504, 756]
    for call in calls:
        assert call["fold_count"] == 3
        assert call["alpha"] == pytest.approx(0.05 / 240)
        assert call["request_alpha"] == pytest.approx(call["alpha"])
        assert call["resamples"] >= 4800
        assert call["request_resamples"] == call["resamples"]
        assert call["registry"] is _REGISTRY_SENTINEL
    assert baseline["study_complete"] is True
    assert baseline["artifact_published"] is False
    assert baseline["status"] == "RESEARCH_ONLY"
    assert baseline["selected_family"] == TAIL_LAMBDARANK_FAMILY
    assert baseline["recommended_lookback_sessions"] == 504
    # The winner owns strictly positive base/stress/matched lower bounds.
    winner = baseline["candidates"][0]["families"][TAIL_LAMBDARANK_FAMILY]
    certificate = winner["certificate"]
    assert certificate["base_lower_cagr"] > 0
    assert certificate["stress_lower_cagr"] > 0
    assert certificate["matched_lower_excess_cagr"] > 0
    assert winner["best_tail_excess_lower_bound"] > 0
    # Identical selection under mutated holdout labels proves the lock.
    assert flipped["selected_family"] == baseline["selected_family"]
    assert (
        flipped["recommended_lookback_sessions"]
        == baseline["recommended_lookback_sessions"]
    )
    assert flipped["next_action"] == baseline["next_action"]
    assert baseline["next_action"] == "rerun-qualified-family"


def _passing_negative_stress() -> dict[str, object]:
    return {
        "passed": True,
        "reasons": [],
        "base_lower_cagr": -0.001,
        "stress_lower_cagr": -0.002,
        "matched_lower_excess_cagr": 0.09,
        "mdd": 0.21,
        "observed_intervals": 720,
        "invested_intervals": 650,
        "filled_orders": 189,
    }


def test_economic_family_03_no_positive_candidate_selected(monkeypatch) -> None:
    """A negative-stress certificate never wins even with larger nominal values."""
    responses = {
        lookback: _window_result(
            {
                ELASTIC_NET_FAMILY: _family_summary(
                    ELASTIC_NET_FAMILY,
                    qualified=False,
                    failure_stage="execution-economics-insufficient",
                    certificate=_passing_negative_stress(),
                ),
                TAIL_LAMBDARANK_FAMILY: _family_summary(
                    TAIL_LAMBDARANK_FAMILY,
                    qualified=False,
                    failure_stage="execution-economics-insufficient",
                    certificate=_passing_negative_stress(),
                ),
            }
        )
        for lookback in (504, 756)
    }
    _install_candidate(monkeypatch, lambda lookback: responses[lookback])

    payload = evaluate_economic_family_study(
        _data(1952), _cost_bound_request(), _settings(), registry=_REGISTRY_SENTINEL
    )

    assert payload["selected_family"] is None
    assert payload["recommended_lookback_sessions"] is None
    assert payload["recommended_is_expanding"] is False
    assert payload["next_action"] == "execution-economics-insufficient"


def test_economic_family_04_all_cash_on_insufficient_economics(monkeypatch) -> None:
    """ECONOMIC_FAMILY_04_ALL_CASH_ON_INSUFFICIENT_ECONOMICS.

    When every candidate dies from non-positive oracle capacity, zero fills,
    or non-positive stress lower growth, the result stays all-cash research
    evidence: no selected family, no publication, one deterministic next
    action drawn from the declared failure vocabulary.
    """
    mixed = {
        504: _window_result(
            {
                ELASTIC_NET_FAMILY: _family_summary(
                    ELASTIC_NET_FAMILY,
                    qualified=False,
                    failure_stage="no-label-capacity",
                    certificate=_failing_certificate(),
                ),
                TAIL_LAMBDARANK_FAMILY: _family_summary(
                    TAIL_LAMBDARANK_FAMILY,
                    qualified=False,
                    failure_stage="execution-economics-insufficient",
                    certificate=_failing_certificate(),
                ),
            }
        ),
        756: _window_result(
            {
                ELASTIC_NET_FAMILY: _family_summary(
                    ELASTIC_NET_FAMILY,
                    qualified=False,
                    failure_stage="execution-economics-insufficient",
                    certificate=_failing_certificate(),
                ),
                TAIL_LAMBDARANK_FAMILY: _family_summary(
                    TAIL_LAMBDARANK_FAMILY,
                    qualified=False,
                    failure_stage="execution-economics-insufficient",
                    certificate=_failing_certificate(),
                ),
            }
        ),
    }
    _install_candidate(monkeypatch, lambda lookback: mixed[lookback])
    payload = evaluate_economic_family_study(
        _data(1952), _cost_bound_request(), _settings(), registry=_REGISTRY_SENTINEL
    )

    assert payload["status"] == "RESEARCH_ONLY"
    assert payload["artifact_published"] is False
    assert payload["selected_family"] is None
    assert payload["next_action"] in _FAILURE_ACTIONS

    all_oracle = {
        lookback: _window_result(
            {
                family: _family_summary(
                    family,
                    qualified=False,
                    failure_stage="no-label-capacity",
                    certificate=_failing_certificate(),
                )
                for family in (ELASTIC_NET_FAMILY, TAIL_LAMBDARANK_FAMILY)
            }
        )
        for lookback in (504, 756)
    }
    _install_candidate(monkeypatch, lambda lookback: all_oracle[lookback])
    payload = evaluate_economic_family_study(
        _data(1952), _cost_bound_request(), _settings(), registry=_REGISTRY_SENTINEL
    )
    assert payload["selected_family"] is None
    assert payload["next_action"] == "no-label-capacity"

    all_tail = {
        lookback: _window_result(
            {
                family: _family_summary(
                    family,
                    qualified=False,
                    failure_stage="tail-capture-insufficient",
                    certificate=_failing_certificate(),
                )
                for family in (ELASTIC_NET_FAMILY, TAIL_LAMBDARANK_FAMILY)
            }
        )
        for lookback in (504, 756)
    }
    _install_candidate(monkeypatch, lambda lookback: all_tail[lookback])
    payload = evaluate_economic_family_study(
        _data(1952), _cost_bound_request(), _settings(), registry=_REGISTRY_SENTINEL
    )
    assert payload["selected_family"] is None
    assert payload["next_action"] == "tail-capture-insufficient"


def test_economic_family_study_rejects_missing_cost_evidence() -> None:
    request = NetAlphaTrainingRequest(artifact_id="econ_nocost")
    with pytest.raises(ValueError, match="cost-evidence-required"):
        evaluate_economic_family_study(
            _data(1952),
            request,
            _settings(candidate_lookback_sessions=(1260,)),
            registry=_REGISTRY_SENTINEL,
        )



def _rawnet_fixture(
    n_sessions: int = 60,
    per_session: int = 30,
    seed: int = 7,
    horizon_sessions: int = 10,
):
    """Synthetic panel + labels with feature-correlated net utility."""
    import numpy as np
    from datetime import UTC, datetime, timedelta

    from src.core.datasets import DatasetManifest
    from src.core.instruments import AssetKind
    from legacy.stocks.ml.contracts import (
        ExecutionFrontierSettings,
        NetAlphaResearchData,
        NetAlphaTrainingRequest,
    )
    from legacy.stocks.research.folds import Fold

    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for s in range(n_sessions):
        session = start + timedelta(days=s)
        rows.extend(
            {
                "instrument_id": f"KRX:{t + 1:05d}",
                "session": session,
                "feature__a": float(rng.normal()),
                "feature__b": float(rng.normal()),
                "open": 10000.0 + t,
                "adtv_20d": 5.0e9,
                "volatility_20d": 0.03,
            }
            for t in range(per_session)
        )
    frame = pl.DataFrame(rows).sort(["session", "instrument_id"]).with_columns(
        pl.col("session").rank("dense").cast(pl.Int64).alias("session_index")
    )
    label_rows: list[dict[str, object]] = []
    for row in frame.iter_rows(named=True):
        utility = 0.002 * row["feature__a"] + float(rng.normal(scale=0.001))
        label_rows.append(
            {
                "instrument_id": row["instrument_id"],
                "session": row["session"],
                "net_alpha_target": float(rng.normal(scale=0.5)),
                "risk_residual": utility + 0.001,
                "reference_cost": 0.001,
                "label_available_time": row["session"]
                + timedelta(days=horizon_sessions),
            }
        )
    manifest = DatasetManifest(
        asset_kind=AssetKind.STOCK,
        schema_version="v1",
        schema_hash="h",
        provider_version="p",
        universe_policy_version="u",
        universe_policy_hash="u",
        feature_set="stock_net_alpha_v1",
        feature_set_hash="f",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=horizon_sessions,
        time_start=start,
        time_end=start + timedelta(days=n_sessions),
        generated_time=start + timedelta(days=n_sessions),
        row_count=len(rows),
    )
    data = NetAlphaResearchData(
        feature_frame=frame,
        labels_by_horizon={horizon_sessions: pl.DataFrame(label_rows)},
        manifest=manifest,
    )
    request = NetAlphaTrainingRequest(
        artifact_id="rawnet",
        candidate_horizon_sessions=(horizon_sessions,),
        execution_frontier=ExecutionFrontierSettings(
            candidate_horizon_sessions=(horizon_sessions,),
            candidate_rebalance_frequency_sessions=(5,),
            candidate_top_k=(12,),
        ),
        model_threads=1,
    )
    session_index = frame["session_index"].to_numpy()
    folds: list[Fold] = []
    for i in range(4):
        val_start = 40 + 4 * i
        val_rows = [
            int(pos)
            for pos in range(frame.height)
            if val_start <= session_index[pos] < val_start + 4
        ]
        train_rows = [
            int(pos) for pos in range(frame.height) if session_index[pos] < val_start - 3
        ]
        folds.append(
            Fold(
                train_mask=train_rows,
                validation_mask=val_rows,
                train_label_end=val_start - 3,
                validation_decision_start=val_start,
                segment_id=i,
                validation_sessions=(val_start, val_start + 3),
            )
        )
    return frame, folds, data, request, ("feature__a", "feature__b")


def test_rawnet_lgbm_02_oof_shape_and_determinism() -> None:
    """SCENARIO_RAWNET_LGBM_02_OOF_SHAPE_AND_DETERMINISM."""
    pre_holdout, folds, data, request, learner_columns = _rawnet_fixture()
    oof1, labeled1 = economic_research.fit_rawnet_lgbm_oof(
        pre_holdout, folds, data, request, learner_columns, 10
    )
    from legacy.stocks.ml.models import SCORE_COLUMN

    assert not oof1.is_empty()
    assert {"instrument_id", "session", SCORE_COLUMN}.issubset(oof1.columns)
    assert oof1[SCORE_COLUMN].null_count() == 0
    assert bool(oof1[SCORE_COLUMN].is_finite().all())
    expected_keys: set[tuple[str, object]] = set()
    for fold in folds:
        validation = pre_holdout[fold.validation_mask]
        for inst, sess in zip(
            validation[_LABEL_ID].to_list(),
            validation[_LABEL_SESSION].to_list(),
            strict=True,
        ):
            expected_keys.add((inst, sess))
    got_keys = {
        (inst, sess)
        for inst, sess in zip(
            oof1[_LABEL_ID].to_list(), oof1[_LABEL_SESSION].to_list(), strict=True
        )
    }
    assert got_keys == expected_keys
    assert oof1.height == len(expected_keys)
    for column in (
        "risk_residual",
        "reference_cost",
        "realized_net_return",
        "label_available_time",
    ):
        assert column in labeled1.columns
    assert labeled1.height == oof1.height
    oof2, labeled2 = economic_research.fit_rawnet_lgbm_oof(
        pre_holdout, folds, data, request, learner_columns, 10
    )
    assert oof1.equals(oof2)
    assert labeled1.equals(labeled2)


def test_rawnet_lgbm_03_seed_bagging_rank_average() -> None:
    """SCENARIO_RAWNET_LGBM_03_SEED_BAGGING_RANK_AVERAGE."""
    import polars.testing as pl_testing

    from legacy.stocks.ml.models import SCORE_COLUMN

    keys = {"instrument_id": ["A", "B", "C", "A", "B", "C"]}
    sessions = {"session": [1, 1, 1, 2, 2, 2]}
    seed_a = pl.DataFrame({**keys, **sessions, SCORE_COLUMN: [0.1, 0.5, 0.9, 0.2, 0.4, 0.6]})
    seed_b = pl.DataFrame({**keys, **sessions, SCORE_COLUMN: [0.9, 0.5, 0.1, 0.6, 0.4, 0.2]})
    seed_c = pl.DataFrame({**keys, **sessions, SCORE_COLUMN: [0.3, 0.6, 0.2, 0.8, 0.1, 0.5]})
    averaged = economic_research._rank_average_seed_scores((seed_a, seed_b, seed_c))
    # Session-1 ascending percentile ranks per seed:
    #   seed_a A/B/C = 0/.5/1 · seed_b = 1/.5/0 · seed_c(A=.3,B=.6,C=.2) = .5/1/0
    # Session-2: seed_a = 0/.5/1 · seed_b = 1/.5/0 · seed_c(.8,.1,.5) = 1/0/.5
    manual = pl.DataFrame(
        {
            "instrument_id": ["A", "B", "C", "A", "B", "C"],
            "session": [1, 1, 1, 2, 2, 2],
        }
    ).with_columns(
        pl.Series(
            SCORE_COLUMN,
            [
                (0.0 + 1.0 + 0.5) / 3.0,
                (0.5 + 0.5 + 1.0) / 3.0,
                (1.0 + 0.0 + 0.0) / 3.0,
                (0.0 + 1.0 + 1.0) / 3.0,
                (0.5 + 0.5 + 0.0) / 3.0,
                (1.0 + 0.0 + 0.5) / 3.0,
            ],
        )
    )
    averaged_sorted = averaged.sort(["instrument_id", "session"])
    manual_sorted = manual.sort(["instrument_id", "session"])
    pl_testing.assert_frame_equal(
        averaged_sorted.select(manual_sorted.columns), manual_sorted, check_dtypes=False
    )
    missing_row = seed_c.filter(pl.col("instrument_id") != "C")
    with pytest.raises(ValueError, match="seed"):
        economic_research._rank_average_seed_scores((seed_a, seed_b, missing_row))
    with pytest.raises(ValueError, match="seed"):
        economic_research._rank_average_seed_scores((seed_a,))

    pre_holdout, folds, data, request, learner_columns = _rawnet_fixture()
    from dataclasses import replace

    oof_base, _ = economic_research.fit_rawnet_lgbm_oof(
        pre_holdout, folds, data, request, learner_columns, 10
    )
    oof_shifted, _ = economic_research.fit_rawnet_lgbm_oof(
        pre_holdout,
        folds,
        data,
        replace(request, seed=request.seed + 100),
        learner_columns,
        10,
    )
    shifted_scores = oof_shifted.rename({SCORE_COLUMN: "shifted_score"})
    joined = oof_base.join(
        shifted_scores, on=["instrument_id", "session"], how="inner"
    )
    assert joined.height == oof_base.height
    assert (
        float((joined[SCORE_COLUMN] - joined["shifted_score"]).abs().max()) > 1e-9
    )


def test_economic_family_study_rejects_sub_annual_windows(monkeypatch) -> None:
    _install_candidate(monkeypatch, lambda lookback: pytest.fail("must not run"))
    with pytest.raises(ValueError, match="annualization_sessions"):
        evaluate_economic_family_study(
            _data(4000),
            _cost_bound_request(),
            _settings(
                candidate_lookback_sessions=(126,),
                common_min_train_sessions=126,
            ),
            registry=_REGISTRY_SENTINEL,
        )


def test_economic_family_study_folds_fail_closed(monkeypatch) -> None:
    calls = _install_candidate(monkeypatch, lambda lookback: pytest.fail("must not run"))
    payload = evaluate_economic_family_study(
        _data(1400), _cost_bound_request(), _settings(), registry=_REGISTRY_SENTINEL
    )
    assert calls == []
    assert payload["study_complete"] is False
    assert payload["next_action"] == "no-label-capacity"
    assert payload["rejection_reason_counts"].get(
        "insufficient-common-window-calendar"
    ) == 1


def _tail_route_fixture():
    from datetime import UTC, datetime, timedelta

    from src.core.datasets import DatasetManifest
    from src.core.instruments import AssetKind
    from legacy.stocks.ml.contracts import ExecutionFrontierSettings, NetAlphaResearchData, NetAlphaTrainingRequest
    from legacy.stocks.research.folds import Fold

    n_sessions = 60
    per_session = 8
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []  # noqa: PERF401
    for s in range(n_sessions):
        session = start + timedelta(days=s)
        # first 3 instruments are gross winners, rest are others
        for t in range(per_session):
            rows.append(  # noqa: PERF401
                {
                    "instrument_id": chr(ord("A") + t) if t < 8 else f"X{t}",
                    "session": session,
                    "feature__a": float(t),
                    "feature__b": float(s),
                    "open": 10000.0,
                    "adtv_20d": 1e9,
                    "volatility_20d": 0.02,
                }
            )
    frame = pl.DataFrame(rows).sort(["session", "instrument_id"]).with_columns(
        pl.col("session").rank("dense").cast(pl.Int64).alias("session_index")
    )
    # labels: A,B gross high, C residual high but gross low
    # We craft per-instrument gross and residual such that gross ranking A/B top, residual ranking B/C top
    # For each session, A: gross 0.06 residual -0.02, B: gross 0.04 residual 0.03, C: gross 0.01 residual 0.02, others: gross 0.0 residual 0.0
    label_rows = []
    gross_map = {"A": 0.06, "B": 0.04, "C": 0.01, "D": 0.0, "E": 0.0, "F": 0.0, "G": 0.0, "H": 0.0}
    resid_map = {"A": -0.02, "B": 0.03, "C": 0.02, "D": 0.0, "E": 0.0, "F": 0.0, "G": 0.0, "H": 0.0}
    for row in frame.iter_rows(named=True):
        iid = row["instrument_id"]
        label_rows.append(
            {
                "instrument_id": iid,
                "session": row["session"],
                "net_alpha_target": gross_map.get(iid, 0.0) - 0.0,
                "label_available_time": row["session"] + timedelta(days=10),
                "gross_return": gross_map.get(iid, 0.0),
                "risk_residual": resid_map.get(iid, 0.0),
                "reference_cost": 0.0,
            }
        )
    manifest = DatasetManifest(
        asset_kind=AssetKind.STOCK,
        schema_version="v1",
        schema_hash="h",
        provider_version="p",
        universe_policy_version="u",
        universe_policy_hash="u",
        feature_set="stock_net_alpha_v1",
        feature_set_hash="f",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=10,
        time_start=start,
        time_end=start + timedelta(days=n_sessions),
        generated_time=start,
        row_count=frame.height,
    )
    data = NetAlphaResearchData(feature_frame=frame, labels_by_horizon={10: pl.DataFrame(label_rows)}, manifest=manifest)
    request = NetAlphaTrainingRequest(
        artifact_id="tail_route",
        candidate_horizon_sessions=(10,),
        execution_frontier=ExecutionFrontierSettings(candidate_horizon_sessions=(10,), candidate_rebalance_frequency_sessions=(5,), candidate_top_k=(2,)),
        route_objective=SimpleNamespace(kind="unhedged_absolute"),
    )

    folds = []
    for i in range(4):
        val_start = 30 + i * 5
        train_mask = [int(idx) for idx in range(frame.height) if frame["session_index"][idx] < val_start - 2]
        val_mask = [int(idx) for idx in range(frame.height) if val_start <= frame["session_index"][idx] < val_start + 5]
        folds.append(Fold(train_mask=train_mask, validation_mask=val_mask, train_label_end=val_start - 2, validation_decision_start=val_start, segment_id=i))
    pre_holdout = frame
    learner_columns = ("feature__a", "feature__b")
    return pre_holdout, folds, data, request, learner_columns


def test_fit_tail_lambdarank_oof_unhedged_uses_route_relevance(monkeypatch) -> None:
    # Given
    pre_holdout, folds, data, request, learner_columns = _tail_route_fixture()
    captured: dict[str, object] = {}
    original = economic_research._lambda_rank_matrices
    def spy(frame, columns, top_k, *, route):
        captured["selected"] = set(build_route_tail_relevance(frame, route=route, top_k=top_k).filter(pl.col("relevance") == 1)["instrument_id"].to_list())
        return original(frame, columns, top_k, route=route)
    monkeypatch.setattr(economic_research, "_lambda_rank_matrices", spy)
    # When
    _oof, labeled = economic_research.fit_tail_lambdarank_oof(
        pre_holdout, folds, data, request, learner_columns, 10, 2
    )
    # Then
    assert captured["selected"] == {"A", "B"}
    assert "gross_return" in labeled.columns
    assert labeled["gross_return"].null_count() == 0


def test_settings_reject_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="ascending"):
        _settings(candidate_lookback_sessions=(756, 504))
    with pytest.raises(ValueError, match="final position"):
        _settings(candidate_lookback_sessions=(None, 504))
    with pytest.raises(ValueError, match="maximum"):
        _settings(common_min_train_sessions=503)
    with pytest.raises(ValueError, match="unknown model families"):
        EconomicFamilyStudySettings(model_families=("mystery_ranker",))
    with pytest.raises(ValueError, match="unique"):
        EconomicFamilyStudySettings(model_families=(ELASTIC_NET_FAMILY, ELASTIC_NET_FAMILY))


def _objective_frames(
    sessions: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    utilities = {"A": 0.90, "B": 0.80, "C": 0.05, "D": -0.02}
    scores = {"A": 9.0, "B": 8.0, "C": 7.0, "D": 6.0}
    label_rows = []
    scored_rows = []
    for session in range(1, sessions + 1):
        for name in utilities:
            label_rows.append((name, session, utilities[name], 0.0))
            scored_rows.append({_LABEL_ID: name, _LABEL_SESSION: session, "predicted_net_alpha": scores[name]})
    labels = pl.DataFrame(
        {
            _LABEL_ID: [r[0] for r in label_rows],
            _LABEL_SESSION: [r[1] for r in label_rows],
            "risk_residual": [r[2] for r in label_rows],
            "reference_cost": [r[3] for r in label_rows],
        }
    )
    return labels, pl.DataFrame(scored_rows)


def test_ECONOMIC_FAMILY_05_holdout_blind_integrity(monkeypatch) -> None:
    """ECONOMIC_FAMILY_05_HOLDOUT_BLIND_INTEGRITY.

    Runtime research validates only the rows its OOF joins actually produce:
    corrupting locked-holdout labels cannot change any tail-capture evidence,
    an invalid OOF join raises the typed integrity error and rejects the
    window candidate with invalid-oof-economic-utility, and the study maps
    that failure to next_action repair-label-integrity instead of
    no-label-capacity.
    """
    labels, scored = _objective_frames(sessions=12)
    oof_scored = scored.filter(pl.col(_LABEL_SESSION) <= 9)
    kwargs = {
        "top_k": 2,
        "bootstrap_alpha": 0.05,
        "bootstrap_resamples": 200,
        "seed": 5,
    }
    baseline = measure_tail_capture(oof_scored, labels, **kwargs)

    holdout_only_corruption = labels.with_columns(
        pl.when(pl.col(_LABEL_SESSION) >= 10)  # locked forward holdout
        .then(float("nan"))
        .otherwise(pl.col("risk_residual"))
        .alias("risk_residual")
    )
    blinded = measure_tail_capture(oof_scored, holdout_only_corruption, **kwargs)
    assert blinded == baseline

    joined_corruption = labels.with_columns(
        pl.when(pl.col(_LABEL_SESSION) == 3)  # inside the OOF join
        .then(None)
        .otherwise(pl.col("risk_residual"))
        .alias("risk_residual")
    )
    with pytest.raises(InvalidOofEconomicUtilityError):
        measure_tail_capture(oof_scored, joined_corruption, **kwargs)

    def explode(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise InvalidOofEconomicUtilityError("null oof utility")

    monkeypatch.setattr(economic_research, "_evaluate_window_body", explode)
    data = SimpleNamespace(
        feature_frame=pl.DataFrame({"session": list(range(400))}),
        labels_by_horizon={
            10: pl.DataFrame(
                {
                    _LABEL_ID: ["A"],
                    _LABEL_SESSION: [1],
                    "risk_residual": [0.1],
                    "reference_cost": [0.0],
                }
            )
        },
    )
    from legacy.stocks.ml.contracts import ExecutionFrontierSettings

    result = evaluate_economic_window_candidate(
        data,
        _cost_bound_request(
            candidate_horizon_sessions=(10,),
            execution_frontier=ExecutionFrontierSettings(candidate_horizon_sessions=(10,)),
        ),
        _settings(),
        registry=_REGISTRY_SENTINEL,
    )
    assert result["study_complete"] is False
    assert result["selected_family"] is None
    assert result["rejection_reason_counts"] == {"invalid-oof-economic-utility": 1}

    def integrity_rejection(lookback: object) -> dict[str, object]:
        return {
            "status": "RESEARCH_ONLY",
            "artifact_published": False,
            "study_complete": False,
            "fold_count": 0,
            "candidates_evaluated": 0,
            "selected_family": None,
            "certificate": None,
            "families": {},
            "rejection_reason_counts": {"invalid-oof-economic-utility": 1},
        }

    _install_candidate(monkeypatch, integrity_rejection)
    payload = evaluate_economic_family_study(
        _data(1952), _cost_bound_request(), _settings(), registry=_REGISTRY_SENTINEL
    )
    assert payload["selected_family"] is None
    assert payload["next_action"] == "repair-label-integrity"
