"""Walk-forward scorer causality, ensemble and contract invariants."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

import src.research.model as model_mod
from src.data.research_protocol import WindowAuthorization, WindowError
from src.research.model import ScoreMatrix, ScorerConfig, _ensemble_average, walk_forward_scores
from src.research.panel import FEATURE_NAMES, FeaturePanel, feature_names_for

N_SESS = 1150
N_INST = 6


def _sessions(n: int = N_SESS) -> tuple[date, ...]:
    start = date(2016, 1, 1)
    return tuple(start + timedelta(days=i) for i in range(n))


def _synthetic_panel(n_s: int = N_SESS, n_n: int = N_INST, seed: int = 3) -> FeaturePanel:
    rng = np.random.default_rng(seed)
    feats = {
        name: np.ascontiguousarray(rng.standard_normal((n_s, n_n)), dtype=np.float32)
        for name in FEATURE_NAMES
    }
    labels = {
        h: np.ascontiguousarray(rng.standard_normal((n_s, n_n)) * 0.02, dtype=np.float32)
        for h in (5, 10, 21)
    }
    return FeaturePanel(features=feats, labels=labels, last_row=n_s - 1)


def _tiny_config(**overrides: object) -> ScorerConfig:
    kwargs: dict[str, object] = {
        "horizons": (5, 10),
        "num_boost_round": 5,
        "min_data_in_leaf": 2,
        "min_train_rows": 30,
        "min_cross_section": 3,
        "winsor_low_pct": 1.0,
        "winsor_high_pct": 99.0,
        "first_test_year": 2018,
        "num_threads": 1,
    }
    kwargs.update(overrides)
    return ScorerConfig(**kwargs)  # type: ignore[arg-type]


def _auth(sess: tuple[date, ...], end: date | None = None) -> WindowAuthorization:
    return WindowAuthorization(start=sess[0], end=end or sess[-1])


def _universe(n_s: int = N_SESS, n_n: int = N_INST) -> np.ndarray:
    return np.ones((n_s, n_n), dtype=bool)


def _equal_with_nan(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    return bool(
        a.shape == b.shape
        and np.array_equal(np.isnan(a), np.isnan(b))
        and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])
    )


class _StubBooster:
    def __init__(self, scale: float = 0.0, offset: float = 0.0) -> None:
        self._scale = scale
        self._offset = offset

    def predict(self, data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float64)
        if self._scale == 0.0 and self._offset == 0.0:
            return np.zeros(arr.shape[0], dtype=np.float64)
        return np.asarray(arr[:, 0] * self._scale + self._offset, dtype=np.float64)


def test_year_scores_ignore_the_future() -> None:
    sess = _sessions()
    panel = _synthetic_panel()
    cfg = _tiny_config()
    base = walk_forward_scores(panel, _universe(), sess, cfg, test_years=(2018, 2019), authorization=_auth(sess))
    first_2019 = next(i for i, day in enumerate(sess) if day.year == 2019)
    noisy_feats = dict(panel.features)
    rng = np.random.default_rng(99)
    for name, arr in panel.features.items():
        blown = np.asarray(arr, dtype=np.float64).copy()
        blown[first_2019:, :] = rng.standard_normal((N_SESS - first_2019, N_INST)) * 999.0
        noisy_feats[name] = np.ascontiguousarray(blown, dtype=np.float32)
    noisy_labels = dict(panel.labels)
    for h, arr in panel.labels.items():
        blown = np.asarray(arr, dtype=np.float64).copy()
        blown[first_2019:, :] = rng.standard_normal((N_SESS - first_2019, N_INST)) * 999.0
        noisy_labels[h] = np.ascontiguousarray(blown, dtype=np.float32)
    corrupted = FeaturePanel(features=noisy_feats, labels=noisy_labels, last_row=panel.last_row)
    other = walk_forward_scores(
        corrupted, _universe(), sess, cfg, test_years=(2018, 2019), authorization=_auth(sess)
    )
    in_2018 = np.array([day.year == 2018 for day in sess])
    assert _equal_with_nan(
        np.asarray(base.scores)[in_2018, :], np.asarray(other.scores)[in_2018, :]
    )


def test_purge_excludes_overlapping_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    sess = _sessions()
    panel = _synthetic_panel()
    cfg = _tiny_config(horizons=(5,))
    captured: list[np.ndarray] = []

    def _fake(tx: np.ndarray, ty: np.ndarray, params: object, rounds: int, seed: int, rows: object = None) -> object:
        captured.append(np.asarray(rows).copy())
        return _StubBooster()

    monkeypatch.setattr(model_mod, "_train_booster", _fake)
    walk_forward_scores(panel, _universe(), sess, cfg, test_years=(2018,), authorization=_auth(sess))
    assert len(captured) == 1
    first = next(i for i, day in enumerate(sess) if day.year == 2018)
    assert bool(np.all(np.asarray(captured[0]) + 5 + cfg.purge_extra_sessions < first))


def test_ensemble_uses_all_horizons(monkeypatch: pytest.MonkeyPatch) -> None:
    sess = _sessions()
    panel = _synthetic_panel()
    cfg = _tiny_config(horizons=(5, 10))
    calls = 0

    def _fake(tx: np.ndarray, ty: np.ndarray, params: object, rounds: int, seed: int, rows: object = None) -> object:
        nonlocal calls
        calls += 1
        return _StubBooster()

    monkeypatch.setattr(model_mod, "_train_booster", _fake)
    walk_forward_scores(panel, _universe(), sess, cfg, test_years=(2018,), authorization=_auth(sess))
    assert calls == 2


def test_deterministic_rerun() -> None:
    sess = _sessions()
    panel = _synthetic_panel()
    uni = _universe()
    first = walk_forward_scores(
        panel, uni, sess, _tiny_config(num_threads=1), test_years=(2018,), authorization=_auth(sess)
    )
    second = walk_forward_scores(
        panel, uni, sess, _tiny_config(num_threads=2), test_years=(2018,), authorization=_auth(sess)
    )
    assert _equal_with_nan(np.asarray(first.scores), np.asarray(second.scores))


def test_universe_and_year_masking() -> None:
    sess = _sessions()
    panel = _synthetic_panel()
    uni = _universe()
    uni[:, 0] = False
    out = walk_forward_scores(
        panel, uni, sess, _tiny_config(), test_years=(2018, 2019), authorization=_auth(sess)
    )
    assert isinstance(out, ScoreMatrix)
    scores = np.asarray(out.scores)
    assert bool(np.all(np.isnan(scores[:, 0])))
    wanted = np.array([day.year in (2018, 2019) for day in sess])
    assert bool(np.all(np.isnan(scores[~wanted, :])))
    assert bool(np.isfinite(scores[wanted, 1:]).any())
    assert out.test_years == (2018, 2019)
    assert out.config_hash == _tiny_config().config_hash
    assert out.last_row == panel.last_row


def test_thin_cross_section_yields_nan() -> None:
    sess = _sessions()
    panel = _synthetic_panel()
    uni = _universe()
    thin = next(i for i, day in enumerate(sess) if day.year == 2018) + 10
    uni[thin, :] = False
    uni[thin, :2] = True
    uni[thin + 5, :] = False
    out = walk_forward_scores(
        panel, uni, sess, _tiny_config(), test_years=(2018, 2019), authorization=_auth(sess)
    )
    scores = np.asarray(out.scores)
    assert bool(np.all(np.isnan(scores[thin, :])))
    assert bool(np.all(np.isnan(scores[thin + 5, :])))
    assert bool(np.isfinite(scores[thin + 1, 2:]).any())


def test_fold_without_enough_data_fails_closed() -> None:
    sess = _sessions()
    with pytest.raises(ValueError, match="min_train_rows"):
        walk_forward_scores(
            _synthetic_panel(),
            _universe(),
            sess,
            _tiny_config(min_train_rows=10**9),
            test_years=(2018,),
            authorization=_auth(sess),
        )


def test_sealed_segment_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    sess = _sessions()
    calls = 0

    def _fake(tx: np.ndarray, ty: np.ndarray, params: object, rounds: int, seed: int, rows: object = None) -> object:
        nonlocal calls
        calls += 1
        return _StubBooster()

    monkeypatch.setattr(model_mod, "_train_booster", _fake)
    with pytest.raises(WindowError, match="ends after"):
        walk_forward_scores(
            _synthetic_panel(),
            _universe(),
            sess,
            _tiny_config(),
            test_years=(2018,),
            authorization=_auth(sess, end=date(2018, 6, 30)),
        )
    assert calls == 0


def test_ensemble_is_horizon_normalized_mean() -> None:
    slow = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0], [np.nan, 1.0, 2.0, 3.0]])
    fast = slow * 100.0 + 50.0
    got = np.asarray(_ensemble_average([slow, fast], min_count=2))
    assert got.shape == slow.shape

    def _z(row: np.ndarray) -> np.ndarray:
        mask = np.isfinite(row)
        vals = row[mask]
        return (vals - vals.mean()) / vals.std()

    assert got[0] == pytest.approx(_z(slow[0]))
    assert bool(np.all(np.isnan(got[1])))
    expected_last = np.concatenate([[np.nan], _z(np.array([1.0, 2.0, 3.0]))])
    assert bool(np.array_equal(np.isnan(got[2]), np.isnan(expected_last)))
    assert got[2][1:] == pytest.approx(expected_last[1:])


def test_ensemble_scale_invariance_through_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    sess = _sessions()
    panel = _synthetic_panel()
    states = {"calls": 0}

    def _fake(tx: np.ndarray, ty: np.ndarray, params: object, rounds: int, seed: int, rows: object = None) -> object:
        states["calls"] += 1
        scale = 1.0 if states["calls"] % 2 else 1000.0
        return _StubBooster(scale=scale, offset=50.0)

    monkeypatch.setattr(model_mod, "_train_booster", _fake)
    out = walk_forward_scores(
        panel, _universe(), sess, _tiny_config(), test_years=(2018,), authorization=_auth(sess)
    )
    assert states["calls"] == 2
    assert bool(np.isfinite(np.asarray(out.scores)).any())


def test_ascending_years_only() -> None:
    sess = _sessions()
    panel = _synthetic_panel()
    uni = _universe()
    cfg = _tiny_config()
    with pytest.raises(ValueError, match="ascending"):
        walk_forward_scores(panel, uni, sess, cfg, test_years=(2019, 2018), authorization=_auth(sess))
    with pytest.raises(ValueError, match="first_test_year"):
        walk_forward_scores(panel, uni, sess, cfg, test_years=(2017,), authorization=_auth(sess))
    with pytest.raises(ValueError, match="non-empty"):
        walk_forward_scores(panel, uni, sess, cfg, test_years=(), authorization=_auth(sess))


def test_config_validators_reject_bad_values() -> None:
    with pytest.raises(ValueError, match="horizons"):
        ScorerConfig(horizons=())
    with pytest.raises(ValueError, match="horizons"):
        ScorerConfig(horizons=(0,))
    with pytest.raises(ValueError, match="winsor"):
        ScorerConfig(winsor_low_pct=-1.0)
    with pytest.raises(ValueError, match="winsor"):
        ScorerConfig(winsor_high_pct=101.0)


def test_shape_and_year_guards_fail_closed() -> None:
    sess = _sessions()
    panel = _synthetic_panel()
    uni = _universe()
    cfg = _tiny_config()
    auth = _auth(sess)
    bad_feats = dict(panel.features)
    bad_feats["ret_1d"] = np.zeros((10, N_INST), dtype=np.float32)
    with pytest.raises(ValueError, match="mismatched"):
        walk_forward_scores(
            FeaturePanel(features=bad_feats, labels=panel.labels, last_row=panel.last_row),
            uni,
            sess,
            cfg,
            test_years=(2018,),
            authorization=auth,
        )
    with pytest.raises(ValueError, match="universe shape"):
        walk_forward_scores(
            panel, np.ones((5, N_INST), dtype=bool), sess, cfg, test_years=(2018,), authorization=auth
        )
    with pytest.raises(ValueError, match="last_row"):
        walk_forward_scores(panel, uni, sess[:100], cfg, test_years=(2018,), authorization=auth)
    with pytest.raises(ValueError, match="no sessions"):
        walk_forward_scores(panel, uni, sess, cfg, test_years=(2030,), authorization=auth)
    bad_labels = dict(panel.labels)
    bad_labels[5] = np.zeros((10, N_INST), dtype=np.float32)
    with pytest.raises(ValueError, match="wrong shape"):
        walk_forward_scores(
            FeaturePanel(features=panel.features, labels=bad_labels, last_row=panel.last_row),
            uni,
            sess,
            cfg,
            test_years=(2018,),
            authorization=auth,
        )


def test_rank_and_target_helpers_edge_cases() -> None:
    mats = [np.ones((4, 3)), np.ones((4, 3)) * 2.0]
    empty_uni = np.zeros((4, 3), dtype=bool)
    x, r, i = model_mod._rank_stack(mats, empty_uni, [0, 1, 2, 3], 2)
    assert x.shape == (0, 2)
    assert r.size == 0
    assert i.size == 0
    t = model_mod._winsor_z_by_date(
        np.ones((4, 3)), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), 1.0, 99.0, 2
    )
    assert t.size == 0
    lab = np.full((4, 3), np.nan)
    lab[1, :] = 0.05
    rows = np.array([0, 0, 1, 1, 1])
    cols = np.array([0, 1, 0, 1, 2])
    thin = model_mod._winsor_z_by_date(lab, rows, cols, 1.0, 99.0, 2)
    assert bool(np.all(np.isnan(thin)))


def test_weekly_rows_reject_bad_window() -> None:
    sess = _sessions(30)
    assert model_mod._weekly_rows(sess, lo=0, hi=30) == model_mod._weekly_rows(sess, lo=0, hi=30)
    with pytest.raises(ValueError, match="within"):
        model_mod._weekly_rows(sess, lo=-1, hi=10)
    with pytest.raises(ValueError, match="within"):
        model_mod._weekly_rows(sess, lo=0, hi=31)


def _scores_digest(scores: np.ndarray) -> str:
    import hashlib

    arr = np.asarray(scores)
    canon = np.where(np.isnan(arr), 0.0, arr)
    return hashlib.sha256(np.ascontiguousarray(canon).tobytes()).hexdigest()


def test_scores_bitwise_golden() -> None:
    sess = _sessions()
    out = walk_forward_scores(
        _synthetic_panel(),
        _universe(),
        sess,
        _tiny_config(),
        test_years=(2018, 2019),
        authorization=_auth(sess),
    )
    assert _scores_digest(np.asarray(out.scores)) == (
        "09ac305158002f2bc78c1df638d92cae1374f0f318ecd3fdd8e35b8bf7a79bd3"
    )


def test_no_full_float64_feature_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    import gc
    import tracemalloc

    n_s, n_n = 300, 200
    start = date(2017, 3, 8)
    sess = tuple(start + timedelta(days=i) for i in range(n_s))
    rng = np.random.default_rng(3)
    feats = {
        name: np.ascontiguousarray(rng.standard_normal((n_s, n_n)), dtype=np.float32)
        for name in FEATURE_NAMES
    }
    labels = {5: np.ascontiguousarray(rng.standard_normal((n_s, n_n)) * 0.02, dtype=np.float32)}
    panel = FeaturePanel(features=feats, labels=labels, last_row=n_s - 1)
    uni = np.zeros((n_s, n_n), dtype=bool)
    uni[:14, :] = True
    uni[-1, :] = True
    monkeypatch.setattr(model_mod, "_train_booster", lambda *a, **k: _StubBooster())
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    walk_forward_scores(
        panel,
        uni,
        sess,
        _tiny_config(horizons=(5,)),
        test_years=(2018,),
        authorization=_auth(sess),
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak - before < 0.25 * 62 * n_s * n_n * 8


def test_default_feature_set_keeps_config_hash() -> None:
    import json

    default = ScorerConfig()
    explicit = ScorerConfig(feature_set="full62")
    assert default.canonical_json() == explicit.canonical_json()
    assert "feature_set" not in json.loads(default.canonical_json())
    legacy_payload = default.model_dump(mode="json", exclude={"feature_set"})
    assert default.canonical_json() == json.dumps(legacy_payload, sort_keys=True, separators=(",", ":"))
    assert default.config_hash == explicit.config_hash


def test_non_default_feature_set_changes_identity() -> None:
    import json

    other = ScorerConfig(feature_set="dedup52_v1")
    assert json.loads(other.canonical_json())["feature_set"] == "dedup52_v1"
    assert other.config_hash != ScorerConfig().config_hash


def test_unknown_feature_set_rejected() -> None:
    with pytest.raises(ValueError, match="feature_set"):
        ScorerConfig(feature_set="x")


def test_scorer_uses_only_the_configured_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    sess = _sessions()
    panel = _synthetic_panel()
    widths: list[int] = []
    names = feature_names_for("dedup52_v1")
    expected = np.stack(
        [
            ((values[:, :, None] > values[:, None, :]).sum(axis=2) + 0.5) / N_INST
            for name in names
            for values in (panel.features[name],)
        ],
        axis=2,
    )
    prediction_rows = [r for r, day in enumerate(sess) if day.year == 2018]
    prediction_widths: list[int] = []

    class _OrderedBooster(_StubBooster):
        def predict(self, data: np.ndarray) -> np.ndarray:
            prediction_widths.append(data.shape[1])
            np.testing.assert_array_equal(data, expected[prediction_rows].reshape(-1, 52))
            return super().predict(data)

    def _fake(tx: np.ndarray, ty: np.ndarray, params: object, rounds: int, seed: int, rows: object = None) -> object:
        widths.append(int(np.asarray(tx).shape[1]))
        train_rows = np.asarray(rows, dtype=int)
        instruments = np.tile(np.arange(N_INST), len(train_rows) // N_INST)
        np.testing.assert_array_equal(tx, expected[train_rows, instruments])
        return _OrderedBooster(scale=1.0)

    monkeypatch.setattr(model_mod, "_train_booster", _fake)
    walk_forward_scores(
        panel,
        _universe(),
        sess,
        _tiny_config(feature_set="dedup52_v1"),
        test_years=(2018,),
        authorization=_auth(sess),
    )
    assert widths
    assert all(w == 52 for w in widths)
    assert prediction_widths == widths


def test_unused_features_cannot_influence_scores() -> None:
    sess = _sessions()
    panel = _synthetic_panel()
    base = walk_forward_scores(
        panel,
        _universe(),
        sess,
        _tiny_config(feature_set="dedup52_v1"),
        test_years=(2018,),
        authorization=_auth(sess),
    )
    assert np.isfinite(base.scores).any()
    noisy_feats = dict(panel.features)
    rng = np.random.default_rng(1234)
    blown = rng.standard_normal((N_SESS, N_INST))
    noisy_feats["vol60"] = np.ascontiguousarray(blown, dtype=np.float32)
    noisy = FeaturePanel(features=noisy_feats, labels=panel.labels, last_row=panel.last_row)
    other = walk_forward_scores(
        noisy,
        _universe(),
        sess,
        _tiny_config(feature_set="dedup52_v1"),
        test_years=(2018,),
        authorization=_auth(sess),
    )
    assert _equal_with_nan(np.asarray(base.scores), np.asarray(other.scores))


def test_missing_configured_feature_fails_closed() -> None:
    sess = _sessions()
    panel = _synthetic_panel()
    feats = dict(panel.features)
    del feats["ivol60"]
    pruned = FeaturePanel(features=feats, labels=panel.labels, last_row=panel.last_row)
    with pytest.raises(KeyError, match="ivol60"):
        walk_forward_scores(
            pruned,
            _universe(),
            sess,
            _tiny_config(feature_set="dedup52_v1"),
            test_years=(2018,),
            authorization=_auth(sess),
        )


def test_prediction_design_built_once_per_year(monkeypatch: pytest.MonkeyPatch) -> None:
    import weakref

    sess = _sessions()
    panel = _synthetic_panel()
    cfg = _tiny_config(horizons=(5, 10, 21))
    calls = 0
    orig = model_mod._rank_stack
    previous_prediction: weakref.ReferenceType[np.ndarray] | None = None

    def _counting(*args: object, **kwargs: object) -> object:
        nonlocal calls, previous_prediction
        if previous_prediction is not None:
            assert previous_prediction() is None
        calls += 1
        result = orig(*args, **kwargs)  # type: ignore[arg-type]
        if calls > 1:
            previous_prediction = weakref.ref(result[0])
        return result

    monkeypatch.setattr(model_mod, "_rank_stack", _counting)
    monkeypatch.setattr(model_mod, "_train_booster", lambda *a, **k: _StubBooster())
    walk_forward_scores(panel, _universe(), sess, cfg, test_years=(2018, 2019), authorization=_auth(sess))
    assert calls == 1 + 2


def test_booster_thread_count_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    sess = _sessions()
    panel = _synthetic_panel()
    cfg = _tiny_config(horizons=(5, 10), num_threads=4)
    seen: list[object] = []

    def _fake(tx: np.ndarray, ty: np.ndarray, params: object, rounds: int, seed: int, rows: object = None) -> object:
        seen.append(params["num_threads"])  # type: ignore[index]
        return _StubBooster()

    monkeypatch.setattr(model_mod, "_train_booster", _fake)
    walk_forward_scores(panel, _universe(), sess, cfg, test_years=(2018, 2019), authorization=_auth(sess))
    assert len(seen) == 2 * 2
    assert all(v == 4 for v in seen)


def test_fold_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    sess = _sessions()
    panel = _synthetic_panel()
    cfg = _tiny_config(horizons=(5, 10))
    failure = RuntimeError("boom-fold")
    failing_seed = (cfg.seed + 10 * 100003 + 2018 * 101) % 2147483647
    predictions: list[int] = []

    class _RecordingBooster(_StubBooster):
        def predict(self, data: np.ndarray) -> np.ndarray:
            predictions.append(len(data))
            return super().predict(data)

    def _fake(tx: np.ndarray, ty: np.ndarray, params: object, rounds: int, seed: int, rows: object = None) -> object:
        if seed == failing_seed:
            raise failure
        return _RecordingBooster()

    monkeypatch.setattr(model_mod, "_train_booster", _fake)
    with pytest.raises(RuntimeError, match="boom-fold") as caught:
        walk_forward_scores(panel, _universe(), sess, cfg, test_years=(2018,), authorization=_auth(sess))
    assert caught.value is failure
    assert not predictions


def test_fold_workers_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    assert model_mod._fold_workers(8) == 1
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert model_mod._fold_workers(4) == 2
    assert model_mod._fold_workers(8) == 1
    assert model_mod._fold_workers(16) == 1
    assert model_mod._fold_workers(0) == 1
    assert model_mod._fold_workers(-1) == 1
    monkeypatch.setattr(os, "cpu_count", lambda: None)  # type: ignore[return-value]
    assert model_mod._fold_workers(4) == 1


def test_concurrent_equals_sequential(monkeypatch: pytest.MonkeyPatch) -> None:
    sess = _sessions()
    panel = _synthetic_panel()
    uni = _universe()
    cfg = _tiny_config(horizons=(5, 10, 21), num_threads=4)
    monkeypatch.setattr(model_mod, "_fold_workers", lambda n: 1)
    seq = walk_forward_scores(panel, uni, sess, cfg, test_years=(2018, 2019), authorization=_auth(sess))
    monkeypatch.setattr(model_mod, "_fold_workers", lambda n: 3)
    conc = walk_forward_scores(panel, uni, sess, cfg, test_years=(2018, 2019), authorization=_auth(sess))
    assert np.isfinite(seq.scores).any()
    assert seq.scores.tobytes() == conc.scores.tobytes()


def test_folds_overlap_and_predictions_wait_in_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    from threading import Barrier, get_ident

    sess = _sessions()
    cfg = _tiny_config(horizons=(5, 10, 21))
    caller = get_ident()
    barrier = Barrier(3, timeout=10)
    completed: list[int] = []

    class _CallerBooster(_StubBooster):
        def predict(self, data: np.ndarray) -> np.ndarray:
            assert get_ident() == caller
            assert len(completed) == 3
            return super().predict(data)

    def _fake(tx: np.ndarray, ty: np.ndarray, params: object, rounds: int, seed: int, rows: object = None) -> object:
        assert get_ident() != caller
        barrier.wait()
        completed.append(seed)
        return _CallerBooster(scale=1.0)

    monkeypatch.setattr(model_mod, "_fold_workers", lambda n: 3)
    monkeypatch.setattr(model_mod, "_train_booster", _fake)
    out = walk_forward_scores(
        _synthetic_panel(), _universe(), sess, cfg, test_years=(2018,), authorization=_auth(sess)
    )
    assert np.isfinite(out.scores).any()


def test_all_year_folds_validated_before_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    sess = _sessions()
    panel = _synthetic_panel()
    labels = dict(panel.labels)
    labels[21] = np.full_like(labels[21], np.nan)
    invalid = FeaturePanel(features=panel.features, labels=labels, last_row=panel.last_row)
    calls: list[int] = []

    def _fake(tx: np.ndarray, ty: np.ndarray, params: object, rounds: int, seed: int, rows: object = None) -> object:
        calls.append(seed)
        return _StubBooster()

    monkeypatch.setattr(model_mod, "_train_booster", _fake)
    with pytest.raises(ValueError, match="fold year=2018 horizon=21 has 0 rows below min_train_rows"):
        walk_forward_scores(
            invalid, _universe(), sess, _tiny_config(horizons=(5, 10, 21)),
            test_years=(2018,), authorization=_auth(sess)
        )
    assert not calls
