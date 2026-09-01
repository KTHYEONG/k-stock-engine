"""Universe rescope scenarios: SCENARIO_RESCOPE_KERNEL_*, SCENARIO_RESCOPE_SETTINGS_VALIDATION,
SCENARIO_RESCOPE_REQUEST_FINGERPRINT_PARITY."""
from __future__ import annotations

import json
from datetime import UTC, datetime

import polars as pl
import pytest

from legacy.stocks.ml.contracts import NetAlphaTrainingRequest, UniverseRescopeSettings
from legacy.stocks.ml.result_ledger import _project_request
from legacy.stocks.ml.universe_rescope import apply_universe_rescope


def _session_frame(
    sessions: int = 3,
    instruments: int = 20,
    *,
    with_market_cap: bool = True,
    with_trading_value: bool = False,
) -> pl.LazyFrame:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for s in range(sessions):
        session = base.replace(day=1 + s)
        for t in range(instruments):
            row: dict[str, object] = {
                "instrument_id": f"KRX:{t + 1:05d}",
                "session": session,
            }
            if with_market_cap:
                row["market_cap"] = float((t + 1) * 1_000_000_000)
            if with_trading_value:
                row["trading_value"] = float((t + 1) * 500_000_000)
            rows.append(row)
    return pl.LazyFrame(rows)


def _kept_keys(frame: pl.DataFrame) -> set[tuple[str, datetime]]:
    return set(
        zip(frame["instrument_id"].to_list(), frame["session"].to_list(), strict=True)
    )


def test_band_filter_keeps_top_quartile_per_session() -> None:
    """SCENARIO_RESCOPE_KERNEL_BAND_FILTER."""
    settings = UniverseRescopeSettings(market_cap_quantile_lo=0.75)
    lf = _session_frame()
    scoped, diagnostics = apply_universe_rescope(lf, settings)
    frame = scoped.collect()

    assert frame.height == 15
    assert diagnostics["kept_row_count"] == 15
    assert diagnostics["dropped_row_count"] == 45
    assert diagnostics["kept_row_fraction"] == pytest.approx(0.25)
    assert diagnostics["kept_session_instrument_p50"] == 5
    assert isinstance(diagnostics["fingerprint"], str)

    # Exactly the top-5 market_cap names survive in every session.
    for s in range(3):
        session = datetime(2024, 1, 1, tzinfo=UTC).replace(day=1 + s)
        kept = {
            int(code.split(":")[1])
            for code in frame.filter(pl.col("session") == session)[
                "instrument_id"
            ].to_list()
        }
        assert kept == {16, 17, 18, 19, 20}

    reordered, reordered_diag = apply_universe_rescope(
        _session_frame().collect().sample(fraction=1.0, shuffle=True, seed=7).lazy(),
        settings,
    )
    assert _kept_keys(reordered.collect()) == _kept_keys(frame)
    assert reordered_diag["fingerprint"] == diagnostics["fingerprint"]


def test_floor_and_adtv_ceiling_are_conjunctive() -> None:
    """SCENARIO_RESCOPE_KERNEL_FLOOR_AND_ADTV_CEILING."""
    lf = _session_frame(with_trading_value=True)

    floored, floor_diag = apply_universe_rescope(
        lf,
        UniverseRescopeSettings(
            market_cap_quantile_lo=0.0,
            min_market_cap_krw=15_000_000_000.0,
        ),
    )
    frame = floored.collect()
    # Band lo=0 keeps all; the 15T floor retains instruments 15..20 only.
    assert _kept_keys(frame) == _kept_keys(
        lf.filter(pl.col("market_cap") >= 15_000_000_000.0).collect()
    )
    assert frame.height == 18
    assert floor_diag["kept_row_fraction"] == pytest.approx(18 / 60)

    ceilinged, _ = apply_universe_rescope(
        lf,
        UniverseRescopeSettings(
            market_cap_quantile_lo=0.0,
            max_adtv_quantile=0.5,
        ),
    )
    ceiling_frame = ceilinged.collect()
    assert _kept_keys(ceiling_frame) == _kept_keys(
        lf.filter(pl.col("trading_value") <= 11 * 500_000_000).collect()
    )

    combined, _ = apply_universe_rescope(
        lf,
        UniverseRescopeSettings(
            market_cap_quantile_lo=0.75,
            min_market_cap_krw=21_000_000_000.0,
            max_adtv_quantile=0.5,
        ),
    )
    expected = (
        lf.with_columns(
            ((pl.col("market_cap").rank("ordinal").over("session") - 1)
             / pl.col("market_cap").len().over("session")).alias("__frac"),
            ((pl.col("trading_value").rank("ordinal").over("session") - 1)
             / pl.col("trading_value").len().over("session")).alias("__afrac"),
        )
        .filter(
            (pl.col("__frac") >= 0.75)
            & (pl.col("market_cap") >= 21_000_000_000.0)
            & (pl.col("__afrac") <= 0.5)
        )
        .drop("__frac", "__afrac")
        .collect()
    )
    assert _kept_keys(combined.collect()) == _kept_keys(expected)

    with pytest.raises(ValueError, match="trading_value"):
        apply_universe_rescope(
            _session_frame(),
            UniverseRescopeSettings(market_cap_quantile_lo=0.0, max_adtv_quantile=0.5),
        ).collect_schema()


def test_fail_closed_missing_column_and_none_passthrough() -> None:
    """SCENARIO_RESCOPE_FAIL_CLOSED_MISSING_COLUMN."""
    bare = _session_frame(with_market_cap=False)
    with pytest.raises(ValueError, match="market_cap"):
        apply_universe_rescope(bare, UniverseRescopeSettings()).collect_schema()

    passthrough_lf = _session_frame()
    returned, diagnostics = apply_universe_rescope(passthrough_lf, None)
    assert returned is passthrough_lf
    assert diagnostics == {}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"market_cap_quantile_lo": 1.0},
        {"market_cap_quantile_lo": -0.1},
        {"market_cap_quantile_hi": 0.0},
        {"market_cap_quantile_hi": 1.2},
        {"market_cap_quantile_lo": 0.8, "market_cap_quantile_hi": 0.8},
        {"market_cap_quantile_lo": 0.9, "market_cap_quantile_hi": 0.8},
        {"min_market_cap_krw": 0.0},
        {"min_market_cap_krw": -5.0},
        {"max_adtv_quantile": 0.0},
        {"max_adtv_quantile": 1.5},
        {"market_cap_quantile_lo": float("nan")},
        {"min_market_cap_krw": float("inf")},
    ],
)
def test_settings_validation_rejects_invalid_inputs(kwargs: dict[str, object]) -> None:
    """SCENARIO_RESCOPE_SETTINGS_VALIDATION."""
    with pytest.raises(ValueError, match=r"must be|quantile"):
        UniverseRescopeSettings(**kwargs)  # type: ignore[arg-type]


def test_fingerprint_is_stable_and_field_sensitive() -> None:
    """SCENARIO_RESCOPE_SETTINGS_VALIDATION."""
    base = UniverseRescopeSettings()
    twin = UniverseRescopeSettings()
    other = UniverseRescopeSettings(market_cap_quantile_lo=0.6)
    assert base.fingerprint == twin.fingerprint
    assert base.fingerprint != other.fingerprint


def test_project_request_fingerprint_parity() -> None:
    """SCENARIO_RESCOPE_REQUEST_FINGERPRINT_PARITY."""
    off_a = _project_request(NetAlphaTrainingRequest(artifact_id="rescope_off_a"))
    off_b = _project_request(NetAlphaTrainingRequest(artifact_id="rescope_off_b"))
    assert json.dumps(off_a, sort_keys=True) == json.dumps(off_b, sort_keys=True)
    assert "universe_rescope" not in off_a

    on = _project_request(
        NetAlphaTrainingRequest(
            artifact_id="rescope_on",
            universe_rescope=UniverseRescopeSettings(),
        )
    )
    block = on["universe_rescope"]
    assert isinstance(block, dict)
    assert block["market_cap_quantile_lo"] == 0.75
    assert block["market_cap_quantile_hi"] == 1.0
    assert block["min_market_cap_krw"] is None
    assert block["max_adtv_quantile"] is None
    assert isinstance(block["fingerprint"], str)
    assert on["request_fingerprint"] != off_a["request_fingerprint"]
