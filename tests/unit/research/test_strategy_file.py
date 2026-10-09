"""Strategy file ``extends`` resolution and market-constant filling invariants."""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any

import pytest

FUTURES = Path("config/market/futures.toml")

_POLICY = {
    "family": "ml_sleeve_hedge",
    "n": 20,
    "keep_rank_multiple": 3.0,
    "rebalance_every_sessions": 5,
    "trend_min_ret21": 0.0,
    "min_units_per_slot": 3,
    "universe": {"min_adtv20_krw": 500000000, "min_price_krw": 1000},
}
_SCORER = {
    "horizons": [5, 10, 21],
    "num_boost_round": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 800,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 50.0,
    "seed": 11,
    "num_threads": 8,
    "min_cross_section": 50,
    "winsor_low_pct": 1.0,
    "winsor_high_pct": 99.0,
    "purge_extra_sessions": 2,
    "min_train_rows": 20000,
    "first_test_year": 2018,
}
_BOOK = {"sleeves": 5, "stock_capital_fraction": 0.75}
_HEDGE = {
    "hedge_ratio": 1.0,
    "beta_window_sessions": 120,
    "beta_min_sessions": 40,
    "beta_cap": 2.0,
    "rebalance_every_sessions": 5,
    "use_futures": True,
    "contract_multiplier_krw": 10000,
    "initial_margin_rate": 0.2175,
    "margin_buffer_rate": 0.10,
    "margin_topup_trigger_fraction": 0.75,
    "futures_cost_rate": 0.0003,
    "inverse_cost_rate": 0.0007,
    "resize_sell_cost_rate": 0.0025,
    "resize_buy_cost_rate": 0.0005,
    "futures_tax_rate": 0.11,
    "futures_annual_deduction_krw": 2500000,
    "inverse_tax_rate": 0.154,
}
_TREND = {
    "ma_sessions": 100,
    "long_fraction": 1.0,
    "short_fraction": 0.5,
    "rebalance_every_sessions": 1,
    "contract_multiplier_krw": 50000,
    "initial_margin_rate": 0.2175,
    "margin_buffer_rate": 0.10,
    "margin_topup_trigger_fraction": 0.75,
    "futures_cost_rate": 0.0003,
    "futures_tax_rate": 0.11,
    "futures_annual_deduction_krw": 2500000,
}
_REGIME = {
    "tsmom_horizons": [21, 63, 126, 252],
    "target_vol": 0.10,
    "max_fraction": 1.5,
    "vol_window_sessions": 60,
    "rebalance_every_sessions": 1,
    "contract_multiplier_krw": 10000,
    "initial_margin_rate": 0.2175,
    "margin_buffer_rate": 0.10,
    "margin_topup_trigger_fraction": 0.75,
    "futures_cost_rate": 0.0003,
    "futures_tax_rate": 0.11,
    "futures_annual_deduction_krw": 2500000,
}
_HEDGE_SUPPLIED = {
    "contract_multiplier_krw",
    "initial_margin_rate",
    "futures_cost_rate",
    "inverse_cost_rate",
    "resize_sell_cost_rate",
    "resize_buy_cost_rate",
    "futures_tax_rate",
    "futures_annual_deduction_krw",
    "inverse_tax_rate",
}
_TREND_SUPPLIED = {
    "futures_tax_rate",
    "futures_annual_deduction_krw",
    "futures_cost_rate",
    "contract_multiplier_krw",
    "initial_margin_rate",
}


def _emit(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        return f"[{', '.join(_emit(item) for item in value)}]"
    return str(value)


def _dump(sections: dict[str, Any]) -> str:
    lines: list[str] = []
    for name, block in sections.items():
        if isinstance(block, dict):
            lines.append(f"[{name}]")
            for key, value in block.items():
                if isinstance(value, dict):
                    lines.append(f"[{name}.{key}]")
                    for sub, item in value.items():
                        lines.append(f"{sub} = {_emit(item)}")
                else:
                    lines.append(f"{key} = {_emit(value)}")
        else:
            lines.append(f"{name} = {_emit(block)}")
    return "\n".join([*lines, ""])


def _write(
    tmp_path: Path,
    name: str,
    *,
    tables: tuple[str, ...] = ("hedge",),
    strip: bool = False,
    hedge_ratio: float = 1.0,
    extra: str = "",
) -> Path:
    supplied = {"hedge": _HEDGE_SUPPLIED, "trend_overlay": _TREND_SUPPLIED, "regime_hedge": _TREND_SUPPLIED}
    blocks: dict[str, Any] = {"policy": dict(_POLICY), "scorer": dict(_SCORER), "book": dict(_BOOK)}
    for table, full in (("hedge", _HEDGE), ("trend_overlay", _TREND), ("regime_hedge", _REGIME)):
        if table in tables:
            block = dict(full)
            if table == "hedge":
                block["hedge_ratio"] = hedge_ratio
            if strip:
                block = {k: v for k, v in block.items() if k not in supplied[table]}
            blocks[table] = block
    path = tmp_path / name
    path.write_text(_dump(blocks) + extra, encoding="utf-8")
    return path


def _load(path: Path, futures: Path = FUTURES) -> Any:
    from src.research.pipeline import load_strategy_spec

    return load_strategy_spec(path, futures_constants=futures)


@pytest.mark.parametrize("tables", [("hedge", "trend_overlay", "regime_hedge"), ("hedge",), ("hedge", "trend_overlay")])
def test_stripped_file_hashes_like_full_file(tmp_path: Path, tables: tuple[str, ...]) -> None:
    from src.research.pipeline import StrategySpec

    ratio = 0.0 if len(tables) > 1 else 1.0
    full = _write(tmp_path, "full.toml", tables=tables, hedge_ratio=ratio)
    stripped = _write(tmp_path, "stripped.toml", tables=tables, strip=True, hedge_ratio=ratio)
    original = StrategySpec.model_validate(tomllib.loads(full.read_text(encoding="utf-8")))
    assert _load(full).spec_hash == _load(stripped).spec_hash == original.spec_hash


def test_explicit_value_beats_market_file(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    stripped = _write(tmp_path, "s.toml", strip=True)
    raw = tomllib.loads(stripped.read_text(encoding="utf-8"))
    raw["hedge"]["initial_margin_rate"] = 0.3
    explicit = tmp_path / "explicit.toml"
    explicit.write_text(_dump(dict(raw)), encoding="utf-8")
    caplog.set_level(logging.INFO, logger="src.research.strategy_file")
    spec = _load(explicit)
    assert spec.hedge.initial_margin_rate == 0.3
    assert spec.hedge.futures_tax_rate == 0.11
    assert [r.getMessage() for r in caplog.records] == [
        "[DATA] strategy overrides market constant table=hedge key=initial_margin_rate"
    ]


def test_no_table_is_invented(tmp_path: Path) -> None:
    spec = _load(_write(tmp_path, "s.toml", strip=True))
    assert spec.trend_overlay is None
    assert spec.regime_hedge is None


def test_engine_policy_keys_are_not_filled(tmp_path: Path) -> None:
    blocks: dict[str, Any] = {"policy": dict(_POLICY), "scorer": dict(_SCORER), "book": dict(_BOOK)}
    hedge = {k: v for k, v in _HEDGE.items() if k not in _HEDGE_SUPPLIED}
    hedge["hedge_ratio"] = 0.0
    trend = {k: v for k, v in _TREND.items() if k not in _TREND_SUPPLIED and k != "margin_buffer_rate"}
    blocks["hedge"] = hedge
    blocks["trend_overlay"] = trend
    path = tmp_path / "s.toml"
    path.write_text(_dump(blocks), encoding="utf-8")
    with pytest.raises(ValueError, match="trend_overlay"):
        _load(path)


def test_extends_merges_tables_and_replaces_scalars(tmp_path: Path) -> None:
    from src.research.strategy_file import resolve_strategy_tables

    _write(tmp_path, "base.toml")
    child = tmp_path / "child.toml"
    child.write_text(
        'extends = "base.toml"\n[policy]\nn = 7\n[policy.universe]\nmin_price_krw = 2000\n'
        '[scorer]\nseeds = [11, 12, 13]\n', encoding="utf-8"
    )
    merged = resolve_strategy_tables(child, futures_constants=FUTURES)
    assert merged["policy"]["n"] == 7
    assert merged["policy"]["family"] == "ml_sleeve_hedge"
    assert merged["policy"]["universe"] == {"min_adtv20_krw": 500000000, "min_price_krw": 2000}
    assert merged["scorer"]["seeds"] == [11, 12, 13]
    assert merged["scorer"]["seed"] == 11
    assert "extends" not in merged
    spec = _load(child)
    assert spec.policy.n == 7
    assert spec.scorer.seeds == (11, 12, 13)


def test_extends_resolves_relative_to_declaring_file(tmp_path: Path) -> None:
    from src.research.strategy_file import resolve_strategy_tables

    (tmp_path / "challenge").mkdir()
    _write(tmp_path, "champion.toml")
    child = tmp_path / "challenge" / "child.toml"
    child.write_text('extends = "../champion.toml"\n[policy]\nn = 9\n', encoding="utf-8")
    merged = resolve_strategy_tables(child, futures_constants=FUTURES)
    assert merged["policy"]["n"] == 9


def test_arrays_are_replaced_whole(tmp_path: Path) -> None:
    from src.research.strategy_file import resolve_strategy_tables

    base = _write(tmp_path, "base.toml", tables=("hedge", "trend_overlay", "regime_hedge"), hedge_ratio=0.0)
    base.write_text(base.read_text(encoding="utf-8").replace("[21, 63, 126, 252]", "[21, 63]"), encoding="utf-8")
    child = tmp_path / "child.toml"
    child.write_text(
        'extends = "base.toml"\n[regime_hedge]\ntsmom_horizons = [5]\n', encoding="utf-8"
    )
    merged = resolve_strategy_tables(child, futures_constants=FUTURES)
    assert merged["regime_hedge"]["tsmom_horizons"] == [5]
    assert _load(child).regime_hedge.tsmom_horizons == (5,)


@pytest.mark.parametrize(
    "body",
    [
        'extends = "/tmp/abs.toml"\n[policy]\nn = 1\n',
        'extends = "missing.toml"\n[policy]\nn = 1\n',
        "extends = 5\n[policy]\nn = 1\n",
    ],
)
def test_extends_misuse_fails_closed(tmp_path: Path, body: str) -> None:
    from src.research.strategy_file import resolve_strategy_tables

    _write(tmp_path, "base.toml")
    child = tmp_path / "child.toml"
    child.write_text(body + _dump({"scorer": dict(_SCORER), "book": dict(_BOOK), "hedge": dict(_HEDGE)}), encoding="utf-8")
    with pytest.raises(ValueError, match="extends"):
        resolve_strategy_tables(child, futures_constants=FUTURES)


def test_extends_cycle_and_long_chain_fail_closed(tmp_path: Path) -> None:
    from src.research.strategy_file import resolve_strategy_tables

    first = tmp_path / "a.toml"
    second = tmp_path / "b.toml"
    first.write_text('extends = "b.toml"\n[policy]\nn = 1\n', encoding="utf-8")
    second.write_text('extends = "a.toml"\n[policy]\nn = 2\n', encoding="utf-8")
    with pytest.raises(ValueError, match="cycle"):
        resolve_strategy_tables(first, futures_constants=FUTURES)

    previous = _write(tmp_path, "c4.toml")
    for depth in range(4):
        if depth == 3:
            assert _load(previous).policy.n == 1
        current = tmp_path / f"c{3 - depth}.toml"
        current.write_text(f'extends = "{previous.name}"\n[policy]\nn = 1\n', encoding="utf-8")
        previous = current
    with pytest.raises(ValueError, match="exceeds 4"):
        resolve_strategy_tables(previous, futures_constants=FUTURES)


def test_unknown_ancestor_keys_cannot_be_hidden(tmp_path: Path) -> None:
    base = _write(tmp_path, "base.toml", extra="[bogus]\nx = 1\n")
    child = tmp_path / "child.toml"
    child.write_text(f'extends = "{base.name}"\n[policy]\nn = 7\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"unknown strategy keys.*bogus"):
        _load(child)


def test_ancestor_os_errors_keep_their_type_and_path(tmp_path: Path) -> None:
    child = tmp_path / "child.toml"
    (tmp_path / "directory").mkdir()
    child.write_text('extends = "directory"\n', encoding="utf-8")
    with pytest.raises(IsADirectoryError, match="directory"):
        _load(child)


def test_symlink_entry_extends_uses_declaring_directory(tmp_path: Path) -> None:
    (tmp_path / "source").mkdir()
    _write(tmp_path, "base.toml")
    source = tmp_path / "source" / "child.toml"
    source.write_text('extends = "base.toml"\n[policy]\nn = 9\n', encoding="utf-8")
    entry = tmp_path / "child.toml"
    entry.symlink_to(source)
    assert _load(entry).policy.n == 9


def test_strategy_read_errors_behave_as_before(tmp_path: Path) -> None:
    from src.research.strategy_file import resolve_strategy_tables

    with pytest.raises(OSError, match="absent"):
        resolve_strategy_tables(tmp_path / "absent.toml", futures_constants=FUTURES)
    broken = tmp_path / "broken.toml"
    broken.write_text("[[[", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid strategy TOML"):
        resolve_strategy_tables(broken, futures_constants=FUTURES)
    unknown = _write(tmp_path, "u.toml")
    with unknown.open("a", encoding="utf-8") as handle:
        handle.write("[bogus]\nx = 1\n")
    with pytest.raises(ValueError, match="unknown"):
        resolve_strategy_tables(unknown, futures_constants=FUTURES)


def test_scalar_table_is_not_filled(tmp_path: Path) -> None:
    from src.research.strategy_file import resolve_strategy_tables

    blocks: dict[str, Any] = {"policy": dict(_POLICY), "scorer": dict(_SCORER), "book": dict(_BOOK)}
    path = tmp_path / "s.toml"
    path.write_text("hedge = 1\n" + _dump(blocks), encoding="utf-8")
    assert resolve_strategy_tables(path, futures_constants=FUTURES)["hedge"] == 1


def test_default_futures_constants_match_explicit(tmp_path: Path) -> None:
    from src.research.pipeline import load_strategy_spec

    stripped = _write(tmp_path, "s.toml", strip=True)
    assert load_strategy_spec(stripped).spec_hash == _load(stripped).spec_hash


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda text: text.replace("inverse_tax_rate = 0.154\n", ""), "inverse_tax_rate"),
        (lambda text: text.replace("[tax]\n", "[tax]\nbogus_key = 1\n", 1), "bogus_key"),
        (lambda text: text + "\n[nope]\nx = 1\n", "nope"),
        (
            lambda text: text.replace(
                "[cost]\nfutures_cost_rate = 0.0003\ninverse_cost_rate = 0.0007\n"
                "resize_sell_cost_rate = 0.0025\nresize_buy_cost_rate = 0.0005\n",
                "",
            ),
            "cost",
        ),
        (lambda text: text.replace("futures_cost_rate = 0.0003", "futures_cost_rate = -0.0003"), "futures_cost_rate"),
        (lambda text: text.replace("futures_cost_rate = 0.0003", "futures_cost_rate = inf"), "futures_cost_rate"),
        (lambda text: text.replace("futures_cost_rate = 0.0003", "futures_cost_rate = nan"), "futures_cost_rate"),
        (lambda text: text.replace("futures_cost_rate = 0.0003", "futures_cost_rate = " + "9" * 400), "futures_cost_rate"),
        (lambda text: text.replace("futures_tax_rate = 0.11", "futures_tax_rate = true", 1), "futures_tax_rate"),
        (lambda text: text.replace("futures_tax_rate = 0.11", 'futures_tax_rate = "high"', 1), "futures_tax_rate"),
        (
            lambda text: text.replace("futures_annual_deduction_krw = 2500000", "futures_annual_deduction_krw = 1.5"),
            "futures_annual_deduction_krw",
        ),
        (
            lambda text: text.replace("futures_annual_deduction_krw = 2500000", "futures_annual_deduction_krw = -1"),
            "futures_annual_deduction_krw",
        ),
    ],
)
def test_market_file_errors_fail_closed(
    tmp_path: Path, mutate: Any, match: str
) -> None:
    source = FUTURES.read_text(encoding="utf-8")
    market = tmp_path / "futures.toml"
    market.write_text(mutate(source), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        _load(_write(tmp_path, "s.toml", strip=True), futures=market)


def test_market_file_read_errors_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="futures constants"):
        _load(_write(tmp_path, "s.toml", strip=True), futures=tmp_path / "absent.toml")
    broken = tmp_path / "futures.toml"
    broken.write_text("data_root = [\n", encoding="utf-8")
    with pytest.raises(ValueError, match="futures constants"):
        _load(_write(tmp_path, "s.toml", strip=True), futures=broken)


_FIXTURES = Path("tests/fixtures/strategies")


def _fixture_specs() -> list[Any]:
    from src.research.pipeline import load_strategy_spec

    specs = [load_strategy_spec(path) for path in sorted(_FIXTURES.glob("*.toml"))]
    assert len(specs) >= 9
    return specs


def test_render_round_trip_keeps_the_hash(tmp_path: Path) -> None:
    from src.research.champion import with_seeds
    from src.research.pipeline import load_strategy_spec
    from src.research.strategy_file import render_strategy_toml

    specs = _fixture_specs()
    seeded = with_seeds(specs[0], (1, 2))
    assert seeded.scorer.seeds == (1, 2)
    for spec in [*specs, seeded]:
        text = render_strategy_toml(spec, futures_constants=FUTURES)
        out = tmp_path / "rendered.toml"
        out.write_text(text, encoding="utf-8")
        assert load_strategy_spec(out).spec_hash == spec.spec_hash


def test_render_omits_only_matching_market_keys(tmp_path: Path) -> None:
    from src.research.strategy_file import render_strategy_toml

    spec = _load(_write(tmp_path, "s.toml", strip=True))
    assert spec.hedge.initial_margin_rate == 0.2175
    altered = spec.model_copy(
        update={"hedge": spec.hedge.model_copy(update={"initial_margin_rate": 0.3})}
    )
    text = render_strategy_toml(altered, futures_constants=FUTURES)
    assert "initial_margin_rate = 0.3" in text
    for key in (
        "futures_tax_rate",
        "futures_annual_deduction_krw",
        "inverse_tax_rate",
        "futures_cost_rate",
        "inverse_cost_rate",
        "resize_sell_cost_rate",
        "resize_buy_cost_rate",
        "contract_multiplier_krw",
    ):
        assert f"{key} =" not in text


def test_toml_string_escaping_round_trips() -> None:
    import tomllib

    from src.research.strategy_file import _toml_string, _toml_value

    tricky = 'a"b\\c\b\t\n\f\r\x01e'
    rendered = _toml_string(tricky)
    assert rendered == '"a\\"b\\\\c\\b\\t\\n\\f\\r\\u0001e"'
    assert tomllib.loads(f"x = {rendered}\n")["x"] == tricky
    assert _toml_value(True) == "true"
    with pytest.raises(ValueError, match="unsupported TOML scalar"):
        _toml_value({"nested": 1})


def test_render_is_deterministic_and_comment_free() -> None:
    from src.research.strategy_file import render_strategy_toml

    spec = _fixture_specs()[0]
    first = render_strategy_toml(spec, futures_constants=FUTURES)
    second = render_strategy_toml(spec, futures_constants=FUTURES)
    assert first == second
    assert first.startswith(
        "# Generated by 'research champion --sync-file'; edit challenge files instead of this one.\n\n"
    )
    assert first.count("#") == 1


def test_render_preserves_signed_zero_market_override(tmp_path: Path) -> None:
    from src.research.strategy_file import render_strategy_toml

    market = tmp_path / "futures.toml"
    market.write_text(
        FUTURES.read_text(encoding="utf-8").replace("futures_cost_rate = 0.0003", "futures_cost_rate = 0.0"),
        encoding="utf-8",
    )
    path = _write(tmp_path, "strategy.toml", strip=True)
    path.write_text(path.read_text(encoding="utf-8") + "futures_cost_rate = -0.0\n", encoding="utf-8")
    spec = _load(path, market)
    text = render_strategy_toml(spec, futures_constants=market)
    assert "futures_cost_rate = -0.0" in text
    rendered = tmp_path / "rendered.toml"
    rendered.write_text(text, encoding="utf-8")
    assert _load(rendered, market).spec_hash == spec.spec_hash
