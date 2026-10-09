"""Protocol v5 binding and window-guard invariants."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.data.research_protocol import (
    WindowError,
    WindowGuard,
    load_research_protocol,
)


def _protocol():  # type: ignore[no-untyped-def]
    from src.data.research_scope import load_research_scope

    scope = load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    return load_research_protocol(Path("config/research/protocol.toml"), scope), scope


def test_protocol_v7_loads() -> None:
    protocol, _ = _protocol()
    assert protocol.version == "research-protocol-v7"
    assert protocol.evaluation_start == date(2017, 4, 1)
    assert protocol.sessions_per_year == 252
    assert protocol.primary_capital_krw == 100_000_000
    assert protocol.evaluation.block_sessions == 63
    assert protocol.evaluation.draws == 4000
    assert protocol.evaluation.seed == 20261001
    assert protocol.evaluation.horizon_sessions == 1260
    assert protocol.evaluation.objective_quantile == pytest.approx(0.10)
    assert protocol.evaluation.ruin_mdd_limit == pytest.approx(-0.5)
    assert protocol.evaluation.max_p_ruin == pytest.approx(1.0)
    assert protocol.evaluation.max_p_growth_le_zero == pytest.approx(0.05)
    assert tuple(protocol.evaluation.report_mdd_limits) == (-0.3, -0.5)
    assert protocol.evaluation.recent_sessions == 252
    assert tuple(protocol.scenarios.cost_grid_ticks) == (0.0, 0.5, 1.0, 1.5)
    assert protocol.scenarios.stress_extra_slippage == pytest.approx(0.001)
    assert protocol.scenarios.stress_hedge_extra_cost == pytest.approx(0.0005)
    assert protocol.scenarios.stress_delay_sessions == 1
    assert protocol.scenarios.placebo_seed == 7
    assert protocol.scenarios.perturbation_cuts == 3
    assert protocol.scenarios.perturbation_seed == 7
    assert protocol.champion.alpha == pytest.approx(0.05)
    assert protocol.champion.require_neighbors is True
    assert protocol.champion.paired_horizon == "full"
    assert protocol.champion.multiplicity == "bonferroni_decisions"
    assert protocol.champion.noninferiority_margin == pytest.approx(0.03)
    assert protocol.champion.tail_quantile == pytest.approx(0.05)
    assert protocol.champion.tail_block_sessions == 21
    assert protocol.champion.champion_seeds == (11, 12, 13)


def test_content_hash_stable() -> None:
    first, _ = _protocol()
    second, _ = _protocol()
    assert first.content_hash == second.content_hash
    assert len(first.content_hash) == 64


def test_unknown_protocol_key_rejected(tmp_path: Path) -> None:
    from src.config import ConfigError
    from src.data.research_scope import load_research_scope

    scope = load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    base = Path("config/research/protocol.toml").read_text(encoding="utf-8")
    leftover = tmp_path / "leftover.toml"
    leftover.write_text(base + "\n[criteria]\nmin_dsr = 0.95\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_research_protocol(leftover, scope)


def test_window_guard_bounds() -> None:
    protocol, _ = _protocol()
    guard = WindowGuard(protocol=protocol, last_session=date(2026, 9, 30))
    auth = guard.authorize(start=date(2018, 1, 2), end=date(2026, 9, 30))
    assert (auth.start, auth.end) == (date(2018, 1, 2), date(2026, 9, 30))
    with pytest.raises(WindowError):
        guard.authorize(start=date(2018, 1, 2), end=date(2026, 10, 1))
    with pytest.raises(WindowError):
        guard.authorize(start=date(2017, 3, 31), end=date(2026, 9, 30))
    with pytest.raises(ValueError, match="start <= end"):
        guard.authorize(start=date(2020, 1, 2), end=date(2020, 1, 1))


def test_evaluation_policy_converts_to_research() -> None:
    from src.research.evaluation import EvaluationPolicy

    protocol, _ = _protocol()
    policy = EvaluationPolicy.model_validate(protocol.evaluation.model_dump())
    assert policy.block_sessions == 63
    assert policy.draws == 4000
    assert policy.seed == 20261001


def test_load_missing_and_invalid_protocol(tmp_path: Path) -> None:
    from src.config import ConfigError
    from src.data.research_scope import load_research_scope

    scope = load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    with pytest.raises(ConfigError):
        load_research_protocol(tmp_path / "absent.toml", scope)
    broken = tmp_path / "broken.toml"
    broken.write_text("version = [\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_research_protocol(broken, scope)
    unknown = tmp_path / "unknown.toml"
    base = Path("config/research/protocol.toml").read_text(encoding="utf-8")
    unknown.write_text(base + '\nunknown_key = "x"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_research_protocol(unknown, scope)
    bad_start = tmp_path / "bad-start.toml"
    bad_start.write_text(
        base.replace('evaluation_start = "2017-04-01"', 'evaluation_start = "not-a-date"'),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_research_protocol(bad_start, scope)
    early = tmp_path / "early.toml"
    early.write_text(
        base.replace('evaluation_start = "2017-04-01"', 'evaluation_start = "2016-01-01"'),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_research_protocol(early, scope)


def test_missing_table_rejected(tmp_path: Path) -> None:
    from src.config import ConfigError
    from src.data.research_scope import load_research_scope

    scope = load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    base = Path("config/research/protocol.toml").read_text(encoding="utf-8")
    head, _, _ = base.partition("[evaluation]")
    for table in ("scenarios", "champion"):
        stripped = tmp_path / f"without-{table}.toml"
        lines = [
            line
            for line in head.splitlines()
            if not line.startswith(f"[{table}]")
        ]
        stripped.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_research_protocol(stripped, scope)


def test_load_protocol_domains(tmp_path: Path) -> None:
    from src.config import ConfigError
    from src.data.research_scope import load_research_scope

    scope = load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    base = Path("config/research/protocol.toml").read_text(encoding="utf-8")

    def _bad(old: str, new: str) -> Path:
        path = tmp_path / f"bad-{abs(hash(old + new)) % 100000}.toml"
        path.write_text(base.replace(old, new), encoding="utf-8")
        return path

    with pytest.raises(ConfigError):
        load_research_protocol(_bad("sessions_per_year = 252", "sessions_per_year = 0"), scope)
    with pytest.raises(ConfigError):
        load_research_protocol(
            _bad("primary_capital_krw = 100_000_000", "primary_capital_krw = 0"), scope
        )
    with pytest.raises(ConfigError):
        load_research_protocol(_bad("draws = 4000", "draws = 0"), scope)
    with pytest.raises(ConfigError):
        load_research_protocol(_bad("objective_quantile = 0.10", "objective_quantile = 1.5"), scope)
    with pytest.raises(ConfigError):
        load_research_protocol(_bad("ruin_mdd_limit = -0.5", "ruin_mdd_limit = 0.0"), scope)
    with pytest.raises(ConfigError):
        load_research_protocol(_bad("max_p_ruin = 1.0", "max_p_ruin = 2.0"), scope)
    with pytest.raises(ConfigError):
        load_research_protocol(_bad("cost_grid_ticks = [0.0, 0.5, 1.0, 1.5]", "cost_grid_ticks = []"), scope)
    with pytest.raises(ConfigError):
        load_research_protocol(
            _bad("stress_extra_slippage = 0.001", "stress_extra_slippage = -0.1"), scope
        )
    with pytest.raises(ConfigError):
        load_research_protocol(_bad("alpha = 0.05", "alpha = 0.9"), scope)
    with pytest.raises(ConfigError):
        load_research_protocol(
            _bad("require_neighbors = true", 'require_neighbors = "maybe"'), scope
        )


def test_v4_file_keeps_v4_semantics(tmp_path: Path) -> None:
    from src.data.research_scope import load_research_scope

    scope = load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    lines = [
        line
        for line in Path("config/research/protocol.toml").read_text(encoding="utf-8").splitlines()
        if line.strip() not in ('paired_horizon = "full"', 'multiplicity = "bonferroni_decisions"')
    ]
    v4_path = tmp_path / "protocol-v4.toml"
    v4_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    protocol = load_research_protocol(v4_path, scope)
    assert protocol.champion.paired_horizon == "evaluation"
    assert protocol.champion.multiplicity == "none"


def _without_lines(path: Path, tmp_path: Path, name: str, keys: tuple[str, ...]) -> Path:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith(keys)
    ]
    out = tmp_path / name
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def test_champion_keys_round_trip_and_change_the_hash(tmp_path: Path) -> None:
    from src.data.research_scope import load_research_scope

    scope = load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    path = Path("config/research/protocol.toml")
    protocol = load_research_protocol(path, scope)
    assert protocol.champion.paired_horizon == "full"
    assert protocol.champion.multiplicity == "bonferroni_decisions"
    assert protocol.champion.noninferiority_margin == pytest.approx(0.03)
    for name, keys in (
        ("v4-keys.toml", ("paired_horizon", "multiplicity")),
        ("v5-keys.toml", ("noninferiority_margin",)),
    ):
        stripped = load_research_protocol(_without_lines(path, tmp_path, name, keys), scope)
        assert stripped.content_hash != protocol.content_hash
    assert load_research_protocol(
        _without_lines(path, tmp_path, "no-margin.toml", ("noninferiority_margin",)), scope
    ).champion.noninferiority_margin is None


def test_unknown_champion_key_rejected(tmp_path: Path) -> None:
    from src.config import ConfigError
    from src.data.research_scope import load_research_scope

    scope = load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    base = Path("config/research/protocol.toml").read_text(encoding="utf-8")
    bad = tmp_path / "bad-horizon.toml"
    bad.write_text(base.replace('paired_horizon = "full"', 'paired_horizon = "5y"'), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_research_protocol(bad, scope)


def test_v5_file_disables_noninferiority_path(tmp_path: Path) -> None:
    from src.data.research_scope import load_research_scope

    scope = load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    base = Path("config/research/protocol.toml").read_text(encoding="utf-8")
    lines = [
        line
        for line in base.splitlines()
        if not line.strip().startswith("noninferiority_margin")
    ]
    v5_path = tmp_path / "protocol-v5.toml"
    v5_path.write_text("\n".join(lines).replace('version = "research-protocol-v7"', 'version = "research-protocol-v5"') + "\n", encoding="utf-8")
    protocol = load_research_protocol(v5_path, scope)
    assert protocol.champion.noninferiority_margin is None
    assert protocol.champion.tail_quantile == 0.05
    assert protocol.champion.tail_block_sessions == 21
    v6_protocol = load_research_protocol(Path("config/research/protocol.toml"), scope)
    assert protocol.content_hash != v6_protocol.content_hash


def test_out_of_range_margin_rejected(tmp_path: Path) -> None:
    from src.config import ConfigError
    from src.data.research_scope import load_research_scope

    scope = load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    base = Path("config/research/protocol.toml").read_text(encoding="utf-8")

    def _with_margin(val: str) -> Path:
        p = tmp_path / f"margin-{abs(hash(val))}.toml"
        p.write_text(base.replace("[champion]\n", f"[champion]\nnoninferiority_margin = {val}\n"), encoding="utf-8")
        return p

    with pytest.raises(ConfigError):
        load_research_protocol(_with_margin("0.0"), scope)
    with pytest.raises(ConfigError):
        load_research_protocol(_with_margin("0.5"), scope)


def test_invalid_champion_seeds_rejected(tmp_path: Path) -> None:
    from src.config import ConfigError
    from src.data.research_scope import load_research_scope

    scope = load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    base = Path("config/research/protocol.toml").read_text(encoding="utf-8")

    def _with_seeds(val: str) -> Path:
        path = tmp_path / f"champion-seeds-{abs(hash(val)) % 100000}.toml"
        path.write_text(base.replace("champion_seeds = [11, 12, 13]", f"champion_seeds = {val}"), encoding="utf-8")
        return path

    for bad in ("[11]", "[11, 11]", "[11, -1]", "[true, 12]"):
        with pytest.raises(ConfigError):
            load_research_protocol(_with_seeds(bad), scope)


def test_champion_seeds_change_the_protocol_hash() -> None:
    protocol, _ = _protocol()
    reseeded = protocol.model_copy(
        update={"champion": protocol.champion.model_copy(update={"champion_seeds": (11, 12, 14)})}
    )
    assert reseeded.content_hash != protocol.content_hash
    cleared = protocol.model_copy(update={"champion": protocol.champion.model_copy(update={"champion_seeds": ()})})
    assert cleared.content_hash != protocol.content_hash
    assert cleared.champion.champion_seeds == ()


def test_champion_seeds_accept_empty_and_tuple_forms() -> None:
    from src.data.research_protocol import ChampionPolicy

    kw: dict[str, object] = {"alpha": 0.05, "require_neighbors": True}
    assert ChampionPolicy(**kw, champion_seeds=()).champion_seeds == ()  # type: ignore[arg-type]
    assert ChampionPolicy(**kw, champion_seeds=[]).champion_seeds == ()  # type: ignore[arg-type]
    assert ChampionPolicy(**kw, champion_seeds=(11, 12, 13)).champion_seeds == (11, 12, 13)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="champion_seeds"):
        ChampionPolicy(**kw, champion_seeds=11)  # type: ignore[arg-type]

