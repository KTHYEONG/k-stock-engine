"""Research data-window guard, report-card policy and scenario bindings (protocol v5)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config.errors import ConfigError
from src.core.pit import PITDataError
from src.data.research_scope import ResearchScope

__all__ = [
    "ChampionPolicy",
    "EvaluationPolicySettings",
    "ResearchProtocol",
    "ScenarioPolicy",
    "WindowAuthorization",
    "WindowError",
    "WindowGuard",
    "load_research_protocol",
]

MddLimit = Annotated[float, Field(gt=-1.0, lt=0.0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
Rate = Annotated[float, Field(ge=0.0, lt=1.0)]
Ticks = Annotated[float, Field(ge=0.0, lt=100.0)]


class WindowError(PITDataError):
    """A run window reaches outside certified data."""


class EvaluationPolicySettings(BaseModel):
    """Data-layer mirror of the report-card policy (same fields as research EvaluationPolicy)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sessions_per_year: PositiveInt
    block_sessions: PositiveInt
    draws: PositiveInt
    seed: NonNegativeInt
    horizon_sessions: PositiveInt
    objective_quantile: Annotated[float, Field(gt=0.0, lt=1.0)]
    ruin_mdd_limit: MddLimit
    max_p_ruin: Probability
    max_p_growth_le_zero: Probability
    report_mdd_limits: Annotated[tuple[MddLimit, ...], Field(min_length=1)]
    recent_sessions: PositiveInt


class ScenarioPolicy(BaseModel):
    """Account-engine scenario definitions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cost_grid_ticks: Annotated[tuple[Ticks, ...], Field(min_length=1)]
    stress_extra_slippage: Rate
    stress_hedge_extra_cost: Rate
    stress_delay_sessions: NonNegativeInt
    placebo_seed: NonNegativeInt
    perturbation_cuts: PositiveInt
    perturbation_seed: NonNegativeInt


class ChampionPolicy(BaseModel):
    """Champion/challenger promotion rule parameters.

    ``paired_horizon``: ``"evaluation"`` bootstraps paired paths of ``evaluation.horizon_sessions`` (v4 behaviour);
    ``"full"`` uses the full shared sample length. Why full: the decision compares expected growth, whose sampling
    error shrinks with the whole sample; a 5-year path adds future-realisation noise that is not about which
    strategy is better.
    ``multiplicity``: ``"none"`` (v4) or ``"bonferroni_decisions"``. The latter tests at ``alpha / (1 + m)``, where m
    is the number of saved decisions naming the current champion. Why: every challenge on the same champion is
    another draw at the same data, so the family-wise error must be paid explicitly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha: Annotated[float, Field(gt=0.0, lt=0.5)]
    require_neighbors: bool
    paired_horizon: Literal["evaluation", "full"] = "evaluation"
    multiplicity: Literal["none", "bonferroni_decisions"] = "none"


class ResearchProtocol(BaseModel):
    """Data window, report-card policy and scenarios of the research program (v5).

    There is no sealed segment: any window between ``evaluation_start`` and the last certified session may
    be evaluated any number of times. Selection bias is handled by champion/challenger paired comparison
    plus neighbor plateaus and the cost grid, not by withholding history.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    evaluation_start: date
    sessions_per_year: int
    primary_capital_krw: int
    evaluation: EvaluationPolicySettings
    scenarios: ScenarioPolicy
    champion: ChampionPolicy

    @property
    def content_hash(self) -> str:
        payload = {
            "champion": self.champion.model_dump(mode="json"),
            "evaluation": self.evaluation.model_dump(mode="json"),
            "evaluation_start": self.evaluation_start.isoformat(),
            "primary_capital_krw": self.primary_capital_krw,
            "scenarios": self.scenarios.model_dump(mode="json"),
            "sessions_per_year": self.sessions_per_year,
            "version": self.version,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_research_protocol(path: Path, scope: ResearchScope) -> ResearchProtocol:
    """Load the protocol TOML and bind the scope evidence start.

    Raises:
        ConfigError: file missing or invalid, unknown keys, ``evaluation_start`` not after
            ``scope.evidence_start``, non-positive capital/sessions, or invalid table values.
    """
    import tomllib

    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"research protocol is missing: {path}") from exc
    except ValueError as exc:
        raise ConfigError(f"research protocol is invalid TOML: {path}") from exc
    allowed_top = {
        "version",
        "evaluation_start",
        "sessions_per_year",
        "primary_capital_krw",
        "evaluation",
        "scenarios",
        "champion",
    }
    unknown = set(raw) - allowed_top
    if unknown:
        raise ConfigError(f"research protocol has unknown keys: {sorted(unknown)}")
    try:
        evaluation_start = date.fromisoformat(str(raw.get("evaluation_start")))
    except (ValueError, TypeError) as exc:
        raise ConfigError(
            f"research protocol evaluation_start is invalid: {raw.get('evaluation_start')!r}"
        ) from exc
    if evaluation_start <= scope.evidence_start:
        raise ConfigError("research protocol evaluation_start must be after scope.evidence_start")
    for table in ("evaluation", "scenarios", "champion"):
        if not isinstance(raw.get(table), dict):
            raise ConfigError(f"research protocol must declare [{table}]")
    try:
        evaluation_raw = {"sessions_per_year": int(raw["sessions_per_year"]), **raw["evaluation"]}
        evaluation = EvaluationPolicySettings.model_validate(evaluation_raw)
        scenarios = ScenarioPolicy.model_validate(raw["scenarios"])
        champion = ChampionPolicy.model_validate(raw["champion"])
        protocol = ResearchProtocol(
            version=str(raw.get("version")),
            evaluation_start=evaluation_start,
            sessions_per_year=int(raw["sessions_per_year"]),
            primary_capital_krw=int(raw["primary_capital_krw"]),
            evaluation=evaluation,
            scenarios=scenarios,
            champion=champion,
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise ConfigError(f"research protocol is invalid: {exc}") from exc
    if protocol.sessions_per_year <= 0:
        raise ConfigError("research protocol sessions_per_year must be positive")
    if protocol.primary_capital_krw <= 0:
        raise ConfigError("research protocol primary_capital_krw must be positive")
    return protocol


@dataclass(frozen=True, slots=True)
class WindowAuthorization:
    """Proof that a run window lies inside certified data: [evaluation_start, last certified session]."""

    start: date
    end: date


class WindowGuard:
    """Authorizes run windows against the protocol start and the last certified session."""

    def __init__(self, *, protocol: ResearchProtocol, last_session: date) -> None:
        self._protocol = protocol
        self._last_session = last_session

    def authorize(self, *, start: date, end: date) -> WindowAuthorization:
        """ValueError if start > end; WindowError if start < evaluation_start or end > last_session."""
        if start > end:
            raise ValueError(f"run window [{start}, {end}] must satisfy start <= end")
        if start < self._protocol.evaluation_start or end > self._last_session:
            raise WindowError(
                f"run window [{start}, {end}] is outside "
                f"[{self._protocol.evaluation_start}, {self._last_session}]"
            )
        return WindowAuthorization(start=start, end=end)
