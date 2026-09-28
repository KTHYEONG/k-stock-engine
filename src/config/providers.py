"""Provider throughput and quota policy loaded from ``config/providers.toml``."""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveFloat, PositiveInt, field_validator, model_validator

from src.config.errors import ConfigError
from src.config.runtime import RuntimeConfig

__all__ = [
    "DartKeyPolicy",
    "DartPolicy",
    "DisclosureFilter",
    "DividendPlausibilityPolicy",
    "DocumentParserPolicy",
    "KisPolicy",
    "KrxPolicy",
    "LsPolicy",
    "ProviderPolicy",
    "RunnerPolicy",
    "disclosure_filter_for_code",
    "load_provider_policy",
]

# Sustained 15 rps was clean and 30 rps dropped the connection: 20 rps (0.05 s) is the floor.
MIN_DART_INTERVAL_SECONDS = 0.05


@dataclass(frozen=True, slots=True)
class DisclosureFilter:
    """One OpenDART ``list.json`` filter.

    OpenDART separates the publication type (one letter, ``pblntf_ty``) from
    the detail type (one letter plus three digits, ``pblntf_detail_ty``).
    Sending a type as a detail code is not rejected by the API; it silently
    returns no data, so the form is fixed when the configuration is read.
    """

    code: str
    parameter: Literal["pblntf_ty", "pblntf_detail_ty"]


_TYPE_CODE_PATTERN = re.compile(r"^[A-J]$")
_DETAIL_CODE_PATTERN = re.compile(r"^[A-J][0-9]{3}$")


def disclosure_filter_for_code(code: str) -> DisclosureFilter:
    """Map one configured disclosure code to its OpenDART parameter."""
    text = str(code).strip()
    if _TYPE_CODE_PATTERN.fullmatch(text):
        return DisclosureFilter(code=text, parameter="pblntf_ty")
    if _DETAIL_CODE_PATTERN.fullmatch(text):
        return DisclosureFilter(code=text, parameter="pblntf_detail_ty")
    raise ConfigError(f"invalid disclosure type {code!r}: expected [A-J] or [A-J]NNN")


class DartKeyPolicy(BaseModel):
    """Budget and pacing of one OpenDART key, metered in its own quota ledger."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    daily_limit: PositiveInt
    daily_budget: PositiveInt
    daily_reserve: NonNegativeInt
    min_interval_seconds: PositiveFloat
    max_workers: PositiveInt
    default: bool = False

    @model_validator(mode="after")
    def _check_policy(self) -> DartKeyPolicy:
        if self.daily_budget >= self.daily_limit:
            raise ValueError(f"invalid daily_budget {self.daily_budget!r}: must be below {self.daily_limit}")
        if self.daily_reserve >= self.daily_budget:
            raise ValueError("daily_reserve must be below daily_budget")
        if self.min_interval_seconds < MIN_DART_INTERVAL_SECONDS:
            raise ValueError(
                f"min_interval_seconds below {MIN_DART_INTERVAL_SECONDS} exceeds the measured safe rate"
            )
        return self


class DividendPlausibilityPolicy(BaseModel):
    """Gate for dividend decisions entering the Silver event table.

    An implausible decision is withheld, never repaired. The checks target gross misreads (a total
    or a share count taken as DPS), not ordinary noise: treasury shares receive no dividend, so the
    printed total may cover fewer than the listed shares, and the printed market yield uses the
    average price of the week before the record date, not the pre-ex close.

    Attributes:
        max_yield: Hard ceiling on DPS over the pre-ex close.
        max_yield_ratio: Largest allowed ratio between the implied and printed yield, either way.
        min_paid_share_fraction: Smallest share of listed common shares the printed total may pay.
        correction_window_days: Maximum record-date distance for a correction to replace a decision.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_yield: float = 0.5
    max_yield_ratio: float = 2.0
    min_paid_share_fraction: float = 0.5
    correction_window_days: int = 45

    @field_validator("max_yield", "max_yield_ratio", "min_paid_share_fraction")
    @classmethod
    def _check_ratio(cls, value: float) -> float:
        if float(value) <= 0:
            raise ValueError(f"invalid dividend plausibility ratio {value!r}: must be positive")
        return float(value)

    @field_validator("correction_window_days")
    @classmethod
    def _check_window(cls, value: int) -> int:
        if isinstance(value, bool) or int(value) < 0:
            raise ValueError(f"invalid correction_window_days {value!r}: must be a non-negative integer")
        return int(value)


class DocumentParserPolicy(BaseModel):
    """Gate for filing-document facts entering Silver."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trusted: bool = False
    min_precision: float = 0.995
    benchmark_sample: int = 300

    @field_validator("min_precision")
    @classmethod
    def _check_precision(cls, value: float) -> float:
        if not 0 < float(value) <= 1:
            raise ValueError(f"invalid min_precision {value!r}: must be in (0, 1]")
        return float(value)

    @field_validator("benchmark_sample")
    @classmethod
    def _check_sample(cls, value: int) -> int:
        if isinstance(value, bool) or int(value) < 1:
            raise ValueError(f"invalid benchmark_sample {value!r}: must be a positive integer")
        return int(value)


class DartPolicy(BaseModel):
    """Shared DART collection policy plus one entry per declared key."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    circuit_threshold: PositiveInt
    requests_per_identity: PositiveInt
    batch_identities: PositiveInt
    shared_ip_avoid_windows_kst: list[tuple[str, str]] = []
    disclosure_types: tuple[str, ...] = ("A", "I001")
    corp_codes_max_age_days: PositiveInt = 30
    document_parser: DocumentParserPolicy = DocumentParserPolicy()
    keys: dict[str, DartKeyPolicy]
    @field_validator("keys")
    @classmethod
    def _check_keys(cls, value: dict[str, DartKeyPolicy]) -> dict[str, DartKeyPolicy]:
        if not value:
            raise ValueError("dart.keys must declare at least one key")
        return value

    @field_validator("disclosure_types")
    @classmethod
    def _check_disclosure_types(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value if item.strip())
        if not cleaned:
            raise ConfigError("dart.disclosure_types must declare at least one list type")
        for item in cleaned:
            if not (_TYPE_CODE_PATTERN.fullmatch(item) or _DETAIL_CODE_PATTERN.fullmatch(item)):
                raise ConfigError(f"invalid disclosure type {item!r}: expected [A-J] or [A-J]NNN")
        return cleaned

    @property
    def disclosure_filters(self) -> tuple[DisclosureFilter, ...]:
        """Configured codes mapped to the OpenDART parameter each one requires."""
        return tuple(disclosure_filter_for_code(code) for code in self.disclosure_types)

    @field_validator("shared_ip_avoid_windows_kst")
    @classmethod
    def _check_windows(cls, value: list[tuple[str, str]]) -> list[tuple[str, str]]:
        import re

        for window in value:
            if len(tuple(window)) != 2 or any(not re.fullmatch(r"[0-2]\d:[0-5]\d", end) for end in window):
                raise ValueError(f"invalid avoid window {window!r}: expected HH:MM pairs")
        return value

    def dart_key(self, key_env: str) -> DartKeyPolicy:
        """Return the declared policy for a key's environment variable.

        Raises:
            ConfigError: ``key_env`` is not declared.
        """
        try:
            return self.keys[key_env]
        except KeyError:
            raise ConfigError(f"DART key {key_env!r} has no declared policy in providers.toml") from None


class KisPolicy(BaseModel):
    """KIS credential env names and throughput policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    app_key_env: str
    app_secret_env: str
    account_no_env: str
    account_product_code_env: str
    env_env: str
    circuit_threshold: PositiveInt
    daily_limit: PositiveInt
    investor_flow_rows_per_page: PositiveInt
    min_interval_seconds: PositiveFloat
    max_attempts: PositiveInt
    retryable_msg_codes: tuple[str, ...] = ()

    @field_validator(
        "app_key_env",
        "app_secret_env",
        "account_no_env",
        "account_product_code_env",
        "env_env",
    )
    @classmethod
    def _check_env_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("KIS env names must be non-empty")
        return value


class KrxPolicy(BaseModel):
    """KRX credential env name and throughput policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key_env: str
    circuit_threshold: PositiveInt
    min_interval_seconds: PositiveFloat
    daily_limit: PositiveInt

    @field_validator("api_key_env")
    @classmethod
    def _check_env_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("KRX env names must be non-empty")
        return value


class LsPolicy(BaseModel):
    """LS credential env names and throughput policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    app_key_env: str
    app_secret_env: str
    circuit_threshold: PositiveInt
    max_sessions_per_request: PositiveInt
    min_interval_seconds: PositiveFloat
    daily_limit: PositiveInt
    retryable_rsp_codes: tuple[str, ...] = ()

    @field_validator("app_key_env", "app_secret_env")
    @classmethod
    def _check_env_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("LS env names must be non-empty")
        return value


@dataclass(frozen=True, slots=True)
class RunnerPolicy:
    """Provider-specific safety limits the job runner enforces."""

    quota_provider: str
    daily_budget: int
    daily_reserve: int
    circuit_threshold: int
    avoid_windows_kst: tuple[tuple[str, str], ...]  # empty for providers without a shared-IP window

    def __post_init__(self) -> None:
        if not self.quota_provider.strip():
            raise ValueError("runner policy requires a quota provider")
        if isinstance(self.daily_budget, bool) or int(self.daily_budget) < 1:
            raise ValueError(f"invalid daily_budget {self.daily_budget!r}: must be a positive integer")
        if isinstance(self.daily_reserve, bool) or int(self.daily_reserve) < 0:
            raise ValueError(f"invalid daily_reserve {self.daily_reserve!r}: must be a non-negative integer")
        if isinstance(self.circuit_threshold, bool) or int(self.circuit_threshold) < 1:
            raise ValueError(
                f"invalid circuit_threshold {self.circuit_threshold!r}: must be a positive integer"
            )


class ProviderPolicy(BaseModel):
    """Every provider's throughput and quota policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dart: DartPolicy
    kis: KisPolicy
    krx: KrxPolicy
    ls: LsPolicy
    dividends: DividendPlausibilityPolicy = DividendPlausibilityPolicy()

    @property
    def primary_key_env(self) -> str:
        """Environment variable of the primary DART key (first declared)."""
        return next(iter(self.dart.keys))

    @property
    def default_key_env(self) -> str:
        """Environment variable of the default DART collection key.

        The first declared key marked ``default = true`` wins; without any
        marker the primary key stays the default.
        """
        for key_env, policy in self.dart.keys.items():
            if policy.default:
                return key_env
        return self.primary_key_env

    def dart_key(self, key_env: str) -> DartKeyPolicy:
        """Return the declared policy for a key's environment variable.

        Raises:
            ConfigError: ``key_env`` is not declared.
        """
        return self.dart.dart_key(key_env)

    def runner(self, provider: str, *, key_env: str | None = None) -> RunnerPolicy:
        """Return the safety limits the job runner enforces for one provider.

        Args:
            provider: ``"dart"``, ``"kis"``, ``"krx"`` or ``"ls"``.
            key_env: DART key selecting its budget and reserve; the default
                key applies when omitted. The DART ``quota_provider`` is a
                ``"DART"`` placeholder here: the per-key ledger name depends on
                the key secret, so the DART job context replaces it with the
                resolved ledger.

        Raises:
            ConfigError: ``provider`` is unknown or the DART ``key_env`` is
                not declared.
        """
        name = provider.strip().lower()
        if name == "dart":
            resolved = key_env or self.default_key_env
            key_policy = self.dart.dart_key(resolved)
            return RunnerPolicy(
                quota_provider="DART",
                daily_budget=key_policy.daily_budget,
                daily_reserve=key_policy.daily_reserve,
                circuit_threshold=self.dart.circuit_threshold,
                avoid_windows_kst=tuple(self.dart.shared_ip_avoid_windows_kst),
            )
        if name == "kis":
            return RunnerPolicy(
                quota_provider="KIS",
                daily_budget=self.kis.daily_limit,
                daily_reserve=0,
                circuit_threshold=self.kis.circuit_threshold,
                avoid_windows_kst=(),
            )
        if name == "krx":
            return RunnerPolicy(
                quota_provider="KRX",
                daily_budget=self.krx.daily_limit,
                daily_reserve=0,
                circuit_threshold=self.krx.circuit_threshold,
                avoid_windows_kst=(),
            )
        if name == "ls":
            return RunnerPolicy(
                quota_provider="LS",
                daily_budget=self.ls.daily_limit,
                daily_reserve=0,
                circuit_threshold=self.ls.circuit_threshold,
                avoid_windows_kst=(),
            )
        raise ConfigError(f"unknown provider for runner policy: {provider!r}")


def load_provider_policy(runtime: RuntimeConfig) -> ProviderPolicy:
    """Load ``config/providers.toml`` next to the runtime config.

    Raises:
        ConfigError: file missing, unknown key, or invalid policy values.
    """
    path = Path(runtime.repo_root) / "config" / "providers.toml"
    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"provider policy is missing: {path}") from exc
    except ValueError as exc:
        raise ConfigError(f"provider policy is invalid TOML: {path}") from exc
    try:
        return ProviderPolicy.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(f"provider policy is invalid: {exc}") from exc
