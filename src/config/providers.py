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
    "DbsecPolicy",
    "DisclosureFilter",
    "DividendPlausibilityPolicy",
    "DocumentParserPolicy",
    "EarningsReleasePolicy",
    "KindPolicy",
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


class EarningsReleasePolicy(BaseModel):
    """Acceptance policy for early earnings releases entering Silver.

    A preliminary result whose labelled fiscal quarter is implausibly far from its receipt date is
    withheld, never re-dated: the label is the only period evidence in the filing, and a stale or
    mislabelled period would place numbers at the wrong point in time.

    Attributes:
        max_filing_lag_days: Largest allowed distance in days from the labelled quarter end to the
            receipt date of an original (non-correction) preliminary filing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_filing_lag_days: int = 135

    @field_validator("max_filing_lag_days")
    @classmethod
    def _check_lag(cls, value: int) -> int:
        if isinstance(value, bool) or int(value) < 1:
            raise ValueError(f"invalid max_filing_lag_days {value!r}: must be a positive integer")
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
    host_min_interval_seconds: PositiveFloat
    """Minimum spacing between any two DART HTTP attempts from this host, across all keys and processes.

    Why: OpenDART blocks by IP, so per-key pacing alone multiplies the host rate by the number of key
    processes. Must be >= ``MIN_DART_INTERVAL_SECONDS`` (the measured per-key floor).
    """
    shared_ip_avoid_windows_kst: list[tuple[str, str]] = []
    disclosure_types: tuple[str, ...] = ("A", "I001")
    corp_codes_max_age_days: PositiveInt = 30
    document_parser: DocumentParserPolicy = DocumentParserPolicy()
    keys: dict[str, DartKeyPolicy]
    @field_validator("host_min_interval_seconds")
    @classmethod
    def _check_host_interval(cls, value: float) -> float:
        if float(value) < MIN_DART_INTERVAL_SECONDS:
            raise ValueError(
                f"host_min_interval_seconds below {MIN_DART_INTERVAL_SECONDS} exceeds the measured safe rate"
            )
        return float(value)

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
    pool_slot: PositiveInt | None = None
    admission_class: Literal["critical", "standard", "bulk"] = "bulk"
    admission_dir_env: str = "KIS_TOKEN_CACHE_DIR"
    require_shared_admission: bool = False

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


class DbsecPolicy(BaseModel):
    """DB Securities credential env names, host pacing intervals and response-code classes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    app_key_env: str
    app_secret_env: str
    circuit_threshold: PositiveInt
    daily_limit: PositiveInt
    app_min_interval_seconds: PositiveFloat
    scope_min_interval_seconds: dict[Literal["chart", "quote_master"], PositiveFloat]
    token_min_interval_seconds: PositiveFloat
    token_cache_dir_env: str
    retryable_rsp_codes: tuple[str, ...]
    auth_rsp_codes: tuple[str, ...]
    session_final_time_kst: str
    lookback_sessions: PositiveInt
    closing_bar_count: int
    chunk_units: PositiveInt
    equity_market_names: tuple[str, ...]
    master_valid_until_kst: str

    @field_validator("app_key_env", "app_secret_env", "token_cache_dir_env")
    @classmethod
    def _check_env_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("DB env names must be non-empty")
        return value

    @field_validator("scope_min_interval_seconds")
    @classmethod
    def _check_scopes(
        cls, value: dict[Literal["chart", "quote_master"], PositiveFloat]
    ) -> dict[Literal["chart", "quote_master"], PositiveFloat]:
        if set(value) != {"chart", "quote_master"}:
            raise ValueError("scope_min_interval_seconds must declare chart and quote_master")
        return value

    @field_validator("token_min_interval_seconds")
    @classmethod
    def _check_token_interval(cls, value: float) -> float:
        if float(value) < 60.0:
            raise ValueError(f"invalid token_min_interval_seconds {value!r}: must be >= 60.0")
        return float(value)

    @field_validator("session_final_time_kst", "master_valid_until_kst")
    @classmethod
    def _check_hhmm(cls, value: str) -> str:
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", str(value)):
            raise ValueError(f"invalid KST time {value!r}: expected HH:MM")
        return str(value)

    @field_validator("closing_bar_count")
    @classmethod
    def _check_bar_count(cls, value: int) -> int:
        if isinstance(value, bool) or not 1 <= int(value) <= 2000:
            raise ValueError(f"invalid closing_bar_count {value!r}: expected [1, 2000]")
        return int(value)

    @field_validator("equity_market_names")
    @classmethod
    def _check_market_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
        if not cleaned or len(cleaned) != len(tuple(value)):
            raise ValueError("equity_market_names must be non-empty with no blank entries")
        return cleaned

    @model_validator(mode="after")
    def _check_code_classes(self) -> DbsecPolicy:
        overlap = set(self.retryable_rsp_codes) & set(self.auth_rsp_codes)
        if overlap:
            raise ValueError(f"retryable and auth rsp codes overlap: {sorted(overlap)}")
        return self


class KindPolicy(BaseModel):
    """KIND collection policy: pacing, ledger bound and the notice vocabulary to search.

    Attributes:
        circuit_threshold: Consecutive transport failures that abort a run.
        min_interval_seconds: Minimum spacing between requests to a public web service.
        daily_limit: Ledger bound for one KST day.
        search_keywords: Title keywords searched per window; each keyword is its own window key.
        document_titles: Normalized titles (brackets and spaces removed) whose bodies are fetched.
        max_stale_sessions: Observed sessions KIND evidence may lag the decision horizon.
        relay_window_lookback_days: Closed windows ending within this many days are re-fetched.
        relay_document_lookback_days: Notices disclosed within this many days get bodies fetched.
        relay_max_age_hours: A bundle older than this is refused.
        relay_clock_skew_seconds: Tolerated producer-clock lead over the importer clock.
        relay_attempts: Producer retries for one inconsistent window.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    circuit_threshold: PositiveInt
    min_interval_seconds: PositiveFloat
    daily_limit: PositiveInt
    search_keywords: tuple[str, ...]
    document_titles: tuple[str, ...]
    max_stale_sessions: NonNegativeInt
    relay_window_lookback_days: PositiveInt = 14
    relay_document_lookback_days: PositiveInt = 14
    relay_max_age_hours: PositiveInt = 36
    relay_clock_skew_seconds: NonNegativeInt = 300
    relay_attempts: PositiveInt = 3

    @field_validator("search_keywords", "document_titles")
    @classmethod
    def _check_vocabulary(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
        if not cleaned or len(cleaned) != len(tuple(value)):
            raise ValueError("KIND vocabulary must be non-empty with no blank entries")
        return cleaned


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
    dbsec: DbsecPolicy
    kind: KindPolicy
    dividends: DividendPlausibilityPolicy = DividendPlausibilityPolicy()
    earnings_releases: EarningsReleasePolicy = EarningsReleasePolicy()

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
            provider: ``"dart"``, ``"kis"``, ``"krx"``, ``"ls"``, ``"dbsec"`` or ``"kind"``.
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
        if name == "dbsec":
            return RunnerPolicy(
                quota_provider="DBSEC",
                daily_budget=self.dbsec.daily_limit,
                daily_reserve=0,
                circuit_threshold=self.dbsec.circuit_threshold,
                avoid_windows_kst=(),
            )
        if name == "kind":
            return RunnerPolicy(
                quota_provider="KIND",
                daily_budget=self.kind.daily_limit,
                daily_reserve=0,
                circuit_threshold=self.kind.circuit_threshold,
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
