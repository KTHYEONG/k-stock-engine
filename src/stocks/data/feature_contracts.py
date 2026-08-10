"""Feature contracts: typed, immutable declarations for reusable feature panels.

A feature contract declares names, types, source fields, observation and
availability rules, lookback, null policy, and a dependency hash. Feature panels
are label-free projections of one base-panel version; raw/derived name
duplicates must be resolved by an explicit rule or the projection fails closed
(spec acceptance criterion 4).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256

import polars as pl

_TARGET_PREFIXES = ("target_", "label_")


class FeatureSourceKind(StrEnum):
    RAW = "raw"
    DERIVED = "derived"


@dataclass(frozen=True, slots=True)
class DuplicateRule:
    """Explicit resolution for source fields that would collapse to one name.

    ``canonical`` names the surviving source field; ``alternatives`` must not
    be projected for the same feature. Without a rule, a duplicate lineage is
    rejected rather than guessed.
    """

    canonical: str
    alternatives: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.canonical:
            raise ValueError("duplicate rule requires a canonical source field")
        if self.canonical in self.alternatives:
            raise ValueError("canonical field must not appear in alternatives")


@dataclass(frozen=True, slots=True)
class FeatureContract:
    """One feature: names, types, source field, timing rules, and null policy."""

    name: str
    data_type: str
    source_field: str
    source_kind: FeatureSourceKind
    observation_rule: str
    availability_rule: str
    lookback_sessions: int
    null_policy: str
    dependency_hash: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("feature name must be non-empty")
        if self.name.startswith(_TARGET_PREFIXES):
            raise ValueError(f"feature name {self.name!r} uses a reserved target prefix")
        if not self.source_field:
            raise ValueError(f"feature {self.name} requires a source_field")
        if self.lookback_sessions < 0:
            raise ValueError("lookback_sessions must be non-negative")

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "source_field": self.source_field,
            "source_kind": self.source_kind.value,
            "observation_rule": self.observation_rule,
            "availability_rule": self.availability_rule,
            "lookback_sessions": self.lookback_sessions,
            "null_policy": self.null_policy,
            "dependency_hash": self.dependency_hash,
        }


def feature_dependency_hash(fields: Mapping[str, object]) -> str:
    """Deterministic dependency hash binding the contract's own declarations."""
    payload = dict(fields)
    payload.pop("dependency_hash", None)
    return sha256(
        f"{payload['name']}:{payload['data_type']}:{payload['source_field']}:"
        f"{payload['source_kind']}:{payload['observation_rule']}:"
        f"{payload['availability_rule']}:{payload['lookback_sessions']}:"
        f"{payload['null_policy']}".encode()
    ).hexdigest()


def make_feature_contract(
    *,
    name: str,
    data_type: str = "Float64",
    source_field: str,
    source_kind: FeatureSourceKind = FeatureSourceKind.RAW,
    observation_rule: str = "session_close",
    availability_rule: str = "next_session_open",
    lookback_sessions: int = 0,
    null_policy: str = "retain_null",
) -> FeatureContract:
    """Build a contract with a deterministic dependency hash."""
    fields = {
        "name": name,
        "data_type": data_type,
        "source_field": source_field,
        "source_kind": source_kind.value,
        "observation_rule": observation_rule,
        "availability_rule": availability_rule,
        "lookback_sessions": lookback_sessions,
        "null_policy": null_policy,
    }
    dependency_hash = feature_dependency_hash(fields)
    return FeatureContract(
        name=name,
        data_type=data_type,
        source_field=source_field,
        source_kind=source_kind,
        observation_rule=observation_rule,
        availability_rule=availability_rule,
        lookback_sessions=lookback_sessions,
        null_policy=null_policy,
        dependency_hash=dependency_hash,
    )


def feature_set_hash(contracts: tuple[FeatureContract, ...]) -> str:
    """Deterministic fingerprint of an ordered feature-contract set."""
    payload = "\n".join(
        "\t".join(
            (
                contract.name,
                contract.data_type,
                contract.source_field,
                contract.dependency_hash,
            )
        )
        for contract in contracts
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def feature_contract_book_from_allowlist(
    version: str,
    allowlist: tuple[str, ...],
    duplicate_rules: tuple[DuplicateRule, ...] = (),
) -> FeatureContractBook:
    """Build a contract book whose contracts mirror a frozen allowlist."""
    contracts = tuple(
        make_feature_contract(name=name, source_field=name) for name in allowlist
    )
    return FeatureContractBook(version=version, contracts=contracts, duplicate_rules=duplicate_rules)


def resolve_raw_source_names(
    source_columns: tuple[str, ...],
    duplicate_rules: tuple[DuplicateRule, ...],
) -> tuple[str, ...]:
    """Resolve raw/derived name duplicates to a canonical column set.

    When a canonical source and its alternatives are all present, only the
    canonical survives. When the canonical is absent, a single alternative is
    retained; two or more alternatives with no canonical are ambiguous and
    rejected. Columns not covered by any rule pass through unchanged.
    """
    result = list(source_columns)
    for rule in duplicate_rules:
        present_alternatives = [a for a in rule.alternatives if a in result]
        if rule.canonical in result:
            result = [c for c in result if c not in rule.alternatives]
        elif len(present_alternatives) > 1:
            raise ValueError(
                f"ambiguous duplicate lineage for {rule.canonical}: {present_alternatives}"
            )
    return tuple(result)


def contracts_to_json(contracts: tuple[FeatureContract, ...]) -> list[dict[str, object]]:
    return [contract.to_json() for contract in contracts]


def contracts_from_json(payload: list[dict[str, object]]) -> tuple[FeatureContract, ...]:
    contracts: list[FeatureContract] = []
    for raw in payload:
        source_kind = FeatureSourceKind(str(raw["source_kind"]))
        contracts.append(
            FeatureContract(
                name=str(raw["name"]),
                data_type=str(raw["data_type"]),
                source_field=str(raw["source_field"]),
                source_kind=source_kind,
                observation_rule=str(raw["observation_rule"]),
                availability_rule=str(raw["availability_rule"]),
                lookback_sessions=int(str(raw["lookback_sessions"])),
                null_policy=str(raw["null_policy"]),
                dependency_hash=str(raw.get("dependency_hash", "")),
            )
        )
    return tuple(contracts)


class FeatureContractBook:
    """A versioned, immutable set of feature contracts and duplicate rules."""

    def __init__(
        self,
        *,
        version: str,
        contracts: tuple[FeatureContract, ...],
        duplicate_rules: tuple[DuplicateRule, ...] = (),
    ):
        if not version:
            raise ValueError("feature contract book requires a version")
        self.version = version
        self.contracts = contracts
        self.duplicate_rules = duplicate_rules
        by_name: dict[str, FeatureContract] = {}
        for contract in contracts:
            if contract.name in by_name:
                raise ValueError(f"duplicate feature contract for {contract.name}")
            by_name[contract.name] = contract
        self._by_name = by_name

    @property
    def schema_hash(self) -> str:
        return feature_set_hash(self.contracts)

    def contract_for(self, name: str) -> FeatureContract:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise ValueError(f"no feature contract for {name!r}") from exc

    def resolve_source_names(
        self, source_columns: tuple[str, ...]
    ) -> dict[str, FeatureContract]:
        """Resolve raw source columns to canonical feature contracts.

        Duplicate rule groups resolve first: the canonical source wins when
        present, otherwise a single alternative survives; two or more
        alternatives with no canonical are ambiguous and rejected. At most one
        contract is emitted per rule group, so a shadowed alternative never
        yields a duplicate feature. Sources outside any rule group pass through
        when present.
        """
        present = set(source_columns)
        resolved: dict[str, FeatureContract] = {}
        grouped = self._grouped_sources()
        handled: set[str] = set()

        for rule in self.duplicate_rules:
            canonical = rule.canonical
            alternatives_present = sorted(set(rule.alternatives) & present)
            if canonical in present:
                winner = canonical
            elif len(alternatives_present) == 1:
                winner = alternatives_present[0]
            elif len(alternatives_present) > 1:
                raise ValueError(
                    f"ambiguous duplicate lineage for {canonical}: {alternatives_present}"
                )
            else:
                continue
            group_contracts = [
                c for c in self.contracts if c.source_field == canonical
                or c.source_field in rule.alternatives
            ]
            chosen = next(
                (c for c in group_contracts if c.source_field == canonical), None
            ) or next((c for c in group_contracts if c.source_field == winner), None)
            if chosen is not None:
                resolved[winner] = chosen
                handled.add(chosen.name)

        for contract in self.contracts:
            if contract.name in handled or contract.source_field in grouped:
                continue
            if contract.source_field in present:
                resolved[contract.source_field] = contract
        return resolved

    def _rule_for(self, source: str) -> DuplicateRule | None:
        for rule in self.duplicate_rules:
            if rule.canonical == source or source in rule.alternatives:
                return rule
        return None

    def _grouped_sources(self) -> set[str]:
        grouped: set[str] = set()
        for rule in self.duplicate_rules:
            grouped.add(rule.canonical)
            grouped.update(rule.alternatives)
        return grouped

    def project(
        self,
        base_panel: pl.DataFrame,
        source_prefix: str = "raw__",
    ) -> pl.DataFrame:
        """Project a wide, label-free ``feature__*`` frame from a base panel.

        Only columns carrying the configured source prefix are eligible; any
        ``target_``/``label_`` source column is rejected outright. Unknown or
        missing source columns for a declared contract are an error, never a
        silent drop.
        """
        source_columns = tuple(
            column[len(source_prefix) :]
            for column in base_panel.columns
            if column.startswith(source_prefix)
        )
        if any(column.startswith(_TARGET_PREFIXES) for column in source_columns):
            raise ValueError("feature panel projection rejects target_/label_ columns")
        resolved = self.resolve_source_names(source_columns)
        features = [
            pl.col(f"{source_prefix}{source}").alias(f"feature__{contract.name}")
            for source, contract in sorted(resolved.items(), key=lambda item: item[1].name)
        ]
        identity = [c for c in ("instrument_id", "session") if c in base_panel.columns]
        missing = [c for c in identity if c not in base_panel.columns]
        if missing:
            raise ValueError(f"base panel missing identity columns {missing}")
        return base_panel.select([*identity, *features])
