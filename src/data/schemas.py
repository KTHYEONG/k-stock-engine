"""PIT enums, immutable values, schema and primary-key registry."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path


class PITDataError(ValueError):
    """Domain error for PIT dataset validation failures."""


class EvidenceKind(StrEnum):
    CALENDAR = "calendar"
    SECURITY_MASTER = "security_master"
    DAILY_MARKET = "daily_market"
    INVESTOR_FLOW = "investor_flow"
    FINANCIAL_FACTS = "financial_facts"
    CORPORATE_ACTIONS = "corporate_actions"
    DISCLOSURES = "disclosures"
    HISTORICAL_COSTS = "historical_costs"


class SilverTable(StrEnum):
    CALENDAR = "calendar"
    SECURITY_MASTER = "security_master"
    DAILY_MARKET = "daily_market"
    INVESTOR_FLOW = "investor_flow"
    FINANCIAL_FACTS = "financial_facts"
    CORPORATE_ACTIONS = "corporate_actions"
    DISCLOSURES = "disclosures"
    HISTORICAL_COSTS = "historical_costs"


@dataclass(frozen=True, slots=True)
class BronzeReceipt:
    kind: EvidenceKind
    content_hash: str
    source_path: str
    retrieved_at: datetime
    ingested_at: datetime
    payload_path: Path
    metadata_path: Path


@dataclass(frozen=True, slots=True)
class PITSnapshotRequest:
    decision_time: datetime
    required_tables: frozenset[SilverTable]


@dataclass(frozen=True, slots=True)
class CertificationReport:
    certification: object  # DatasetCertification
    report_hash: str
    coverage_start: date
    coverage_end: date
    source_hashes: Mapping[EvidenceKind, str]
