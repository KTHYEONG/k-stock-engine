"""Point-in-time shared contracts owned by core."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path

from src.core.datasets import DatasetCertification


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
    LIFECYCLE_EVENTS = "lifecycle_events"


class SilverTable(StrEnum):
    CALENDAR = "calendar"
    SECURITY_MASTER = "security_master"
    DAILY_MARKET = "daily_market"
    INVESTOR_FLOW = "investor_flow"
    FINANCIAL_FACTS = "financial_facts"
    CORPORATE_ACTIONS = "corporate_actions"
    DISCLOSURES = "disclosures"
    HISTORICAL_COSTS = "historical_costs"
    LIFECYCLE_EVENTS = "lifecycle_events"


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
    certification: DatasetCertification
    report_hash: str
    coverage_start: date
    coverage_end: date
    source_hashes: Mapping[EvidenceKind, str]
