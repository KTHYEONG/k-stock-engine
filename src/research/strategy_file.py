"""Merged strategy tables after ``extends`` resolution and market-constant filling."""

from __future__ import annotations

import logging
import math
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.config.errors import ConfigError

__all__ = ["resolve_strategy_tables"]

_LOG = logging.getLogger(__name__)

_MAX_CHAIN_FILES = 4

_ALLOWED_TOP = frozenset({"policy", "scorer", "book", "hedge", "trend_overlay", "regime_hedge", "extends"})

_TAX_SECTION = "tax"
_COST_SECTION = "cost"
_KQ150_SECTION = "kosdaq150_futures"
_MINI_SECTION = "mini_kospi200_futures"

_INT_KEYS = frozenset({"futures_annual_deduction_krw", "contract_multiplier_krw"})


def _read_strategy_toml(path: Path) -> dict[str, Any]:
    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    except ValueError as exc:
        raise ValueError(f"invalid strategy TOML: {path}: {exc}") from exc
    return raw


def _check_top_level(raw: dict[str, Any]) -> None:
    unknown = set(raw) - _ALLOWED_TOP
    if unknown:
        raise ValueError(f"unknown strategy keys: {sorted(unknown)}")


def _deep_merge(base: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, value in base.items():
        if key not in child:
            merged[key] = deepcopy(value)
        elif isinstance(value, dict) and isinstance(child[key], dict):
            merged[key] = _deep_merge(value, child[key])
        else:
            merged[key] = deepcopy(child[key])
    for key, value in child.items():
        if key not in base:
            merged[key] = deepcopy(value)
    return merged


def _load_market_sections(path: Path) -> dict[str, dict[str, Any]]:
    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"futures constants are missing: {path}") from exc
    except ValueError as exc:
        raise ConfigError(f"futures constants are invalid TOML: {path}: {exc}") from exc
    expected: dict[str, tuple[str, ...]] = {
        _TAX_SECTION: ("futures_tax_rate", "futures_annual_deduction_krw", "inverse_tax_rate"),
        _COST_SECTION: (
            "futures_cost_rate",
            "inverse_cost_rate",
            "resize_sell_cost_rate",
            "resize_buy_cost_rate",
        ),
        _KQ150_SECTION: ("contract_multiplier_krw", "initial_margin_rate"),
        _MINI_SECTION: ("contract_multiplier_krw", "initial_margin_rate"),
    }
    unknown_sections = set(raw) - set(expected)
    if unknown_sections:
        raise ConfigError(f"futures constants has an unknown section: {sorted(unknown_sections)}")
    sections: dict[str, dict[str, Any]] = {}
    for section, keys in expected.items():
        block = raw.get(section)
        if not isinstance(block, dict):
            raise ConfigError(f"futures constants are missing section: {section!r}")
        unknown = set(block) - set(keys)
        if unknown:
            raise ConfigError(f"futures constants has an unknown key: {sorted(unknown)}")
        missing = [key for key in keys if key not in block]
        if missing:
            raise ConfigError(f"futures constants are missing keys: {missing}")
        sections[section] = dict(block)
    for block in sections.values():
        for key, value in block.items():
            if isinstance(value, bool):
                raise ConfigError(f"futures constants {key!r} must be a number")
            if key in _INT_KEYS:
                if not isinstance(value, int):
                    raise ConfigError(f"futures constants {key!r} must be an int")
                if value < 0:
                    raise ConfigError(f"futures constants {key!r} must be >= 0")
            else:
                if not isinstance(value, (int, float)):
                    raise ConfigError(f"futures constants {key!r} must be a number")
                try:
                    number = float(value)
                except OverflowError as exc:
                    raise ConfigError(f"futures constants {key!r} must be finite and >= 0") from exc
                if not math.isfinite(number) or number < 0.0:
                    raise ConfigError(f"futures constants {key!r} must be finite and >= 0")
                block[key] = number
    return sections


def _supply_map(sections: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    tax = sections[_TAX_SECTION]
    cost = sections[_COST_SECTION]
    kq150 = sections[_KQ150_SECTION]
    mini = sections[_MINI_SECTION]
    return {
        "hedge": {**tax, **cost, **kq150},
        "trend_overlay": {
            "futures_tax_rate": tax["futures_tax_rate"],
            "futures_annual_deduction_krw": tax["futures_annual_deduction_krw"],
            "futures_cost_rate": cost["futures_cost_rate"],
            **mini,
        },
        "regime_hedge": {
            "futures_tax_rate": tax["futures_tax_rate"],
            "futures_annual_deduction_krw": tax["futures_annual_deduction_krw"],
            "futures_cost_rate": cost["futures_cost_rate"],
            **kq150,
        },
    }


def resolve_strategy_tables(path: Path, *, futures_constants: Path) -> dict[str, Any]:
    """Merged strategy tables of ``path`` after ``extends`` resolution and market-constant filling."""
    entry = Path(path)
    chain_paths: list[Path] = []
    chain_docs: list[dict[str, Any]] = []
    seen: set[Path] = set()
    current = entry.absolute()
    for _ in range(_MAX_CHAIN_FILES):
        resolved = current.resolve()
        chain_paths.append(resolved)
        if resolved in seen:
            raise ValueError(f"strategy extends cycle: {' -> '.join(str(p) for p in chain_paths)}")
        seen.add(resolved)
        try:
            raw = _read_strategy_toml(current)
        except FileNotFoundError:
            if not chain_docs:
                raise
            raise ValueError(
                f"strategy extends target is missing: {' -> '.join(str(p) for p in chain_paths)}"
            ) from None
        _check_top_level(raw)
        chain_docs.append(raw)
        target = raw.get("extends")
        if target is None:
            break
        if not isinstance(target, str):
            raise ValueError(
                f"strategy extends must be a relative path string: {' -> '.join(str(p) for p in chain_paths)}"
            )
        candidate = Path(target)
        if candidate.is_absolute():
            raise ValueError(
                f"strategy extends must be a relative path: {' -> '.join(str(p) for p in chain_paths)}"
            )
        current = current.parent / candidate
    else:
        chain_paths.append(current.resolve())
        raise ValueError(
            f"strategy extends chain exceeds {_MAX_CHAIN_FILES} files: "
            f"{' -> '.join(str(p) for p in chain_paths)}"
        )
    merged: dict[str, Any] = {}
    for doc in reversed(chain_docs):
        child = {key: value for key, value in doc.items() if key != "extends"}
        merged = _deep_merge(merged, child)
    sections = _load_market_sections(Path(futures_constants))
    supply = _supply_map(sections)
    for table, values in supply.items():
        block = merged.get(table)
        if not isinstance(block, dict):
            continue
        for key, value in values.items():
            if key not in block:
                block[key] = deepcopy(value)
            else:
                _LOG.info("[DATA] strategy overrides market constant table=%s key=%s", table, key)
    return merged
