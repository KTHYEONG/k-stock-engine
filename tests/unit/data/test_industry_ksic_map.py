"""Invariant guards for the unanimous KSIC-to-industry mapping."""

from __future__ import annotations

import pytest

from src.data.industry_ksic_map import learn_ksic_industry_mapping
from src.data.schemas import PITDataError


def test_lookup_maps_unanimous_code_with_distinct_ticker_support() -> None:
    mapping = learn_ksic_industry_mapping([
        ("005930", "032604", "전기·전자"),
        ("000660", "032604", "전기·전자"),
        ("035420", "032604", "전기·전자"),
    ])

    assert mapping.lookup("032604") == ("전기·전자", 3)


def test_repeated_observations_of_one_ticker_do_not_inflate_support() -> None:
    mapping = learn_ksic_industry_mapping([
        ("005930", "032604", "전기·전자"),
        ("005930", "032604", "전기·전자"),
    ])

    assert mapping.lookup("032604") == ("전기·전자", 1)


def test_conflicting_code_is_unusable() -> None:
    mapping = learn_ksic_industry_mapping([
        ("005930", "032604", "전기·전자"),
        ("000660", "032604", "화학"),
    ])

    assert "032604" in mapping.conflicting_ksic
    assert "032604" not in mapping.industry_by_ksic
    assert mapping.lookup("032604") is None


def test_lookup_does_not_generalize_prefixes() -> None:
    mapping = learn_ksic_industry_mapping([("005930", "032604", "전기·전자")])

    assert mapping.lookup("032605") is None
    assert mapping.lookup("0326") is None


def test_lookup_rejects_malformed_codes_without_raising() -> None:
    mapping = learn_ksic_industry_mapping([("005930", "032604", "전기·전자")])

    assert mapping.lookup("") is None
    assert mapping.lookup("32604") is None
    assert mapping.lookup("03260a") is None
    assert mapping.lookup(32604) is None  # type: ignore[arg-type]


def test_learn_rejects_malformed_code_or_blank_industry() -> None:
    with pytest.raises(PITDataError):
        learn_ksic_industry_mapping([("005930", "32604", "전기·전자")])
    with pytest.raises(PITDataError):
        learn_ksic_industry_mapping([("005930", "03260a", "전기·전자")])
    with pytest.raises(PITDataError):
        learn_ksic_industry_mapping([("005930", "032604", "   ")])


def test_learn_rejects_contradicting_observations_for_one_ticker() -> None:
    with pytest.raises(PITDataError, match="005930"):
        learn_ksic_industry_mapping([
            ("005930", "032604", "전기·전자"),
            ("005930", "032604", "화학"),
        ])


def test_learn_is_independent_of_observation_order() -> None:
    observations = [
        ("005930", "032604", "전기·전자"),
        ("000660", "032604", "전기·전자"),
        ("051910", "011101", "화학"),
        ("035420", "032604", "전기·전자"),
    ]

    forward = learn_ksic_industry_mapping(observations)
    backward = learn_ksic_industry_mapping(list(reversed(observations)))

    assert forward == backward
    assert forward.lookup("032604") == ("전기·전자", 3)
    assert forward.lookup("011101") == ("화학", 1)
