"""PLAN-03-PURGED-WALK-FORWARD: Purged walk-forward fold isolation."""
from __future__ import annotations

import polars as pl
import pytest

from src.stocks.research.folds import PurgedWalkForward
from tests.fixtures.stocks.helpers import stock_instrument_df


class TestPurgedWalkForward:
    def test_minimum_training_window_offsets_first_validation_block(self) -> None:
        df = stock_instrument_df(n_sessions=60, n_tickers=3, horizon=5)
        splitter = PurgedWalkForward(
            n_folds=1,
            label_horizon_sessions=5,
            embargo_sessions=2,
            session_column="session_index",
            validation_window_sessions=10,
            min_train_sessions=20,
        )

        folds = splitter.split(df)

        assert folds
        fold = folds[0]
        assert fold.num_train() >= 20 * 3
        assert fold.validation_decision_start >= 20 + 5 + 2
        assert fold.train_label_end < fold.validation_decision_start

    def test_no_training_label_interval_intersects_validation(self) -> None:
        df = stock_instrument_df(n_sessions=60, n_tickers=3, horizon=5)
        splitter = PurgedWalkForward(
            n_folds=3, label_horizon_sessions=5, embargo_sessions=2, session_column="session_index"
        )
        folds = splitter.split(df)

        assert folds, "expected at least one fold"
        for fold in folds:
            assert (
                fold.train_label_end < fold.validation_decision_start
            ), "training label interval must terminate before validation decisions"

    def test_embargo_extends_isolation(self) -> None:
        df = stock_instrument_df(n_sessions=60, n_tickers=3, horizon=5)
        no_embargo = PurgedWalkForward(
            n_folds=3, label_horizon_sessions=5, embargo_sessions=0, session_column="session_index"
        )
        with_embargo = PurgedWalkForward(
            n_folds=3, label_horizon_sessions=5, embargo_sessions=2, session_column="session_index"
        )
        for fold_plain, fold_emb in zip(
            no_embargo.split(df), with_embargo.split(df), strict=True
        ):
            assert fold_emb.train_label_end <= fold_plain.train_label_end

    def test_fold_splits_are_disjoint_and_exhaustive_by_row(self) -> None:
        df = stock_instrument_df(n_sessions=60, n_tickers=3, horizon=5)
        splitter = PurgedWalkForward(
            n_folds=3, label_horizon_sessions=5, embargo_sessions=0, session_column="session_index"
        )
        folds = splitter.split(df)
        n_rows = df.height
        for fold in folds:
            assert len(fold.train_mask) + len(fold.validation_mask) <= n_rows
            overlap = set(fold.train_mask) & set(fold.validation_mask)
            assert not overlap, "train and validation rows must be disjoint"

    def test_non_monotonic_timestamps_fail_closed(self) -> None:
        df = stock_instrument_df(n_sessions=10, n_tickers=1)
        # duplicate observation for the same instrument/session is a temporal violation
        dup = pl.concat([df, df.filter(pl.col("session_index") == 3)])
        with pytest.raises(ValueError, match="duplicate"):
            PurgedWalkForward(
                n_folds=2, label_horizon_sessions=2, embargo_sessions=0
            ).split(dup)

    def test_requires_session_column(self) -> None:
        df = stock_instrument_df().drop("session_index")
        with pytest.raises(ValueError, match="session_index"):
            PurgedWalkForward(
                n_folds=2, label_horizon_sessions=2, embargo_sessions=0
            ).split(df)

    def test_insufficient_sessions_fail_closed(self) -> None:
        df = stock_instrument_df(n_sessions=2, n_tickers=1)
        with pytest.raises(ValueError, match="folds"):
            PurgedWalkForward(
                n_folds=3, label_horizon_sessions=2, embargo_sessions=0
            ).split(df)

    def test_inner_folds_never_expose_outer_validation_rows(self) -> None:
        df = stock_instrument_df(n_sessions=60, n_tickers=3, horizon=5)
        splitter = PurgedWalkForward(
            n_folds=3, label_horizon_sessions=5, embargo_sessions=2, session_column="session_index"
        )
        outer = splitter.split(df)
        assert outer
        train_slice = df[outer[-1].train_mask]
        inner = splitter.inner_folds(train_slice)
        outer_val_sessions = set(
            df[outer[-1].validation_mask]["session_index"].unique().to_list()
        )
        assert inner
        for fold in inner:
            assert not (set(fold.train_mask) & set(fold.validation_mask))
            assert fold.train_label_end < fold.validation_decision_start
            inner_val_sessions = set(
                train_slice[fold.validation_mask]["session_index"].unique().to_list()
            )
            assert not (inner_val_sessions & outer_val_sessions)

    def test_holdout_is_pinned_and_reuse_is_rejected(self) -> None:
        df = stock_instrument_df(n_sessions=60, n_tickers=3, horizon=5)
        splitter = PurgedWalkForward(
            n_folds=3, label_horizon_sessions=5, embargo_sessions=2, session_column="session_index"
        )
        holdout = splitter.holdout(df, 10)
        assert len(holdout.validation_mask) > 0
        assert holdout.train_label_end < holdout.validation_decision_start
        splitter.mark_holdout_inspected("candidate_v1")
        with pytest.raises(ValueError, match="already inspected"):
            splitter.pin_holdout("candidate_v1")
