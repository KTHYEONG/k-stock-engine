import sys

from src.data.cli import _parse_args


def test_collect_command_requires_immutable_plan_id(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["stock-data", "collect", "--plan-id", "plan-a"])

    args = _parse_args()

    assert args.command == "collect"
    assert args.plan_id == "plan-a"


def test_run_backtest_refuses_without_resolved_execution_components(tmp_path) -> None:
    from argparse import Namespace

    import pytest

    from src.data.cli import _dispatch_backtest
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match="requires resolved Gold artifact"):
        _dispatch_backtest(Namespace(gold_root=tmp_path))
