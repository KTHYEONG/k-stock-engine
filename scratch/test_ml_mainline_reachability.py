"""Static reachability audit for the stock ML default path."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def main() -> None:
    train_cli = _read("src/stocks/cli/train.py")
    materializer = _read("src/stocks/data/research_v2.py")
    trainer = _read("src/stocks/workflows/train_model.py")
    legacy_files = (
        "src/stocks/research/lambdarank.py",
        "src/stocks/workflows/candidate_search.py",
        "src/stocks/workflows/economic_selection.py",
        "src/stocks/workflows/training_run_store.py",
    )
    report = {
        "cli_composes_v2": "feature_set=STOCK_ALPHA_V2_FEATURE_SET" in train_cli,
        "cli_mentions_lambdarank": "LambdaRank" in train_cli,
        "v3_materializer_builds_v2_features": (
            "def materialize_stock_alpha_v3_snapshot" in materializer
            and "feature_set=STOCK_ALPHA_V2_FEATURE_SET" in materializer
        ),
        "v3_dispatch_exists": "if manifest.feature_set == STOCK_ALPHA_V3_FEATURE_SET" in trainer,
        "v3_discards_transformed_frame": "_transformed, learner_columns" in trainer,
        "legacy_train_model_lines": len(trainer.splitlines()),
        "horizon_shorthand_occurrences": trainer.count("h{route.horizon}")
        + trainer.count("h{route_horizon}"),
        "legacy_file_lines": {
            path: len(_read(path).splitlines()) for path in legacy_files
        },
    }
    for key, value in report.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
