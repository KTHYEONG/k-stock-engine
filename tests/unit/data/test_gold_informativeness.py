def test_evaluate_feature_coverage_reports_non_null_ratios() -> None:
    import polars as pl

    from src.data.gold_informativeness import FeatureCoverageFloor, evaluate_feature_coverage

    frame = pl.DataFrame({"champion_score": [1.0, None, 3.0, None], "rank": [1, 2, 3, 4]})
    floors = (FeatureCoverageFloor(column="champion_score", min_non_null_ratio=0.5),)

    report = evaluate_feature_coverage(frame=frame, floors=floors)

    assert report.ratios["champion_score"] == 0.5
    assert report.violations == ()

def test_certify_informative_gold_rejects_all_null_feature() -> None:
    import polars as pl
    import pytest

    from src.data.gold_informativeness import FeatureCoverageFloor, certify_informative_gold
    from src.data.schemas import PITDataError

    frame = pl.DataFrame({"champion_score": [None, None, None], "rank": [None, None, None]})
    floors = (FeatureCoverageFloor(column="champion_score", min_non_null_ratio=0.2),)

    with pytest.raises(PITDataError, match="champion_score"):
        certify_informative_gold(frame=frame, floors=floors, dataset_label="champion_scores/gold-x")

def test_certify_informative_gold_rejects_below_floor_breadth() -> None:
    import polars as pl
    import pytest

    from src.data.gold_informativeness import FeatureCoverageFloor, certify_informative_gold
    from src.data.schemas import PITDataError

    frame = pl.DataFrame({"champion_score": [1.0] + [None] * 99})
    floors = (FeatureCoverageFloor(column="champion_score", min_non_null_ratio=0.2),)

    with pytest.raises(PITDataError, match=r"0\.01"):
        certify_informative_gold(frame=frame, floors=floors, dataset_label="champion_scores/gold-x")

def test_certify_informative_gold_accepts_frame_meeting_floors() -> None:
    import polars as pl

    from src.data.gold_informativeness import FeatureCoverageFloor, certify_informative_gold

    frame = pl.DataFrame({"champion_score": [1.0, 2.0, None, 4.0]})
    floors = (FeatureCoverageFloor(column="champion_score", min_non_null_ratio=0.5),)

    report = certify_informative_gold(
        frame=frame, floors=floors, dataset_label="champion_scores/gold-x"
    )

    assert report.violations == ()
    assert report.ratios["champion_score"] == 0.75

def test_certify_informative_gold_rejects_missing_declared_column() -> None:
    import polars as pl
    import pytest

    from src.data.gold_informativeness import FeatureCoverageFloor, certify_informative_gold
    from src.data.schemas import PITDataError

    frame = pl.DataFrame({"rank": [1, 2, 3]})
    floors = (FeatureCoverageFloor(column="champion_score", min_non_null_ratio=0.2),)

    with pytest.raises(PITDataError, match="missing declared feature column"):
        certify_informative_gold(frame=frame, floors=floors, dataset_label="champion_scores/gold-x")

def test_certify_informative_gold_rejects_empty_frame() -> None:
    import polars as pl
    import pytest

    from src.data.gold_informativeness import FeatureCoverageFloor, certify_informative_gold
    from src.data.schemas import PITDataError

    frame = pl.DataFrame(schema={"champion_score": pl.Float64})
    floors = (FeatureCoverageFloor(column="champion_score", min_non_null_ratio=0.2),)

    with pytest.raises(PITDataError, match="no rows"):
        certify_informative_gold(frame=frame, floors=floors, dataset_label="champion_scores/gold-x")

def test_feature_coverage_floor_rejects_ratio_outside_unit_interval() -> None:
    import pytest

    from src.data.gold_informativeness import FeatureCoverageFloor

    with pytest.raises(ValueError, match="min_non_null_ratio"):
        FeatureCoverageFloor(column="champion_score", min_non_null_ratio=1.5)
    with pytest.raises(ValueError, match="min_non_null_ratio"):
        FeatureCoverageFloor(column="champion_score", min_non_null_ratio=-0.1)
    with pytest.raises(ValueError, match="column"):
        FeatureCoverageFloor(column="", min_non_null_ratio=0.2)

def test_champion_score_coverage_floors_declare_breadth_minimum() -> None:
    from src.data.gold_informativeness import CHAMPION_SCORE_COVERAGE_FLOORS

    columns = {floor.column: floor.min_non_null_ratio for floor in CHAMPION_SCORE_COVERAGE_FLOORS}

    assert columns == {"champion_score": 0.2, "rank": 0.2}

