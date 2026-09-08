from datetime import UTC, datetime
from pathlib import Path

from src.core.datasets import DatasetCertification
from src.features.contracts import QvefFeaturePolicy, QvefFeatureRow
from src.features.materialize import materialize_qvef_features


def test_materialize_qvef_features_succeeds(tmp_path: Path) -> None:
    now = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)
    policy = QvefFeaturePolicy(version="v1.0")
    row = QvefFeatureRow(
        decision_session=datetime(2024, 1, 2, tzinfo=UTC),
        instrument_id="KRX:005930",
        sector="Technology",
        gross_profitability=0.2,
        roe=0.15,
        cfo_to_assets=0.1,
        book_to_price=0.8,
        earnings_to_price=0.08,
        operating_income_change=0.05,
        sales_growth=0.1,
        operating_margin_change=0.02,
        foreign_flow_5=0.01,
        foreign_flow_20=0.03,
        quality_score=0.5,
        value_score=0.6,
        earnings_score=0.4,
        foreign_flow_score=0.3,
        component_presence=("quality", "value"),
        source_available_at=((now.isoformat(), now),),
        policy_version="v1.0",
    )

    out_path = materialize_qvef_features(
        (row,),
        root=tmp_path,
        dataset_id="test_ds",
        decision_time=now,
        policy=policy,
        provider_version="p1.0",
        calendar_hash="cal_hash",
        master_hash="master_hash",
        quality_report_hash="qr_hash",
        certification=DatasetCertification.PRODUCTION,
    )

    assert out_path.exists()
    assert (out_path / "dataset_manifest.json").exists()
    assert (out_path / "content_manifest.json").exists()
