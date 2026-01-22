import polars as pl
from pathlib import Path
from config.base import PROCESSED_DATA_DIR

class FeatureStore:
    """Parquet 기반 피처 스토리지 관리 클래스"""
    
    def __init__(self, base_path: Path = PROCESSED_DATA_DIR / "features"):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def save_features(self, df: pl.DataFrame, partition_cols: list[str] = ["year", "date"]):
        """피처를 Parquet 파티션(year/date)으로 저장"""
        if df.is_empty():
            return
            
        # year 컬럼이 없으면 date 컬럼에서 추출
        if "year" not in df.columns and "date" in df.columns:
            # date가 데이터 타입에 따라 처리 (String or Date)
            if df["date"].dtype == pl.Utf8:
                df = df.with_columns(pl.col("date").str.slice(0, 4).alias("year"))
            else:
                df = df.with_columns(pl.col("date").dt.year().cast(pl.Utf8).alias("year"))

        df.write_parquet(
            self.base_path,
            partition_by=partition_cols,
            compression="snappy"
        )
        
    def get_existing_dates(self) -> list[str]:
        """이미 저장된 파티션 날짜 목록 반환 (YYYYMMDD 형식)"""
        # 하위 모든 단계에서 date= 폴더를 찾음
        existing_dates = []
        for p in self.base_path.glob("**/date=*"):
            if p.is_dir():
                date_val = p.name.split("=")[-1]
                clean_date = date_val.replace("-", "").replace(":", "")
                existing_dates.append(clean_date)
        return list(set(existing_dates)) # 중복 제거
        
    def load_features(self, start_date: str = None, end_date: str = None) -> pl.DataFrame:
        """파티셔닝된 피처 로드 (Lazy 추천)"""
        q = pl.scan_parquet(self.base_path / "**" / "*.parquet")
        
        if start_date:
            q = q.filter(pl.col("date") >= start_date)
        if end_date:
            q = q.filter(pl.col("date") <= end_date)
            
        return q.collect()
