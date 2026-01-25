import polars as pl
from pathlib import Path
from datetime import datetime
from config.base import PROCESSED_DATA_DIR
import logging

logger = logging.getLogger("data.feature_store")

class FeatureStore:
    """Parquet 기반 피처 스토리지 관리 클래스"""
    
    def __init__(self, base_path: Path = PROCESSED_DATA_DIR / "features"):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def save_features(self, df: pl.DataFrame | pl.LazyFrame, partition_cols: list[str] = ["year"]):
        # LazyFrame인 경우 파티셔닝 전처리를 위해 일부 collect가 필요할 수 있으나, 
        # 가급적 전체를 collect하여 저장하는 것이 안전 (write_parquet는 DataFrame 필요)
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        # date 컬럼을 Date 타입으로 변환
        if "date" in df.columns:
            if df["date"].dtype == pl.Datetime:
                df = df.with_columns(pl.col("date").cast(pl.Date))
            elif df["date"].dtype == pl.Utf8:
                df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y%m%d", strict=False))

        # year 컬럼이 없으면 date 컬럼에서 추출
        if "year" not in df.columns and "date" in df.columns:
            df = df.with_columns(pl.col("date").dt.year().cast(pl.Utf8).alias("year"))

        # 과도한 파티셔닝 방지: ["year"]만 기본값으로 사용
        df.write_parquet(
            self.base_path,
            partition_by=partition_cols,
            compression="snappy",
            use_pyarrow=True # 속도 및 호환성 향상
        )
        
    def get_existing_dates(self) -> list[str]:
        try:
            import glob
            all_files = glob.glob(str(self.base_path / "**" / "*.parquet"), recursive=True)
            if not all_files:
                return []
            
            return (
                pl.concat([pl.scan_parquet(f) for f in all_files], how="diagonal")
                .select("date")
                .unique()
                .collect()
                .get_column("date")
                .dt.strftime("%Y%m%d")
                .to_list()
            )
        except:
            return []
        
    def load_features(self, start_date: str = None, end_date: str = None) -> pl.LazyFrame:
        """파티셔닝된 피처 로드 (LazyFrame 반환)"""
        if start_date and end_date:
            start_year = int(start_date[:4])
            end_year = int(end_date[:4])
            
            paths = []
            for y in range(start_year, end_year + 1):
                year_path = self.base_path / f"year={y}" / "**" / "*.parquet"
                paths.append(str(year_path))
            scan_path = paths
        else:
            scan_path = self.base_path / "**" / "*.parquet"

        try:
            import glob
            all_files = []
            if isinstance(scan_path, list):
                for p in scan_path:
                    all_files.extend(glob.glob(p, recursive=True))
            else:
                all_files = glob.glob(str(scan_path), recursive=True)
            
            if not all_files:
                return pl.LazyFrame()
            
            # Diagonal concat of LazyFrames handles varying schemas across files
            # compatible with older polars versions
            q = pl.concat([pl.scan_parquet(f) for f in all_files], how="diagonal")
            
            if start_date:
                dt_start = datetime.strptime(start_date, "%Y%m%d").date()
                q = q.filter(pl.col("date") >= dt_start)
            if end_date:
                dt_end = datetime.strptime(end_date, "%Y%m%d").date()
                q = q.filter(pl.col("date") <= dt_end)
                
            return q
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            return pl.LazyFrame()
