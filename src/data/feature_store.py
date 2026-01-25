import polars as pl
from pathlib import Path
from datetime import datetime
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
            q = pl.scan_parquet(scan_path)
            
            if start_date:
                dt_start = datetime.strptime(start_date, "%Y%m%d")
                q = q.filter(pl.col("date") >= dt_start)
            if end_date:
                dt_end = datetime.strptime(end_date, "%Y%m%d")
                q = q.filter(pl.col("date") <= dt_end)
                
            return q.collect()
        except Exception:
            # 스키마 불일치(Schema Mismatch) 대비 Safe Mode: Diagonal Concat
            import glob
            
            all_files = []
            if isinstance(scan_path, list):
                for p in scan_path:
                    # glob은 문자열 경로를 받으므로 변환
                    all_files.extend(glob.glob(p, recursive=True))
            else:
                all_files = glob.glob(str(scan_path), recursive=True)
            
            if not all_files:
                return pl.DataFrame()
            
            dfs = []
            for f in all_files:
                try:
                    df = pl.read_parquet(f)
                    dfs.append(df)
                except:
                    continue
            
            if not dfs:
                return pl.DataFrame()
                
            # how="diagonal"은 서로 다른 컬럼을 null로 채우며 합침
            full_df = pl.concat(dfs, how="diagonal")
            
            # 필터링 적용
            if start_date:
                full_df = full_df.filter(pl.col("date") >= datetime.strptime(start_date, "%Y%m%d"))
            if end_date:
                full_df = full_df.filter(pl.col("date") <= datetime.strptime(end_date, "%Y%m%d"))
                
            return full_df
