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
        
    def save_features(self, df: pl.DataFrame | pl.LazyFrame, partition_cols: list[str] = ["year", "date"]):
        # LazyFrame인 경우 파티셔닝 전처리를 위해 일부 collect가 필요할 수 있으나, 
        # 가급적 전체를 collect하여 저장하는 것이 안전 (write_parquet는 DataFrame 필요)
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        if df.is_empty():
            return

        # date 컬럼을 Date 타입으로 변환
        if "date" in df.columns:
            if df["date"].dtype == pl.Datetime:
                df = df.with_columns(pl.col("date").cast(pl.Date))
            elif df["date"].dtype == pl.Utf8:
                df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y%m%d", strict=False))

        # year 컬럼이 없으면 date 컬럼에서 추출
        if "year" not in df.columns and "date" in df.columns:
            df = df.with_columns(pl.col("date").dt.year().cast(pl.Utf8).alias("year"))

        # [CRITICAL FIX] 덮어쓰기 방지를 위해 개별 날짜별로 유니크한 파일명으로 저장
        import uuid
        
        # 데이터에 포함된 날짜별로 그룹화하여 저장
        for (year, date), group in df.group_by(["year", "date"]):
            date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
            partition_path = self.base_path / f"year={year}" / f"date={date_str}"
            partition_path.mkdir(parents=True, exist_ok=True)
            
            # 유니크한 파일명 생성 (예: data_a1b2c3d4.parquet)
            file_id = uuid.uuid4().hex[:8]
            file_path = partition_path / f"data_{file_id}.parquet"
            
            group.write_parquet(file_path, compression="snappy")
        
    def get_existing_dates(self) -> list[str]:
        """폴더 구조를 기반으로 이미 수집된 날짜 목록을 빠르게 반환"""
        try:
            # Recursive glob으로 모든 date= 파티션 폴더를 찾음
            # 이 방식은 파일을 열지 않으므로 매우 빠름
            existing_dates = []
            for date_path in self.base_path.glob("year=*/date=*"):
                # "date=2021-04-09" 형식에서 날짜 문자열 추출
                date_str = date_path.name.split("=")[-1].replace("-", "")
                if len(date_str) == 8:
                    existing_dates.append(date_str)
            
            return sorted(list(set(existing_dates)))
        except Exception as e:
            logger.warning(f"Failed to scan existing dates from folder structure: {e}")
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

    def load_features(self, start_date: str = None, end_date: str = None) -> pl.LazyFrame:
        """파티셔닝된 피처 로드 (Plan Explosion 방지 최적화)"""
        if start_date and end_date:
            start_year = int(start_date[:4])
            end_year = int(end_date[:4])
            years = list(range(start_year, end_year + 1))
        else:
            # 존재함 폴더 기반으로 연도 추출
            import re
            years = []
            for p in self.base_path.glob("year=*"):
                match = re.search(r"year=(\d{4})", p.name)
                if match:
                    years.append(int(match.group(1)))
        
        year_ldfs = []
        for y in years:
            year_path = self.base_path / f"year={y}"
            if not year_path.exists():
                continue
            
            # 연도 내의 모든 parquet 파일을 하나의 scan으로 시도 (성능 최적화)
            # 하위 폴더(date=...)가 있는 경우를 위해 Recursive Glob 사용
            try:
                # Polars는 리스트 형식의 경로를 지원함 (가장 효율적)
                import glob
                files = glob.glob(str(year_path / "**" / "*.parquet"), recursive=True)
                if not files:
                    continue
                
                # 연도별로 하나의 유닛으로 묶음
                # 스키마가 다르면 여기서 에러가 발생하거나, 추후 collect 시 발생할 수 있음
                # collect 시 발생하면 이 try-except가 잡지 못하므로, 안전하게 diagonal concat을 기본으로 쓰거나
                # 여기서는 명시적으로 allow_missing_columns=True 같은 옵션이 없어서
                # Ragged Schema가 의심되면 무조건 Diagonal Concat을 써야 함.
                
                # 현재 에러는 scan 시점이 아니라 파이프라인 구성 시점에 검증하다 터지는 것.
                # 따라서 여기서는 Fast Scan 대신 Diagonal Concat을 기본으로 사용하는 것이 가장 안전함.
                # 성능 차이가 크지 않다면 안정성을 택함.
                
                # 하지만 성능 최적화를 위해 일단 시도하고, 실패하면 fallback
                yldf = pl.scan_parquet(files)
                
                # 로드 직후 불필요한 중복 컬럼(_right) 제거하여 플랜 경량화
                curr_cols = yldf.collect_schema().names()
                cols_to_drop = [c for c in curr_cols if c.endswith("_right")]
                if cols_to_drop:
                    yldf = yldf.drop(cols_to_drop)
                    
                year_ldfs.append(yldf)
            except Exception as e:
                # logger.warning(f"Year {y} fast scan failed (Schema Mismatch?), falling back to diagonal: {e}")
                # 스키마가 다른 파일들이 섞여 있는 경우 개별 스캔 후 합침
                files = glob.glob(str(year_path / "**" / "*.parquet"), recursive=True)
                if files:
                    yldf = pl.concat([pl.scan_parquet(f) for f in files], how="diagonal")
                    # 여기서도 _right 제거 (스키마 수집은 비용이 들지만 필요함)
                    try:
                         # concat된 LazyFrame은 즉시 스키마 확인이 어려울 수 있으니 try 감쌈
                        curr_cols = yldf.collect_schema().names()
                        cols_to_drop = [c for c in curr_cols if c.endswith("_right")]
                        if cols_to_drop:
                            yldf = yldf.drop(cols_to_drop)
                    except:
                        pass
                    year_ldfs.append(yldf)

        if not year_ldfs:
            return pl.LazyFrame()

        # 연도별로 묶인 단위들을 합침 (전체 파일 개수보다 훨씬 적은 유닛)
        q = pl.concat(year_ldfs, how="diagonal")

        # [CRITICAL FIX] 중복 데이터 제거 (로드 시점에 ticker, date 기준으로 최신/유일 데이터 보장)
        # scan 시점에는 unique를 쓸 수 없으므로, collect 직전에 적용되도록 필터링 구조에 추가
        q = q.unique(subset=["ticker", "date"], keep="last")

        # 필터링 적용
        if start_date:
            dt_start = datetime.strptime(start_date, "%Y%m%d").date()
            q = q.filter(pl.col("date") >= dt_start)
        if end_date:
            dt_end = datetime.strptime(end_date, "%Y%m%d").date()
            q = q.filter(pl.col("date") <= dt_end)
            
        return q
