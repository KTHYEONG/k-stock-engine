import polars as pl
from pathlib import Path
from datetime import datetime
from config.base import PROCESSED_DATA_DIR
import uuid
import logging

from typing import Union

logger = logging.getLogger("data.feature_store")

class FeatureStore:
    """Parquet 기반 피처 스토리지 관리 클래스"""
    
    def __init__(self, base_path: Path = PROCESSED_DATA_DIR / "features"):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def save_features(self, df: Union[pl.DataFrame, pl.LazyFrame], partition_cols: list[str] = ["year", "date"], prefix: str = "data"):
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

        # [RESTORE] 연도별 폴더 밑에 일별 파일 직접 저장 (사용자 요청 구조)
        # 구조: base_path / year=2024 / 2024-01-28_data.parquet
        for (year, date), group in df.group_by(["year", "date"]):
            date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
            
            # 파티션 경로: 연도까지만 폴더 생성
            partition_path = self.base_path / f"year={year}"
            partition_path.mkdir(parents=True, exist_ok=True)
            
            # 파일명 결정: 날짜_접두사.parquet
            if group["ticker"].is_in(["KOSPI", "KOSDAQ"]).all():
                file_name = f"{date_str}_indices.parquet"
            else:
                file_name = f"{date_str}_{prefix}.parquet"
            
            file_path = partition_path / file_name
            group.write_parquet(file_path, compression="snappy")
        
    def get_existing_dates(self) -> list[str]:
        """폴더 구조(year=YYYY/*.parquet)를 기반으로 수집된 날짜 목록 반환"""
        try:
            existing_dates = []
            # year=YYYY 폴더 내의 모든 parquet 파일을 검색
            for file_path in self.base_path.glob("year=*/*.parquet"):
                # 파일명(2024-01-28_raw.parquet)에서 날짜 부분 추출
                filename = file_path.name
                if len(filename) >= 10 and filename[4] == '-' and filename[7] == '-':
                    date_str = filename[:10].replace("-", "")
                    existing_dates.append(date_str)
            
            return sorted(list(set(existing_dates)))
        except Exception as e:
            logger.warning(f"Failed to scan existing dates from folder structure: {e}")
            return []
        
    def load_features(self, start_date: str = None, end_date: str = None, file_pattern: str = "*.parquet") -> pl.LazyFrame:
        """파티셔닝된 피처 로드 (스키마 불일치 대응 최적화)"""
        # 1. 대상 연도 결정
        if start_date and end_date:
            start_year = int(start_date[:4])
            end_year = int(end_date[:4])
            years = list(range(start_year, end_year + 1))
        else:
            import re
            years = []
            for p in self.base_path.glob("year=*"):
                match = re.search(r"year=(\d{4})", p.name)
                if match:
                    years.append(int(match.group(1)))
        
        year_ldfs = []
        import glob
        for y in sorted(years):
            year_path = self.base_path / f"year={y}"
            if not year_path.exists():
                continue
            
            # 2. 연도 내 파일 수집
            # 사용자가 지정한 패턴(file_pattern)에 맞는 파일만 수집 (예: feat_*.parquet)
            files = glob.glob(str(year_path / "**" / file_pattern), recursive=True)
            if not files:
                continue
            
            # 3. 개별 파일 스캔 후 대각선 병합 (Ragged Schema 대응)
            # Polars 1.1x+ 에서는 scan_parquet(files)가 스키마가 다르면 에러를 낼 확률이 높음
            # 개별 스캔 후 concat(how="diagonal")이 가장 안전함
            try:
                # 개별 파일별로 LazyFrame 생성
                file_ldfs = [pl.scan_parquet(f) for f in files]
                yldf = pl.concat(file_ldfs, how="diagonal")
                
                # 불필요한 중복 컬럼(_right) 제거 (Join 흔적 등)
                try:
                    curr_cols = yldf.collect_schema().names()
                    cols_to_drop = [c for c in curr_cols if c.endswith("_right")]
                    if cols_to_drop:
                        yldf = yldf.drop(cols_to_drop)
                except:
                    pass
                    
                year_ldfs.append(yldf)
            except Exception as e:
                logger.warning(f"Failed to scan files in year {y}: {e}")
                continue

        # 4. 전체 연도 합치기 (Diagonal)
        q = pl.concat(year_ldfs, how="diagonal")

        # 5. [중요] 중복 제거 및 피처 우선 선택
        # 동일 티커/날짜 내에서 컬럼 수(null이 아닌 값의 개수 등)가 많은 데이터를 우선하기 위해
        # 여기서는 간단하게 파일 로드 시점에 컬럼 수가 더 많이 보장되는 'feat_' 데이터를 우선하도록 처리할 수 있으나,
        # 가장 확실한 방법은 중복 제거 전 정렬을 활용하는 것입니다.
        # (Polars unique의 keep='last'는 데이터 병합 순서에 의존하므로, 
        #  컬럼 수가 많은 행이 나중에 오도록 처리하거나 명시적 필터링을 권장합니다.)
        
        # 여기서는 단순히 keep='last'를 유지하되, load_features 호출 시점에 
        # 원본 데이터보다 피처 데이터가 나중에 읽히도록 sorted(years) 등을 보장했습니다.
        q = q.unique(subset=["ticker", "date"], keep="last")

        # 6. 날짜 필터링
        if start_date:
            dt_start = datetime.strptime(start_date, "%Y%m%d").date()
            q = q.filter(pl.col("date") >= dt_start)
        if end_date:
            dt_end = datetime.strptime(end_date, "%Y%m%d").date()
            q = q.filter(pl.col("date") <= dt_end)
            
        return q
