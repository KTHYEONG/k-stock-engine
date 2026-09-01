import polars as pl
from pathlib import Path
import sys
from datetime import datetime

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from legacy.stock_yetirank_v1.data.feature_store import FeatureStore
from legacy.config.base import DATA_DIR

def check_readiness(start_date: str = "20160101", end_date: str = "20251231"):
    """
    피처 엔지니어링 수행 전, 필수 데이터(시세, 재무, 지수)가 모두 준비되었는지 확인
    """
    print(f"\n[Feature Engineering Readiness Check]")
    print(f"Target Period: {start_date} ~ {end_date}\n")
    
    store = FeatureStore()
    all_passed = True
    
    # 1. 시세 데이터 (Market Data) 확인
    print("1. Checking Market Data (Price/Volume)...")
    try:
        ldf = store.load_features(start_date=start_date, end_date=end_date)
        
        # 데이터 존재 여부와 스키마 확인을 위해 1줄만 로드 (메모리 절약)
        peek_df = ldf.limit(1).collect()
        
        if peek_df.is_empty():
            print("   ❌ No stock data found for this period.")
            all_passed = False
        else:
            # 전체 개수는 Lazy 연산으로 수행 (Metadata 활용 또는 최적화된 스캔)
            row_count = ldf.select(pl.len()).collect()[0, 0]
            stock_count = ldf.select(pl.col("ticker").n_unique()).collect()[0, 0]
            
            print(f"   ✅ Found {row_count} rows ({stock_count} unique stocks).")
            
            # 필수 컬럼 체크
            required_cols = ["open", "high", "low", "close", "volume", "market_cap"]
            missing = [c for c in required_cols if c not in peek_df.columns]
            if missing:
                print(f"   ❌ Missing core columns: {missing}")
                all_passed = False
            else:
                print("   ✅ Core columns (OHLCV) present.")
    except Exception as e:
        print(f"   ❌ Error loading market data: {e}")
        all_passed = False

    # 2. 지수 데이터 (Indices) 확인
    print("\n2. Checking Index Data (KOSPI/KOSDAQ)...")
    try:
        # 지수 데이터는 120일 이동평균 계산을 위해 과거 데이터가 있는지 확인해야 함
        past_start = (datetime.strptime(start_date, "%Y%m%d").replace(year=int(start_date[:4])-1)).strftime("%Y%m%d")
        
        idx_ldf = store.load_features(start_date=past_start, end_date=end_date)
        idx_ldf = idx_ldf.filter(pl.col("ticker").is_in(["KOSPI", "KOSDAQ"]))
        
        # 그룹별 카운트 계산 (데이터 로드 없이 집계)
        counts_df = idx_ldf.group_by("ticker").len().collect()
        
        if counts_df.is_empty():
            print("   ❌ No index data found.")
            all_passed = False
        else:
            kospi_row = counts_df.filter(pl.col("ticker") == "KOSPI")
            kosdaq_row = counts_df.filter(pl.col("ticker") == "KOSDAQ")
            
            # .len() 집계 결과의 컬럼명은 'len'입니다.
            kospi_cnt = kospi_row["len"][0] if not kospi_row.is_empty() else 0
            kosdaq_cnt = kosdaq_row["len"][0] if not kosdaq_row.is_empty() else 0
            
            print(f"   ✅ Found index history: KOSPI({kospi_cnt}), KOSDAQ({kosdaq_cnt})")
            
            if kospi_cnt < 200 or kosdaq_cnt < 200:
                print("   ⚠️  Warning: Index history might be too short for long-term MA calculation.")
    except Exception as e:
        print(f"   ❌ Error loading index data: {e}")
        all_passed = False

    # 3. 재무 데이터 (Financials) 확인
    print("\n3. Checking Financial Data (OpenDART/File)...")
    fin_path = DATA_DIR / "financials.parquet"
    if not fin_path.exists():
        print(f"   ❌ Financial data file not found at: {fin_path}")
        all_passed = False
    else:
        try:
            fin_df = pl.read_parquet(fin_path)
            if fin_df.is_empty():
                print("   ❌ Financial data file is empty.")
                all_passed = False
            else:
                print(f"   ✅ Found financial records: {len(fin_df)}")
                
                # 필수 컬럼 체크 (feature_engineer 사용시 필요)
                # date 또는 disclosure_date, year 중 하나는 있어야 함
                has_date = "disclosure_date" in fin_df.columns or "date" in fin_df.columns or "year" in fin_df.columns
                if not has_date:
                    print("   ❌ Missing date column (disclosure_date/date/year) in financials.")
                    all_passed = False
                
                req_fin_cols = ["total_assets", "net_income", "total_equity"]
                missing_fin = [c for c in req_fin_cols if c not in fin_df.columns]
                if missing_fin:
                    print(f"   ❌ Missing financial metrics: {missing_fin}")
                    all_passed = False
                else:
                    print("   ✅ Key financial metrics present.")
        except Exception as e:
            print(f"   ❌ Error reading financial file: {e}")
            all_passed = False

    # 최종 결과
    print("\n" + "="*40)
    if all_passed:
        print("🎉 [READY] All data checks passed! Safe to run Feature Engineering.")
    else:
        print("⛔ [NOT READY] Some data is missing or invalid. Please fix errors above.")
    print("="*40)

if __name__ == "__main__":
    check_readiness()
