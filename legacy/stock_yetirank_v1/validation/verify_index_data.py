import polars as pl
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from legacy.stock_yetirank_v1.data.feature_store import FeatureStore

def verify_indices():
    store = FeatureStore()
    
    print("Checking stored index data (KOSPI/KOSDAQ)...")
    
    try:
        # 지수 데이터 로드 (최근 2년치 정도 확인)
        df = store.load_features(start_date="20240101", end_date="20241231")
        
        if df.is_empty():
            print("No data found in FeatureStore.")
            return

        # 지수 데이터만 필터링
        indices_df = df.filter(pl.col("ticker").is_in(["KOSPI", "KOSDAQ"]))
        
        if indices_df.is_empty():
            print("❌ No index data (KOSPI/KOSDAQ) found in the store.")
            print(f"Sample tickers available in store: {df['ticker'].unique().head().to_list()}")
            return

        # 티커별 데이터 개수 확인
        stats = indices_df.group_by("ticker").agg(
            pl.count("date").alias("count"),
            pl.col("date").min().alias("start_date"),
            pl.col("date").max().alias("end_date")
        ).sort("ticker")
        
        print("\n[Index Data Statistics]")
        print(stats)
        
        print("\n[Recent Data Sample (KOSPI)]")
        print(indices_df.filter(pl.col("ticker") == "KOSPI").sort("date").tail(5))
        
        print("\n✅ Index data verification completed.")

    except Exception as e:
        print(f"❌ Verification failed: {e}")

if __name__ == "__main__":
    verify_indices()
