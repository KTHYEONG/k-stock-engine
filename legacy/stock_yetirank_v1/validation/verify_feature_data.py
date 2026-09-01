import polars as pl
from pathlib import Path
import sys
import numpy as np

# 프로젝트 루트 경로 계산
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from legacy.stock_yetirank_v1.data.feature_store import FeatureStore

def main():
    store = FeatureStore()
    
    # 1. Load Data (Recent sample for speed, or full year if needed)
    # 2024년 데이터로 검증
    year = "2025"
    print(f"Loading data for verification (Year: {year})...")
    
    try:
        # feat_*.parquet 패턴을 사용하여 피처 엔지니어링 결과물만 로드
        ldf = store.load_features(start_date=f"{year}0101", end_date=f"{year}1231", file_pattern="feat_*.parquet")
        
        # Check if data exists (Lazy)
        peek_df = ldf.limit(1).collect()
        if peek_df.is_empty():
            print(f"❌ No data found for {year}.")
            return
            
        row_count = ldf.select(pl.len()).collect()[0, 0]
        print(f"✅ Loaded {row_count} rows.")
        
        # Schema for existence check
        schema_cols = ldf.collect_schema().names()

        # 2. Define Expected Feature Groups
        feature_groups = {
            "Technical": [
                "log_return_5d", "volatility_60d", "disparity_60d", "disparity_120d",
                "volume_ratio_20d", "intraday_vol", "amihud_20d"
            ],
            "Flow": [
                "net_purchase_total", "np_mkt_cap", "np_vol", "z_flow"
            ],
            "Fundamental": [
                "bp_ratio", "ep_ratio", "roe", "debt_ratio", 
                "capital_erosion_rate", "relative_trend_score"
            ],
            "Target": [
                "target_return_5d", "target_rank"
            ]
        }

        # 3. Validation Loop
        all_passed = True
        
        for group, features in feature_groups.items():
            print(f"\n[{group} Features Check]")
            
            missing = [f for f in features if f not in schema_cols]
            if missing:
                print(f"⚠️  Missing Columns: {missing}")
                all_passed = False
            
            existing = [f for f in features if f in schema_cols]
            if not existing:
                continue
                
            # Calculate Stats (Null, Inf, Mean, Std) - Lazy Aggregation
            stats_exprs = []
            for f in existing:
                stats_exprs.extend([
                    pl.col(f).null_count().alias(f"{f}_null"),
                    pl.col(f).mean().alias(f"{f}_mean"),
                    pl.col(f).std().alias(f"{f}_std")
                ])
                
            stats = ldf.select(stats_exprs).collect()
            
            # Print Formatted Stats
            print(f"{'Feature':<20} | {'Null %':<10} | {'Mean':<10} | {'Std':<10}")
            print("-" * 60)
            
            for f in existing:
                null_cnt = stats[f"{f}_null"][0]
                mean_val = stats[f"{f}_mean"][0]
                std_val = stats[f"{f}_std"][0]
                
                # Handle None values elegantly
                mean_val = mean_val if mean_val is not None else float('nan')
                std_val = std_val if std_val is not None else float('nan')
                
                null_pct = (null_cnt / row_count) * 100
                
                print(f"{f:<20} | {null_pct:>9.2f}% | {mean_val:>10.4f} | {std_val:>10.4f}")
                
                # Check for anomalies
                if null_pct > 20:
                    print(f"   ⚠️  High Null Rate: {f} ({null_pct:.1f}%)")
                
                # Check for infinite values (using strict logic if possible, or just checking min/max manually later)
                
        # 4. Target Distribution Check
        if "target_rank" in schema_cols:
            print("\n[Target Distribution Check]")
            target_stats = ldf.select([
                pl.col("target_rank").min().alias("min"),
                pl.col("target_rank").max().alias("max"),
                pl.col("target_rank").mean().alias("mean")
            ]).collect()
            
            min_val = target_stats['min'][0]
            max_val = target_stats['max'][0]
            mean_val = target_stats['mean'][0]
            
            print(f"Target Rank (0~1): Min={min_val:.4f}, Max={max_val:.4f}, Mean={mean_val:.4f}")
            
            if min_val is not None and max_val is not None:
                if not (0 <= min_val <= 0.1) or not (0.9 <= max_val <= 1.0):
                    print("⚠️  Target Rank scaling looks suspicious (should be roughly 0 to 1).")
                
        print("\n[Verification Summary]")
        if all_passed:
            print("✅ All expected feature groups exist.")
        else:
            print("⚠️  Some features are missing or problematic.")

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
