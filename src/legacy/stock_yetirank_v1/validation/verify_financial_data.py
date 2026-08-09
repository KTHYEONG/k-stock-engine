import polars as pl
import os
from pathlib import Path
from datetime import datetime

def verify_financial_data():
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "financials.parquet"
    
    print("=" * 60)
    print(f" Financial Data Verification: {data_path}")
    print("=" * 60)
    
    if not data_path.exists():
        print(f"[ERROR] File not found: {data_path}")
        return

    # 1. Load Data
    try:
        df = pl.read_parquet(data_path)
    except Exception as e:
        print(f"[ERROR] Failed to read parquet: {e}")
        return

    # 2. Basic Info
    print(f"[*] Total Records: {len(df)}")
    print(f"[*] Total Columns: {len(df.columns)}")
    print(f"[*] Column Names: {df.columns}")
    
    # 3. Required Columns Check
    required_cols = [
        "ticker", "year", "reprt_code", 
        "total_assets", "total_equity", "capital", 
        "revenue", "operating_income", "net_income"
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[WARN] Missing Columns: {missing_cols}")
    else:
        print("[OK] All required columns present.")

    # 4. Data Distribution (Year & Report Code)
    print("\n[ Distribution by Year ]")
    year_dist = df.group_by("year").count().sort("year")
    print(year_dist)
    
    print("\n[ Distribution by Report Code ]")
    # 11013: 1Q, 11012: Half, 11014: 3Q, 11011: Annual
    rpt_dist = df.group_by("reprt_code").count().sort("reprt_code")
    print(rpt_dist)

    # 5. Integrity Check (Duplicates)
    duplicates = df.filter(pl.struct(["ticker", "year", "reprt_code"]).is_duplicated())
    if len(duplicates) > 0:
        print(f"\n[WARN] Found {len(duplicates)} duplicate records (ticker, year, reprt_code).")
    else:
        print("\n[OK] No duplicates found.")

    # 6. Quality Check (Nulls & Zeros)
    print("\n[ Data Quality (Missing/Zero Values) ]")
    quality_metrics = []
    for col in ["total_equity", "revenue", "net_income"]:
        if col in df.columns:
            null_count = df.select(pl.col(col).is_null().sum()).item()
            zero_count = df.filter(pl.col(col) == 0).height
            quality_metrics.append({
                "Column": col,
                "Nulls": null_count,
                "Zeros": zero_count,
                "Zero%": f"{(zero_count/len(df))*100:.2f}%"
            })
    print(pl.DataFrame(quality_metrics))

    # 7. Sample Check
    print("\n[ Sample Data (Top 5) ]")
    print(df.head(5))

    print("\n" + "=" * 60)
    print(" Verification Completed")
    print("=" * 60)

if __name__ == "__main__":
    verify_financial_data()
