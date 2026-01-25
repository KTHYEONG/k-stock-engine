import polars as pl
from pathlib import Path
import sys

# 프로젝트 루트 경로 계산
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# 데이터 저장 경로 설정
file_path = PROJECT_ROOT / "data" / "processed" / "features" / "year=2024"

print(f"Checking data at: {file_path}")

if not file_path.exists():
    print(f"Error: Path not found! {file_path}")
else:
    try:
        # 방법 1: 스키마 이슈를 피하기 위해 가장 최근에 생성된 파티션 파일 하나만 읽기
        all_parquet_files = sorted(list(file_path.rglob("*.parquet")))
        
        if not all_parquet_files:
            print("No parquet files found in the directory.")
        else:
            target_file = all_parquet_files[-1] # 가장 최신 파일
            print(f"Reading sample file: {target_file.relative_to(PROJECT_ROOT)}")
            
            df = pl.read_parquet(target_file)

            # 피처 엔지니어링으로 추가된 피처 목록 정의
            engineered_features = {
                "Technical": ["log_return_1d", "log_return_5d", "log_return_20d", "log_return_60d", "log_return_120d", 
                             "volatility_20d", "volatility_60d", "disparity_5d", "disparity_20d", "disparity_60d", 
                             "disparity_120d", "volume_ratio_5d", "volume_ratio_20d", "volume_ratio_60d", 
                             "intraday_vol", "amihud_20d"],
                "Flow/Supply": ["net_purchase_total", "np_mkt_cap", "np_vol", "np_cum_60d", "z_flow"],
                "Fundamental": ["bp_ratio", "ep_ratio", "sp_ratio", "op_ratio", "roe", "debt_ratio", "relative_trend_score"],
                "Universe/Filter": ["avg_trading_value_5d", "sector_count", "vol_rank", "vol_percentile"],
                "Target": ["target_return_5d", "target_rank"]
            }

            print(f"\n[Engineered Features Check]")
            
            all_added = []
            for category, features in engineered_features.items():
                found = [f for f in features if f in df.columns]
                all_added.extend(found)
                print(f"- {category}: {len(found)}/{len(features)} found")
                if found:
                    print(f"  > {found}")

            # 추가된 피처들만 따로 보기 (데이터 샘플)
            print(f"\n[Sample Data: Only Engineered Features]")
            # 상위 5개 행에 대해 추가된 컬럼들만 출력
            if all_added:
                # 출력 편의를 위해 ticker와 date를 포함시켜 출력
                display_cols = ["date", "ticker"] + all_added[:10] # 너무 많으면 잘릴 수 있으니 10개만 우선 확인
                print(df.select(display_cols).head())
            else:
                print("No engineered features found in the current file.")

    except Exception as e:
        print(f"\nFailed to read data: {e}")
        print("\nTip: 만약 전체 데이터를 읽고 싶다면, 모든 날짜에 대해 feature_engineer.py를 실행하여 스키마를 통일시켜야 합니다.")