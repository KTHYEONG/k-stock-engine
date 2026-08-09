
import sys
import os

# 프로젝트 루트 경로를 sys.path에 추가하여 src 모듈 임포트 가능하게 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # src/의 상위 (루트)가 아니라, verify_investor_data.py가 src/에 있다면 project_root는 c:/.../my_stock_traider
# src 폴더가 c:/.../src 구조이므로
# c:/.../src/verify_investor_data.py -> dirname -> c:/.../src
# dirname -> c:/.../my_stock_traider
# 이렇게 되어야 하는데, import src... 하려면 c:/.../my_stock_traider가 sys.path에 있어야 함.

sys.path.append(project_root)

from src.legacy.stock_yetirank_v1.data.collectors.investor_data_collector import InvestorDataCollector
import polars as pl


def verify_investor_data():
    log_file = os.path.join(project_root, "verification_result.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("🚀 Starting verification script...\n")
        
        try:
            collector = InvestorDataCollector()
            
            # 1. 테스트 날짜: 2024년 1월 2일 (개장일)
            target_date = "20240102"
            
            # 2. 데이터 수집
            f.write(f"Collecting data for {target_date}...\n")
            df = collector.collect_daily_investor_net_buy(target_date, market="KOSPI")
            
            if df.is_empty():
                f.write("❌ Error: Verification Failed. DataFrame is empty.\n")
                return
                
            f.write(f"✅ Data collected. Shape: {df.shape}\n")
            f.write(f"Columns: {df.columns}\n")
            
            # 3. 특정 종목(삼성전자 005930) 확인
            samsung_code = "005930"
            
            # Ticker 컬럼 이름 확인
            ticker_col = "ticker" if "ticker" in df.columns else "티커"
            
            samsung_data = df.filter(pl.col(ticker_col) == samsung_code)
            
            if samsung_data.is_empty():
                f.write(f"❌ Error: Samsung Electronics ({samsung_code}) not found in data.\n")
                f.write("Head of data:\n")
                f.write(str(df.head()) + "\n")
                return

            f.write("\n🔍 삼성전자 (005930) 데이터 확인:\n")
            f.write(str(samsung_data) + "\n")
            
            # 4. 데이터 값 검증
            required_cols = ["foreign_net_buy", "institution_net_buy"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                f.write(f"Warning: Expected columns {missing_cols} not found. Check column mapping.\n")
                f.write(f"Current columns: {df.columns}\n")
            else:
                f.write("✅ Essential columns (foreign_net_buy, institution_net_buy) exist.\n")
                
                # 값 출력
                f_net = samsung_data.select("foreign_net_buy").item()
                i_net = samsung_data.select("institution_net_buy").item()
                
                f.write(f"   - Foreign Net Buy: {f_net:,}\n")
                f.write(f"   - Institution Net Buy: {i_net:,}\n")
                
            f.write("\n✅ Verification Completed.\n")
            
        except Exception as e:
            f.write(f"❌ Exception occurred: {str(e)}\n")
            import traceback
            f.write(traceback.format_exc())

if __name__ == "__main__":
    verify_investor_data()
