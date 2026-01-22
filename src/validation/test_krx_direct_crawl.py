import requests
import pandas as pd
import time
from datetime import datetime

class KRXDirectCollector:
    def __init__(self):
        self.session = requests.Session()
        # 로그인 우회를 위한 outerLoader 헤더 설정
        self.base_url = "http://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd?menuId=MDC0201020303"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": self.base_url,
        }
        self.session.headers.update(self.headers)
        self._initialize_session()

    def _initialize_session(self):
        """outerLoader 페이지를 방문하여 유효한 세션 쿠키를 획득"""
        try:
            print(f"세션 초기화 중... ({self.base_url})")
            self.session.get(self.base_url, timeout=10)
        except Exception as e:
            print(f"세션 초기화 실패: {e}")

    def get_investor_net_buy(self, date_str, market="ALL"):
        """
        특정 날짜의 투자자별(기관/외국인 등) 순매수 데이터를 수집합니다.
        
        Args:
            date_str (str): "YYYYMMDD"
            market (str): "ALL", "KOSPI", "KOSDAQ"
            
        Returns:
            pd.DataFrame: [티커, 종목명, 종가, 외국인순매수, 기관순매수] 등이 포함된 데이터프레임
        """
        
        # 투자자 코드 매핑 (MDCSTAT02401 기준)
        # 1000: 금융투자, 2000: 보험, 3000: 투신, 3100: 사모, 4000: 은행, 5000: 기타금융
        # 6000: 연기금, 7050: 기관합계, 8000: 개인, 9000: 외국인
        target_investors = {
            "기관합계": "7050",
            "외국인": "9000",
            "개인": "8000",
            "연기금": "6000"
        }
        
        market_map = {"ALL": "ALL", "KOSPI": "STK", "KOSDAQ": "KSQ"}
        mkt_id = market_map.get(market.upper(), "ALL")
        
        merged_df = None
        
        for inv_name, inv_code in target_investors.items():
            print(f"[{inv_name}({inv_code})] 데이터 수집 중...")
            df = self._fetch_net_buy_by_investor(date_str, mkt_id, inv_code)
            
            if df is None:
                continue
                
            # 컬럼명 정리: 'NETBID_TRDVAL' -> '{inv_name}_순매수'
            # 필요한 컬럼: ISU_SRT_CD(티커), ISU_NM(종목명), TDD_CLSPRC(종가), NETBID_TRDVAL(순매수대금)
            cols_map = {
                'ISU_SRT_CD': 'ticker',
                'ISU_NM': '종목명',
                'TDD_CLSPRC': '종가',
                'NETBID_TRDVAL': f'{inv_name}_순매수'
            }
            
            # 존재하는 컬럼만 선택
            avail_cols = [c for c in cols_map.keys() if c in df.columns]
            subset = df[avail_cols].rename(columns=cols_map)
            
            # 숫자형 변환 (천단위 콤마 제거)
            for col in subset.columns:
                if col not in ['ticker', '종목명']:
                    subset[col] = pd.to_numeric(subset[col].astype(str).str.replace(',', ''), errors='coerce')

            if merged_df is None:
                merged_df = subset
            else:
                # 티커 기준으로 병합 (종목명, 종가 등 중복 컬럼은 제거하거나 suffix 처리)
                # 데이터가 완전하다면 ticker를 인덱스로 놓고 병합하는게 유리
                subset = subset[['ticker', f'{inv_name}_순매수']]
                merged_df = pd.merge(merged_df, subset, on='ticker', how='outer')
            
            time.sleep(0.5) # 과도한 요청 방지
            
        return merged_df

    def _fetch_net_buy_by_investor(self, date_str, mkt_id, inv_code):
        url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
        
        params = {
            "bld": "dbms/MDC/STAT/standard/MDCSTAT02401", # 투자자별 순매수 상위(전체)
            "mktId": mkt_id,
            "invstTpCd": inv_code,
            "strtDd": date_str,
            "endDd": date_str,
            "share": "1",    # 1:주식수, 2:거래량 (여기선 거래대금 위주로 받을 것이라면 money=1 중요)
            "money": "1",    # 1:거래대금
            "csvxls_isNo": "false",
        }
        
        try:
            res = self.session.post(url, data=params, timeout=10)
            if res.status_code != 200:
                print(f"HTTP {res.status_code} Error")
                return None
                
            data = res.json()
            if "OutBlock_1" in data:
                return pd.DataFrame(data["OutBlock_1"])
            elif "output" in data:
                return pd.DataFrame(data["output"])
            else:
                return None
        except Exception as e:
            print(f"Error fetching {inv_code}: {e}")
            return None

# 실행 테스트
if __name__ == "__main__":
    collector = KRXDirectCollector()
    
    # 2026-01-21 (평일) 데이터 수집 시도
    target_date = "20260121" 
    print(f"Collecting Investor Data for {target_date}...")
    
    result_df = collector.get_investor_net_buy(target_date)
    
    if result_df is not None and not result_df.empty:
        print(f"\n[Result Overview] Rows: {len(result_df)}")
        print(result_df.head())
        
        # 삼성전자(005930) 데이터 확인
        samsung = result_df[result_df['ticker'] == '005930']
        if not samsung.empty:
            print("\n[삼성전자(005930) Data]")
            print(samsung.iloc[0])
    else:
        print("Failed to collect data.")
