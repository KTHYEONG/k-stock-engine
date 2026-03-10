import requests
import pandas as pd
import time

class DirectKRXCollector:
    """
    Directly scrapes KRX data for Investor Net Purchases (MDCSTAT02401).
    Bypasses pykrx to avoid library issues and allow custom parameter tuning.
    Uses outerLoader strategy to bypass login requirements.
    """
    def __init__(self):
        self.api_url = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
        self.base_url = "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd?menuId=MDC0201020303"
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd?menuId=MDC0201020303",
            "Origin": "https://data.krx.co.kr",
            "Host": "data.krx.co.kr",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Accept": "application/json, text/javascript, */*; q=0.01",
        }
        self.session.headers.update(self.headers)
        self._is_initialized = False

    def _initialize_session(self, force=False):
        """outerLoader 페이지를 방문하여 유효한 세션 쿠키 획득"""
        if self._is_initialized and not force:
            return
        try:
            self.session.get(self.base_url, timeout=10)
            self._is_initialized = True
        except Exception as e:
            print(f"DirectKRXCollector session initialization failed: {e}")

    def get_net_purchases_by_date(self, date_str, market="ALL", investor="FOREIGNER"):
        """
        특정 날짜와 투자자 유형에 대한 순매수 데이터 수집
        """
        # 시작 전 매번 세션 확인 (필요시)
        self._initialize_session()
        
        inv_code_map = {
            "FOREIGNER": "9000",
            "INSTITUTION": "7050",
            "INDIVIDUAL": "8000",
            "PENSION": "6000"
        }
        inv_code = inv_code_map.get(investor.upper())
        if not inv_code:
            raise ValueError(f"Unknown investor type: {investor}")

        mkt_map = {"KOSPI": "STK", "KOSDAQ": "KSQ", "ALL": "ALL"}
        mkt_id = mkt_map.get(market.upper(), "ALL")
        
        params = {
            "bld": "dbms/MDC/STAT/standard/MDCSTAT02401",
            "mktId": mkt_id,
            "invstTpCd": inv_code,
            "strtDd": date_str,
            "endDd": date_str,
            "share": "1",
            "money": "1",
            "csvxls_isNo": "false",
        }
        
        # 최대 2회 시도 (세션 만료 대응)
        for attempt in range(2):
            try:
                response = self.session.post(self.api_url, data=params, timeout=15)
                
                # Bad Request (400) 발생 시 세션 재초기화 후 재시도
                if response.status_code == 400 and attempt == 0:
                    self._initialize_session(force=True)
                    continue
                
                response.raise_for_status()
                data = response.json()
                records = data.get("output") or data.get("OutBlock_1")
                
                if records:
                    df = pd.DataFrame(records)
                    result_df = df[['ISU_SRT_CD', 'NETBID_TRDVAL', 'NETBID_TRDVOL']].copy()
                    result_df = result_df.rename(columns={
                        'ISU_SRT_CD': 'ticker',
                        'NETBID_TRDVAL': 'net_buy_value',
                        'NETBID_TRDVOL': 'net_buy_volume'
                    })
                    result_df['net_buy_value'] = pd.to_numeric(result_df['net_buy_value'].astype(str).str.replace(',', ''), errors='coerce')
                    result_df['net_buy_volume'] = pd.to_numeric(result_df['net_buy_volume'].astype(str).str.replace(',', ''), errors='coerce')
                    return result_df.set_index('ticker')
                
                return pd.DataFrame()
                    
            except Exception as e:
                if attempt == 0:
                    self._initialize_session(force=True)
                    continue
                print(f"Error fetching KRX data for {date_str} {investor}: {e}")
                return pd.DataFrame()
        
        return pd.DataFrame()


