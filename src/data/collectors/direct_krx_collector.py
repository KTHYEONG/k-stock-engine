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
        self.api_url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
        self.base_url = "http://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd?menuId=MDC0201020303"
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": self.base_url,
        }
        self.session.headers.update(self.headers)
        self._is_initialized = False

    def _initialize_session(self):
        """outerLoader 페이지를 방문하여 유효한 세션 쿠키 획득"""
        if self._is_initialized:
            return
        try:
            self.session.get(self.base_url, timeout=10)
            self._is_initialized = True
        except Exception as e:
            print(f"DirectKRXCollector session initialization failed: {e}")

    def get_net_purchases_by_date(self, date_str, market="ALL", investor="FOREIGNER"):
        """
        Fetch net purchases for a specific date and investor type.
        
        Args:
            date_str (str): YYYYMMDD
            market (str): 'KOSPI', 'KOSDAQ', 'ALL'
            investor (str): 'FOREIGNER', 'INSTITUTION', 'INDIVIDUAL', 'PENSION'
            
        Returns:
            pd.DataFrame: DataFrame with ticker index and net buy columns
        """
        self._initialize_session()
        
        # Investor Code Mapping
        # 9000: Foreigner (외국인), 7050: Institution (기관합계), 8000: Individual (개인), 6000: Pension (연기금)
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
        
        try:
            response = self.session.post(self.api_url, data=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # KRX JSON structure typically has 'output' or 'OutBlock_1' list
            records = data.get("output") or data.get("OutBlock_1")
            
            if records:
                df = pd.DataFrame(records)
                
                # Standardize columns
                result_df = df[['ISU_SRT_CD', 'NETBID_TRDVAL', 'NETBID_TRDVOL']].copy()
                result_df = result_df.rename(columns={
                    'ISU_SRT_CD': 'ticker',
                    'NETBID_TRDVAL': 'net_buy_value',
                    'NETBID_TRDVOL': 'net_buy_volume'
                })
                
                # Convert numeric
                result_df['net_buy_value'] = pd.to_numeric(result_df['net_buy_value'].astype(str).str.replace(',', ''), errors='coerce')
                result_df['net_buy_volume'] = pd.to_numeric(result_df['net_buy_volume'].astype(str).str.replace(',', ''), errors='coerce')
                
                return result_df.set_index('ticker')
            
            return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching KRX data for {date_str} {investor}: {e}")
            return pd.DataFrame()


