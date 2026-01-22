"""KRX OpenAPI 기반 시장 데이터 수집기 (requests 직접 호출 방식)"""

import os
import logging
import requests
import pandas as pd
import polars as pl
from datetime import datetime
from typing import Optional
import time

logger = logging.getLogger("data.collectors.krx_openapi")


class KRXOpenAPICollector:
    """KRX OpenAPI 직접 호출 데이터 수집기
    
    pykrx-openapi 라이브러리 대신 requests를 사용하여 직접 API 호출
    Python 3.9 환경에서도 작동하도록 설계
    """
    
    BASE_URL = "https://data-dbg.krx.co.kr/svc/apis"
    
    # 엔드포인트 중앙 관리 (사용자 제공 URL 및 최신 규격 반영)
    ENDPOINTS = {
        "KOSPI_STOCK": "sto/stk_bydd_trd",
        "KOSDAQ_STOCK": "sto/ksq_bydd_trd",
        "KOSPI_INDEX": "idx/kospi_dd_trd",
        "KOSDAQ_INDEX": "idx/kosdaq_dd_trd",
        "KRX_INDEX": "idx/krx_dd_trd",
        "BOND_INDEX": "idx/bon_dd_trd",
        "KOSPI_INFO": "sto/stk_isu_base_info",
        "KOSDAQ_INFO": "sto/ksq_isu_base_info",
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: KRX OpenAPI 키. None이면 환경변수 KRX_OPENAPI_KEY 사용
        """
        self.api_key = api_key or os.getenv("KRX_OPENAPI_KEY")
        
        if not self.api_key:
            raise ValueError("KRX_OPENAPI_KEY not found in environment variables")
        
        self.headers = {
            "AUTH_KEY": self.api_key,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        
        # Rate limiting & Usage tracking
        self.last_request_time = 0
        self.min_request_interval = 0.2  # 초당 최대 5회 요청
        self.daily_limit = 10000         # 공식 1일 한도
        self.request_count = 0           # 현재 세션 호출 횟수
        
        # Session 재사용 (Keep-Alive)
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        logger.info(f"KRX OpenAPI Collector initialized. Daily limit: {self.daily_limit}")
        
    def _rate_limit(self):
        """API 요청 속도 및 일일 한도 제한 체크"""
        if self.request_count >= self.daily_limit:
            logger.error(f"Daily API limit ({self.daily_limit}) reached! Stopping.")
            raise RuntimeError(f"KRX API daily limit of {self.daily_limit} reached.")

        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        self.last_request_time = time.time()
        self.request_count += 1
        
        if self.request_count % 100 == 0:
            logger.info(f"API Usage: {self.request_count}/{self.daily_limit} calls made today.")
    
    def _make_request(self, endpoint: str, params: dict) -> dict:
        """API 요청 실행
        
        Args:
            endpoint: API 엔드포인트 경로
            params: 요청 파라미터
            
        Returns:
            dict: JSON 응답
        """
        self._rate_limit()
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            # Session 사용으로 3Way Handshake 오버헤드 제거
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                raise Exception("401 Unauthorized: Invalid API key")
            elif response.status_code == 404:
                raise Exception(f"404 Not Found: Endpoint {endpoint} does not exist")
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
                
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
        
    def collect_stock_daily_trade(self, date_str: str, market: str = "ALL") -> pl.DataFrame:
        """
        주식 일별 거래 데이터를 수집
        
        Args:
            date_str: "YYYYMMDD" 형식
            market: "KOSPI", "KOSDAQ", "ALL" (기본값)
            
        Returns:
            pl.DataFrame: 거래 데이터
        """
        logger.info(f"Collecting stock daily trade for {date_str}, market={market}")
        
        try:
            trade_df = self._collect_trade_data(date_str, market)
            
            if trade_df.is_empty():
                return pl.DataFrame()
            
            logger.info(f"Successfully collected {len(trade_df)} stocks for {date_str}")
            return trade_df
            
        except Exception as e:
            logger.error(f"Failed to collect stock data for {date_str}: {e}")
            return pl.DataFrame()

    
    def _collect_trade_data(self, date_str: str, market: str) -> pl.DataFrame:
        """KRX OpenAPI로 거래 데이터 수집 (내부 메서드)"""
        try:
            records = []
            params = {"basDd": date_str}
            
            # KOSPI 데이터 수집
            if market in ["KOSPI", "ALL"]:
                try:
                    logger.info(f"Fetching KOSPI data for {date_str}...")
                    kospi_data = self._make_request(self.ENDPOINTS["KOSPI_STOCK"], params)
                    kospi_records = kospi_data.get("OutBlock_1", [])
                    
                    for rec in kospi_records:
                        rec["MARKET"] = "KOSPI"
                    records.extend(kospi_records)
                    logger.info(f"Collected {len(kospi_records)} KOSPI stocks")
                except Exception as e:
                    logger.warning(f"KOSPI data collection failed: {e}")
            
            # KOSDAQ 데이터 수집
            if market in ["KOSDAQ", "ALL"]:
                try:
                    logger.info(f"Fetching KOSDAQ data for {date_str}...")
                    kosdaq_data = self._make_request(self.ENDPOINTS["KOSDAQ_STOCK"], params)
                    kosdaq_records = kosdaq_data.get("OutBlock_1", [])
                    
                    for rec in kosdaq_records:
                        rec["MARKET"] = "KOSDAQ"
                    records.extend(kosdaq_records)
                    logger.info(f"Collected {len(kosdaq_records)} KOSDAQ stocks")
                except Exception as e:
                    logger.warning(f"KOSDAQ data collection failed: {e}")
            
            if not records:
                logger.warning(f"No stock trade data for {date_str}")
                return pl.DataFrame()
            
            # Pandas로 변환
            df = pd.DataFrame(records)
            
            # 컬럼명 매핑 (KRX API 필드명 -> 표준명)
            column_mapping = self._get_stock_trade_column_mapping()
            df = df.rename(columns=column_mapping)
            
            # Date 컬럼 추가
            if "date" not in df.columns:
                date_obj = datetime.strptime(date_str, "%Y%m%d")
                df["date"] = date_obj
            
            # Polars로 변환
            pl_df = pl.from_pandas(df)
            
            # 타입 캐스팅
            pl_df = self._cast_types(pl_df)
            
            # 필수 컬럼 검증
            if "ticker" not in pl_df.columns:
                logger.error("Critical column 'ticker' missing from KRX OpenAPI response. Returning empty.")
                return pl.DataFrame()
            
            return pl_df
            
        except Exception as e:
            logger.error(f"Trade data collection failed: {e}")
            return pl.DataFrame()

    def collect_stock_base_info(self, date_str: str, market: str = "ALL") -> pl.DataFrame:
        """
        주식 기본정보 수집 (종목명, 업종, 상장일 등)
        
        Args:
            date_str: "YYYYMMDD" 형식
            market: "KOSPI", "KOSDAQ", "ALL"
            
        Returns:
            pl.DataFrame: 종목 기본 정보
        """
        logger.info(f"Collecting stock base info for {date_str}, market={market}")
        
        try:
            records = []
            params = {"basDd": date_str}
            
            if market in ["KOSPI", "ALL"]:
                kospi_data = self._make_request(self.ENDPOINTS["KOSPI_INFO"], params)
                kospi_records = kospi_data.get("OutBlock_1", [])
                for rec in kospi_records:
                    rec["MARKET"] = "KOSPI"
                    rec["BAS_DD"] = date_str  # 서버 응답에 없으므로 수동 주입
                records.extend(kospi_records)
            
            if market in ["KOSDAQ", "ALL"]:
                kosdaq_data = self._make_request(self.ENDPOINTS["KOSDAQ_INFO"], params)
                kosdaq_records = kosdaq_data.get("OutBlock_1", [])
                for rec in kosdaq_records:
                    rec["MARKET"] = "KOSDAQ"
                    rec["BAS_DD"] = date_str  # 서버 응답에 없으므로 수동 주입
                records.extend(kosdaq_records)
            
            if not records:
                logger.warning(f"No stock base info for {date_str}")
                return pl.DataFrame()
            
            df = pd.DataFrame(records)
            
            # 컬럼 매핑
            column_mapping = self._get_stock_baseinfo_column_mapping()
            df = df.rename(columns=column_mapping)
            
            pl_df = pl.from_pandas(df)
            
            logger.info(f"Collected base info for {len(pl_df)} stocks")
            return pl_df
            
        except Exception as e:
            logger.error(f"Failed to collect stock base info: {e}")
            return pl.DataFrame()
    
    def collect_market_indices(self, date_str: str) -> pl.DataFrame:
        """
        시장 지수 데이터 수집 (KOSPI, KOSDAQ 지수)
        
        Args:
            date_str: "YYYYMMDD" 형식
            
        Returns:
            pl.DataFrame: 지수 데이터
        """
        logger.info(f"Collecting market indices for {date_str}")
        
        try:
            records = []
            params = {"basDd": date_str}
            
            # KOSPI 지수
            kospi_idx = self._make_request(self.ENDPOINTS["KOSPI_INDEX"], params)
            kospi_records = kospi_idx.get("OutBlock_1", [])
            for rec in kospi_records:
                rec["INDEX_TYPE"] = "KOSPI"
            records.extend(kospi_records)
            
            # KOSDAQ 지수
            kosdaq_idx = self._make_request(self.ENDPOINTS["KOSDAQ_INDEX"], params)
            kosdaq_records = kosdaq_idx.get("OutBlock_1", [])
            for rec in kosdaq_records:
                rec["INDEX_TYPE"] = "KOSDAQ"
            records.extend(kosdaq_records)
            
            if not records:
                logger.warning(f"No index data for {date_str}")
                return pl.DataFrame()
            
            df = pd.DataFrame(records)
            pl_df = pl.from_pandas(df)
            
            logger.info(f"Collected {len(pl_df)} index records")
            return pl_df
            
        except Exception as e:
            logger.error(f"Failed to collect market indices: {e}")
            return pl.DataFrame()
    
    def _get_stock_trade_column_mapping(self) -> dict:
        """KRX OpenAPI 응답 필드명을 표준 컬럼명으로 매핑"""
        return {
            "BAS_DD": "date",
            "ISU_CD": "ticker",
            "ISU_NM": "name",
            "MKT_NM": "market",

            "TDD_OPNPRC": "open",
            "TDD_HGPRC": "high", 
            "TDD_LWPRC": "low",
            "TDD_CLSPRC": "close",
            "ACC_TRDVOL": "volume",
            "ACC_TRDVAL": "trading_value",
            "MKTCAP": "market_cap",
            "LIST_SHRS": "shares_outstanding",
            "FLUC_RT": "fluc_rate",
            "CMPPREVDD_PRC": "change",
        }
    
    def _get_stock_baseinfo_column_mapping(self) -> dict:
        """종목 기본정보 컬럼 매핑"""
        return {
            "BAS_DD": "date",
            "ISU_SRT_CD": "ticker",        # 테스트 결과 존재 확인
            "ISU_ABBRV": "name",          # 테스트 결과 존재 확인
            "ISU_CD": "full_code",        # 테스트 결과 존재 확인 (표준코드)
            "MARKET": "market",           # 수동 주입 데이터
            "LIST_DD": "listing_date",    # 테스트 결과 존재 확인
            "KIND_STKCERT_TP_NM": "stock_type", # 주권구분 (테스트 결과 기반)
        }
    
    def _cast_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """Polars DataFrame 타입 캐스팅"""
        if df.is_empty():
            return df
        
        casts = []
        
        # 문자열 필드
        for col in ["ticker", "name", "full_code", "market"]:
            if col in df.columns:
                casts.append(pl.col(col).cast(pl.Utf8))
        
        # 숫자 필드
        for col in ["open", "high", "low", "close", "volume", "trading_value", 
                   "market_cap", "shares_outstanding", "fluc_rate"]:
            if col in df.columns:
                casts.append(pl.col(col).cast(pl.Float64))
        
        # 날짜 필드
        if "date" in df.columns and df["date"].dtype != pl.Datetime:
            if df["date"].dtype == pl.Utf8:
                casts.append(pl.col("date").str.strptime(pl.Datetime, "%Y%m%d"))
        
        if casts:
            df = df.with_columns(casts)
        
        return df
    
    def calculate_derived_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """파생 지표 계산 (회전율 등)"""
        if df.is_empty():
            return df
        
        # Turnover Ratio = Volume / Shares Outstanding
        if "shares_outstanding" in df.columns and "volume" in df.columns:
            df = df.with_columns([
                (pl.col("volume") / pl.col("shares_outstanding").fill_null(1).replace(0, 1))
                .alias("turnover_ratio")
            ])
        
        return df
