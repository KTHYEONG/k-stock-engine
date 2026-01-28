"""KRX OpenAPI 기반 시장 데이터 수집기 (requests 직접 호출 방식)"""

import os
import logging
import requests
import pandas as pd
import polars as pl
from datetime import datetime
from typing import Optional
import time
import asyncio

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
        "ETF_TRADE": "etp/etf_bydd_trd",
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
        
        
        self._lock = asyncio.Lock()
        
        # Rate limiting & Usage tracking
        self.last_request_time = 0
        self.min_request_interval = 0.2  # 초당 최대 5회 요청
        self.daily_limit = 10000         # 공식 1일 한도
        self.request_count = 0           # 현재 세션 호출 횟수
        
        # Session 재사용 (Keep-Alive)
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        logger.info(f"KRX OpenAPI Collector initialized. Daily limit: {self.daily_limit}")
        
    async def _rate_limit(self):
        """API 요청 속도 및 일일 한도 제한 체크 (Async Safe)"""
        async with self._lock:
            if self.request_count >= self.daily_limit:
                logger.error(f"Daily API limit ({self.daily_limit}) reached! Stopping.")
                raise RuntimeError(f"KRX API daily limit of {self.daily_limit} reached.")

            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - elapsed)
            
            self.last_request_time = time.time()
            self.request_count += 1
            
            if self.request_count % 100 == 0:
                logger.info(f"API Usage: {self.request_count}/{self.daily_limit} calls made today.")
    
    async def _make_request(self, endpoint: str, params: dict, retry_count: int = 3) -> dict:
        """API 요청 실행 (Async + Retry for 429)"""
        for i in range(retry_count):
            await self._rate_limit()
            url = f"{self.BASE_URL}/{endpoint}"
            
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: self.session.get(url, params=params, timeout=30)
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Too Many Requests - 지수 함수적 대기 후 재시도
                    wait_time = (i + 1) * 2 
                    logger.warning(f"HTTP 429 detected. Waiting {wait_time}s and retrying ({i+1}/{retry_count})...")
                    await asyncio.sleep(wait_time)
                    continue
                elif response.status_code == 401:
                    raise Exception("401 Unauthorized: Invalid API key")
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                    
            except requests.RequestException as e:
                if i < retry_count - 1:
                    await asyncio.sleep(1)
                    continue
                logger.error(f"Request failed after {retry_count} retries: {e}")
                raise
        
        raise Exception(f"Failed to get response after {retry_count} retries (Last status: 429)")
        
    async def collect_stock_daily_trade(self, date_str: str, market: str = "ALL") -> pl.DataFrame:
        """
        주식 일별 거래 데이터를 수집 (Async)
        
        Args:
            date_str: "YYYYMMDD" 형식
            market: "KOSPI", "KOSDAQ", "ALL" (기본값)
            
        Returns:
            pl.DataFrame: 거래 데이터
        """
        logger.info(f"Collecting stock daily trade for {date_str}, market={market}")
        
        try:
            trade_df = await self._collect_trade_data(date_str, market)
            
            if trade_df.is_empty():
                return pl.DataFrame()
            
            logger.info(f"Successfully collected {len(trade_df)} stocks for {date_str}")
            return trade_df
            
        except Exception as e:
            logger.error(f"Failed to collect stock data for {date_str}: {e}")
            return pl.DataFrame()

    
    async def _collect_trade_data(self, date_str: str, market: str) -> pl.DataFrame:
        """KRX OpenAPI로 거래 데이터 수집 (내부 메서드, Async)"""
        try:
            records = []
            params = {"basDd": date_str}
            
            # KOSPI 데이터 수집
            if market in ["KOSPI", "ALL"]:
                try:
                    logger.info(f"Fetching KOSPI data for {date_str}...")
                    kospi_data = await self._make_request(self.ENDPOINTS["KOSPI_STOCK"], params)
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
                    kosdaq_data = await self._make_request(self.ENDPOINTS["KOSDAQ_STOCK"], params)
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

    async def collect_etf_daily_trade(self, date_str: str) -> pl.DataFrame:
        """
        ETF 일별 매매 데이터 수집 (Async)
        ENDPOINT: etp/etf_bydd_trd
        
        Args:
            date_str: "YYYYMMDD" 형식
            
        Returns:
            pl.DataFrame: ETF 거래 데이터
        """
        logger.info(f"Collecting ETF daily trade for {date_str}")
        
        try:
            params = {"basDd": date_str}
            
            # ETF 데이터 수집
            etf_data = await self._make_request(self.ENDPOINTS["ETF_TRADE"], params)
            records = etf_data.get("OutBlock_1", [])
            
            if not records:
                logger.warning(f"No ETF trade data for {date_str}")
                return pl.DataFrame()
            
            # Pandas로 변환
            df = pd.DataFrame(records)
            
            # 컬럼 매핑 (ETF API 응답 -> 표준명)
            # API 응답 필드는 주식과 유사하나 확인 필요. 통상적으로 ISU_CD, ISU_NM, TDD_CLSPRC 등 사용.
            # ETF API 필드 추정: ISU_CD, ISU_NM, TDD_CLSPRC, ACC_TRDVOL, NAV, etc.
            # 여기서는 주식과 공통된 필드 위주로 매핑하고 ETF 특화 필드(NAV 등)는 필요시 추가.
            column_mapping = self._get_stock_trade_column_mapping()
            # ETF 특화 컬럼 추가 매핑이 필요할 수 있음 (예: NAV -> nav)
            # 현재는 기본 가격 정보 위주로 매핑.
            
            df = df.rename(columns=column_mapping)
            
            # Date 컬럼 추가
            if "date" not in df.columns:
                date_obj = datetime.strptime(date_str, "%Y%m%d")
                df["date"] = date_obj
            
            # Polars로 변환
            pl_df = pl.from_pandas(df)
            
            # 타입 캐스팅
            pl_df = self._cast_types(pl_df)
            
            logger.info(f"Collected {len(pl_df)} ETFs for {date_str}")
            return pl_df
            
        except Exception as e:
            logger.error(f"Failed to collect ETF data for {date_str}: {e}")
            return pl.DataFrame()

    async def collect_stock_base_info(self, date_str: str, market: str = "ALL") -> pl.DataFrame:
        """
        주식 기본정보 수집 (종목명, 업종, 상장일 등) (Async)
        
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
                kospi_data = await self._make_request(self.ENDPOINTS["KOSPI_INFO"], params)
                kospi_records = kospi_data.get("OutBlock_1", [])
                for rec in kospi_records:
                    rec["MARKET"] = "KOSPI"
                    rec["BAS_DD"] = date_str  # 서버 응답에 없으므로 수동 주입
                records.extend(kospi_records)
            
            if market in ["KOSDAQ", "ALL"]:
                kosdaq_data = await self._make_request(self.ENDPOINTS["KOSDAQ_INFO"], params)
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
    
    async def collect_market_indices(self, date_str: str) -> pl.DataFrame:
        """
        시장 지수 데이터 수집 (KOSPI, KOSDAQ 지수) (Async)
        
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
            kospi_idx = await self._make_request(self.ENDPOINTS["KOSPI_INDEX"], params)
            kospi_records = kospi_idx.get("OutBlock_1", [])
            for rec in kospi_records:
                rec["INDEX_TYPE"] = "KOSPI"
            records.extend(kospi_records)
            
            # KOSDAQ 지수
            kosdaq_idx = await self._make_request(self.ENDPOINTS["KOSDAQ_INDEX"], params)
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
                # '-' 값이 들어있는 경우 '0'으로 치환 후 캐스팅
                casts.append(
                    pl.col(col).cast(pl.Utf8).str.replace("-", "0").cast(pl.Float64)
                )
        
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
