import os
import polars as pl
import logging
import asyncio
from datetime import datetime
logger = logging.getLogger("data.collectors.market")

class MarketDataCollector:
    """KRX OpenAPI 직접 호출 기반 시장 데이터 수집기
    
    KRX OpenAPI를 직접 호출하여 시세 데이터를 수집하고,
    Naver Finance를 통해 섹터(업종) 정보를 보완합니다.
    """
    
    def __init__(self, use_openapi: bool = True):
        """
        Args:
            use_openapi: True면 KRX OpenAPI 사용 (환경변수 KRX_OPENAPI_KEY 필요)
        """
        self.openapi_collector = None
        self.naver_collector = None
        self.dart_collector = None
        
        # KRX OpenAPI 초기화
        if use_openapi and os.getenv("KRX_OPENAPI_KEY"):
            try:
                from .krx_openapi_collector import KRXOpenAPICollector
                self.openapi_collector = KRXOpenAPICollector()
                logger.info("Using KRX OpenAPI for data collection")
            except Exception as e:
                logger.error(f"Failed to initialize KRX OpenAPI: {e}.")
                raise RuntimeError("KRX OpenAPI initialization failed")
        else:
             raise ValueError("KRX_OPENAPI_KEY not found or use_openapi is False. This system now requires KRX OpenAPI.")
             
        # Naver Collector 초기화 (섹터 정보 수집용)
        try:
            from .naver_finance_collector import NaverFinanceCollector
            self.naver_collector = NaverFinanceCollector(concurrency=10)
        except Exception as e:
            logger.warning(f"Failed to initialize NaverFinanceCollector: {e}")

        # OpenDART Collector 초기화 (로컬 데이터 로드용)
        try:
            from .dart_collector import OpenDartCollector
            self.dart_collector = OpenDartCollector() # API Key는 .env에서 자동 로드
        except Exception as e:
            logger.warning(f"Failed to initialize OpenDartCollector: {e}")
            self.dart_collector = None

        # Investor Collector 초기화
        try:
            from .investor_data_collector import InvestorDataCollector
            self.investor_collector = InvestorDataCollector()
        except Exception as e:
            logger.warning(f"Failed to initialize InvestorDataCollector: {e}")
            self.investor_collector = None
            
        # [Cache] 섹터 정보 캐시
        self._sector_cache = None
        
    async def collect_daily_data(self, date_str: str) -> pl.DataFrame:
        """
        특정 날짜의 전 종목 시세 데이터를 수집 (Async)
        
        Args:
            date_str (str): "YYYYMMDD" 형식
            
        Returns:
            pl.DataFrame: 수집된 일별 스냅샷 데이터
        """
        import time
        overall_start = time.time()
        logger.info(f"Collecting market data for {date_str} via KRX OpenAPI...")
        
        if not self.openapi_collector:
             logger.error("OpenAPI collector not initialized.")
             return pl.DataFrame()

        try:
            # 1. 시세 데이터 수집 (KRX OpenAPI - Sync)
            krx_start = time.time()
            pl_df = self.openapi_collector.collect_stock_daily_trade(date_str, market="ALL")
            krx_end = time.time()
            
            if pl_df.is_empty():
                 logger.warning(f"OpenAPI returned empty data in {krx_end - krx_start:.2f}s.")
                 return pl.DataFrame()
                 
            logger.info(f"Collected {len(pl_df)} stocks via KRX OpenAPI in {krx_end - krx_start:.2f}s")
            
            # 2. 섹터(업종) 데이터 수집 (Naver Finance Crawling - Async)
            if self.naver_collector:
                try:
                    naver_start = time.time()
                    logger.info("Starting sector mapping collection via Naver Finance...")
                    
                    # 섹터 정보는 날짜에 관계 없이 현재 기준 최신 매핑을 가져옴
                    if self._sector_cache is not None:
                        sector_df = self._sector_cache
                    else:
                        sector_df = await self.naver_collector.collect_sector_mapping()
                        self._sector_cache = sector_df  # 캐싱
                    
                    naver_end = time.time()
                    logger.info(f"Sector collection finished in {naver_end - naver_start:.2f}s")
                    
                    if not sector_df.is_empty():
                        merge_start = time.time()
                        # 병합 (Left Join)
                        pl_df = pl_df.join(sector_df, on="ticker", how="left")
                        
                        # 섹터가 없는 경우 'Unknown' 처리
                        pl_df = pl_df.with_columns(pl.col("sector").fill_null("Unknown"))
                                
                        logger.info(f"Merged sector data successfully in {time.time() - merge_start:.2f}s")
                    else:
                        logger.warning("Sector mapping returned empty. Filling with 'Unknown'.")
                        pl_df = pl_df.with_columns(pl.lit("Unknown").alias("sector"))
                        
                except Exception as e:
                    logger.error(f"Failed to collect sectors via Naver: {e}")
                    if "sector" not in pl_df.columns:
                        pl_df = pl_df.with_columns(pl.lit("Unknown").alias("sector"))
            # 3. 재무 데이터 병합 및 지표 계산 (Local DART Data)
            if self.dart_collector:
                try:
                    dart_start = time.time()
                    # 로컬에 저장된 재무 데이터 로드 (시점 일치 반영)
                    financial_df = self.dart_collector.load_financial_data(as_of_date=date_str)
                    
                    if not financial_df.is_empty():
                        # [FIX] year 컬럼 충돌 방지 (재무 데이터의 year는 회계연도이므로 fiscal_year로 변경)
                        if "year" in financial_df.columns:
                            financial_df = financial_df.rename({"year": "fiscal_year"})

                        # 병합 (Left Join)
                        pl_df = pl_df.join(financial_df, on="ticker", how="left")
                        
                        # 지표 계산: PBR, PER, BPS, EPS, 자본잠식률 등
                        pl_df = self._calculate_financial_metrics(pl_df)
                        
                        logger.info(f"Merged DART financial data and calculated metrics in {time.time() - dart_start:.2f}s")
                    else:
                        logger.warning("Local financial data (DART) is empty. Please run financial collection first.")
                        pl_df = self._fill_placeholders(pl_df)
                except Exception as e:
                    logger.error(f"Failed to process DART data: {e}")
                    pl_df = self._fill_placeholders(pl_df)
            else:
                pl_df = self._fill_placeholders(pl_df)

            # 4. 투자자별 순매수 데이터 수집 및 병합 (KRX Direct Scraper)
            if self.investor_collector:
                try:
                    inv_start = time.time()
                    logger.info(f"Collecting investor net buy data for {date_str}...")
                    investor_df = self.investor_collector.collect_daily_investor_net_buy(date_str)
                    
                    if not investor_df.is_empty():
                        # date 컬럼 삭제 (이미 pl_df에 어울리는 date가 있음)
                        if "date" in investor_df.columns:
                            investor_df = investor_df.drop("date")
                            
                        # 병합 (Left Join)
                        pl_df = pl_df.join(investor_df, on="ticker", how="left")
                        
                        # 순매수 데이터 결측치 0 처리
                        pl_df = self._fill_investor_placeholders(pl_df)
                        
                        logger.info(f"Merged investor data in {time.time() - inv_start:.2f}s")
                    else:
                        logger.warning("Investor data is empty. Filling with 0.")
                        pl_df = self._fill_investor_placeholders(pl_df)
                except Exception as e:
                    logger.error(f"Failed to collect investor data: {e}")
                    pl_df = self._fill_investor_placeholders(pl_df)
            else:
                pl_df = self._fill_investor_placeholders(pl_df)

            logger.info(f"Total collection time for {date_str}: {time.time() - overall_start:.2f}s")
            return pl_df
        except Exception as e:
            logger.error(f"KRX OpenAPI failed: {e}")
            return pl.DataFrame()

    async def sync_all_indices(self, count: int = 3000) -> pl.DataFrame:
        """네이버 금융을 통해 KOSPI/KOSDAQ 지수 데이터를 일괄 수집 및 갱신"""
        if not self.naver_collector:
            logger.error("Naver collector not initialized.")
            return pl.DataFrame()
            
        logger.info(f"Syncing all indices (last {count} days)...")
        
        # 병렬 수집
        kospi_task = self.naver_collector.collect_index_data("KOSPI", count=count)
        kosdaq_task = self.naver_collector.collect_index_data("KOSDAQ", count=count)
        
        results = await asyncio.gather(kospi_task, kosdaq_task)
        
        df_list = [df for df in results if not df.is_empty()]
        if not df_list:
            return pl.DataFrame()
            
        combined_df = pl.concat(df_list)
        
        # 누락된 컬럼 placeholder 채우기 (기존 스키마 유지)
        combined_df = self._fill_placeholders(combined_df)
        
        return combined_df

    def collect_market_indices(self, date_str: str) -> pl.DataFrame:
        """KOSPI/KOSDAQ 지수 데이터 수집 및 표준화"""
        if not self.openapi_collector:
            return pl.DataFrame()
        try:
            df = self.openapi_collector.collect_market_indices(date_str)
            if df.is_empty(): return df
            mapping = {
                "BAS_DD": "date", "INDEX_TYPE": "ticker",
                "CLSPRC_IDX": "close", "OPNPRC_IDX": "open",
                "HGPRC_IDX": "high", "LWPRC_IDX": "low",
                "ACC_TRDVOL": "volume", "ACC_TRDVAL": "trading_value"
            }
            current_mapping = {k: v for k, v in mapping.items() if k in df.columns}
            df = df.rename(current_mapping)
            df = self.openapi_collector._cast_types(df)
            return df
        except Exception as e:
            logger.error(f"Failed to collect market indices: {e}")
            return pl.DataFrame()
            
    def _calculate_financial_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """수집된 재무/시세 데이터를 바탕으로 밸류에이션 및 건전성 지표 계산"""
        # 1. BPS (Book-value Per Share) = 자본총계 / 상장주식수
        if "total_equity" in df.columns and "shares_outstanding" in df.columns:
            df = df.with_columns([
                (pl.col("total_equity") / pl.col("shares_outstanding").replace(0, 1)).alias("bps_calc")
            ])
            # BPS가 없으면 계산값 사용
            df = df.with_columns(pl.col("bps_calc").alias("bps"))
            
        # 2. EPS (Earnings Per Share) = 당기순이익 / 상장주식수
        if "net_income" in df.columns and "shares_outstanding" in df.columns:
            df = df.with_columns([
                (pl.col("net_income") / pl.col("shares_outstanding").replace(0, 1)).alias("eps_calc")
            ])
            df = df.with_columns(pl.col("eps_calc").alias("eps"))

        # 3. PBR = Close / BPS
        if "close" in df.columns and "bps" in df.columns:
            df = df.with_columns([
                (pl.col("close") / pl.col("bps").replace(0, 1)).alias("pbr")
            ])
            
        # 4. PER = Close / EPS
        if "close" in df.columns and "eps" in df.columns:
            df = df.with_columns([
                (pl.col("close") / pl.col("eps").replace(0, 1)).alias("per")
            ])

        # 5. ROE (Return On Equity) = 당기순이익 / 자본총계 * 100
        if "net_income" in df.columns and "total_equity" in df.columns:
            df = df.with_columns([
                (pl.col("net_income") / pl.col("total_equity").replace(0, 1) * 100).alias("roe")
            ])

        # 6. 자본잠식률 = (자본금 - 자본총계) / 자본금 * 100
        if "capital" in df.columns and "total_equity" in df.columns:
            df = df.with_columns([
                ((pl.col("capital") - pl.col("total_equity")) / pl.col("capital").replace(0, 1) * 100).alias("capital_erosion_rate")
            ])

        # Null 값 0.0 처리
        metrics = ["per", "pbr", "bps", "eps", "roe", "capital_erosion_rate", "div"]
        for m in metrics:
            if m not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias(m))
            else:
                df = df.with_columns(pl.col(m).fill_null(0.0))
                
        return df

    def _fill_investor_placeholders(self, df: pl.DataFrame) -> pl.DataFrame:
        """순매수 데이터 컬럼을 0으로 초기화"""
        inv_cols = ["foreign_net_buy", "institution_net_buy", "individual_net_buy", "pension_net_buy"]
        for col in inv_cols:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0).cast(pl.Int64).alias(col))
        return df

    def _fill_placeholders(self, df: pl.DataFrame) -> pl.DataFrame:
        """기존 파이프라인 호환성을 위해 펀더멘털 컬럼을 0.0으로 초기화"""
        fund_cols = ["per", "pbr", "bps", "eps", "div", "roe", "capital_erosion_rate", "fiscal_year"]
        for col in fund_cols:
            if col not in df.columns:
                if col == "fiscal_year":
                    df = df.with_columns(pl.lit("0000").alias(col)) # 기본값 설정
                else:
                    df = df.with_columns(pl.lit(0.0).alias(col))
        return df


    def calculate_derived_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """수집 직후 계산 가능한 파생 지표 (회전율 등) 추가"""
        if df.is_empty():
            return df
            
        # Turnover Ratio = Volume / Shares Outstanding
        if "shares_outstanding" in df.columns and "volume" in df.columns:
            df = df.with_columns([
                (pl.col("volume") / pl.col("shares_outstanding").replace(0, 1)).alias("turnover_ratio")
            ])
            
        return df
