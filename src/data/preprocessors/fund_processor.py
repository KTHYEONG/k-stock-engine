import polars as pl
from src.data.preprocessors.base_processor import BaseProcessor
from src.data.feature_store import FeatureStore
from config.base import DATA_DIR
import logging

logger = logging.getLogger("preprocessors.fund")

class FundProcessor(BaseProcessor):
    """
    재무 지표 & 시장 상대 추세 (Fundamental & Relative Trend) 전처리기
    
    Features:
    1. Inverse Ratios: bp_ratio, ep_ratio, sp_ratio, op_ratio
    2. Stability: roe, debt_ratio
    3. Relative Trend: (Stock/MA120) / (Index/MA120) - Benchmark Mapping (KOSPI/KOSDAQ)
    """
    
    def __init__(self, financials_path=DATA_DIR / "financials.parquet"):
        self.financials_path = financials_path
        self._fund_df = None
        self.store = FeatureStore()

    def _load_financials(self) -> pl.DataFrame:
        if self._fund_df is None:
            if not self.financials_path.exists():
                logger.error(f"Financials not found at {self.financials_path}")
                return pl.DataFrame()
            
            self._fund_df = pl.read_parquet(self.financials_path)
            
            # disclosure_date 컬럼이 없으면 대체 컬럼 찾기
            if "disclosure_date" not in self._fund_df.columns:
                if "date" in self._fund_df.columns:
                     self._fund_df = self._fund_df.with_columns(pl.col("date").alias("disclosure_date"))
                elif "year" in self._fund_df.columns:
                     # year만 있는 경우 연말(1231)을 기준으로 날짜 생성
                     self._fund_df = self._fund_df.with_columns(
                        (pl.col("year").cast(pl.Utf8) + "1231").alias("disclosure_date")
                     )

            if "disclosure_date" in self._fund_df.columns:
                # date format: YYYYMMDD string -> Datetime
                self._fund_df = self._fund_df.with_columns(
                    pl.col("disclosure_date").cast(pl.Utf8).str.strptime(pl.Datetime, "%Y%m%d")
                )
        return self._fund_df

    def _load_indices(self, start_date, end_date) -> pl.DataFrame:
        """KOSPI, KOSDAQ 지수 로드 및 MA120 계산"""
        idx_df = self.store.load_features(start_date=start_date, end_date=end_date)
        # Filter only indices
        idx_df = idx_df.filter(pl.col("ticker").is_in(["KOSPI", "KOSDAQ"]))
        
        if idx_df.is_empty():
            return pl.DataFrame()
            
        # MA120 계산
        idx_df = idx_df.sort(["ticker", "date"]).with_columns([
            pl.col("close").rolling_mean(window_size=120).over("ticker").alias("idx_ma120")
        ])
        
        # relative_basis = close / ma120
        idx_df = idx_df.with_columns([
            (pl.col("close") / pl.col("idx_ma120")).alias("idx_relative_basis")
        ])
        
        return idx_df.select(["date", "ticker", "idx_relative_basis"])

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        # Pre-check: Ensure date is Datetime
        if df["date"].dtype == pl.Utf8:
            df = df.with_columns(pl.col("date").str.strptime(pl.Datetime, "%Y%m%d"))

        # 1. PiT Financial Merge
        fund_data = self._load_financials()
        if not fund_data.is_empty():
            df = df.sort("date")
            fund_data = fund_data.sort("disclosure_date")
            df = df.join_asof(
                fund_data,
                left_on="date",
                right_on="disclosure_date",
                by="ticker"
            )

        # 2. Fundamental Metrics (Inverse Ratios)
        df = df.with_columns([
            (pl.col("total_equity") / pl.col("market_cap").replace(0, None)).alias("bp_ratio"),
            (pl.col("net_income") / pl.col("market_cap").replace(0, None)).alias("ep_ratio"),
            (pl.col("revenue") / pl.col("market_cap").replace(0, None)).alias("sp_ratio"),
            (pl.col("operating_income") / pl.col("market_cap").replace(0, None)).alias("op_ratio"),
            (pl.col("net_income") / pl.col("total_equity").replace(0, None)).alias("roe"),
            (pl.col("total_liabilities") / pl.col("total_equity").replace(0, None)).alias("debt_ratio")
        ])

        # 3. Relative Trend (Benchmark Match)
        start_date = df["date"].min().strftime("%Y%m%d")
        end_date = df["date"].max().strftime("%Y%m%d")
        idx_df = self._load_indices(start_date, end_date)
        
        if not idx_df.is_empty():
            # Stock MA120 (already in TechProcessor as disparity_120d, but let's recalculate if not present)
            if "disparity_120d" not in df.columns:
                df = df.sort(["ticker", "date"]).with_columns([
                    (pl.col("close") / pl.col("close").rolling_mean(window_size=120).over("ticker")).alias("disparity_120d")
                ])
            
            # Join benchmark index based on market
            # market: KOSPI, KOSDAQ
            df = df.join(
                idx_df,
                left_on=["date", "market"],
                right_on=["date", "ticker"],
                how="left"
            )
            
            # Relative Trend = Stock Basis / Index Basis
            df = df.with_columns([
                (pl.col("disparity_120d") / pl.col("idx_relative_basis")).alias("relative_trend_score")
            ]).drop("idx_relative_basis")

        return df
