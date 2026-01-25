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

    def _load_financials(self) -> pl.LazyFrame:
        if self._fund_df is None:
            if not self.financials_path.exists():
                logger.error(f"Financials not found at {self.financials_path}")
                return pl.LazyFrame()
            
            self._fund_df = pl.scan_parquet(self.financials_path)
            
            # capital -> capital_stock 매핑
            cols = self._fund_df.collect_schema().names()
            if "capital" in cols:
                self._fund_df = self._fund_df.with_columns(pl.col("capital").alias("capital_stock"))
            elif "capital_stock" not in cols:
                # 자본금이 없으면 자본잠식률 계산 불가 -> 임시 0 처리
                self._fund_df = self._fund_df.with_columns(pl.lit(None).cast(pl.Float64).alias("capital_stock"))

            # disclosure_date 컬럼 처리 (Lazy)
            if "disclosure_date" not in cols:
                if "date" in cols:
                    self._fund_df = self._fund_df.with_columns(pl.col("date").alias("disclosure_date"))
                elif "year" in cols:
                    self._fund_df = self._fund_df.with_columns(
                        (pl.col("year").cast(pl.Utf8) + "1231").alias("disclosure_date")
                    )

            self._fund_df = self._fund_df.with_columns(
                pl.col("disclosure_date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)
            )
        return self._fund_df

    def _load_indices(self, start_date: pl.Expr, end_date: pl.Expr) -> pl.LazyFrame:
        """KOSPI, KOSDAQ 지수 로드 및 MA120 계산 (Lazy)"""
        # FeatureStore.load_features가 이제 LazyFrame을 반환함
        idx_df = self.store.load_features() 
        
        # Filter only indices
        idx_df = idx_df.filter(pl.col("ticker").is_in(["KOSPI", "KOSDAQ"]))
            
        # MA120 및 Relative Basis 계산 (Lazy)
        idx_df = idx_df.sort(["ticker", "date"]).with_columns([
            pl.col("close").rolling_mean(window_size=120).over("ticker").alias("idx_ma120")
        ]).with_columns([
            (pl.col("close") / pl.col("idx_ma120")).alias("idx_relative_basis")
        ])
        
        return idx_df.select(["date", "ticker", "idx_relative_basis"])

    def process(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # Pre-check: Ensure date is Date (Lazy friendly cast)
        df = df.with_columns(pl.col("date").cast(pl.Date))

        # 1. PiT Financial Merge
        fund_data = self._load_financials()
        
        # 중복 컬럼 방지: 조인 전 fund_data에서 불필요한 year 제거
        # 이미 df에 year가 있으므로 financials의 year는 필요 없음
        fund_data = fund_data.drop("year") if "year" in fund_data.collect_schema().names() else fund_data
        
        # 반복 실행 시 기존에 생성된 _right 컬럼이 있다면 제거 (충돌 방지)
        cols_to_drop = [c for c in df.collect_schema().names() if c.endswith("_right")]
        if cols_to_drop:
            df = df.drop(cols_to_drop)

        df = df.sort("date")
        fund_data = fund_data.sort("disclosure_date")
        df = df.join_asof(
            fund_data,
            left_on="date",
            right_on="disclosure_date",
            by="ticker"
        )

        # 2. Market/Market Cap Column Normalization (Schema Consistency)
        schema_cols = df.collect_schema().names()
        
        # market 컬럼 정규화 (MARKET -> market)
        if "market" not in schema_cols:
            if "MARKET" in schema_cols:
                df = df.rename({"MARKET": "market"})
            else:
                # 최악의 경우 종목코드로 판단 (임시: 6자리면 KOSPI/KOSDAQ 가능성 높음)
                # 여기서는 안전하게 'KOSPI'로 기본값 부여 (지수 매칭용)
                df = df.with_columns(pl.lit("KOSPI").alias("market"))
        
        # market_cap 컬럼 정규화
        if "market_cap" not in schema_cols:
            if "MKTCAP" in schema_cols:
                df = df.with_columns(pl.col("MKTCAP").alias("market_cap"))
            elif "MarCap" in schema_cols:
                df = df.with_columns(pl.col("MarCap").alias("market_cap"))
            else:
                df = df.with_columns(pl.col("close").alias("market_cap"))

        # 3. Fundamental Metrics (Inverse Ratios)
        df = df.with_columns([
            (pl.col("total_equity") / pl.col("market_cap").replace(0, None)).alias("bp_ratio"),
            (pl.col("net_income") / pl.col("market_cap").replace(0, None)).alias("ep_ratio"),
            (pl.col("revenue") / pl.col("market_cap").replace(0, None)).alias("sp_ratio"),
            (pl.col("operating_income") / pl.col("market_cap").replace(0, None)).alias("op_ratio"),
            (pl.col("net_income") / pl.col("total_equity").replace(0, None)).alias("roe"),
            (pl.col("total_liabilities") / pl.col("total_equity").replace(0, None)).alias("debt_ratio"),
            # 자본잠식률: (자본금 - 자본총계) / 자본금
            ((pl.col("capital_stock") - pl.col("total_equity")) / pl.col("capital_stock").replace(0, None)).fill_null(0).alias("capital_erosion_rate")
        ])

        # 4. Relative Trend (Benchmark Match)
        # 지수 데이터 로드 (Lazy)
        idx_df = self._load_indices(None, None).with_columns(
            pl.col("idx_relative_basis").replace(0, None).fill_null(1.0) # 0/Null 방어
        )
        
        # Stock MA120 (Safe)
        df = df.sort(["ticker", "date"]).with_columns([
            (pl.col("close") / pl.col("close").rolling_mean(window_size=120).over("ticker").replace(0, None)).fill_null(1.0).alias("disparity_120d")
        ])
        
        # Join benchmark index based on market
        df = df.join(
            idx_df,
            left_on=["date", "market"],
            right_on=["date", "ticker"],
            how="left"
        )
        
        # Relative Trend = Stock Basis / Index Basis
        df = df.with_columns([
            (pl.col("disparity_120d") / pl.col("idx_relative_basis").replace(0, None)).fill_null(1.0).alias("relative_trend_score")
        ]).drop("idx_relative_basis")

        return df
