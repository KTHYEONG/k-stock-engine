import unittest
import polars as pl
from legacy.stock_yetirank_v1.filters.quality import apply_quality_filter
from legacy.stock_yetirank_v1.filters.liquidity import apply_liquidity_filter
from legacy.stock_yetirank_v1.filters.volatility import apply_volatility_filter
from legacy.stock_yetirank_v1.filters.pipeline import UniverseFilter

class TestLayer1Filters(unittest.TestCase):
    
    def setUp(self):
        # Dummy Data 생성
        self.df = pl.DataFrame({
            "ticker": ["000010", "000020", "000030", "000040", "000050"],
            "date": ["2026-01-01"] * 5,
            
            # Quality Cols
            "capital_erosion_rate": [0.0, 20.0, 0.0, 0.0, 60.0],
            "operating_income": [1.0, 1.0, -1.0, 1.0, 1.0],
            
            # Liquidity Cols
            "adtv_20d": [50e9, 50e9, 50e9, 50e9, 1e9],
            "close": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
            
            # Volatility Cols
            "min_vol_5d": [1000, 1000, 1000, 0, 1000],
        })
        
    def test_quality_filter(self):
        filtered = apply_quality_filter(self.df)
        tickers = filtered["ticker"].to_list()
        
        # A002(자본잠식률 > 0), A003(영업손실), A005(자본잠식률 > 0): Fail
        # A004는 재무 건전성 필터를 통과한다.
        assert tickers == ["000010", "000040"]
        
    def test_liquidity_filter(self):
        filtered = apply_liquidity_filter(self.df)
        tickers = filtered["ticker"].to_list()
        
        # A005(ADTV < 50억): Fail
        assert tickers == ["000010", "000020", "000030", "000040"]

    def test_volatility_filter(self):
        filtered = apply_volatility_filter(self.df)
        tickers = filtered["ticker"].to_list()
        
        # A004(최근 5거래일 최소 거래량 0): Fail
        assert "000010" in tickers
        assert "000040" not in tickers
        
    def test_pipeline(self):
        # 모든 필터 통과해야 하는 종목: A001만 남아야 함?
        # A002: Quality Fail
        # A003: Quality Fail
        # A004: Volatility Fail
        # A005: Quality/Liquidity Fail
        
        pipeline = UniverseFilter()
        result = pipeline.apply_all(self.df)
        tickers = result["ticker"].to_list()
        
        assert len(tickers) == 1
        assert tickers[0] == "000010"

if __name__ == '__main__':
    unittest.main()
