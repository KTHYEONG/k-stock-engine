import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import unittest
import polars as pl
from src.filters.quality import apply_quality_filter
from src.filters.liquidity import apply_liquidity_filter
from src.filters.volatility import apply_volatility_filter
from src.filters.pipeline import UniverseFilter

class TestLayer1Filters(unittest.TestCase):
    
    def setUp(self):
        # Dummy Data 생성
        self.df = pl.DataFrame({
            "ticker": ["A001", "A002", "A003", "A004", "A005"],
            "date": ["2026-01-01"] * 5,
            
            # Quality Cols
            "pbr": [1.2, 0.05, 0.0, None, 0.5],  # 002(Low PBR), 003(Zero), 004(Null) -> Should fail
            "capital_erosion_rate": [10.0, 20.0, 0.0, 0.0, 60.0], # 005(Erosion > 50) -> Should fail
            
            # Liquidity Cols
            "market_cap": [200e9, 200e9, 200e9, 200e9, 5e9], # 005(Low Cap < 100억) -> Should fail
            "turnover_ratio": [0.1, 0.1, 0.6, 0.1, 0.1], # 003(High Turnover > 0.5) -> Should fail
            "trading_value": [50e9, 50e9, 50e9, 50e9, 50e9],
            
            # Volatility Cols
            "volume": [1000, 1000, 1000, 0, 1000], # 004(Volume=0) -> Should fail
        })
        
    def test_quality_filter(self):
        filtered = apply_quality_filter(self.df)
        tickers = filtered["ticker"].to_list()
        
        # A002(PBR 0.05 < 0.1): Fail
        # A003(PBR 0.0): Fail
        # A004(PBR Null): Fail
        # A005(Erosion 60): Fail
        # Expected: Only A001
        self.assertIn("A001", tickers)
        self.assertNotIn("A002", tickers)
        self.assertNotIn("A003", tickers)
        self.assertNotIn("A005", tickers)
        
    def test_liquidity_filter(self):
        filtered = apply_liquidity_filter(self.df)
        tickers = filtered["ticker"].to_list()
        
        # A003(Turnover 0.6 > 0.5): Fail
        # A005(Cap 50억 < 100억): Fail
        # Expected: A001, A002, A004
        self.assertIn("A001", tickers)
        self.assertIn("A002", tickers)
        self.assertNotIn("A003", tickers)
        self.assertNotIn("A005", tickers)

    def test_volatility_filter(self):
        filtered = apply_volatility_filter(self.df)
        tickers = filtered["ticker"].to_list()
        
        # A004(Volume 0): Fail
        self.assertIn("A001", tickers)
        self.assertNotIn("A004", tickers)
        
    def test_pipeline(self):
        # 모든 필터 통과해야 하는 종목: A001만 남아야 함?
        # A002: Qual Fail
        # A003: Qual Fail, Liq Fail
        # A004: Qual Fail, Vol Fail
        # A005: Qual Fail, Liq Fail
        
        pipeline = UniverseFilter()
        result = pipeline.apply_all(self.df)
        tickers = result["ticker"].to_list()
        
        self.assertEqual(len(tickers), 1)
        self.assertEqual(tickers[0], "A001")

if __name__ == '__main__':
    unittest.main()
