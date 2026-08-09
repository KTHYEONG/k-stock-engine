"""
Foreign/Institution Net Buy Data Collector
Collects daily investor net purchase data using DirectKRXCollector (bypassing pykrx).
"""

import logging
import pandas as pd
import polars as pl
from datetime import datetime
from src.data.collectors.direct_krx_collector import DirectKRXCollector

logger = logging.getLogger("data.collectors.investor")

class InvestorDataCollector:
    """
    Collector for Investor Net Buy Data (Foreigner, Institution, Individual, Pension).
    Uses DirectKRXCollector to fetch data from KRX interface directly.
    """
    
    def __init__(self):
        self.manual_collector = DirectKRXCollector()
        logger.info("InvestorDataCollector initialized using DirectKRXCollector")
        
    async def collect_daily_investor_net_buy(self, date_str: str, market: str = "ALL") -> pl.DataFrame:
        """
        Collects investor net buy data for a specific date across all tickers (Async).
        Merges Foreigner, Institution, Individual, and Pension data.
        """
        import asyncio
        logger.info(f"Collecting investor net buy data for {date_str}, market={market}")
        
        investor_map = [
            ("FOREIGNER", "foreign_net_buy"),
            ("INSTITUTION", "institution_net_buy"),
            ("INDIVIDUAL", "individual_net_buy"),
            ("PENSION", "pension_net_buy")
        ]
        
        async def fetch_one(inv_type, col_name):
            try:
                # DirectKRXCollector uses requests (blocking), so run in thread
                loop = asyncio.get_event_loop()
                df = await loop.run_in_executor(
                    None, 
                    lambda: self.manual_collector.get_net_purchases_by_date(date_str, market, inv_type)
                )
                if df.empty:
                    return None
                return df[['net_buy_value']].rename(columns={'net_buy_value': col_name})
            except Exception as e:
                logger.warning(f"Failed to fetch {inv_type} on {date_str}: {e}")
                return None

        # Parallelize 4 investor types
        tasks = [fetch_one(inv_type, col_name) for inv_type, col_name in investor_map]
        results = await asyncio.gather(*tasks)
        
        merged_df = None
        for res in results:
            if res is not None:
                if merged_df is None:
                    merged_df = res
                else:
                    merged_df = merged_df.join(res, how='outer')
        
        try:
            if merged_df is None or merged_df.empty:
                logger.error(f"Total investor data is empty for {date_str}")
                return pl.DataFrame()
            
            # Post-processing
            merged_df.index.name = 'ticker'
            merged_df = merged_df.reset_index()
            merged_df = merged_df.fillna(0)
            merged_df['date'] = datetime.strptime(date_str, "%Y%m%d")
            
            pl_df = pl.from_pandas(merged_df)
            pl_df = pl_df.with_columns(pl.col("ticker").cast(pl.Utf8))
            
            for _, col_name in investor_map:
                if col_name in pl_df.columns:
                    pl_df = pl_df.with_columns(pl.col(col_name).cast(pl.Int64))
            
            logger.info(f"Successfully collected investor data for {len(pl_df)} stocks")
            return pl_df
            
        except Exception as e:
            logger.error(f"Failed to process investor data for {date_str}: {e}")
            return pl.DataFrame()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = InvestorDataCollector()
    result = collector.collect_daily_investor_net_buy("20240102")
    print(result)
