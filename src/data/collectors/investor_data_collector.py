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
        
    def collect_daily_investor_net_buy(self, date_str: str, market: str = "ALL") -> pl.DataFrame:
        """
        Collects investor net buy data for a specific date across all tickers.
        Merges Foreigner, Institution, Individual, and Pension data.
        
        Args:
            date_str: "YYYYMMDD"
            market: "KOSPI", "KOSDAQ", "ALL" (default)
            
        Returns:
            pl.DataFrame: DataFrame with columns: 
                          [ticker, date, foreign_net_buy, institution_net_buy, individual_net_buy, pension_net_buy]
        """
        logger.info(f"Collecting investor net buy data for {date_str}, market={market}")
        
        # Define investors to collect and their target column names
        investor_map = [
            ("FOREIGNER", "foreign_net_buy"),
            ("INSTITUTION", "institution_net_buy"),
            ("INDIVIDUAL", "individual_net_buy"),
            ("PENSION", "pension_net_buy")
        ]
        
        merged_df = None
        
        try:
            for inv_type, col_name in investor_map:
                logger.debug(f"Fetching {inv_type} data...")
                df = self.manual_collector.get_net_purchases_by_date(date_str, market, inv_type)
                
                if df.empty:
                    logger.warning(f"No data for {inv_type} on {date_str}")
                    continue
                
                # df index is 'ticker', cols: 'net_buy_value', 'net_buy_volume'
                # We typically strive for 'Value' (Trading Value) in net buy analysis
                # Rename 'net_buy_value' to target column
                df_subset = df[['net_buy_value']].rename(columns={'net_buy_value': col_name})
                
                if merged_df is None:
                    merged_df = df_subset
                else:
                    # Outer join to include all tickers
                    merged_df = merged_df.join(df_subset, how='outer')
            
            if merged_df is None or merged_df.empty:
                logger.error(f"Total investor data is empty for {date_str}")
                return pl.DataFrame()
            
            # Post-processing
            merged_df.index.name = 'ticker'
            merged_df = merged_df.reset_index()
            
            # Fill NaNs with 0
            merged_df = merged_df.fillna(0)
            
            # Add date column
            merged_df['date'] = datetime.strptime(date_str, "%Y%m%d")
            
            # Convert to Polars
            pl_df = pl.from_pandas(merged_df)
            
            # Cast types
            # Ticker -> String, Net Buys -> Int64
            pl_df = pl_df.with_columns(pl.col("ticker").cast(pl.Utf8))
            
            for _, col_name in investor_map:
                if col_name in pl_df.columns:
                    pl_df = pl_df.with_columns(pl.col(col_name).cast(pl.Int64))
            
            logger.info(f"Successfully collected investor data for {len(pl_df)} stocks")
            return pl_df
            
        except Exception as e:
            logger.error(f"Failed to collect investor data for {date_str}: {e}")
            return pl.DataFrame()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = InvestorDataCollector()
    result = collector.collect_daily_investor_net_buy("20240102")
    print(result)
