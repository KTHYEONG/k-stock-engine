
import numpy as np
from numba import njit, float64, int64
from typing import Tuple

@njit(fastmath=True, cache=True)
def backtest_etf_numba(
    rows: float64[:, :], 
    params_sl_atr: float64,
    params_tp_atr: float64,
    params_lev_hurst: float64, 
    params_lev_natr: float64,
    params_ts_trigger: float64,
    params_ts_dist: float64,
    fee_rate: float64 = 0.0003
) -> Tuple[float64, float64, float64, int64, int64, float64, float64, float64[:, :], float64[:]]:
    """
    Revised Numba Backtester: Open-High-Low-Close (OHLC) Reality Check
    
    Data Schema (rows) - 20 Columns:
    0: signal_trigger (T-1)
    1: risk_mult (T-1)
    2: natr (T-1)
    3: hurst (T-1)
    
    [Assets: 0=B1, 1=I1, 2=B2, 3=I2] -> Offsets: 4, 8, 12, 16
    For each asset i (Base col = 4 + i*4):
    +0: Gap Return (Prev Close -> Open)
    +1: Open to High %
    +2: Open to Low %
    +3: Open to Close %
    """
    n = len(rows)
    
    # --- State ---
    position = 0 # 0: None, 1: Bull, -1: Bear
    held_asset_idx = -1 # 0:B1, 1:I1, 2:B2, 3:I2 (Internal 0-3 index)
    
    equity = 1.0
    peak_equity = 1.0 # Global peak for MDD
    mdd = 0.0
    
    # Trade State
    trade_entry_equity = 0.0     # Equity at entry (after fee)
    trade_peak_pnl = 0.0         # Max PnL during trade (for Trailing)
    entry_natr = 0.0
    
    # Metrics
    trades = 0
    wins = 0
    total_gross_win = 0.0
    total_gross_loss = 0.0
    
    trade_rets = np.zeros(10000, dtype=np.float64)
    daily_rets = np.zeros(n, dtype=np.float64) # Log equity change per day
    
    prev_equity_for_daily = 1.0
    
    for i in range(n):
        # 1. Update with Gap (Prev Close -> Current Open)
        # ---------------------------------------------
        gap_pnl = 0.0
        
        if position != 0:
            # Get Gap Return of held asset
            # Asset Base Index: 4 + held_asset_idx * 4
            base_col = 4 + held_asset_idx * 4
            gap_ret = rows[i, base_col] # Gap
            
            # Apply Gap
            equity *= (1.0 + gap_ret)
            gap_pnl = gap_ret
        
        # 2. Trading Decision (At Open)
        # ---------------------------------------------
        # We trade at OPEN. So existing position is closed at Open (already gapped), 
        # new position opened at Open.
        
        sig = int(rows[i, 0])
        curr_natr = rows[i, 2] # T-1
        curr_hurst = rows[i, 3] # T-1
        risk = rows[i, 1]
        
        just_closed = False
        
        # Check Signal Switch or Risk Management Exit (from gap)
        # Note: If gap caused massive loss exceeding SL, we might theoretically exit at Open.
        # But let's handle "Signal Switch" as priority at Open.
        
        if sig != position:
            # A. Close Existing
            if position != 0:
                # Sell at Open price
                # Fee
                equity *= (1.0 - fee_rate)
                
                # Record Trade
                tr_ret = (equity - trade_entry_equity) / trade_entry_equity
                if trades < 10000: trade_rets[trades] = tr_ret
                trades += 1
                
                if tr_ret > 0:
                    wins += 1
                    total_gross_win += (equity - trade_entry_equity)
                else:
                    total_gross_loss += abs(equity - trade_entry_equity)
                
                position = 0
                held_asset_idx = -1
                trade_peak_pnl = 0.0
                just_closed = True
            
            # B. Open New (if valid signal)
            if sig != 0:
                # Select Asset
                # Bull(1): B1(0) or B2(2)
                # Bear(-1): I1(1) or I2(3)
                use_2x = (curr_hurst > params_lev_hurst) and (curr_natr < params_lev_natr)
                
                if sig == 1:
                    held_asset_idx = 2 if use_2x else 0
                else:
                    held_asset_idx = 3 if use_2x else 1
                
                # Fee (Buy at Open)
                equity *= (1.0 - fee_rate) # Simplified: reducing equity instead of units
                
                # Init Trade State
                position = sig
                trade_entry_equity = equity
                entry_natr = curr_natr
                trade_peak_pnl = 0.0 # Reset
                
                # Risk Sizing (Simplified: Leverage handled by asset selection, risk_mult used for cash/equity mix?)
                # Current logic assumes 100% equity invested in ETF selected. 
                # If risk < 1.0, we simulate by simply holding cash? 
                # For this specific ETF logic, we assume full exposure to the chosen ETF 
                # (since 'risk_mult' was 1.0 mostly, or controlled by asset choice)
                # TODO: Implement partial exposure later if needed.
                
                just_closed = False # New trade is active now

        # 3. Intraday Monitor (Open -> High/Low)
        # ---------------------------------------------
        if position != 0 and (not just_closed): 
            # We are holding an asset through the day
            base_col = 4 + held_asset_idx * 4
            
            # Intraday moves relative to Open
            # NOTE: These are returns: (H-O)/O, (L-O)/O
            o_h = rows[i, base_col + 1]
            o_l = rows[i, base_col + 2]
            
            # Logic:
            # For Long ETF (Bull or Bear ETF are both "bought"):
            # We want Price Up. High is good, Low is bad.
            # (Bear ETF goes up when Index goes down, but here we hold the ETF itself, so we want ETF price UP)
            
            # Current PnL relative to Open
            # But SL is based on Trade Entry.
            # We need to approximate Trade PnL.
            # Trade PnL ~= (Current_Price - Entry_Price) / Entry_Price
            # Current_Price = Open * (1 + move)
            # Entry_Price = Open_at_entry
            
            # Precise: trade_pnl = (equity * (1+move) - trade_entry_equity) / trade_entry_equity
            
            # Thresholds
            real_natr = entry_natr if entry_natr > 0 else 1.0
            
            # SL/TP Percentages
            sl_pct = (params_sl_atr * real_natr) / 100.0
            tp_pct = (params_tp_atr * real_natr) / 100.0
            
            # Calculate potential PnL at Low and High
            # Worst case (Low) check first for SL
            pnl_at_low = (equity * (1.0 + o_l) - trade_entry_equity) / trade_entry_equity
            pnl_at_high = (equity * (1.0 + o_h) - trade_entry_equity) / trade_entry_equity
            
            # Trailing State Update (Virtual High)
            if pnl_at_high > trade_peak_pnl:
                trade_peak_pnl = pnl_at_high
            
            # --- Stop Logic ---
            executed_exit = False
            exit_pnl_pct = 0.0
            
            # A. Break-even check
            be_trigger = (3.0 * real_natr) / 100.0
            active_sl_pct = sl_pct
            if trade_peak_pnl > be_trigger:
                active_sl_pct = -0.001 # Profit protect (0.1% profit)
                # Note: active_sl_pct is used as: if pnl < -active_sl_pct (for normal SL)
                # For BE: if pnl < +0.1% (triggered)
                # Let's standardize: limit_pnl = -sl_pct normally.
                # If BE active, limit_pnl = +0.001
            
            # Standard Limit
            if trade_peak_pnl > be_trigger:
                limit_pnl = 0.001
            else:
                limit_pnl = -sl_pct
            
            # B. Trailing Stop
            ts_trigger = (params_ts_trigger * real_natr) / 100.0
            ts_dist = (params_ts_dist * real_natr) / 100.0
            ts_limit = trade_peak_pnl - ts_dist
            
            # Check Failure (SL or TS)
            # We check if Low touched the limit
            
            fail_triggered = False
            trigger_price_ret = 0.0 # Return relative to Open to exit at
            
            # 1. Check Standard SL / BreakEven
            if pnl_at_low < limit_pnl:
                fail_triggered = True
                # Execution Price?
                # If gap already killed it, we exit at Open (handled above? No, we are in intraday).
                # If Open was above SL, but Low is below SL -> Exited at SL.
                # solve: (equity*(1+ret) - entry)/entry = limit
                # equity*(1+ret) = entry*(1+limit)
                # 1+ret = (entry/equity)*(1+limit)
                # ret = (entry/equity)*(1+limit) - 1
                required_ret_from_open = (trade_entry_equity / equity) * (1.0 + limit_pnl) - 1.0
                
                # But can we execute at SL? Yes, unless gap skipped it.
                # If Open was already below SL, we should have exited at Open?
                # Gap check:
                current_pnl_at_open = (equity - trade_entry_equity) / trade_entry_equity
                if current_pnl_at_open < limit_pnl:
                    # Gapped below SL. Exit at Open.
                    trigger_price_ret = 0.0 # Open
                else:
                    # Hit during day
                    trigger_price_ret = required_ret_from_open
            
            # 2. Check Trailing Stop
            if (not fail_triggered) and (trade_peak_pnl > ts_trigger):
                if pnl_at_low < ts_limit:
                    fail_triggered = True
                    required_ret_from_open = (trade_entry_equity / equity) * (1.0 + ts_limit) - 1.0
                    
                    # Gap Check TS
                    current_pnl_at_open = (equity - trade_entry_equity) / trade_entry_equity
                    if current_pnl_at_open < ts_limit:
                        trigger_price_ret = 0.0
                    else:
                        trigger_price_ret = required_ret_from_open

            # 3. Check TP
            # If High > TP
            success_triggered = False
            if pnl_at_high > tp_pct:
                success_triggered = True
                # Exit at TP
                # Gap check
                current_pnl_at_open = (equity - trade_entry_equity) / trade_entry_equity
                if current_pnl_at_open > tp_pct:
                    trigger_price_ret = 0.0 # Exit at Open (Gap up)
                else:
                    required_ret_from_open = (trade_entry_equity / equity) * (1.0 + tp_pct) - 1.0
                    trigger_price_ret = required_ret_from_open
            
            # Priority: If both SL and TP touched? 
            # Pessimistic: SL first.
            if fail_triggered:
                # Execution
                equity *= (1.0 + trigger_price_ret)
                executed_exit = True
            elif success_triggered:
                equity *= (1.0 + trigger_price_ret)
                executed_exit = True
                
            if executed_exit:
                equity *= (1.0 - fee_rate)
                
                tr_ret = (equity - trade_entry_equity) / trade_entry_equity
                if trades < 10000: trade_rets[trades] = tr_ret
                trades += 1
                
                if tr_ret > 0:
                    wins += 1
                    total_gross_win += (equity - trade_entry_equity)
                else:
                    total_gross_loss += abs(equity - trade_entry_equity)
                
                position = 0
                held_asset_idx = -1
                trade_peak_pnl = 0.0
                just_closed = True
                
        # 4. Close (Overnight)
        # ---------------------------------------------
        if position != 0 and (not just_closed):
            base_col = 4 + held_asset_idx * 4
            oc_ret = rows[i, base_col + 3] # Open to Close
            
            equity *= (1.0 + oc_ret)
            # End of day check logic could go here, but next loop starts with Gap
            
        # 5. Daily Stats
        # ---------------------------------------------
        daily_ret = (equity - prev_equity_for_daily) / prev_equity_for_daily
        daily_rets[i] = daily_ret
        prev_equity_for_daily = equity
        
        # MDD
        if equity > peak_equity:
            peak_equity = equity
        dd = (equity - peak_equity) / peak_equity
        if dd < mdd:
            mdd = dd
            
    # Params for output
    duration_years = n / 252.0
    cagr = 0.0
    if duration_years > 0 and equity > 0:
        cagr = (equity ** (1.0 / duration_years)) - 1.0
        
    win_rate = 0.0
    if trades > 0:
        win_rate = (wins / trades) * 100.0
        
    pf = 0.0
    if total_gross_loss > 0:
        pf = total_gross_win / total_gross_loss
    elif total_gross_win > 0:
        pf = 99.0
        
    final_trade_rets = trade_rets[:trades].reshape(1, trades)
    
    return cagr, mdd, equity - 1.0, trades, wins, win_rate, pf, final_trade_rets, daily_rets
