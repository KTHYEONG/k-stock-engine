import numpy as np
from numba import njit, float64, int64
from typing import Tuple

@njit(nogil=True, cache=True)
def backtest_etf_numba(
    arr: float64[:, :],
    ibs_exit: float64,
    max_hold_days: int64,
    stop_loss_pct: float64,
    initial_balance: float64,
    fee_rate: float64
):
    """
    arr cols:
    0: sig (T-1) -> 1 for Bull, -1 for Bear
    1: idx_ibs (T)
    2: idx_close (T)
    3: b1_open
    4: b1_high
    5: b1_low
    6: b1_close
    7: i1_open
    8: i1_high
    9: i1_low
    10: i1_close
    """
    n = len(arr)
    balance = initial_balance
    equity_curve = np.zeros(n)
    
    in_position = False
    asset_type = 0 # 1=Bull, -1=Bear
    entry_price = 0.0
    entry_idx = 0
    amount = 0.0
    entry_fee_stored = 0.0
    
    max_trades = 10000
    trades = np.zeros((max_trades, 8))
    trade_count = 0
    
    for i in range(n):
        c_sig = arr[i, 0]
        c_idx_ibs = arr[i, 1]
        c_idx_close = arr[i, 2]
        
        prev_idx_close = arr[i-1, 2] if i > 0 else c_idx_close
        
        b1_o, b1_h, b1_l, b1_c = arr[i, 3], arr[i, 4], arr[i, 5], arr[i, 6]
        i1_o, i1_h, i1_l, i1_c = arr[i, 7], arr[i, 8], arr[i, 9], arr[i, 10]
        
        bar_processed = False
        
        # 1. Check Exit
        if in_position:
            if asset_type == 1:
                c_o, c_h, c_l, c_c = b1_o, b1_h, b1_l, b1_c
            else:
                c_o, c_h, c_l, c_c = i1_o, i1_h, i1_l, i1_c
                
            hold_days = i - entry_idx
            
            exit_triggered = False
            exit_price = 0.0
            
            # Hard Stop (Intraday)
            stop_price = entry_price * (1.0 - stop_loss_pct)
            if c_o <= stop_price:
                exit_price = c_o
                exit_triggered = True
            elif c_l <= stop_price:
                exit_price = stop_price
                exit_triggered = True
            else:
                # Time Stop
                if hold_days >= max_hold_days:
                    exit_price = c_c
                    exit_triggered = True
                
                # Take Profit 1: Intraday Strength (IBS)
                elif asset_type == 1 and c_idx_ibs >= ibs_exit:
                    exit_price = c_c
                    exit_triggered = True
                elif asset_type == -1 and c_idx_ibs <= (1.0 - ibs_exit):
                    exit_price = c_c
                    exit_triggered = True
                    
            if exit_triggered:
                pnl = (exit_price - entry_price) * amount
                exit_fee = amount * exit_price * fee_rate
                pnl -= exit_fee
                balance += (amount * entry_price) + pnl
                
                if trade_count < max_trades:
                    trades[trade_count] = [entry_idx, i, asset_type, entry_price, exit_price, pnl, amount, entry_fee_stored]
                    trade_count += 1
                in_position = False
                bar_processed = True
                
        # 2. Check Entry
        if not in_position and not bar_processed and c_sig != 0:
            target_capital = balance * 0.99 # 99% 투입 (수수료 및 슬리피지 여유)
            
            if c_sig == 1 and b1_o > 0:
                fill_price = b1_o
                amount = target_capital / fill_price
                entry_fee = target_capital * fee_rate
                balance -= (target_capital + entry_fee)
                entry_fee_stored = entry_fee
                
                in_position = True
                asset_type = 1
                entry_price = fill_price
                entry_idx = i
                
            elif c_sig == -1 and i1_o > 0:
                fill_price = i1_o
                amount = target_capital / fill_price
                entry_fee = target_capital * fee_rate
                balance -= (target_capital + entry_fee)
                entry_fee_stored = entry_fee
                
                in_position = True
                asset_type = -1
                entry_price = fill_price
                entry_idx = i

        # Update Equity
        if in_position:
            if asset_type == 1:
                c_c = b1_c
            else:
                c_c = i1_c
            unrealized = (c_c - entry_price) * amount
            equity_curve[i] = balance + (amount * entry_price) + unrealized
        else:
            equity_curve[i] = balance
            
    # Force close at end
    if in_position and n > 0:
        last_idx = n - 1
        if asset_type == 1:
            c_c = arr[last_idx, 6]
        else:
            c_c = arr[last_idx, 10]
        exit_price = c_c
        pnl = (exit_price - entry_price) * amount
        exit_fee = amount * exit_price * fee_rate
        pnl -= exit_fee
        balance += (amount * entry_price) + pnl
        if trade_count < max_trades:
            trades[trade_count] = [entry_idx, last_idx, asset_type, entry_price, exit_price, pnl, amount, entry_fee_stored]
            trade_count += 1
            
    return trades[:trade_count], balance, equity_curve
